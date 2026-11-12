"""Local language models (Qwen3 family) running through HF transformers.

One `LocalLLM` serves both uses of a language model in the system:
  * the **investigator** (log_agents/agents/investigator.py) asks it to *choose one option from a lettered menu*
    (`choose`): the answer is the option whose letter the model scores highest, so there is nothing to parse and
    no malformed output, which is what makes small models usable as agents;
  * the **chat** (log_agents/chat.py) asks it to write answers (`chat`, `stream_chat`).

Models are named like "Qwen/Qwen3-8B" or "Qwen/Qwen3-8B:nf4" (4-bit NF4 via bitsandbytes, so 4B and 8B fit an 8 GB
GPU). Weights are looked up in LAD_MODELS_DIR (plain folders, see scripts/download_models.py) and then in the
Hugging Face cache; nothing is downloaded implicitly (set LAD_ALLOW_DOWNLOAD=1 to allow it).
"""
import gc
import re
import threading
from pathlib import Path

import torch

from . import config


def parse_model_spec(spec):
    """'Qwen/Qwen3-8B' / 'Qwen/Qwen3-8B:nf4' -> (repo id, quantisation None | 'nf4')."""
    repo, _, quant = spec.partition(":")
    if not quant:
        quant = config.MODEL_QUANT.get(repo)
    return repo, (None if quant in (None, "", "bf16", "none") else quant)


def local_dir(repo):
    p = config.MODELS_DIR / repo.split("/")[-1]
    return p if (p / "config.json").exists() else None


def in_hf_cache(repo):
    return (Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))).exists()


def model_available(spec):
    """True if the weights are on this machine (so loading will not need a download)."""
    repo, _ = parse_model_spec(spec)
    return local_dir(repo) is not None or in_hf_cache(repo)


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32


def _load_causal_lm(spec):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    repo, quant = parse_model_spec(spec)
    path = str(local_dir(repo) or repo)
    device, dtype = _device_dtype()
    kw = {"local_files_only": not config.ALLOW_DOWNLOAD}
    if quant == "nf4":
        if device != "cuda":
            raise RuntimeError("4-bit models need a CUDA GPU")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kw)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kw)
    return (model if quant == "nf4" else model.to(device)).eval(), device


def _chat_prompt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# Prompt tokens per forward pass when pre-filling the KV cache (see LocalLLM._prefill).
PREFILL_CHUNK = 1024
CHOOSE_CHUNK = 384


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("<think>", "").replace("</think>", "").strip()


_strip_think = strip_think


class LocalLLM:
    def __init__(self, model_name=config.CHAT_MODEL):
        self.model_name = model_name
        self.repo, self.quant = parse_model_spec(model_name)
        self._model = self._tok = None
        self.device = None
        self._lock = threading.Lock()

    @property
    def label(self):
        return self.repo.split("/")[-1] + (f" ({self.quant})" if self.quant else "")

    def load(self):
        if self._model is None:
            from transformers import AutoTokenizer
            path = str(local_dir(self.repo) or self.repo)
            self._tok = AutoTokenizer.from_pretrained(path, local_files_only=not config.ALLOW_DOWNLOAD)
            self._model, self.device = _load_causal_lm(self.model_name)
        return self

    def unload(self):
        """Free the GPU memory (only one model is kept loaded at a time)."""
        self._model = self._tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode(self, messages):
        self.load()
        text = _chat_prompt(self._tok, messages)
        return self._tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    # ── constrained choice (used by the investigator) ─────────────────────
    @torch.no_grad()
    def choose(self, system, user, labels):
        """Pick one of `labels` (single characters, e.g. 'A'..'H'). Returns (best label, {label: probability}).

        One forward pass over the prompt; the answer is the allowed label with the highest score for the first
        token the model would write. The probabilities are a softmax over the allowed labels only."""
        x = self._encode([{"role": "system", "content": system}, {"role": "user", "content": user}])
        ids = [self._tok.encode(l, add_special_tokens=False)[0] for l in labels]
        with self._lock:
            # chunked prefill keeps the attention memory bounded: a single pass over a ~2k-token prompt pushed the
            # 4-bit 8B model (6 GB of weights) past 8 GB and into Windows' slow shared GPU memory (14x slower)
            cache = self._prefill(x.input_ids, chunk=CHOOSE_CHUNK)
            logits = self._model(input_ids=x.input_ids[:, -1:], past_key_values=cache, use_cache=True,
                                 logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits[ids], dim=0).tolist()
        scores = dict(zip(labels, probs))
        return max(scores, key=scores.get), scores

    # ── free text (used by the chat) ──────────────────────────────────────
    def _gen_kwargs(self, max_new_tokens):
        return dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                    repetition_penalty=1.05, pad_token_id=self._tok.eos_token_id)

    def generate(self, system, user, max_new_tokens=config.CHAT_MAX_NEW_TOKENS):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens)

    @torch.no_grad()
    def chat(self, messages, max_new_tokens=config.CHAT_MAX_NEW_TOKENS):
        """messages: [{"role": "system" | "user" | "assistant", "content": str}, ...] -> the next reply."""
        x = self._encode(messages)
        with self._lock:
            out = self._model.generate(**x, past_key_values=self._prefill(x.input_ids),
                                       **self._gen_kwargs(max_new_tokens))
        return strip_think(self._tok.decode(out[0, x.input_ids.shape[1]:], skip_special_tokens=True))

    def stream_chat(self, messages, max_new_tokens=config.CHAT_MAX_NEW_TOKENS):
        """Like chat(), but yields the reply in pieces while it is generated (for a chat UI)."""
        from transformers import TextIteratorStreamer
        x = self._encode(messages)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
        failure = []

        def work():
            try:
                with self._lock, torch.no_grad():
                    self._model.generate(**x, past_key_values=self._prefill(x.input_ids), streamer=streamer,
                                         **self._gen_kwargs(max_new_tokens))
            except Exception as e:  # end the stream so the caller sees the error instead of waiting forever
                failure.append(e)
                streamer.end()

        threading.Thread(target=work, daemon=True).start()
        yield from streamer
        if failure:
            raise failure[0]

    def _prefill(self, ids, chunk=PREFILL_CHUNK):
        """Run all but the last prompt token through the model in chunks and return the filled KV cache.

        In one pass, attention over the whole prompt needs memory that grows with the square of its length
        (PyTorch on Windows has no flash attention). With the RAG neighbours in the facts the prompt is ~5k
        tokens, which peaked at 9.4 GiB and spilled an 8 GB GPU into slow shared memory (2 tok/s). Chunks keep
        the peak near 4 GiB; generate() then continues from the cache.
        """
        from transformers import DynamicCache
        cache = DynamicCache()
        n = ids.shape[1] - 1
        for start in range(0, n, chunk):
            self._model(input_ids=ids[:, start:min(start + chunk, n)], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        return cache


ChatSLM = LocalLLM      # the chat's historical name
