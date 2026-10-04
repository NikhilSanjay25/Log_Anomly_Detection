"""Small language models running locally through HF transformers.

* InsightSLM     - Qwen3-1.7B (configurable) that writes explanations / impact / recommendations.
* LoraClassifier - the fine-tuned Qwen3-0.6B LoRA from slm_Qwen3_0_6ipynb.ipynb, used by the
                   Anomaly Detection Agent as an advisory second opinion on uncertain sequences.
Both load lazily, use CUDA when available and fall back to CPU.
"""
import json
import re
import threading

import torch

from . import config


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32


def _load_causal_lm(name):
    from transformers import AutoModelForCausalLM
    device, dtype = _device_dtype()
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model.to(device).eval(), device


def _chat_prompt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# Prompt tokens per forward pass when pre-filling the KV cache (see InsightSLM._prefill).
PREFILL_CHUNK = 1024


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("<think>", "").replace("</think>", "").strip()


_strip_think = strip_think


class InsightSLM:
    def __init__(self, model_name=config.INSIGHT_MODEL):
        self.model_name = model_name
        self._model = self._tok = None
        self.device = None
        self._lock = threading.Lock()

    def load(self):
        if self._model is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model, self.device = _load_causal_lm(self.model_name)
        return self

    def _encode(self, messages):
        self.load()
        text = _chat_prompt(self._tok, messages)
        return self._tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    def _gen_kwargs(self, max_new_tokens):
        return dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                    repetition_penalty=1.05, pad_token_id=self._tok.eos_token_id)

    def generate(self, system, user, max_new_tokens=config.INSIGHT_MAX_NEW_TOKENS):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens)

    @torch.no_grad()
    def chat(self, messages, max_new_tokens=config.INSIGHT_MAX_NEW_TOKENS):
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

    def _prefill(self, ids):
        """Run all but the last prompt token through the model in chunks and return the filled KV cache.

        In one pass, attention over the whole prompt needs memory that grows with the square of its length
        (PyTorch on Windows has no flash attention). With the RAG neighbours in the facts the prompt is ~5k
        tokens, which peaked at 9.4 GiB and spilled an 8 GB GPU into slow shared memory (2 tok/s). Chunks keep
        the peak near 4 GiB; generate() then continues from the cache.
        """
        from transformers import DynamicCache
        cache = DynamicCache()
        n = ids.shape[1] - 1
        for start in range(0, n, PREFILL_CHUNK):
            self._model(input_ids=ids[:, start:min(start + PREFILL_CHUNK, n)], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        return cache

    def generate_json(self, system, user, max_new_tokens=config.INSIGHT_MAX_NEW_TOKENS):
        """Returns (parsed dict or None, raw text)."""
        raw = self.generate(system, user, max_new_tokens)
        return extract_json(raw), raw


def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LoraClassifier:
    """Qwen3-0.6B + LoRA fine-tuned to answer exactly 'Normal' or 'Anomaly'."""

    def __init__(self, adapter_dir=config.LORA_ADAPTER_DIR):
        self.adapter_dir = adapter_dir
        self._model = self._tok = None
        self.prompt = None
        self._lock = threading.Lock()  # one cached model is shared by all Streamlit sessions

    @property
    def available(self):
        return (self.adapter_dir / "adapter_config.json").exists()

    def load(self):
        if self._model is None:
            from peft import PeftModel
            from transformers import AutoTokenizer
            exp = json.loads((self.adapter_dir / "experiment_config.json").read_text())
            self.prompt = exp["prompt"]
            base_name = exp.get("base_model", config.LORA_BASE_MODEL)
            try:
                self._tok = AutoTokenizer.from_pretrained(str(self.adapter_dir))
            except (AttributeError, TypeError, ValueError):
                # The adapter was saved with transformers 5.x, which stores extra_special_tokens as a list that
                # transformers 4.x cannot read. LoRA training left the tokenizer unchanged (same special tokens
                # and chat template), so the base model's tokenizer is equivalent.
                self._tok = AutoTokenizer.from_pretrained(base_name)
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            base, self.device = _load_causal_lm(base_name)
            self._model = PeftModel.from_pretrained(base, str(self.adapter_dir)).eval()
        return self

    @torch.no_grad()
    def classify(self, sequences, batch_size=16):
        """sequences: list of event lists -> list of 'Normal' | 'Anomaly' | None (invalid output)."""
        self.load()
        prompts = [_chat_prompt(self._tok, [{"role": "user", "content": self.prompt.format(seq=" ".join(s))}])
                   for s in sequences]
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = self._tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(self.device)
            with self._lock:
                gen = self._model.generate(**batch, max_new_tokens=8, do_sample=False, temperature=None,
                                           top_p=None, top_k=None, pad_token_id=self._tok.pad_token_id)
            for t in self._tok.batch_decode(gen[:, batch.input_ids.shape[1]:], skip_special_tokens=True):
                found = {m.lower() for m in re.findall(r"\b(normal|anomaly)\b", _strip_think(t), re.I)}
                results.append(found.pop().capitalize() if len(found) == 1 else None)
        return results
