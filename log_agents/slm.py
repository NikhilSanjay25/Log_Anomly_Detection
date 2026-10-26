"""The chat language model: Qwen3-1.7B (configurable with LAD_CHAT_MODEL) running locally through HF transformers.

It is the only language model in the system. It answers follow-up questions about a finished run (see
log_agents/chat.py). It loads lazily, uses CUDA when available and falls back to CPU.
"""
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


class ChatSLM:
    def __init__(self, model_name=config.CHAT_MODEL):
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
