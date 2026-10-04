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


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("<think>", "").strip()


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

    @torch.no_grad()
    def generate(self, system, user, max_new_tokens=config.INSIGHT_MAX_NEW_TOKENS):
        self.load()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = _chat_prompt(self._tok, msgs)
        x = self._tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self._lock:
            out = self._model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                                       temperature=None, top_p=None, top_k=None,
                                       repetition_penalty=1.05, pad_token_id=self._tok.eos_token_id)
        return _strip_think(self._tok.decode(out[0, x.input_ids.shape[1]:], skip_special_tokens=True))

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
            self._tok = AutoTokenizer.from_pretrained(str(self.adapter_dir))
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            base, self.device = _load_causal_lm(exp.get("base_model", config.LORA_BASE_MODEL))
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
