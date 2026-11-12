"""Lazily-loaded heavy resources shared by all agents (loaded once per process)."""
import threading

from . import config
from .datasets import PROFILES, get_profile


class Resources:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipelines = {}
        self._llm = None

    def pipeline(self, dataset="hdfs"):
        """The Transformer → FAISS → RF pipeline of one dataset (HDFS, BGL or OpenStack), loaded on first use."""
        with self._lock:
            if dataset not in self._pipelines:
                from .ml import AnomalyPipeline
                self._pipelines[dataset] = AnomalyPipeline(get_profile(dataset))
            return self._pipelines[dataset]

    def llm(self, model_name=None):
        """The language model (investigator and chat). Only one is kept in memory: asking for a different one
        unloads the previous one first, so a 4-bit 8B model and the 1.7B model never compete for 8 GB of VRAM."""
        from .slm import LocalLLM
        model_name = model_name or config.CHAT_MODEL
        with self._lock:
            if self._llm is None or self._llm.model_name != model_name:
                if self._llm is not None:
                    self._llm.unload()
                self._llm = LocalLLM(model_name)
            return self._llm.load()

    def chat_slm(self, model_name=None):
        return self.llm(model_name)

    def status(self):
        from .slm import model_available
        return {"pipelines_loaded": sorted(self._pipelines),
                "llm_loaded": self._llm.model_name if self._llm is not None and self._llm._model is not None else None,
                "chat_loaded": self._llm is not None and self._llm._model is not None,
                "chat_model": config.CHAT_MODEL,
                "models_available": {m: model_available(m) for m in config.LLM_CHOICES},
                "available": {n: p.available for n, p in PROFILES.items()}}
