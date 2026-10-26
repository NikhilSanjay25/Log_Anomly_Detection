"""Lazily-loaded heavy resources shared by all agents (loaded once per process)."""
import threading

from . import config
from .datasets import PROFILES, get_profile


class Resources:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipelines = {}
        self._chat = None

    def pipeline(self, dataset="hdfs"):
        """The Transformer → FAISS → RF pipeline of one dataset (HDFS, BGL or OpenStack), loaded on first use."""
        with self._lock:
            if dataset not in self._pipelines:
                from .ml import AnomalyPipeline
                self._pipelines[dataset] = AnomalyPipeline(get_profile(dataset))
            return self._pipelines[dataset]

    def chat_slm(self):
        """The Qwen model of the follow-up chat (the only language model in the system)."""
        with self._lock:
            if self._chat is None:
                from .slm import ChatSLM
                self._chat = ChatSLM(config.CHAT_MODEL).load()
            return self._chat

    def status(self):
        return {"pipelines_loaded": sorted(self._pipelines), "chat_loaded": self._chat is not None,
                "chat_model": config.CHAT_MODEL,
                "available": {n: p.available for n, p in PROFILES.items()}}
