"""Lazily-loaded heavy resources shared by all agents (loaded once per process)."""
import threading

from . import config


class Resources:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = self._insight = self._lora = None
        self._lora_checked = False

    def pipeline(self):
        with self._lock:
            if self._pipeline is None:
                from .ml import AnomalyPipeline
                self._pipeline = AnomalyPipeline()
            return self._pipeline

    def insight_slm(self):
        with self._lock:
            if self._insight is None:
                from .slm import InsightSLM
                self._insight = InsightSLM(config.INSIGHT_MODEL).load()
            return self._insight

    def lora(self):
        """The LoRA classifier, or None when the adapter folder is missing."""
        with self._lock:
            if not self._lora_checked:
                from .slm import LoraClassifier
                clf = LoraClassifier()
                self._lora = clf if clf.available else None
                self._lora_checked = True
            return self._lora

    def status(self):
        return {"pipeline_loaded": self._pipeline is not None, "insight_slm_loaded": self._insight is not None,
                "insight_model": config.INSIGHT_MODEL, "lora_adapter": str(config.LORA_ADAPTER_DIR),
                "lora_available": (config.LORA_ADAPTER_DIR / "adapter_config.json").exists()}
