"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env, default):
    return Path(os.environ.get(env, default))


# ── ML pipeline artifacts (Transformer → FAISS → RAG → Random Forest) ──────────
ARTIFACTS_DIR = _path("LAD_ARTIFACTS_DIR", ROOT)
BACKBONE_PATH = ARTIFACTS_DIR / "transformer_backbone.pth"
RF_PATH = ARTIFACTS_DIR / "rag_random_forest.joblib"
FAISS_PATH = ARTIFACTS_DIR / "hdfs_faiss_index.index"
EVENT2ID_PATH = ARTIFACTS_DIR / "event2id.joblib"
ID2EVENT_PATH = ARTIFACTS_DIR / "id2event.joblib"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.joblib"
# Built by scripts/build_rag_metadata.py — maps FAISS ids back to sequences/labels
RAG_METADATA_PATH = ARTIFACTS_DIR / "rag_metadata.npz"
EVENT_STATS_PATH = ARTIFACTS_DIR / "event_stats.json"

# Training dataset (only needed to rebuild rag_metadata.npz)
DATASET_DIR = _path("LAD_DATASET_DIR", ROOT / "preprocessed" / "preprocessed")

MAX_SEQ_LEN = 60          # must match training (load_real_hdfs max_len)
RAG_K = int(os.environ.get("LAD_RAG_K", 5))  # RF was trained with k=5 neighbours

# ── SLMs ─────────────────────────────────────────────────────────────────────
INSIGHT_MODEL = os.environ.get("LAD_INSIGHT_MODEL", "Qwen/Qwen3-1.7B")
INSIGHT_MAX_NEW_TOKENS = int(os.environ.get("LAD_INSIGHT_MAX_TOKENS", 700))
# Follow-up chat about a finished run (log_agents/chat.py)
CHAT_MAX_NEW_TOKENS = int(os.environ.get("LAD_CHAT_MAX_TOKENS", 400))
CHAT_MAX_TURNS = int(os.environ.get("LAD_CHAT_MAX_TURNS", 4))     # earlier question/answer pairs kept in the prompt
# Fine-tuned Qwen3-0.6B LoRA classifier from slm_Qwen3_0_6ipynb.ipynb (optional)
LORA_ADAPTER_DIR = _path("LAD_LORA_DIR", ROOT / "qwen3-hdfs-ckpt" / "qwen3-hdfs-lora")
LORA_BASE_MODEL = os.environ.get("LAD_LORA_BASE", "Qwen/Qwen3-0.6B")
# RF probabilities within threshold ± this margin are "uncertain" and get an SLM second opinion
UNCERTAIN_MARGIN = 0.3
LORA_MAX_SEQUENCES = int(os.environ.get("LAD_LORA_MAX", 40))

# ── Storage (medallion layers, run memory, reports) ──────────────────────────
DATA_DIR = _path("LAD_DATA_DIR", ROOT / "data")
MEDALLION_DIR = DATA_DIR / "medallion"
RUNS_DIR = DATA_DIR / "runs"
