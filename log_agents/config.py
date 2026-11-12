"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env, default):
    return Path(os.environ.get(env, default))


# ── HDFS pipeline artifacts (Transformer → FAISS → RAG → Random Forest) ─────────
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

# ── BGL and OpenStack artifacts: artifacts/<dataset>/ (built by scripts/train_dataset.py) ───
ARTIFACTS_ROOT = _path("LAD_ARTIFACTS_ROOT", ROOT / "artifacts")

# Training datasets (only needed to retrain / rebuild metadata)
DATASET_DIR = _path("LAD_DATASET_DIR", ROOT / "preprocessed" / "preprocessed")          # HDFS_v1
BGL_DATASET_DIR = _path("LAD_BGL_DIR", ROOT / "Dataset_BGL")
OPENSTACK_DATASET_DIR = _path("LAD_OPENSTACK_DIR", ROOT / "Dataset_OpenStack")

MAX_SEQ_LEN = 60          # HDFS: must match training (load_real_hdfs max_len)
RAG_K = int(os.environ.get("LAD_RAG_K", 5))  # the Random Forests were trained with k=5 neighbours
BGL_WINDOW = 100          # BGL has no session id: sessions are windows of this many consecutive messages

# RF probabilities within threshold ± this margin are flagged "uncertain" (shown for manual review)
UNCERTAIN_MARGIN = 0.3

# ── Language models (Qwen3, run locally) ─────────────────────────────────────
# One model serves the investigator (picks the next investigation step from a menu) and the chat (writes answers).
# Names look like "Qwen/Qwen3-8B"; a ":nf4" / ":bf16" suffix overrides the default quantisation below.
RULES = "rules"           # AGENT_MODEL value that switches the investigator off: rule-based root-cause analysis only
AGENT_MODEL = os.environ.get("LAD_AGENT_MODEL", "Qwen/Qwen3-4B")      # best accuracy per second in the comparison (README)
CHAT_MODEL = os.environ.get("LAD_CHAT_MODEL", os.environ.get("LAD_INSIGHT_MODEL", "Qwen/Qwen3-1.7B"))
MODEL_QUANT = {"Qwen/Qwen3-1.7B": None, "Qwen/Qwen3-4B": "nf4", "Qwen/Qwen3-8B": "nf4"}   # 4B / 8B need 4-bit on 8 GB
LLM_CHOICES = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
# Plain folders with the weights (scripts/download_models.py); kept out of the OneDrive project folder.
MODELS_DIR = _path("LAD_MODELS_DIR", Path.home() / ".cache" / "lad_models")
ALLOW_DOWNLOAD = os.environ.get("LAD_ALLOW_DOWNLOAD", "") == "1"      # otherwise missing weights are an error

# Investigator limits
AGENT_MAX_STEPS = int(os.environ.get("LAD_AGENT_MAX_STEPS", 5))        # evidence-gathering actions per case
AGENT_MAX_CASES = int(os.environ.get("LAD_AGENT_MAX_CASES", 30))       # cases (groups of identical anomalies) per run
AGENT_DECISION = os.environ.get("LAD_AGENT_DECISION", "listwise_debiased")     # pointwise | listwise | listwise_debiased
AGENT_SHORTLIST = 5                                                    # root causes the rule engine puts on the final menu

# Chat
CHAT_MAX_NEW_TOKENS = int(os.environ.get("LAD_CHAT_MAX_TOKENS", 400))
CHAT_MAX_TURNS = int(os.environ.get("LAD_CHAT_MAX_TURNS", 4))     # earlier question/answer pairs kept in the prompt
CHAT_MAX_LOG_LINES = int(os.environ.get("LAD_CHAT_LOG_LINES", 40))   # raw log lines shown to the model per session
CHAT_MAX_LOG_CHARS = int(os.environ.get("LAD_CHAT_LOG_CHARS", 6000))  # cap on all raw log text added to a question

# ── Storage (medallion layers, run memory, reports) ──────────────────────────
DATA_DIR = _path("LAD_DATA_DIR", ROOT / "data")
MEDALLION_DIR = DATA_DIR / "medallion"
RUNS_DIR = DATA_DIR / "runs"
