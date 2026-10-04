"""Shared run context: the blackboard the Coordinator and all agents read from / write to."""
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from . import config


@dataclass
class Message:
    sender: str
    recipient: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class TraceStep:
    agent: str
    task: str
    status: str = "pending"        # pending | running | ok | fallback | skipped | failed
    summary: str = ""
    attempts: int = 0
    duration_s: float = 0.0
    issues: list = field(default_factory=list)


@dataclass
class RunOptions:
    use_slm_insight: bool = True
    lora_mode: str = "auto"         # auto (uncertain cases only) | all | off
    rag_k: int = config.RAG_K
    threshold: float = 0.5
    persist: bool = True            # write medallion layers + run memory to disk


class RunContext:
    def __init__(self, source_name, text, options=None, origin="upload"):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.source_name = source_name
        self.origin = origin
        self.text = text
        self.options = options or RunOptions()
        self.data = {}              # blackboard: agents publish results here
        self.messages = []
        self.trace = []
        self.warnings = []
        self.started = time.time()

    def send(self, sender, recipient, content):
        self.messages.append(Message(sender, recipient, content))

    def warn(self, msg):
        self.warnings.append(msg)


class RunMemory:
    """Long-term memory: compact summaries of previous runs (JSON files in data/runs)."""

    def __init__(self, directory=config.RUNS_DIR):
        self.dir = directory

    def recent(self, n=5):
        if not self.dir.exists():
            return []
        files = sorted(self.dir.glob("*.json"), reverse=True)[:n]
        out = []
        for f in files:
            try:
                out.append(json.loads(f.read_text(encoding="utf-8"))["summary"])
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return out

    def save(self, run_id, report_dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{run_id}.json").write_text(json.dumps(report_dict, indent=1, default=str), encoding="utf-8")
