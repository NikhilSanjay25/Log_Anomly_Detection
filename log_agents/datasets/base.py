"""Dataset profiles: everything that differs between the supported log families.

One profile per dataset (HDFS, BGL, OpenStack). The agents (identification, preprocessing, medallion, detection,
RCA, insight, chat) are written once against this interface and ask the profile for the parts that depend on the
log family: how to recognise and parse it, what a session is, what each event means, which model artifacts to use,
and which root causes exist.

Sessions are always stored under the column name `block_id` (the HDFS name), whatever they are called in the data:
a block (HDFS), a window of consecutive messages (BGL) or a VM instance (OpenStack). `session_noun` is the word
used in text shown to people.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

EVENT_ID_RE = re.compile(r"\bE\d{1,4}\b")


@dataclass
class ArtifactPaths:
    directory: Path
    backbone: Path
    model_config: Path
    event2id: Path
    id2event: Path
    faiss: Path
    rf: Path
    rag_meta: Path
    event_stats: Path

    def required(self):
        return [self.backbone, self.model_config, self.event2id, self.id2event, self.faiss, self.rf]

    def missing(self):
        return [p.name for p in self.required() if not p.exists()]

    @classmethod
    def standard(cls, directory):
        d = Path(directory)
        return cls(d, d / "transformer_backbone.pth", d / "model_config.joblib", d / "event2id.joblib",
                   d / "id2event.joblib", d / "faiss.index", d / "rf.joblib", d / "rag_metadata.npz",
                   d / "event_stats.json")


class DatasetProfile:
    name = ""
    title = ""
    persona = ""                    # how the chat model introduces itself
    session_noun = "session"
    session_noun_plural = "sessions"
    node_noun = "node"
    formats = {}                    # format key -> description shown to the user
    max_seq_len = 60                # the Transformer keeps the last N events of a session
    extra_features = ()             # numeric Gold columns appended to the Random Forest input
    rf_inputs = "full"              # "full": [embedding, neighbour mean, extras] · "extras": only extra_matrix()
    session_re = re.compile(r"(?!)")  # finds session ids in a free-text question
    node_re = None                  # finds node ids (other than IPs) in text, for the grounding check
    glossary = ""                   # short explanation of the data model, given to the chat model
    normal_description = ""         # what a healthy session looks like (used in the deterministic insight)
    uses_ips_as_nodes = True
    sample_files = ()               # demo inputs shipped in samples/

    # ── artifacts ─────────────────────────────────────────────────────
    @property
    def paths(self):
        raise NotImplementedError

    @property
    def available(self):
        return not self.paths.missing()

    # ── event vocabulary ──────────────────────────────────────────────
    @property
    def event_info(self):
        """event id -> (stage, kind normal|warning|error, description)"""
        raise NotImplementedError

    @property
    def template_text(self):
        raise NotImplementedError

    @property
    def error_events(self):
        return {e for e, (_, kind, _) in self.event_info.items() if kind == "error"}

    @property
    def warning_events(self):
        return {e for e, (_, kind, _) in self.event_info.items() if kind == "warning"}

    def describe(self, eid):
        return self.event_info.get(eid, ("unknown", "warning", "Unknown event"))[2]

    # ── root causes ───────────────────────────────────────────────────
    @property
    def root_causes(self):
        raise NotImplementedError

    # ── parsing hooks (raw text -> Bronze -> Silver -> Gold) ──────────
    def detect(self, lines):
        """{format: score 0..1} for this family's formats on a sample of non-blank lines."""
        raise NotImplementedError

    def clean(self, text, fmt, **kw):
        """Bronze layer: returns (DataFrame, stats dict)."""
        raise NotImplementedError

    def build_silver(self, clean, fmt):
        """Silver layer: returns (DataFrame one row per (session, event), number of unmatched lines)."""
        raise NotImplementedError

    def extra_matrix(self, gold):
        """(N, m) float32 numeric features appended to the Random Forest input, or None."""
        if not self.extra_features:
            return None
        return gold[list(self.extra_features)].to_numpy(dtype="float32")

    def add_session_features(self, gold, silver, clean, fmt):
        """Hook to add family-specific Gold columns (labels, timings...). Returns the Gold frame."""
        return gold

    # ── RCA hooks ─────────────────────────────────────────────────────
    def structural_scores(self, counts, present, row, stats):
        """Family-specific structural checks on one anomalous session: (scores {cause: score}, evidence lines)."""
        return {}, []

    def normal_structural_evidence(self, counts, row, stats):
        return []

    # ── helpers ───────────────────────────────────────────────────────
    def enrich_events(self, silver):
        """Add template text, lifecycle stage and severity (from the event vocabulary) to a Silver frame."""
        info = silver.event_id.map(self.event_info)
        return silver.assign(template=silver.event_id.map(self.template_text),
                             stage=info.str[0], severity=info.str[1])

    def __repr__(self):
        return f"<DatasetProfile {self.name}>"


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
