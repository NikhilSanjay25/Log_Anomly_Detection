"""BlueGene/L (BGL) profile.

BGL (LLNL supercomputer, Oliner & Stearley 2007) has one label per log line: "-" for a normal message, or an alert
category tag (KERNDTLB, APPSEV, ...). It has no session id, so a session here is a window of `config.BGL_WINDOW`
consecutive messages, anomalous if it contains at least one alert line.

Line format (LogHub):  <label> <unix time> <date> <node> <time> <node> RAS <component> <level> <message>
The label column is used only as ground truth for evaluation. It is never given to the models.
"""
import json
import re

import numpy as np
import pandas as pd

from .. import config
from .base import read_json
from .mined import LEVEL_KIND, MinedProfile, to_datetime

WS_RE = re.compile(r"\s+")
HEAD_RE = re.compile(r"^\S+ \d{9,11} \d{4}\.\d\d\.\d\d \S+ \d{4}-\d\d-\d\d-\d\d\.\d\d\.\d\d\.\d+ \S+ \S+ ")
TIME_FMT = "%Y-%m-%d-%H.%M.%S.%f"
WORD_RE = re.compile(r"[A-Za-z][A-Za-z_\-]{2,}")
LEVELS = {"INFO", "WARNING", "ERROR", "FATAL", "SEVERE", "FAILURE", "DEBUG"}

# Node locations and long hex words are variable parts of a message, not part of its template
HEX_RE = re.compile(r"(?<![\w])(?=[0-9a-f]*\d)[0-9a-f]{6,}(?![\w])")
LOCATION_RE = re.compile(r"\bR\d{2}-M\d-[NL][0-9A-F]+(?:-[A-Z]:J\d{2}-U\d{2})?(?:-[A-Z])?\b")

FAMILIES = {
    "KERN": {
        "name": "compute-node kernel fault",
        "explanation": "The BlueGene/L compute-node kernel reported a fault (memory, TLB or storage interrupt, "
                       "machine check, recoverable network error or kernel termination).",
        "impact": "Jobs on the affected nodes can fail or be killed; repeated faults on one node point to failing "
                  "hardware (memory, processor, or a torus/tree link).",
        "actions": ["Look up the affected node location and check for earlier faults on the same location",
                    "Run the hardware diagnostics, or reseat or replace the node card, if the same location keeps faulting",
                    "Re-run the job that was on the node elsewhere to separate a hardware fault from an application bug",
                    "Check tree and torus network error counters for the affected midplane"],
    },
    "APP": {
        "name": "application / I/O-node (ciod) fault",
        "explanation": "The I/O node's control and I/O daemon (ciod) reported an error while loading, running, reading "
                       "from or writing for an application.",
        "impact": "The affected job failed to start, lost its control connection or lost its I/O; it usually has to "
                  "be resubmitted.",
        "actions": ["Check the job's output and the ciod log on the I/O node",
                    "Verify that the file systems the job uses are reachable from the I/O node",
                    "Check the control-system network between the service node and the I/O node",
                    "Resubmit the job; if it fails again on the same I/O node, take that node out of service"],
    },
    "LINK": {
        "name": "link card / interconnect fault",
        "explanation": "A link card or the interconnect reported a problem (discovery, cabling or link training).",
        "impact": "Part of the machine may be isolated or unusable until the link is repaired.",
        "actions": ["Inspect the reported link card and its cables", "Re-run link-card discovery on the midplane",
                    "Replace the link card if the fault repeats"],
    },
    "MON": {
        "name": "hardware monitor alert",
        "explanation": "The hardware monitor reported a power, temperature or sensor condition.",
        "impact": "Possible power or cooling problem; the affected hardware may shut down.",
        "actions": ["Check power and cooling for the reported location", "Compare sensor readings with neighbouring cards"],
    },
    "MAS": {
        "name": "machine-state (BGLMaster) event",
        "explanation": "A BlueGene/L master service changed state abnormally.",
        "impact": "A control-system service may have restarted or stopped.",
        "actions": ["Check the BGLMaster logs on the service node", "Verify the control-system services are running"],
    },
    "MMCS": {
        "name": "control-system (MMCS) fault",
        "explanation": "The midplane management and control system reported an error.",
        "impact": "Partition allocation or booting may be affected.",
        "actions": ["Check the MMCS logs on the service node", "Verify the database and partition state"],
    },
}
OTHER_FAMILY = {
    "name": "other alert category",
    "explanation": "The labelled BGL data marks these messages as alerts.",
    "impact": "Unknown; review the log lines around the alert.",
    "actions": ["Review the raw log lines of the window", "Compare with earlier windows that contain the same message"],
}
RARE_PATTERN = {
    "title": "Unusual message pattern (model-detected)",
    "events": set(), "weight": 0.4, "severity": "medium",
    "explanation": "No known alert signature explains it, but the ML model and the retrieved similar historical "
                   "windows indicate this message pattern is characteristic of alert windows.",
    "impact": "Unknown - needs manual review of the window's log lines.",
    "actions": ["Review the raw log lines of the window",
                "Compare with the retrieved similar anomalous windows for a common pattern"],
}


def _shorten(text, n):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0].rstrip(",:;") + " …"


def family_for(tag):
    for prefix, fam in FAMILIES.items():
        if tag.startswith(prefix):
            return fam
    return OTHER_FAMILY


class BGLProfile(MinedProfile):
    name = "bgl"
    title = "BlueGene/L (BGL)"
    persona = "BlueGene/L supercomputer operations engineer"
    session_noun = "window"
    session_noun_plural = "windows"
    node_noun = "node"
    max_seq_len = config.BGL_WINDOW
    # Event ids are not stable over time in BGL: the test period holds alert templates never seen in training, so
    # the id-based Transformer embedding and its neighbours misled the Random Forest (F1 0.63 with them against
    # 0.97 on the word features alone). The forest therefore reads only the word and severity features; the
    # Transformer and FAISS still retrieve similar windows (evidence, uncertainty flag).
    rf_inputs = "extras"
    # Severity counts of the window's lines. Alert types change over time, so a template the model never saw in
    # training still shows up as FATAL/ERROR lines. The label column is never used.
    extra_features = ("n_fatal", "n_error", "n_severe", "n_failure", "n_warning")
    session_re = re.compile(r"\bwin_\d+\b")
    node_re = re.compile(r"\bR\d{2}-M\d-[NL][0-9A-F]+(?:-[A-Z]:J\d{2}-U\d{2})?(?:-[A-Z])?\b")
    drain_masks = [(LOCATION_RE, "<*>"), (HEX_RE, "<*>")]
    formats = {"bgl_raw": f"Raw BGL RAS log lines grouped into windows of {config.BGL_WINDOW} consecutive messages"}
    glossary = (f"BGL logs have no session id, so a session is a window of {config.BGL_WINDOW} consecutive messages "
                "(win_N). A window is anomalous when it contains at least one alert message. Events are message "
                "templates learned from the logs (E<n>), and nodes are BlueGene/L hardware locations such as "
                "R02-M1-N0-C:J12-U11. Alert categories (KERNDTLB, APPSEV, ...) come from the labelled dataset.")
    normal_description = ("Normal BGL windows contain routine informational RAS messages and no alert categories.")
    sample_files = ("bgl_raw_sample.log",)

    # ── word features ─────────────────────────────────────────────────
    # Alert types change over time, so a window can contain templates the model never saw. Their *words* ("error",
    # "failed", "timeout", the log level and component...) still carry the signal, so every window also gets a
    # bag-of-words of its messages' templates. The vocabulary is fixed from the training windows only.
    def _reset_derived(self):
        self._vocab = self._W = self._eidx = None

    @property
    def word_vocab(self):
        if self._vocab is None:
            self._vocab = json.loads((self.dir / "word_features.json").read_text(encoding="utf-8"))["vocab"]
        return self._vocab

    def event_words(self, eid):
        ev = self.events[eid]
        return [w.lower() for w in WORD_RE.findall(ev["template"])] + ["lvl_" + ev["level"].lower(),
                                                                     "cmp_" + ev["component"].lower()]

    def fit_word_vocab(self, train_gold):
        """Vocabulary = words of the templates that occur in the training windows."""
        used = {e for ev in train_gold.events for e in ev}
        self._vocab = sorted({w for e in used for w in self.event_words(e)})
        self._W = None

    def save_word_vocab(self, directory=None):
        (directory or self.dir).joinpath("word_features.json").write_text(
            json.dumps({"vocab": self.word_vocab}), encoding="utf-8")

    def _word_matrix(self):
        if self._W is None:
            ids = sorted(self.events, key=lambda e: int(e[1:]))
            self._eidx = {e: i for i, e in enumerate(ids)}
            vidx = {w: i for i, w in enumerate(self.word_vocab)}
            W = np.zeros((len(ids), len(vidx)), np.float32)
            for e in ids:
                for w in self.event_words(e):
                    if w in vidx:
                        W[self._eidx[e], vidx[w]] += 1
            self._W = W
        return self._W

    def extra_matrix(self, gold):
        """Severity counts of the window's lines + word counts of its messages' templates."""
        W = self._word_matrix()
        counts = np.zeros((len(gold), W.shape[0]), np.float32)
        for i, events in enumerate(gold.events):
            for e in events:
                counts[i, self._eidx[e]] += 1
        base = gold[list(self.extra_features)].to_numpy(dtype=np.float32)
        return np.hstack([base, counts @ W])

    # ── identification ────────────────────────────────────────────────
    def detect(self, lines):
        return {"bgl_raw": sum(1 for ln in lines if HEAD_RE.match(ln)) / len(lines)}

    # ── Bronze ────────────────────────────────────────────────────────
    def clean(self, text, fmt, window_offset=0, window=config.BGL_WINDOW, **kw):
        lines = pd.Series(text.splitlines(), dtype="object")
        n_in = len(lines)
        df = pd.DataFrame({"line_no": range(1, n_in + 1),
                           "raw": lines.str.replace(WS_RE, " ", regex=True).str.strip()})
        blank = df.raw.eq("") | df.raw.str.startswith("#")
        df = df[~blank]
        dup = df.raw.duplicated()
        n_dup = int(dup.sum())
        df = df[~dup]

        parts = df.raw.str.split(" ", n=9, expand=True).reindex(columns=range(10))
        valid = (parts[9].notna() & parts[1].str.fullmatch(r"\d{9,11}", na=False)
                 & parts[2].str.fullmatch(r"\d{4}\.\d\d\.\d\d", na=False))
        noise_examples = df.loc[~valid, "raw"].head(5).tolist()
        n_noise = int((~valid).sum())
        df, parts = df[valid].copy(), parts[valid]

        level = parts[8].str.upper()
        component = parts[7]
        content = parts[9]
        bad = ~level.isin(LEVELS)                      # a few lines have a shifted layout
        swap = bad & component.str.upper().isin(LEVELS)
        level = level.where(~swap, component.str.upper()).where(~(bad & ~swap), "UNKNOWN")
        content = content.where(~bad, parts[8] + " " + parts[9])
        component = component.where(~swap, "UNKNOWN")

        df["label"] = parts[0]
        df["true_label"] = (parts[0] != "-").astype(int)
        df["node"] = parts[3]
        df["timestamp"] = to_datetime(parts[4], TIME_FMT)
        df["component"] = component
        df["level"] = level
        df["content"] = content
        df = df.reset_index(drop=True)
        df["block_id"] = [f"win_{window_offset + i // window:07d}" for i in range(len(df))]
        df = df[["line_no", "raw", "label", "true_label", "block_id", "node", "timestamp", "component", "level",
                 "content"]]
        return df, {
            "input_lines": n_in, "blank_or_comment": int(blank.sum()), "duplicates_removed": n_dup,
            "noise_removed": n_noise, "noise_examples": noise_examples, "clean_records": len(df),
            "windows": int(df.block_id.nunique()), "window_size": window,
            "has_ground_truth": bool(df.true_label.any()),
            "steps": ["whitespace normalisation", "blank/comment removal", "exact-duplicate removal",
                      "field parsing (label, time, node, component, level, message)",
                      "timestamp normalisation", "noise removal (lines that are not BGL RAS records)",
                      f"windowing ({window} consecutive messages per session)"],
        }

    # ── Silver ────────────────────────────────────────────────────────
    def build_silver(self, clean, fmt):
        df = clean.copy()
        df["event_id"] = self.event_for(df.content)
        unmatched = int(df.event_id.isna().sum())
        df = df.dropna(subset=["event_id"])
        df["position"] = df.groupby("block_id").cumcount()
        df["ips"] = [[n] for n in df.node]
        df = df.rename(columns={"true_label": "line_label"})
        silver = df[["block_id", "position", "line_no", "timestamp", "level", "component", "event_id", "ips",
                     "line_label"]].reset_index(drop=True)
        return self.enrich_events(silver), unmatched

    def add_session_features(self, gold, silver, clean, fmt):
        # ground truth for evaluation only: a window is anomalous if it contains an alert line
        gold["true_label"] = silver.groupby("block_id").line_label.max().reindex(gold.block_id).values
        levels = silver.groupby(["block_id", "level"]).size().unstack(fill_value=0)
        for level, col in (("FATAL", "n_fatal"), ("ERROR", "n_error"), ("SEVERE", "n_severe"),
                           ("FAILURE", "n_failure"), ("WARNING", "n_warning")):
            gold[col] = levels[level].reindex(gold.block_id).fillna(0).values if level in levels else 0
        return gold

    # ── RCA ───────────────────────────────────────────────────────────
    def build_root_causes(self):
        causes = {}
        for tag, s in read_json(self.dir / "alert_signatures.json", {}).items():
            fam = family_for(tag)
            causes[tag] = {
                "title": f"{tag}: {_shorten(s['top_template'], 60)}",
                "events": set(s["events"]), "weight": s["weight"], "severity": s["severity"],
                "explanation": f"Alert category {tag} ({fam['name']}). In the labelled BGL data, messages such as "
                               f"'{s['top_template'][:120]}' carry this tag. {fam['explanation']}",
                "impact": fam["impact"], "actions": fam["actions"],
            }
        causes["rare_pattern"] = RARE_PATTERN
        return causes
