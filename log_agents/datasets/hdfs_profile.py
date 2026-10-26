"""HDFS profile: block sessions, the 29 hand-written event templates, three input formats.

The parsing code here is the former HDFS-only code of the Preprocessing agent and the Medallion processor,
moved behind the DatasetProfile interface unchanged.
"""
import io
import re

import pandas as pd

from .. import config, hdfs
from ..knowledge import ROOT_CAUSES
from .base import ArtifactPaths, DatasetProfile

WS_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")


def _is_event_line(line):
    body = line.split(":", 1)[1] if re.match(r"^\s*[\w\-]+\s*:", line) else line
    toks = [t for t in re.split(r"[\s,\[\]]+|->|→", body) if t]
    if not toks:
        return False
    known = sum(1 for t in toks if re.fullmatch(r"[Ee]?\d{1,2}", t) and f"E{int(t.lstrip('Ee'))}" in hdfs.EVENT_INFO)
    return known / len(toks) >= 0.8


def _match_cached(contents):
    """Template-match with a cache keyed on the line 'shape' (ids/IPs/numbers masked)."""
    cache, eids = {}, []
    for c in contents:
        key = DIGITS_RE.sub("0", hdfs.IP_RE.sub("<ip>", hdfs.BLOCK_RE.sub("<blk>", c)))
        if key not in cache:
            cache[key] = hdfs.match_template(c)[0]
        eids.append(cache[key])
    return eids


class HDFSProfile(DatasetProfile):
    name = "hdfs"
    title = "HDFS"
    persona = "senior Hadoop/HDFS site-reliability engineer"
    session_noun = "block"
    session_noun_plural = "blocks"
    node_noun = "DataNode"
    max_seq_len = config.MAX_SEQ_LEN
    session_re = re.compile(r"\b(?:blk_-?\d+|seq_\d+)\b")
    formats = {
        "hdfs_raw": "Raw HDFS log lines: template matching + block-session grouping",
        "hdfs_trace_csv": "HDFS Event_traces CSV: one pre-built session per row (labels used for evaluation if present)",
        "hdfs_events": "HDFS event-ID sequences: one session per line",
    }
    glossary = ("A session is one HDFS block: the ordered events (E1-E29) that DataNodes and the NameNode logged for "
                "that block id. A healthy block is allocated (E22), received by each replica (E5, E9/E6, E11), "
                "stored on the NameNode (E26) and later deleted (E23, E21).")
    normal_description = ("Normal HDFS blocks are allocated, received by each replica, stored on the NameNode and "
                          "later read or deleted.")
    sample_files = ("hdfs_raw_sample.log", "hdfs_sequences.txt", "hdfs_traces_sample.csv")

    @property
    def paths(self):
        return ArtifactPaths(config.ARTIFACTS_DIR, config.BACKBONE_PATH, config.MODEL_CONFIG_PATH,
                             config.EVENT2ID_PATH, config.ID2EVENT_PATH, config.FAISS_PATH, config.RF_PATH,
                             config.RAG_METADATA_PATH, config.EVENT_STATS_PATH)

    @property
    def event_info(self):
        return hdfs.EVENT_INFO

    @property
    def template_text(self):
        return hdfs.TEMPLATE_TEXT

    @property
    def root_causes(self):
        return ROOT_CAUSES

    # ── identification ────────────────────────────────────────────────
    def detect(self, lines):
        n = len(lines)
        first = lines[0].lower()
        scores = {
            "hdfs_trace_csv": 1.0 if ("blockid" in first and "features" in first) else 0.0,
            "hdfs_raw": sum(1 for ln in lines
                            if (m := hdfs.LOG_LINE_RE.match(ln.strip())) and
                            (m["component"].startswith("dfs.") or "blk_" in ln)) / n,
            "hdfs_events": sum(1 for ln in lines if _is_event_line(ln)) / n,
        }
        # Raw lines we couldn't parse but that clearly reference HDFS blocks (only when no format fits)
        if max(scores.values()) < 0.5:
            blk_share = sum(1 for ln in lines if hdfs.BLOCK_RE.search(ln)) / n
            if blk_share >= 0.5:
                scores["hdfs_raw"] = blk_share
        return scores

    # ── Bronze ────────────────────────────────────────────────────────
    def clean(self, text, fmt, **kw):
        return {"hdfs_raw": self._raw, "hdfs_events": self._events, "hdfs_trace_csv": self._csv}[fmt](text)

    def _raw(self, text):
        lines = pd.Series(text.splitlines(), dtype="object")
        n_in = len(lines)
        df = pd.DataFrame({"line_no": range(1, n_in + 1),
                           "raw": lines.str.replace(WS_RE, " ", regex=True).str.strip()})  # normalise whitespace
        blank = df.raw.eq("") | df.raw.str.startswith(("#", "//"))
        df = df[~blank]

        dup = df.raw.duplicated()
        n_dup = int(dup.sum())
        df = df[~dup]

        parts = df.raw.str.extract(hdfs.LOG_LINE_RE)
        df = pd.concat([df, parts], axis=1)
        unparsed = df.content.isna()
        # unparsed lines are kept only if they still reference an HDFS block
        df.loc[unparsed, "content"] = df.loc[unparsed, "raw"]
        has_block = df.content.str.contains(r"blk_-?\d+", regex=True)
        noise = ~has_block
        noise_examples = df.loc[noise, "raw"].head(5).tolist()
        df = df[has_block].copy()

        df["level"] = df.level.fillna("UNKNOWN").str.upper()
        df["component"] = df.component.fillna("unknown")
        df["timestamp"] = pd.to_datetime(df.date.fillna("") + df.time.fillna(""), format="%y%m%d%H%M%S",
                                         errors="coerce")
        df = df[["line_no", "raw", "date", "time", "timestamp", "pid", "level", "component", "content"]]
        return df.reset_index(drop=True), {
            "input_lines": n_in, "blank_or_comment": int(blank.sum()), "duplicates_removed": n_dup,
            "unparsed_kept": int((unparsed & has_block).sum()), "noise_removed": int(noise.sum()),
            "noise_examples": noise_examples, "clean_records": len(df),
            "steps": ["whitespace normalisation", "blank/comment removal", "exact-duplicate removal",
                      "field parsing (date, time, pid, level, component, content)",
                      "timestamp normalisation", "noise removal (lines without an HDFS block id)"],
        }

    def _events(self, text):
        rows, noise, noise_examples = [], 0, []
        n_in = 0
        for i, line in enumerate(text.splitlines(), 1):
            n_in += 1
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            sid, body = None, line
            m = re.match(r"^\s*([\w\-]+)\s*:\s*(.+)$", line)
            if m and not re.fullmatch(r"[Ee]?\d{1,2}", m.group(1)):
                sid, body = m.group(1), m.group(2)
            events = hdfs.parse_event_tokens(body)
            if not events:
                noise += 1
                if len(noise_examples) < 5:
                    noise_examples.append(line)
                continue
            rows.append({"line_no": i, "session_id": sid or f"seq_{len(rows) + 1:04d}", "events": events})
        df = pd.DataFrame(rows, columns=["line_no", "session_id", "events"])
        n_dup = int(df.session_id.duplicated().sum()) if len(df) else 0
        df = df.drop_duplicates("session_id") if len(df) else df
        return df.reset_index(drop=True), {
            "input_lines": n_in, "duplicates_removed": n_dup, "noise_removed": noise,
            "noise_examples": noise_examples, "clean_records": len(df),
            "steps": ["session-id extraction", "event token normalisation (5 → E5, case)",
                      "unknown token removal", "duplicate session removal"],
        }

    def _csv(self, text):
        df = pd.read_csv(io.StringIO(text))
        n_in = len(df)
        cols = {c.lower(): c for c in df.columns}
        out = pd.DataFrame({
            "line_no": range(2, n_in + 2),
            "session_id": df[cols["blockid"]].astype(str).str.strip(),
            "events": df[cols["features"]].astype(str).map(hdfs.parse_event_tokens),
        })
        if "label" in cols:
            lab = df[cols["label"]].astype(str).str.strip().str.lower()
            out["true_label"] = lab.map({"fail": 1, "anomaly": 1, "1": 1, "success": 0, "normal": 0, "0": 0})
        if "latency" in cols:
            out["latency"] = pd.to_numeric(df[cols["latency"]], errors="coerce")
        empty = out.events.map(len) == 0
        noise_examples = [",".join(map(str, row)) for row in df.loc[empty.values].head(5).itertuples(index=False)]
        out = out[~empty]
        dup = out.session_id.duplicated()
        out = out[~dup]
        return out.reset_index(drop=True), {
            "input_lines": n_in, "duplicates_removed": int(dup.sum()), "noise_removed": int(empty.sum()),
            "noise_examples": noise_examples, "clean_records": len(out),
            "has_ground_truth": "true_label" in out.columns,
            "steps": ["CSV parsing", "event list parsing", "label normalisation (Success/Fail → 0/1)",
                      "empty session removal", "duplicate block removal"],
        }

    # ── Silver ────────────────────────────────────────────────────────
    def build_silver(self, clean, fmt):
        if fmt == "hdfs_raw":
            df = clean.copy()
            df["event_id"] = _match_cached(df.content.tolist())
            unmatched = int(df.event_id.isna().sum())
            df = df.dropna(subset=["event_id"])
            # one event per block per line (E21 lines repeat the block id in the file path)
            df["block_id"] = df.content.map(lambda c: list(dict.fromkeys(hdfs.BLOCK_RE.findall(c))))
            df["ips"] = df.content.map(lambda c: sorted(set(hdfs.IP_RE.findall(c))))
            df = df.explode("block_id").reset_index(drop=True)
            df["position"] = df.groupby("block_id").cumcount()
            silver = df[["block_id", "position", "line_no", "timestamp", "level", "component", "event_id", "ips"]]
        else:
            unmatched = 0
            df = clean[["session_id", "events"]].explode("events").rename(
                columns={"session_id": "block_id", "events": "event_id"})
            df["position"] = df.groupby("block_id").cumcount()
            df["timestamp"] = pd.NaT
            df["level"] = "UNKNOWN"
            df["ips"] = [[] for _ in range(len(df))]
            silver = df[["block_id", "position", "timestamp", "level", "event_id", "ips"]].reset_index(drop=True)
        return self.enrich_events(silver), unmatched

    def add_session_features(self, gold, silver, clean, fmt):
        if fmt == "hdfs_trace_csv":
            extra = clean.set_index("session_id")
            for col in ("true_label", "latency"):
                if col in extra:
                    gold[col] = extra[col].reindex(gold.block_id).values
            if "latency" in gold:
                gold["duration_s"] = gold.latency
        return gold

    # ── RCA hooks ─────────────────────────────────────────────────────
    @staticmethod
    def lifecycle(c):
        """Every receiving replica should finish, terminate and be stored. Returns (started, finished, complete)."""
        started, finished = c["E5"], c["E9"] + c["E6"]
        complete = not (finished < started or c["E11"] < c["E9"] or c["E26"] < min(finished, 3))
        return started, finished, complete

    def structural_scores(self, c, present, row, stats):
        scores, evidence = {}, []
        started, finished, complete = self.lifecycle(c)
        if started and not complete:
            scores["incomplete_write"] = ROOT_CAUSES["incomplete_write"]["weight"] * 0.8
            evidence.append(f"Lifecycle: {started} replica write(s) started (E5) but {finished} finished (E9/E6), "
                            f"{c['E11']} PacketResponder terminations (E11), {c['E26']} stored on NameNode (E26)")
        if "E22" not in present and started == 0 and not present & {"E21", "E23"}:
            evidence.append("No block allocation (E22) or write events - session observed only partially")
        p05 = stats.get("normal_length", {}).get("p05", 13)
        if row.n_events < 0.5 * p05:
            scores["abrupt_termination"] = ROOT_CAUSES["abrupt_termination"]["weight"] * 0.9
            evidence.append(f"Only {row.n_events} events vs ≥{int(p05)} in 95% of normal sessions")
        return scores, evidence

    def normal_structural_evidence(self, c, row, stats):
        out = []
        started, finished, complete = self.lifecycle(c)
        if started:
            out.append(("Lifecycle complete: " if complete else "Lifecycle looks incomplete: ")
                        + f"{started} replica write(s) started (E5), {finished} finished (E9/E6), "
                          f"{c['E11']} PacketResponder terminations (E11), {c['E26']} stored on NameNode (E26)")
        return out
