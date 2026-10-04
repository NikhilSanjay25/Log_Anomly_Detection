"""Preprocessing & Cleaning Agent: parsing, normalisation, de-duplication, noise removal, formatting.

Produces ctx.data["clean"]: a DataFrame of cleaned records whose columns depend on the format
  hdfs_raw        line_no, raw, date, time, timestamp, pid, level, component, content
  hdfs_events     line_no, session_id, events (list)
  hdfs_trace_csv  line_no, session_id, events (list), true_label, latency
"""
import io
import re

import pandas as pd

from .. import hdfs
from .base import Agent, AgentError

WS_RE = re.compile(r"\s+")


class PreprocessingAgent(Agent):
    name = "Preprocessing & Cleaning Agent"
    task = "Parse, normalise, de-duplicate, remove noise and format"

    def run(self, ctx):
        fmt = ctx.data["identification"]["format"]
        handler = {"hdfs_raw": self._raw, "hdfs_events": self._events, "hdfs_trace_csv": self._csv}[fmt]
        clean, stats = handler(ctx.text)
        if clean.empty:
            raise AgentError("No usable HDFS records remained after cleaning.")
        ctx.data["clean"] = clean
        ctx.data["preprocessing"] = stats
        return (f"{stats['input_lines']:,} lines → {stats['clean_records']:,} clean records "
                f"(duplicates {stats['duplicates_removed']:,}, noise {stats['noise_removed']:,})")

    # ── raw HDFS log lines ────────────────────────────────────────────────
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

    # ── event-ID sequences, one session per line ──────────────────────────
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

    # ── Event_traces.csv ──────────────────────────────────────────────────
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

    def validate(self, ctx):
        clean = ctx.data.get("clean")
        if clean is None or clean.empty:
            return ["no clean records"]
        if "events" in clean and clean.events.map(len).eq(0).any():
            return ["sessions with no events survived cleaning"]
        return []
