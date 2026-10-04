"""Medallion-architecture data processing.

Bronze: raw cleaned logs (output of the Preprocessing agent, persisted as-is)
Silver: structured + enriched rows - one row per (block, event) with template, lifecycle stage,
        severity, IPs/DataNodes, timestamps
Gold:   aggregated per-block sessions (event sequence, counts, durations, error events, nodes)
        plus correlation tables (DataNode → blocks, time-window activity)
"""
import re

import pandas as pd

from . import config, hdfs
from .agents.base import Agent, AgentError

DIGITS_RE = re.compile(r"\d+")


def _match_cached(contents):
    """Template-match with a cache keyed on the line 'shape' (ids/IPs/numbers masked)."""
    cache = {}
    eids = []
    for c in contents:
        key = DIGITS_RE.sub("0", hdfs.IP_RE.sub("<ip>", hdfs.BLOCK_RE.sub("<blk>", c)))
        if key not in cache:
            cache[key] = hdfs.match_template(c)[0]
        eids.append(cache[key])
    return eids


def build_silver(clean, fmt):
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

    info = silver.event_id.map(hdfs.EVENT_INFO)
    silver = silver.assign(template=silver.event_id.map(hdfs.TEMPLATE_TEXT),
                           stage=info.str[0], severity=info.str[1])
    return silver, unmatched


def build_gold(silver, clean, fmt):
    g = silver.groupby("block_id", sort=False)
    gold = pd.DataFrame({
        "events": g["event_id"].agg(list),
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "warn_lines": g["level"].agg(lambda s: int(s.isin(["WARN", "WARNING", "ERROR", "FATAL"]).sum())),
        "nodes": g["ips"].agg(lambda s: sorted({ip for ips in s for ip in ips})),
    })
    gold["n_events"] = gold.events.map(len)
    gold["sequence"] = gold.events.map(" ".join)
    gold["error_events"] = gold.events.map(lambda e: sorted(set(e) & hdfs.ERROR_EVENTS, key=lambda x: int(x[1:])))
    gold["warning_events"] = gold.events.map(lambda e: sorted(set(e) & hdfs.WARNING_EVENTS, key=lambda x: int(x[1:])))
    gold["duration_s"] = (gold.end - gold.start).dt.total_seconds()
    if fmt == "hdfs_raw":
        gaps = silver.sort_values(["block_id", "timestamp"]).groupby("block_id").timestamp.diff().dt.total_seconds()
        gold["max_gap_s"] = gaps.groupby(silver.block_id).max()
    else:
        gold["max_gap_s"] = float("nan")
    if fmt == "hdfs_trace_csv":
        extra = clean.set_index("session_id")
        for col in ("true_label", "latency"):
            if col in extra:
                gold[col] = extra[col].reindex(gold.index)
        if "latency" in gold:
            gold["duration_s"] = gold.latency
    gold.index.name = "block_id"
    return gold.reset_index()


def build_correlations(silver, gold):
    node_blocks = (gold[["block_id", "nodes"]].explode("nodes").dropna()
                   .groupby("nodes").block_id.nunique().sort_values(ascending=False))
    ts = silver.dropna(subset=["timestamp"])
    per_minute = ts.groupby(ts.timestamp.dt.floor("min")).block_id.nunique() if len(ts) else pd.Series(dtype=int)
    return {"node_blocks": node_blocks, "blocks_per_minute": per_minute}


class MedallionProcessor(Agent):
    name = "Medallion Data Processing"
    task = "Organise data into Bronze, Silver and Gold layers"

    def run(self, ctx):
        fmt = ctx.data["identification"]["format"]
        bronze = ctx.data["clean"]
        silver, unmatched = build_silver(bronze, fmt)
        if silver.empty:
            raise AgentError("No log line matched a known HDFS event template.")
        gold = build_gold(silver, bronze, fmt)
        ctx.data["medallion"] = {"bronze": bronze, "silver": silver, "gold": gold,
                                 "correlations": build_correlations(silver, gold),
                                 "unmatched_lines": unmatched}
        if ctx.options.persist:
            out = config.MEDALLION_DIR / ctx.run_id
            out.mkdir(parents=True, exist_ok=True)
            bronze.to_csv(out / "bronze.csv", index=False)
            silver.to_csv(out / "silver.csv", index=False)
            gold.to_csv(out / "gold.csv", index=False)
            corr = ctx.data["medallion"]["correlations"]
            corr["node_blocks"].rename("blocks").to_csv(out / "gold_node_activity.csv", index_label="node")
            corr["blocks_per_minute"].rename("active_blocks").to_csv(out / "gold_timeline.csv", index_label="minute")
            ctx.data["medallion"]["path"] = str(out)
        msg = f"Bronze {len(bronze):,} rows → Silver {len(silver):,} events → Gold {len(gold):,} block sessions"
        if unmatched:
            msg += f" ({unmatched:,} lines matched no template)"
        return msg

    def validate(self, ctx):
        gold = ctx.data.get("medallion", {}).get("gold")
        if gold is None or gold.empty:
            return ["gold layer is empty"]
        if gold.n_events.min() < 1:
            return ["gold contains empty sessions"]
        return []
