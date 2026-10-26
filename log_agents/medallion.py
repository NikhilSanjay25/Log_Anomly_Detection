"""Medallion-architecture data processing (the same layers for every supported dataset).

Bronze: raw cleaned logs (output of the Preprocessing agent, persisted as-is)
Silver: structured + enriched rows - one row per (session, event) with template, lifecycle stage,
        severity, nodes, timestamps. What a "session" is comes from the dataset profile: an HDFS block,
        a window of BGL messages, or an OpenStack instance.
Gold:   aggregated per-session rows (event sequence, counts, durations, error events, nodes, numeric features)
        plus correlation tables (node -> sessions, time-window activity)
"""
import pandas as pd

from . import config
from .agents.base import Agent, AgentError

WARN_LEVELS = ["WARN", "WARNING", "ERROR", "FATAL", "SEVERE", "FAILURE", "CRITICAL"]


def build_gold(silver, clean, fmt, profile):
    g = silver.groupby("block_id", sort=False)
    gold = pd.DataFrame({
        "events": g["event_id"].agg(list),
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "warn_lines": g["level"].agg(lambda s: int(s.isin(WARN_LEVELS).sum())),
        "nodes": g["ips"].agg(lambda s: sorted({ip for ips in s for ip in ips})),
    })
    errors, warnings_ = profile.error_events, profile.warning_events
    gold["n_events"] = gold.events.map(len)
    gold["sequence"] = gold.events.map(" ".join)
    gold["error_events"] = gold.events.map(lambda e: sorted(set(e) & errors, key=lambda x: int(x[1:])))
    gold["warning_events"] = gold.events.map(lambda e: sorted(set(e) & warnings_, key=lambda x: int(x[1:])))
    gold["duration_s"] = (gold.end - gold.start).dt.total_seconds()
    if silver.timestamp.notna().any():
        gaps = silver.sort_values(["block_id", "timestamp"]).groupby("block_id").timestamp.diff().dt.total_seconds()
        gold["max_gap_s"] = gaps.groupby(silver.block_id).max()
    else:
        gold["max_gap_s"] = float("nan")
    gold.index.name = "block_id"
    gold = gold.reset_index()
    return profile.add_session_features(gold, silver, clean, fmt)


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
        profile = ctx.profile
        fmt = ctx.data["identification"]["format"]
        bronze = ctx.data["clean"]
        silver, unmatched = profile.build_silver(bronze, fmt)
        if silver.empty:
            raise AgentError(f"No log line matched a known {profile.title} event template.")
        gold = build_gold(silver, bronze, fmt, profile)
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
            corr["node_blocks"].rename("sessions").to_csv(out / "gold_node_activity.csv", index_label="node")
            corr["blocks_per_minute"].rename("active_sessions").to_csv(out / "gold_timeline.csv", index_label="minute")
            ctx.data["medallion"]["path"] = str(out)
        noun = profile.session_noun
        msg = f"Bronze {len(bronze):,} rows → Silver {len(silver):,} events → Gold {len(gold):,} {noun} sessions"
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
