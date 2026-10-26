"""Preprocessing & Cleaning Agent: parsing, normalisation, de-duplication, noise removal, formatting.

The dataset profile does the family-specific parsing; this agent runs it and checks the result.
Produces ctx.data["clean"] (the Bronze layer), whose columns depend on the format:
  hdfs_raw        line_no, raw, date, time, timestamp, pid, level, component, content
  hdfs_events     line_no, session_id, events (list)
  hdfs_trace_csv  line_no, session_id, events (list), true_label, latency
  bgl_raw         line_no, raw, label, true_label, block_id (window), node, timestamp, component, level, content
  openstack_raw   line_no, raw, service, timestamp, pid, level, component, block_id (instance), content
"""
from .base import Agent, AgentError


class PreprocessingAgent(Agent):
    name = "Preprocessing & Cleaning Agent"
    task = "Parse, normalise, de-duplicate, remove noise and format"

    def run(self, ctx):
        fmt = ctx.data["identification"]["format"]
        clean, stats = ctx.profile.clean(ctx.text, fmt)
        if clean.empty:
            raise AgentError(f"No usable {ctx.profile.title} records remained after cleaning.")
        ctx.data["clean"] = clean
        ctx.data["preprocessing"] = stats
        return (f"{stats['input_lines']:,} lines → {stats['clean_records']:,} clean records "
                f"(duplicates {stats['duplicates_removed']:,}, noise {stats['noise_removed']:,})")

    def validate(self, ctx):
        clean = ctx.data.get("clean")
        if clean is None or clean.empty:
            return ["no clean records"]
        if "events" in clean and clean.events.map(len).eq(0).any():
            return ["sessions with no events survived cleaning"]
        return []
