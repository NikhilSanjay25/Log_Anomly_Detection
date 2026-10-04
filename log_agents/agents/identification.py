"""Log Identification Agent: works out what kind of log was ingested and picks the workflow."""
import re

from .. import hdfs
from .base import Agent, AgentError

SAMPLE_LINES = 2000

# Non-HDFS families we recognise only to give a helpful rejection message
OTHER_FAMILIES = [
    ("Kubernetes / container JSON logs", re.compile(r'^\s*\{.*"(log|msg|message|level)"\s*:')),
    ("Kubernetes component logs (klog)", re.compile(r"^[IWEF]\d{4} \d\d:\d\d:\d\d\.\d+\s+\d+ \S+\.go:\d+\]")),
    ("BGL (BlueGene/L) logs", re.compile(r"^\S+ \d{10} \d{4}\.\d\d\.\d\d \S+ \d{4}-\d\d-\d\d-")),
    ("Syslog", re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d \S+ \S+")),
    ("Apache / Nginx access logs", re.compile(r'^\S+ \S+ \S+ \[\d\d/\w{3}/\d{4}:')),
    ("ISO-timestamped application logs", re.compile(r"^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d")),
]

WORKFLOWS = {
    # format -> (description, steps)
    "hdfs_raw": "Raw HDFS log lines: template matching + block-session grouping",
    "hdfs_trace_csv": "HDFS Event_traces CSV: one pre-built session per row (labels used for evaluation if present)",
    "hdfs_events": "HDFS event-ID sequences: one session per line",
}


def _is_event_line(line):
    body = line.split(":", 1)[1] if re.match(r"^\s*[\w\-]+\s*:", line) else line
    toks = [t for t in re.split(r"[\s,\[\]]+|->|→", body) if t]
    if not toks:
        return False
    known = sum(1 for t in toks if re.fullmatch(r"[Ee]?\d{1,2}", t) and f"E{int(t.lstrip('Ee'))}" in hdfs.EVENT_INFO)
    return known / len(toks) >= 0.8


class LogIdentificationAgent(Agent):
    name = "Log Identification Agent"
    task = "Identify log type and assign log family / workflow"

    def run(self, ctx):
        lines = [ln for ln in ctx.text.splitlines() if ln.strip()][:SAMPLE_LINES]
        if not lines:
            raise AgentError("The input is empty.")
        n = len(lines)
        first = lines[0].lower()

        scores = {
            "hdfs_trace_csv": 1.0 if ("blockid" in first and "features" in first) else 0.0,
            "hdfs_raw": sum(1 for ln in lines
                            if (m := hdfs.LOG_LINE_RE.match(ln.strip())) and
                            (m["component"].startswith("dfs.") or "blk_" in ln)) / n,
            "hdfs_events": sum(1 for ln in lines if _is_event_line(ln)) / n,
        }
        fmt, conf = max(scores.items(), key=lambda kv: kv[1])
        # Raw lines we couldn't parse but that clearly reference HDFS blocks
        if conf < 0.5:
            blk_share = sum(1 for ln in lines if hdfs.BLOCK_RE.search(ln)) / n
            if blk_share >= 0.5:
                fmt, conf = "hdfs_raw", blk_share

        if conf < 0.5:
            hint = next((fam for fam, rx in OTHER_FAMILIES
                         if sum(1 for ln in lines[:200] if rx.search(ln)) >= 0.5 * min(n, 200)), "an unknown format")
            ctx.data["identification"] = {"family": "unsupported", "format": None, "detected": hint,
                                          "confidence": 1 - conf, "scores": scores}
            raise AgentError(f"Input looks like {hint}, not HDFS. This system currently supports HDFS logs only "
                             "(raw HDFS log lines, Event_traces CSV, or event-ID sequences like 'E5 E22 E11 ...').")

        ctx.data["identification"] = {
            "family": "HDFS", "format": fmt, "confidence": round(conf, 3), "scores": scores,
            "workflow": WORKFLOWS[fmt], "lines_sampled": n,
        }
        return f"HDFS log family, format={fmt} (confidence {conf:.0%})"

    def validate(self, ctx):
        ident = ctx.data.get("identification", {})
        return [] if ident.get("format") in WORKFLOWS else ["no supported workflow assigned"]
