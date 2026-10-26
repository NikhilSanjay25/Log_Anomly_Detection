"""Log Identification Agent: works out which supported dataset the input belongs to and picks the workflow.

Every dataset profile scores the sampled lines for each of its formats (HDFS: raw lines / trace CSV / event
sequences; BGL: raw RAS lines; OpenStack: raw nova lines). The best (dataset, format) above 50% is used.
Anything else is rejected, with a guess at what the input is so the message is useful.
"""
import re

from ..datasets import PROFILES, detect_dataset
from .base import Agent, AgentError

SAMPLE_LINES = 2000
MIN_CONFIDENCE = 0.5

# Families we recognise only to give a helpful rejection message
OTHER_FAMILIES = [
    ("Kubernetes / container JSON logs", re.compile(r'^\s*\{.*"(log|msg|message|level)"\s*:')),
    ("Kubernetes component logs (klog)", re.compile(r"^[IWEF]\d{4} \d\d:\d\d:\d\d\.\d+\s+\d+ \S+\.go:\d+\]")),
    ("Syslog", re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d \S+ \S+")),
    ("Apache / Nginx access logs", re.compile(r'^\S+ \S+ \S+ \[\d\d/\w{3}/\d{4}:')),
    ("ISO-timestamped application logs", re.compile(r"^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d")),
]

SUPPORTED = ("raw HDFS log lines, an HDFS Event_traces.csv or event sequences like 'E5 E22 E11 ...'; "
             "raw BGL (BlueGene/L) RAS log lines; raw OpenStack nova log lines")


class LogIdentificationAgent(Agent):
    name = "Log Identification Agent"
    task = "Identify the dataset and log format, and assign the workflow"

    def run(self, ctx):
        lines = [ln for ln in ctx.text.splitlines() if ln.strip()][:SAMPLE_LINES]
        if not lines:
            raise AgentError("The input is empty.")
        n = len(lines)
        forced = ctx.options.dataset
        profile, fmt, conf, scores = detect_dataset(lines, only=forced)

        if conf < MIN_CONFIDENCE:
            hint = next((fam for fam, rx in OTHER_FAMILIES
                         if sum(1 for ln in lines[:200] if rx.search(ln)) >= 0.5 * min(n, 200)), "an unknown format")
            ctx.data["identification"] = {"family": "unsupported", "format": None, "detected": hint,
                                          "confidence": 1 - conf, "scores": scores}
            if forced:
                raise AgentError(f"The input does not look like {PROFILES[forced].title} logs "
                                 f"(it looks like {hint}). Choose another dataset or use auto-detect.")
            raise AgentError(f"Input looks like {hint}, not a supported log family. This system supports: "
                             f"{SUPPORTED}.")

        ctx.profile = profile
        ctx.data["identification"] = {
            "family": profile.title, "dataset": profile.name, "format": fmt, "confidence": round(conf, 3),
            "scores": scores, "workflow": profile.formats[fmt], "lines_sampled": n,
            "session": profile.session_noun,
        }
        return f"{profile.title} log family, format={fmt} (confidence {conf:.0%})"

    def validate(self, ctx):
        ident = ctx.data.get("identification", {})
        profile = ctx.profile
        return [] if profile is not None and ident.get("format") in profile.formats else ["no supported workflow assigned"]
