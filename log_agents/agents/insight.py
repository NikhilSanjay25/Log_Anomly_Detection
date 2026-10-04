"""Insight Agent: uses an SLM (Qwen3-1.7B) to turn detection + RCA results into explanations,
impact analysis, recommendations and next-best actions.

The SLM is grounded with the RCA evidence and the knowledge base, must answer in JSON, and its
output is validated (schema, severity value, no invented block ids / event ids). If the SLM is
disabled, unavailable or keeps failing validation, a deterministic knowledge-base insight is used.
"""
import json
import re

from .. import hdfs
from .base import Agent

SEVERITIES = {"low", "medium", "high", "critical"}
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text):
    """Numbers in free text, ignoring those inside block ids, IPs and event ids."""
    text = hdfs.BLOCK_RE.sub(" ", text)
    text = hdfs.IP_RE.sub(" ", text)
    text = re.sub(r"E\d{1,3}", " ", text)
    return set(NUMBER_RE.findall(text))
SYSTEM = ("You are a senior Hadoop/HDFS site-reliability engineer. You explain log anomaly analysis results "
          "to operators. Use ONLY the facts provided. Never invent block ids, event ids, node addresses or numbers. "
          "Answer with a single JSON object and nothing else.")
SCHEMA = """{
  "summary": "2-3 sentence overview of what happened",
  "explanation": "why these sessions are anomalous, referring to the specific events and root causes given",
  "impact": {"severity": "low|medium|high|critical", "description": "operational / data-durability impact",
             "affected": ["affected components, nodes or block groups taken from the facts"]},
  "recommendations": ["3-5 concrete remediation steps"],
  "next_best_actions": ["2-4 prioritised actions the operator should take first"]
}"""


def build_facts(ctx):
    det = ctx.data["detection"]["sessions"]
    rca = ctx.data["rca"]
    lines = [f"Log source: {ctx.source_name} (HDFS)",
             f"Block sessions analysed: {len(det)}; anomalous: {int(det.is_anomaly.sum())} "
             f"({det.is_anomaly.mean():.1%})"]
    for c in rca["causes"][:4]:
        lines.append(f"\nROOT CAUSE GROUP: {c['title']} - {c['count']} session(s), severity {c['severity']}, "
                     f"heuristic rule-score share {c['avg_share']:.0%} (not a probability)")
        lines.append(f"  Knowledge base: {c['explanation']}")
        if c["key_events"]:
            lines.append(f"  Key error events: {'; '.join(c['key_events'])}")
        lines.append(f"  Example blocks: {', '.join(c['example_blocks'][:3])}")
        lines.append(f"  Example sequence: {c['example_sequence'][:300]}")
        ex = next(p for p in rca["sessions"] if p["primary_cause"] == c["cause"])
        for e in ex["evidence"][:6]:
            lines.append(f"  Evidence: {e}")
        lines.append(f"  Suggested actions (knowledge base): {' | '.join(c['actions'][:3])}")
    corr = rca["correlations"]
    for n in corr["nodes"][:3]:
        lines.append(f"\nCORRELATION: DataNode {n['node']} is involved in {n['anomalous_blocks']} anomalous of "
                     f"{n['total_blocks']} blocks ({n['anomaly_share']:.0%})")
    for b in corr["time_bursts"][:3]:
        lines.append(f"CORRELATION: burst of {b['anomalies']} anomalies at {b['minute']}")
    for p in corr["patterns"][:2]:
        lines.append(f"CORRELATION: identical anomalous pattern repeated in {p['blocks']} blocks")
    prev = ctx.data.get("memory", [])
    if prev:
        p = prev[0]
        lines.append(f"\nPREVIOUS RUN ({p.get('source')}): {p.get('anomalies')} anomalies in {p.get('sessions')} sessions")
    return "\n".join(lines)


def deterministic_insight(ctx):
    det = ctx.data["detection"]["sessions"]
    rca = ctx.data["rca"]
    n_anom = int(det.is_anomaly.sum())
    if n_anom == 0:
        return {"summary": f"All {len(det)} block sessions look normal; no anomalies were detected.",
                "explanation": "Every session followed the normal HDFS block lifecycle (allocate → receive → "
                               "store → optional read/delete) and resembled normal historical sessions.",
                "impact": {"severity": "low", "description": "No operational impact detected.", "affected": []},
                "recommendations": ["Keep monitoring; re-run analysis on new log windows"],
                "next_best_actions": ["No action required"]}
    causes = rca["causes"]
    if not causes:  # RCA failed or produced nothing: report detection results only
        return {"summary": f"{n_anom} of {len(det)} block sessions ({n_anom / len(det):.1%}) are anomalous. "
                           "Root cause analysis produced no result for this run.",
                "explanation": "The anomaly detector flagged these sessions, but no root-cause evidence is available.",
                "impact": {"severity": "medium", "description": "Unknown - review the anomalous sessions manually.",
                           "affected": []},
                "recommendations": ["Review the anomalous block timelines in the Evidence tab"],
                "next_best_actions": ["Inspect the highest-probability anomalous sessions first"]}
    top = causes[0]
    sev = "critical" if top["severity"] == "high" and n_anom / len(det) > 0.2 else top["severity"]
    affected = [n["node"] for n in rca["correlations"]["nodes"][:5]] or top["example_blocks"][:3]
    recs, seen = [], set()
    for c in causes:
        for a in c["actions"]:
            if a not in seen:
                seen.add(a)
                recs.append(a)
    return {
        "summary": f"{n_anom} of {len(det)} block sessions ({n_anom / len(det):.1%}) are anomalous. "
                   f"The dominant root cause is '{top['title']}' ({top['count']} sessions).",
        "explanation": " ".join(f"{c['title']}: {c['explanation']}" for c in causes[:3]),
        "impact": {"severity": sev, "description": top["impact"], "affected": affected},
        "recommendations": recs[:5],
        "next_best_actions": [c["actions"][0] for c in causes[:3]],
    }


def validate_insight(ins, ctx):
    problems = []
    if not isinstance(ins, dict):
        return ["not a JSON object"]
    for key in ("summary", "explanation", "impact", "recommendations", "next_best_actions"):
        if not ins.get(key):
            problems.append(f"missing '{key}'")
    imp = ins.get("impact")
    if isinstance(imp, dict):
        if str(imp.get("severity", "")).lower() not in SEVERITIES:
            problems.append("invalid severity")
        if not isinstance(imp.get("description", ""), str):
            problems.append("impact.description must be text")
        affected = imp.get("affected", [])
        if not isinstance(affected, list) or not all(isinstance(a, str) for a in affected):
            problems.append("impact.affected must be a list of strings")
    elif imp:
        problems.append("impact must be an object")
    for key in ("summary", "explanation"):
        if ins.get(key) is not None and not isinstance(ins.get(key), str):
            problems.append(f"'{key}' must be text")
    for key in ("recommendations", "next_best_actions"):
        val = ins.get(key)
        if val is not None and (not isinstance(val, list) or not all(isinstance(v, str) and v.strip() for v in val)):
            problems.append(f"'{key}' must be a list of non-empty strings")
    text = json.dumps(ins)
    known_blocks = set(ctx.data["detection"]["sessions"].block_id)
    invented = [b for b in set(hdfs.BLOCK_RE.findall(text)) if b not in known_blocks]
    if invented:
        problems.append(f"mentions unknown block ids {invented[:3]}")
    bad_events = {e for e in re.findall(r"\bE\d{1,3}\b", text) if e not in hdfs.EVENT_INFO}
    if bad_events:
        problems.append(f"mentions unknown events {sorted(bad_events)}")
    known_nodes = {n for nodes in ctx.data["detection"]["sessions"].nodes for n in nodes}
    bad_nodes = {ip for ip in hdfs.IP_RE.findall(text) if ip not in known_nodes}
    if bad_nodes:
        problems.append(f"mentions unknown node addresses {sorted(bad_nodes)[:3]}")
    facts = ctx.data.get("insight_facts")
    if facts:  # every number the SLM states must come from the facts it was given
        prose = " ".join([str(ins.get("summary", "")), str(ins.get("explanation", "")),
                          str((imp or {}).get("description", "")) if isinstance(imp, dict) else ""])
        invented_numbers = _numbers(prose) - _numbers(facts)
        if invented_numbers:
            problems.append(f"states numbers not present in the facts {sorted(invented_numbers)[:5]}")
    return problems


class InsightAgent(Agent):
    name = "Insight Agent"
    task = "Generate explanation, impact analysis, recommendations and next-best actions (SLM)"

    def __init__(self, resources):
        self.res = resources

    def run(self, ctx):
        n_anom = int(ctx.data["detection"]["sessions"].is_anomaly.sum())
        if not ctx.options.use_slm_insight or n_anom == 0:
            ctx.data["insight"] = {**deterministic_insight(ctx), "source": "knowledge base",
                                   "note": "SLM disabled" if n_anom else "no anomalies - SLM not needed"}
            return f"Knowledge-base insight ({ctx.data['insight']['note']})"

        slm = self.res.insight_slm()
        facts = build_facts(ctx)
        ctx.data["insight_facts"] = facts
        user = f"FACTS:\n{facts}\n\nReturn JSON with exactly this structure:\n{SCHEMA}"
        problems, raw = [], ""
        for attempt in range(2):
            try:
                parsed, raw = slm.generate_json(SYSTEM, user if attempt == 0 else
                                                user + "\n\nYour previous answer was invalid: "
                                                + "; ".join(problems) + ". Output valid JSON only.")
            except Exception as e:
                problems = [f"SLM error: {type(e).__name__}: {e}"]
                break
            problems = validate_insight(parsed, ctx)
            if not problems:
                parsed["impact"]["severity"] = parsed["impact"]["severity"].lower()
                ctx.data["insight"] = {**parsed, "source": f"SLM ({slm.model_name})", "attempts": attempt + 1,
                                       "facts": facts}
                return f"SLM insight generated by {slm.model_name} (attempt {attempt + 1})"

        ctx.warn(f"SLM insight rejected ({'; '.join(problems)}); used knowledge-base insight instead.")
        ctx.data["insight"] = {**deterministic_insight(ctx), "source": "knowledge base (SLM fallback)",
                               "slm_problems": problems, "slm_raw": raw[:2000], "facts": facts}
        return "SLM output failed validation - knowledge-base fallback used"

    def validate(self, ctx):
        return validate_insight(ctx.data.get("insight"), ctx)
