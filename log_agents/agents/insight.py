"""Insight Agent: uses an SLM (Qwen3-1.7B) to turn detection + RCA results into explanations,
impact analysis, recommendations and next-best actions.

The SLM is grounded with the RCA evidence, the knowledge base and the top-k similar historical
sessions retrieved by FAISS (for anomalous groups and for representative normal sessions), must answer
in JSON, and its output is validated (schema, severity value, no invented block ids / event ids /
numbers). It explains why sessions are anomalous and, when the run has any, why the rest are normal.
If the SLM is disabled, unavailable or keeps failing validation, a deterministic knowledge-base
insight is used.
"""
import json
import re
from collections import Counter

from .. import config, hdfs
from .base import Agent

SEVERITIES = {"low", "medium", "high", "critical"}
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text):
    """Numbers in free text, ignoring those inside block ids, IPs and event ids."""
    text = hdfs.BLOCK_RE.sub(" ", text)
    text = hdfs.IP_RE.sub(" ", text)
    text = re.sub(r"E\d{1,3}", " ", text)
    return set(NUMBER_RE.findall(text))


def _relation(query, seq):
    """How a retrieved sequence differs from the session's own last MAX_SEQ_LEN events (what the model saw)."""
    query = list(query)[-config.MAX_SEQ_LEN:]
    if query == list(seq):
        return "identical to this session"
    q, s = Counter(query), Counter(seq)
    if q == s:
        return "same events in a different order"

    def fmt(c):
        return ", ".join(f"{e} ×{k}" for e, k in sorted(c.items(), key=lambda kv: int(kv[0][1:])))
    parts = []
    if s - q:
        parts.append(f"has {fmt(s - q)} that this session lacks")
    if q - s:
        parts.append(f"lacks {fmt(q - s)} that this session has")
    return "; ".join(parts)


def describe_neighbours(query, similar):
    """Facts lines for a session's top-k FAISS neighbours. Their block ids are left out on purpose: they are
    not part of this run, so the validator would reject any answer that repeated them."""
    if not similar:
        return []
    groups = {}
    for n in similar:
        groups.setdefault(tuple(n.get("sequence") or ()), []).append(n)
    labels = Counter(n["label"] for n in similar if n.get("label"))
    head = (f"Similar historical sessions to this example (top {len(similar)} FAISS matches in the training "
            f"data): {len(groups)} distinct pattern(s)")
    if labels:
        head += "; " + ", ".join(f"{v} of {len(similar)} labelled {k}" for k, v in labels.most_common())
    lines = [head]
    for i, (seq, ns) in enumerate(groups.items(), 1):
        if not seq:
            lines.append(f"  Match {i}: sequence unavailable (rag_metadata.npz missing)")
            continue
        n, rel = ns[0], _relation(query, seq)
        lines.append(f"  Match {i} ({len(ns)} of {len(similar)} retrieved): {n['occurrences']:,} training "
                     f"session(s) had this exact pattern, {n['anomaly_rate']:.0%} of them anomalous; {rel}")
        if rel != "identical to this session":
            lines.append(f"    Sequence: {' '.join(seq)[:240]}")
    return lines


def normal_neighbour_share(det):
    """e.g. '97%': how normal the retrieved neighbours of the normal sessions are on average (None if unknown)."""
    norm = det[det.is_anomaly == 0]
    if norm.empty or "nb_anomaly_rate" not in norm or norm.nb_anomaly_rate.isna().all():
        return None
    return f"{1 - norm.nb_anomaly_rate.mean():.0%}"


def _normal_sentence(det, prefix):
    share = normal_neighbour_share(det)
    return (prefix + " scored below the anomaly threshold"
            + (f", and their most similar historical sessions were on average {share} normal" if share else "") + ".")
SYSTEM = ("You are a senior Hadoop/HDFS site-reliability engineer. You explain log anomaly analysis results "
          "to operators. Use ONLY the facts provided. Never invent block ids, event ids, node addresses or numbers. "
          "The facts include similar historical sessions: the top matches retrieved by vector search over the "
          "labelled training data, with how often each exact pattern occurred and how often it was anomalous. "
          "Use them as evidence: say what a session shares with past anomalous or normal sessions and how it "
          "differs. Answer with a single JSON object and nothing else.")


def schema(n_anom, n_norm):
    """The JSON the SLM must return. It depends on whether the run has anomalies, normal sessions or both."""
    if n_anom:
        explanation = ("why the anomalous sessions are anomalous, referring to the specific events, root causes "
                       "and how they compare with their similar historical sessions")
    else:
        explanation = ("why these sessions are considered normal, referring to the block lifecycle, the events "
                       "present and how they compare with their similar historical sessions. If a NORMAL EXAMPLE "
                       "contains error or warning events, name them and explain why it was still judged normal")
    fields = ['  "summary": "2-3 sentence overview of what happened, using the session counts exactly as given '
              'in the facts"', f'  "explanation": "{explanation}"']
    if n_anom and n_norm:
        fields.append('  "normal_explanation": "why the remaining sessions were not flagged, using the NORMAL '
                      'SESSIONS facts and their similar historical sessions. If a NORMAL EXAMPLE contains error or '
                      'warning events, name them and explain why it was still judged normal"')
    fields += ['  "impact": {"severity": "low|medium|high|critical", "description": "operational / '
               'data-durability impact",\n             "affected": ["affected components, nodes or block groups '
               'taken from the facts"]}',
               '  "recommendations": ["3-5 concrete remediation steps"]' if n_anom
               else '  "recommendations": ["2-3 monitoring or follow-up steps"]',
               '  "next_best_actions": ["2-4 prioritised actions the operator should take first"]']
    return "{\n" + ",\n".join(fields) + "\n}"


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
        for ln in describe_neighbours(ex["sequence"].split(), ex.get("similar_logs")):
            lines.append(f"  {ln}")
        lines.append(f"  Suggested actions (knowledge base): {' | '.join(c['actions'][:3])}")
    corr = rca["correlations"]
    for n in corr["nodes"][:3]:
        lines.append(f"\nCORRELATION: DataNode {n['node']} is involved in {n['anomalous_blocks']} anomalous of "
                     f"{n['total_blocks']} blocks ({n['anomaly_share']:.0%})")
    for b in corr["time_bursts"][:3]:
        lines.append(f"CORRELATION: burst of {b['anomalies']} anomalies at {b['minute']}")
    for p in corr["patterns"][:2]:
        lines.append(f"CORRELATION: identical anomalous pattern repeated in {p['blocks']} blocks")
    n_norm = int((det.is_anomaly == 0).sum())
    if n_norm:
        share = normal_neighbour_share(det)
        lines.append(f"\nNORMAL SESSIONS: {n_norm} session(s) classified normal"
                     + (f"; their retrieved similar historical sessions are on average {share} normal" if share else ""))
        for p in rca.get("normal_examples", []):
            lines.append(f"\nNORMAL EXAMPLE ({p['why_picked']}): block {p['block_id']}")
            lines.append(f"  Sequence: {p['sequence'][:300]}")
            for e in p["evidence"]:
                lines.append(f"  Evidence: {e}")
            for ln in describe_neighbours(p["sequence"].split(), p.get("similar_logs")):
                lines.append(f"  {ln}")
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
                "explanation": _normal_sentence(det, "Every session")
                               + " Normal HDFS blocks are allocated, received by each replica, stored on the "
                                 "NameNode and later read or deleted.",
                "impact": {"severity": "low", "description": "No operational impact detected.", "affected": []},
                "recommendations": ["Keep monitoring; re-run analysis on new log windows"],
                "next_best_actions": ["No action required"]}
    causes = rca["causes"]
    n_norm = len(det) - n_anom
    normal = {"normal_explanation": _normal_sentence(det, f"The other {n_norm} session(s)")} if n_norm else {}
    if not causes:  # RCA failed or produced nothing: report detection results only
        return {"summary": f"{n_anom} of {len(det)} block sessions ({n_anom / len(det):.1%}) are anomalous. "
                           "Root cause analysis produced no result for this run.",
                "explanation": "The anomaly detector flagged these sessions, but no root-cause evidence is available.",
                **normal,
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
        **normal,
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
    det = ctx.data["detection"]["sessions"]
    n_anom = int(det.is_anomaly.sum())
    if 0 < n_anom < len(det) and not ins.get("normal_explanation"):
        problems.append("missing 'normal_explanation'")
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
    for key in ("summary", "explanation", "normal_explanation"):
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
                          str(ins.get("normal_explanation", "")),
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
        det = ctx.data["detection"]["sessions"]
        n_anom = int(det.is_anomaly.sum())
        if not ctx.options.use_slm_insight:
            ctx.data["insight"] = {**deterministic_insight(ctx), "source": "knowledge base", "note": "SLM disabled"}
            return "Knowledge-base insight (SLM disabled)"

        slm = self.res.insight_slm()
        facts = build_facts(ctx)
        ctx.data["insight_facts"] = facts
        user = f"FACTS:\n{facts}\n\nReturn JSON with exactly this structure:\n{schema(n_anom, len(det) - n_anom)}"
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
