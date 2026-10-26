"""Insight Agent: turns detection + RCA results into an explanation, impact analysis, recommendations and
next-best actions, deterministically from the dataset's knowledge base and the evidence the RCA agent gathered.

No language model is involved here. The Qwen model is used only by the follow-up chat (log_agents/chat.py), which
is grounded with the *facts sheet* built by `build_facts` below and with the relevant raw log lines. The helpers
`describe_neighbours`, `build_facts` and `grounding_issues` are shared with the chat.
"""
import json
import re
from collections import Counter

from .. import config
from ..datasets.base import EVENT_ID_RE
from ..hdfs import IP_RE
from .base import Agent

SEVERITIES = {"low", "medium", "high", "critical"}
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text, profile):
    """Numbers in free text, ignoring those inside session ids, node names, IPs and event ids."""
    text = profile.session_re.sub(" ", text)
    if profile.node_re is not None:
        text = profile.node_re.sub(" ", text)
    text = IP_RE.sub(" ", text)
    text = EVENT_ID_RE.sub(" ", text)
    return set(NUMBER_RE.findall(text))


def _relation(query, seq, max_len):
    """How a retrieved sequence differs from the session's own last `max_len` events (what the model saw)."""
    query = list(query)[-max_len:]
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


def describe_neighbours(query, similar, max_len=config.MAX_SEQ_LEN):
    """Facts lines for a session's top-k FAISS neighbours. Their session ids are left out on purpose: they are
    not part of this run, so the grounding check would flag any answer that repeated them."""
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
        n, rel = ns[0], _relation(query, seq, max_len)
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


def build_facts(ctx):
    """Everything the run established, as plain text. The chat model is told to answer only from this."""
    profile = ctx.profile
    det = ctx.data["detection"]["sessions"]
    rca = ctx.data["rca"]
    noun_pl = profile.session_noun_plural
    lines = [f"Log source: {ctx.source_name} ({profile.title})",
             f"DATA MODEL: {profile.glossary}",
             f"{noun_pl.capitalize()} analysed: {len(det)}; anomalous: {int(det.is_anomaly.sum())} "
             f"({det.is_anomaly.mean():.1%})"]
    for c in rca["causes"][:4]:
        lines.append(f"\nROOT CAUSE GROUP: {c['title']} - {c['count']} session(s), severity {c['severity']}, "
                     f"heuristic rule-score share {c['avg_share']:.0%} (not a probability)")
        lines.append(f"  Knowledge base: {c['explanation']}")
        if c["key_events"]:
            lines.append(f"  Key error events: {'; '.join(c['key_events'])}")
        lines.append(f"  Example {noun_pl}: {', '.join(c['example_blocks'][:3])}")
        lines.append(f"  Example sequence: {c['example_sequence'][:300]}")
        ex = next(p for p in rca["sessions"] if p["primary_cause"] == c["cause"])
        for e in ex["evidence"][:6]:
            lines.append(f"  Evidence: {e}")
        for ln in describe_neighbours(ex["sequence"].split(), ex.get("similar_logs"), profile.max_seq_len):
            lines.append(f"  {ln}")
        lines.append(f"  Suggested actions (knowledge base): {' | '.join(c['actions'][:3])}")
    corr = rca["correlations"]
    for n in corr["nodes"][:3]:
        lines.append(f"\nCORRELATION: {profile.node_noun} {n['node']} is involved in {n['anomalous_blocks']} anomalous of "
                     f"{n['total_blocks']} {noun_pl} ({n['anomaly_share']:.0%})")
    for b in corr["time_bursts"][:3]:
        lines.append(f"CORRELATION: burst of {b['anomalies']} anomalies at {b['minute']}")
    for p in corr["patterns"][:2]:
        lines.append(f"CORRELATION: identical anomalous pattern repeated in {p['blocks']} {noun_pl}")
    n_norm = int((det.is_anomaly == 0).sum())
    if n_norm:
        share = normal_neighbour_share(det)
        lines.append(f"\nNORMAL SESSIONS: {n_norm} session(s) classified normal"
                     + (f"; their retrieved similar historical sessions are on average {share} normal" if share else ""))
        for p in rca.get("normal_examples", []):
            lines.append(f"\nNORMAL EXAMPLE ({p['why_picked']}): {profile.session_noun} {p['block_id']}")
            lines.append(f"  Sequence: {p['sequence'][:300]}")
            for e in p["evidence"]:
                lines.append(f"  Evidence: {e}")
            for ln in describe_neighbours(p["sequence"].split(), p.get("similar_logs"), profile.max_seq_len):
                lines.append(f"  {ln}")
    prev = ctx.data.get("memory", [])
    if prev:
        p = prev[0]
        lines.append(f"\nPREVIOUS RUN ({p.get('source')}): {p.get('anomalies')} anomalies in {p.get('sessions')} sessions")
    return "\n".join(lines)


def _first_actions(causes, n=3):
    """The top action of each leading cause, without repeats (causes of one family share actions)."""
    out = []
    for c in causes:
        for a in c["actions"]:
            if a not in out:
                out.append(a)
                break
        if len(out) == n:
            break
    return out


def deterministic_insight(ctx):
    profile = ctx.profile
    noun, noun_pl = profile.session_noun, profile.session_noun_plural
    det = ctx.data["detection"]["sessions"]
    rca = ctx.data["rca"]
    n_anom = int(det.is_anomaly.sum())
    if n_anom == 0:
        return {"summary": f"All {len(det)} {noun_pl} look normal; no anomalies were detected.",
                "explanation": _normal_sentence(det, f"Every {noun}") + " " + profile.normal_description,
                "impact": {"severity": "low", "description": "No operational impact detected.", "affected": []},
                "recommendations": ["Keep monitoring; re-run analysis on new log windows"],
                "next_best_actions": ["No action required"]}
    causes = rca["causes"]
    n_norm = len(det) - n_anom
    normal = {"normal_explanation": _normal_sentence(det, f"The other {n_norm} {noun}(s)")} if n_norm else {}
    if not causes:  # RCA failed or produced nothing: report detection results only
        return {"summary": f"{n_anom} of {len(det)} {noun_pl} ({n_anom / len(det):.1%}) are anomalous. "
                           "Root cause analysis produced no result for this run.",
                "explanation": "The anomaly detector flagged these sessions, but no root-cause evidence is available.",
                **normal,
                "impact": {"severity": "medium", "description": "Unknown - review the anomalous sessions manually.",
                           "affected": []},
                "recommendations": ["Review the anomalous session timelines in the Evidence tab"],
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
        "summary": f"{n_anom} of {len(det)} {noun_pl} ({n_anom / len(det):.1%}) are anomalous. "
                   f"The dominant root cause is '{top['title']}' ({top['count']} {noun_pl}).",
        "explanation": " ".join(f"{c['title']}: {c['explanation']}" for c in causes[:3]),
        **normal,
        "impact": {"severity": sev, "description": top["impact"], "affected": affected},
        "recommendations": recs[:5],
        "next_best_actions": _first_actions(causes),
    }


def grounding_issues(text, ctx, facts=None, prose=None):
    """What `text` states that this run cannot back up: unknown session ids, event ids, node names or IPs, and
    (when `facts` is given) numbers in `prose` (default: all of `text`) that never appear in the facts.
    Used to check the knowledge-base insight and to flag chat answers."""
    profile = ctx.profile
    det = ctx.data["detection"]["sessions"]
    problems = []
    seen_in_facts = facts or ""

    known_sessions = set(det.block_id)
    invented = [b for b in set(profile.session_re.findall(text)) if b not in known_sessions and b not in seen_in_facts]
    if invented:
        problems.append(f"mentions unknown {profile.session_noun} ids {invented[:3]}")
    bad_events = {e for e in EVENT_ID_RE.findall(text) if e not in profile.event_info}
    if bad_events:
        problems.append(f"mentions unknown events {sorted(bad_events)}")
    known_nodes = {n for nodes in det.nodes for n in nodes}
    bad_nodes = {ip for ip in IP_RE.findall(text) if ip not in known_nodes and ip not in seen_in_facts}
    if profile.node_re is not None:
        bad_nodes |= {n for n in profile.node_re.findall(text) if n not in known_nodes and n not in seen_in_facts}
    if bad_nodes:
        problems.append(f"mentions unknown node addresses {sorted(bad_nodes)[:3]}")
    if facts:
        invented_numbers = _numbers(text if prose is None else prose, profile) - _numbers(facts, profile)
        if invented_numbers:
            problems.append(f"states numbers not present in the facts {sorted(invented_numbers)[:5]}")
    return problems


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
    # every session / event / node must exist in this run, every number in the prose must come from the facts
    prose = " ".join([str(ins.get("summary", "")), str(ins.get("explanation", "")),
                      str(ins.get("normal_explanation", "")),
                      str((imp or {}).get("description", "")) if isinstance(imp, dict) else ""])
    return problems + grounding_issues(json.dumps(ins), ctx, ctx.data.get("insight_facts"), prose)


class InsightAgent(Agent):
    name = "Insight Agent"
    task = "Generate explanation, impact analysis, recommendations and next-best actions (knowledge base)"

    def __init__(self, resources=None):
        self.res = resources

    def run(self, ctx):
        facts = build_facts(ctx)
        ctx.data["insight_facts"] = facts            # the grounding material of the follow-up chat
        ctx.data["insight"] = {**deterministic_insight(ctx), "source": "knowledge base", "facts": facts}
        return "Knowledge-base insight generated from the RCA evidence (the language model is only used in the chat)"

    def validate(self, ctx):
        return validate_insight(ctx.data.get("insight"), ctx)
