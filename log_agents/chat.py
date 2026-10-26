"""Follow-up chat about a finished run. The Qwen model lives here and nowhere else in the system.

Answers are grounded on three things:
  * the run's facts sheet (root causes, evidence, similar historical sessions, normal examples) and the analysis
    the Insight agent already wrote, both in the system prompt;
  * BLOCK DETAILS for every session the question names (its label, evidence, root-cause hypotheses and the FAISS
    neighbours the detector retrieved);
  * RAW LOG LINES chosen for the question: the lines of the named sessions (error and warning lines first), the
    lines of any event id the question mentions, the lines matching the question's keywords, or, when the question
    is general, the lines of the most anomalous sessions.

Answers are not blocked like a report would be, because a conversation should keep going. Instead, anything the
run cannot back up (unknown session ids, event ids, node names / IPs, or numbers absent from what the model was
given, which includes the raw log lines) is returned as a warning for the UI to show next to the answer.
"""
import re

from . import config
from .agents.insight import build_facts, describe_neighbours, grounding_issues
from .datasets.base import EVENT_ID_RE
from .slm import strip_think

SYSTEM = ("You are a {persona} answering follow-up questions about a log anomaly analysis you produced earlier. "
          "Answer ONLY from the FACTS, YOUR EARLIER ANALYSIS and any BLOCK DETAILS or RAW LOG LINES included with a "
          "question. If they do not contain the answer, say so plainly instead of guessing. Never invent session "
          "ids, event ids, node names, IP addresses or numbers. When you rely on a log line, quote it as written. "
          "Similar historical sessions are past sessions retrieved by vector search over the labelled training data. "
          "Root-cause shares are heuristic rule scores, not probabilities. Root-cause groups in the FACTS are ordered "
          "by severity (high, then medium, then low) and then by number of sessions. Be concise: a few sentences or "
          "a short list, in plain text.")
MAX_BLOCKS = 3          # sessions named in one question whose details are added
MAX_LINE_CHARS = 300
STOPWORDS = {"what", "which", "when", "where", "why", "how", "does", "did", "have", "has", "were", "with", "that",
             "this", "these", "those", "from", "there", "their", "about", "into", "than", "then", "them", "they",
             "session", "sessions", "anomaly", "anomalies", "anomalous", "normal", "block", "blocks", "window",
             "windows", "instance", "instances", "log", "logs", "line", "lines", "root", "cause", "causes", "mean",
             "means", "explain", "show", "tell", "happened", "happen", "most", "more", "much", "many", "the", "and"}
GENERAL_HINTS = re.compile(r"anomal|why|cause|fail|error|wrong|happen|worst|top|main|problem|issue|explain|"
                           r"what|which|summar", re.I)


def analysis_text(ins):
    """The report's insight as plain text, so the model can be asked about what it said."""
    imp = ins.get("impact") or {}
    lines = [f"Summary: {ins.get('summary', '')}", f"Explanation: {ins.get('explanation', '')}"]
    if ins.get("normal_explanation"):
        lines.append(f"Why the other sessions look normal: {ins['normal_explanation']}")
    lines.append(f"Impact ({imp.get('severity', '?')}): {imp.get('description', '')}")
    lines += [f"Recommendation: {r}" for r in ins.get("recommendations", [])]
    lines += [f"Next best action: {a}" for a in ins.get("next_best_actions", [])]
    lines.append(f"(This analysis was produced by: {ins.get('source', 'unknown')})")
    return "\n".join(lines)


# ── raw log lines ─────────────────────────────────────────────────────────
def has_raw_logs(ctx):
    med = ctx.data.get("medallion") or {}
    return "raw" in med.get("bronze", ()) and "line_no" in med.get("silver", ())


def _clip(line):
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " …"


def session_lines(ctx, block_id, limit=config.CHAT_MAX_LOG_LINES):
    """Raw log lines of one session: error lines first, then warnings, then the rest, shown in file order.
    Returns (lines, total number of lines of the session)."""
    med = ctx.data["medallion"]
    sv = med["silver"]
    sv = sv[sv.block_id == block_id][["line_no", "severity", "position"]].drop_duplicates("line_no")
    prio = sv.severity.map({"error": 0, "warning": 1}).fillna(2)
    chosen = sv.assign(p=prio).sort_values(["p", "position"]).head(limit).line_no.sort_values()
    raw = med["bronze"].drop_duplicates("line_no").set_index("line_no").raw
    return [_clip(raw[n]) for n in chosen if n in raw.index], len(sv)


def event_lines(ctx, event_ids, limit=12):
    """Raw log lines whose event is one of `event_ids`."""
    med = ctx.data["medallion"]
    sv = med["silver"]
    nums = sv.loc[sv.event_id.isin(event_ids), "line_no"].drop_duplicates().head(limit)
    raw = med["bronze"].drop_duplicates("line_no").set_index("line_no").raw
    return [_clip(raw[n]) for n in nums if n in raw.index]


def keyword_lines(ctx, question, limit=12):
    """Raw log lines containing the most keywords of the question."""
    terms = [t for t in dict.fromkeys(re.findall(r"[A-Za-z0-9_.:\-/]{4,}", question))
             if t.lower() not in STOPWORDS and not EVENT_ID_RE.fullmatch(t)][:6]
    if not terms:
        return []
    raw = ctx.data["medallion"]["bronze"].raw
    hits = sum(raw.str.contains(t, case=False, regex=False).astype(int) for t in terms)
    top = hits[hits > 0].sort_values(ascending=False, kind="stable").head(limit)
    return [_clip(raw[i]) for i in top.index.sort_values()]


def log_context(ctx, question, named):
    """The raw log text added to a question (empty string when the input had no raw lines)."""
    if not has_raw_logs(ctx):
        return ""
    profile = ctx.profile
    sections, budget = [], config.CHAT_MAX_LOG_CHARS

    def add(title, lines, note=""):
        nonlocal budget
        if not lines or budget <= 0:
            return
        body, used = [], 0
        for ln in lines:
            if used + len(ln) > budget:
                break
            body.append(ln)
            used += len(ln) + 1
        budget -= used
        sections.append(f"{title}{note}:\n" + "\n".join(body))

    for b in named[:MAX_BLOCKS]:
        lines, total = session_lines(ctx, b)
        note = f" ({len(lines)} of {total} lines shown, error and warning lines first)" if total > len(lines) else ""
        add(f"RAW LOG LINES of {profile.session_noun} {b}", lines, note)
    if not named:
        events = [e for e in dict.fromkeys(EVENT_ID_RE.findall(question)) if e in profile.event_info][:3]
        for e in events:
            add(f"RAW LOG LINES of event {e} ({profile.describe(e)})", event_lines(ctx, {e}))
        add("RAW LOG LINES matching the question's keywords", keyword_lines(ctx, question))
        if not sections and GENERAL_HINTS.search(question):
            det = ctx.data["detection"]["sessions"]
            for b in det[det.is_anomaly == 1].sort_values("anomaly_proba", ascending=False).block_id.head(2):
                lines, total = session_lines(ctx, b, limit=15)
                add(f"RAW LOG LINES of the most anomalous {profile.session_noun} {b}", lines,
                    f" ({len(lines)} of {total} lines shown, error and warning lines first)"
                    if total > len(lines) else "")
    return "\n\n".join(sections)


def block_details(ctx, rca_agent, block_ids):
    profile = ctx.profile
    out = []
    for b in block_ids[:MAX_BLOCKS]:
        d = rca_agent.describe_block(ctx, b)
        if d is None:
            continue
        lines = [f"{profile.session_noun.upper()} {d['block_id']}: classified {d['label']}, "
                 f"model anomaly probability {d['anomaly_proba']:.1%}",
                 f"  Sequence: {d['sequence'][:400]}"]
        if d["hypotheses"]:
            lines.append("  Root-cause hypotheses (heuristic rule-score shares, not probabilities): "
                         + ", ".join(f"{h['title']} {h['share']:.0%}" for h in d["hypotheses"][:3]))
        lines += [f"  Evidence: {e}" for e in d["evidence"]]
        lines += [f"  {ln}" for ln in describe_neighbours(d["sequence"].split(), d["similar_logs"],
                                                          profile.max_seq_len)]
        out.append("\n".join(lines))
    return "\n\n".join(out)


class RunChat:
    """Conversation about one finished run. `history` keeps each turn as the model saw it (`content`) and as
    the user should see it (`display`), plus grounding warnings for answers (`issues`)."""

    def __init__(self, ctx, rca_agent):
        self.ctx, self.rca = ctx, rca_agent
        facts = ctx.data.get("insight_facts") or build_facts(ctx)
        self.system = (f"{SYSTEM.format(persona=ctx.profile.persona)}\n\nFACTS:\n{facts}\n\n"
                       f"YOUR EARLIER ANALYSIS:\n{analysis_text(ctx.data['insight'])}")
        self.history = []

    def mentioned_blocks(self, question):
        known = set(self.ctx.data["detection"]["sessions"].block_id)
        return [b for b in dict.fromkeys(self.ctx.profile.session_re.findall(question)) if b in known]

    def messages_for(self, question):
        """Returns (messages for the SLM, the question as the model sees it)."""
        named = self.mentioned_blocks(question)
        content = question
        details = block_details(self.ctx, self.rca, named)
        if details:
            content += f"\n\nBLOCK DETAILS:\n{details}"
        logs = log_context(self.ctx, question, named)
        if logs:
            content += f"\n\n{logs}"
        recent = self.history[-2 * config.CHAT_MAX_TURNS:]
        msgs = [{"role": "system", "content": self.system}]
        msgs += [{"role": t["role"], "content": t["content"]} for t in recent]
        return msgs + [{"role": "user", "content": content}], content

    def finish(self, question, content, raw_answer):
        """Clean and check an answer, add the exchange to the history. Returns (answer, issues)."""
        answer = strip_think(raw_answer if isinstance(raw_answer, str) else "".join(map(str, raw_answer)))
        # numbers are allowed if they appear in anything the model was given (not in its own earlier answers)
        given = "\n".join([self.system, content] + [t["content"] for t in self.history if t["role"] == "user"])
        issues = grounding_issues(answer, self.ctx, facts=given)
        self.history += [{"role": "user", "content": content, "display": question, "issues": []},
                         {"role": "assistant", "content": answer, "display": answer, "issues": issues}]
        return answer, issues

    def ask(self, slm, question):
        """Blocking question/answer (CLI and tests). Returns (answer, issues)."""
        msgs, content = self.messages_for(question)
        return self.finish(question, content, slm.chat(msgs, config.CHAT_MAX_NEW_TOKENS))
