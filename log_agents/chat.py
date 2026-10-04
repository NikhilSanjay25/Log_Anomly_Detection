"""Follow-up chat about a finished run: re-question the Insight agent's explanation.

Answers come from the same SLM as the report, grounded on the same facts sheet (root causes, evidence,
similar historical sessions, normal examples) plus the analysis it already wrote. When a question names a
session of this run (blk_... or seq_...), that session's own evidence and FAISS neighbours are added to the
question, so the bot can discuss sessions the summary did not cover.

Answers are not blocked like the report is, because a conversation should keep going. Instead, anything
the run cannot back up (unknown block ids, event ids, node IPs, or numbers absent from what the model was
given) is returned as a warning for the UI to show next to the answer.
"""
import re

from . import config
from .agents.insight import build_facts, describe_neighbours, grounding_issues
from .slm import strip_think

SYSTEM = ("You are a senior Hadoop/HDFS site-reliability engineer answering follow-up questions about a log "
          "anomaly analysis you produced earlier. Answer ONLY from the FACTS, YOUR EARLIER ANALYSIS and any BLOCK "
          "DETAILS included with a question. If they do not contain the answer, say so plainly instead of "
          "guessing. Never invent block ids, event ids, node addresses or numbers. Similar historical sessions "
          "are past sessions retrieved by vector search over the labelled training data. Root-cause shares are "
          "heuristic rule scores, not probabilities. Root-cause groups in the FACTS are ordered by severity (high, "
          "then medium, then low) and then by number of sessions. Be concise: a few sentences or a short list, in "
          "plain text.")
SESSION_RE = re.compile(r"\b(?:blk_-?\d+|seq_\d+)\b")
MAX_BLOCKS = 3          # sessions named in one question whose details are added


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


def block_details(ctx, rca_agent, block_ids):
    out = []
    for b in block_ids[:MAX_BLOCKS]:
        d = rca_agent.describe_block(ctx, b)
        if d is None:
            continue
        lines = [f"BLOCK {d['block_id']}: classified {d['label']}, model anomaly probability {d['anomaly_proba']:.1%}",
                 f"  Sequence: {d['sequence'][:400]}"]
        if d["hypotheses"]:
            lines.append("  Root-cause hypotheses (heuristic rule-score shares, not probabilities): "
                         + ", ".join(f"{h['title']} {h['share']:.0%}" for h in d["hypotheses"][:3]))
        lines += [f"  Evidence: {e}" for e in d["evidence"]]
        lines += [f"  {ln}" for ln in describe_neighbours(d["sequence"].split(), d["similar_logs"])]
        out.append("\n".join(lines))
    return "\n\n".join(out)


class RunChat:
    """Conversation about one finished run. `history` keeps each turn as the model saw it (`content`) and as
    the user should see it (`display`), plus grounding warnings for answers (`issues`)."""

    def __init__(self, ctx, rca_agent):
        self.ctx, self.rca = ctx, rca_agent
        facts = ctx.data.get("insight_facts") or build_facts(ctx)
        self.system = f"{SYSTEM}\n\nFACTS:\n{facts}\n\nYOUR EARLIER ANALYSIS:\n{analysis_text(ctx.data['insight'])}"
        self.history = []

    def mentioned_blocks(self, question):
        known = set(self.ctx.data["detection"]["sessions"].block_id)
        return [b for b in dict.fromkeys(SESSION_RE.findall(question)) if b in known]

    def messages_for(self, question):
        """Returns (messages for the SLM, the question as the model sees it)."""
        content = question
        details = block_details(self.ctx, self.rca, self.mentioned_blocks(question))
        if details:
            content += f"\n\nBLOCK DETAILS:\n{details}"
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
