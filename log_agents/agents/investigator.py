"""Investigator Agent: a language model investigates every anomaly and decides its root cause.

For each case (a group of anomalies with an identical event pattern, one representative investigated per group) the
model runs a short loop. Each turn it is shown the evidence gathered so far and a lettered menu of actions, and it
answers with ONE letter:

    A read_logs  B similar_cases  C event_evidence  D structural_check  E node_correlation  F repeat_pattern
    G rule_ranking (ask the rule engine for its ranking)            H decide (stop and choose the root cause)

The model decides which evidence to gather, in which order, whether to consult the rule engine, and when to stop.
Then it is shown a shortlist of candidate root causes (in a shuffled order, without the rule scores unless it asked
for them) and picks one. Three ways of picking are implemented (`RunOptions.agent_decision`):
  listwise            one menu, the model answers with a letter
  listwise_debiased   the same menu under every rotation of the candidates, probabilities averaged per candidate
  pointwise           each candidate is judged alone ("is this the root cause? Yes / No"), highest P(Yes) wins
Every answer is the option the model scores highest (`LocalLLM.choose`), so nothing is parsed and a small model
cannot produce malformed output.

What the model does NOT do: it does not label sessions anomalous (the trained detector does), it does not
compute the evidence (the tools are ordinary code, the RCA agent's evidence functions), and it cannot invent a
cause (it chooses from the shortlist). Every step, observation and probability is recorded in
`ctx.data["investigation"]` and shown in the UI and the report. If the language model is not available, the
agent falls back to the rule engine and says so. With `agent_model="rules"` it is the rule-based RCA.
"""
import random
import time
import zlib
from collections import Counter, defaultdict

import pandas as pd

from .. import config
from ..chat import has_raw_logs, session_lines
from .base import AgentError
from .insight import describe_neighbours
from .rca import RootCauseAnalysisAgent

ACTIONS = {
    "A": ("read_logs", "read the raw log lines of this {noun} (error and warning lines first)"),
    "B": ("similar_cases", "see how the most similar past {nouns} in the training data were labelled"),
    "C": ("event_evidence", "list the error / warning events present and how often each was anomalous in training"),
    "D": ("structural_check", "run the dataset-specific checks (timings, lifecycle, length)"),
    "E": ("node_correlation", "check whether the same nodes or the same minute are involved in other anomalies of this run"),
    "F": ("repeat_pattern", "count the other anomalies in this run with exactly the same event pattern"),
    "G": ("rule_ranking", "ask the rule engine for its own ranking of the root causes"),
    "H": ("text_match", "find the root causes whose descriptions best match this {noun}'s log messages"),
    "I": ("decide", "stop investigating and choose the root cause now"),
}
TOOL_LETTER = {name: letter for letter, (name, _) in ACTIONS.items()}
SYSTEM = ("You are a {persona}. A detector flagged one {noun} as anomalous and you are investigating why. You work "
          "step by step: each turn you pick exactly ONE action from a lettered menu and answer with its single "
          "letter. Choose the action that will teach you the most, and do not repeat an action. Answer with one "
          "letter and nothing else.")
MAX_OBS_LINES = 6
MAX_OBS_CHARS = 170


def _clip(text, n=MAX_OBS_CHARS):
    return text if len(text) <= n else text[:n - 1] + "…"


class InvestigatorAgent(RootCauseAnalysisAgent):
    name = "Investigator Agent"
    task = "Investigate each anomaly with a language model that chooses the next step, and decide its root cause"

    def __init__(self, resources, llm_factory=None):
        super().__init__(resources)
        self.llm_factory = llm_factory or (lambda model: resources.llm(model))

    # ── agent entry point ─────────────────────────────────────────────────
    def run(self, ctx):
        model = ctx.options.agent_model
        ctx.data["investigation"] = {"model": model, "status": "rules", "episodes": []}
        summary = super().run(ctx)                      # evidence for every session; _decide() below investigates
        inv = ctx.data["investigation"]
        if inv["status"] == "rules":
            return f"{summary} (rule-based: no language model)"
        if inv["status"] == "unavailable":
            return f"{summary} (language model unavailable, rule-based fallback: {inv['reason']})"
        return (f"{summary} · {inv['model_label']} investigated {inv['cases']} case(s) in {inv['model_calls']} "
                f"model calls ({inv['seconds']:.0f}s)")

    def _decide(self, ctx, per_session, correlations, rows):
        model = ctx.options.agent_model
        inv = ctx.data["investigation"]
        if model in (None, "", config.RULES) or not per_session:
            return
        ctx.emit("model_loading", model=model)
        t_load = time.time()
        try:
            llm = self.llm_factory(model)
        except Exception as e:                           # weights missing, no GPU for 4-bit, out of memory...
            inv.update(status="unavailable", reason=f"{type(e).__name__}: {str(e)[:150]}")
            ctx.warn(f"Investigator: could not load {model} ({inv['reason']}); used the rule-based analysis instead.")
            ctx.emit("model_failed", model=model, reason=inv["reason"])
            return
        ctx.emit("model_loaded", model=llm.label, seconds=time.time() - t_load)

        groups = defaultdict(list)
        for p in per_session:
            groups[p["sequence"]].append(p)
        order = sorted(groups.values(), key=lambda g: (-len(g), -g[0]["anomaly_proba"]))
        n_cases = min(len(order), ctx.options.agent_max_cases)
        t0, failures, calls = time.time(), 0, 0
        ctx.emit("investigation_start", model=llm.label, anomalies=len(per_session), groups=len(order),
                 cases=n_cases, noun=ctx.profile.session_noun)
        kb = ctx.profile.root_causes
        for k, g in enumerate(order[:n_cases]):
            rep = g[0]
            ctx.emit("case_start", index=k, total=n_cases, block_id=rep["block_id"], group_size=len(g),
                     probability=rep["anomaly_proba"], noun=ctx.profile.session_noun)
            try:
                ep = self._investigate(ctx, llm, rep, rows[rep["block_id"]], per_session, correlations, len(g),
                                       ctx.options.agent_max_steps)
            except Exception as e:                       # one bad case must not stop the others
                failures += 1
                ctx.warn(f"Investigator: case {rep['block_id']} failed ({type(e).__name__}: {str(e)[:100]}); "
                         "kept the rule-based cause.")
                ctx.emit("case_failed", index=k, block_id=rep["block_id"])
                if failures > 3:
                    raise AgentError("the language model failed on several cases; stopped") from e
                continue
            ep["case"] = len(inv["episodes"])
            inv["episodes"].append(ep)
            calls += ep["model_calls"]
            for p in g:
                p["primary_cause"], p["decided_by"], p["investigation"] = ep["choice"], llm.label, ep["case"]
            ctx.emit("case_done", index=k, total=n_cases, block_id=rep["block_id"], group_size=len(g),
                     choice=ep["choice"], title=kb[ep["choice"]]["title"], rule_top=ep["rule_top"],
                     rule_top_title=kb[ep["rule_top"]]["title"], agrees=ep["choice"] == ep["rule_top"],
                     seconds=ep["seconds"], model_calls=ep["model_calls"])
        for g in order[n_cases:]:
            for p in g:
                p["decided_by"] = "rules (case limit reached)"
        eps = inv["episodes"]
        inv.update(
            status="agent", model_label=llm.label, cases=len(eps), anomalies=len(per_session),
            groups=len(order), model_calls=calls, seconds=time.time() - t0,
            rule_ranking_used=sum(ep["used_rule_ranking"] for ep in eps),
            agrees_with_rule_top=sum(ep["choice"] == ep["rule_top"] for ep in eps),
            mean_steps=(sum(len(ep["steps"]) for ep in eps) / len(eps)) if eps else 0.0)
        if n_cases < len(order):
            ctx.warn(f"Investigator: investigated {n_cases} of {len(order)} anomaly groups; the rest use the "
                     "rule-based cause.")
        ctx.send(self.name, "Insight Agent", f"{len(eps)} case(s) investigated by {llm.label}")
        ctx.emit("investigation_done", cases=len(eps), model_calls=calls, seconds=inv["seconds"],
                 agrees=inv["agrees_with_rule_top"], model=llm.label)

    # ── one investigation ─────────────────────────────────────────────────
    def _investigate(self, ctx, llm, entry, row, per_session, correlations, group_size, max_steps):
        profile = ctx.profile
        noun, nouns = profile.session_noun, profile.session_noun_plural
        system = SYSTEM.format(persona=profile.persona, noun=noun)
        header = (f"CASE: {noun} {entry['block_id']} was flagged anomalous (model probability "
                  f"{entry['anomaly_proba']:.0%}).\nABOUT THE DATA: {profile.glossary}\n")
        evidence, used, steps, calls = [], set(), [], 0
        t0 = time.time()
        tools = self._tools(ctx, entry, row, per_session, correlations)

        for turn in range(max_steps + 1):
            available = [l for l, (name, _) in ACTIONS.items()
                         if name not in used and (name != "decide" or evidence)]
            if len(used) >= max_steps or len(available) <= 1:
                break
            menu = "\n".join(f"{l}) {name}: " + text.format(noun=noun, nouns=nouns)
                             for l, (name, text) in ACTIONS.items() if l in available)
            prompt = (f"{header}EVIDENCE SO FAR:\n" + ("\n".join(evidence) if evidence else "(none yet)")
                      + f"\n\nACTIONS YOU CAN STILL TAKE:\n{menu}\n\nWhich action next? Answer with one letter.")
            letter, probs = llm.choose(system, prompt, available)
            calls += 1
            name = ACTIONS[letter][0]
            options = {ACTIONS[l][0]: round(float(p), 3) for l, p in probs.items()}
            if name == "decide":
                steps.append({"action": name, "probs": _round(probs), "options": options, "observation": ""})
                ctx.emit("step", n=len(steps), action=name, options=options, observation="", max_steps=max_steps)
                break
            obs = tools[name]()
            used.add(name)
            evidence.append(f"[{name}] {obs}")
            steps.append({"action": name, "probs": _round(probs), "options": options, "observation": obs})
            ctx.emit("step", n=len(steps), action=name, options=options, observation=obs, max_steps=max_steps)

        # final decision: a shortlist from the rule engine, shuffled, scores hidden unless the model asked for them
        cand = self.shortlist(entry, row)
        rule_top = entry["hypotheses"][0]["cause"]
        kb = profile.root_causes
        decision = ctx.options.agent_decision
        ctx.emit("deciding", candidates=[kb[c]["title"] for c in cand], method=decision)
        if len(cand) == 1:
            choice, cprobs = cand[0], {cand[0]: 1.0}
        else:
            rng = random.Random(zlib.crc32(entry["block_id"].encode()))
            rng.shuffle(cand)                                  # the order carries no information
            body = f"{header}EVIDENCE:\n" + ("\n".join(evidence) if evidence else "(none gathered)")
            describe = lambda c: f"{kb[c]['title']}: {_clip(kb[c]['explanation'], 150)}"
            if decision == "pointwise":
                cprobs = {}
                for c in cand:
                    _, pr = llm.choose(system, body + f"\n\nPROPOSED ROOT CAUSE: {describe(c)}\n\nGiven the evidence, "
                                       "is this the root cause of the anomaly? Answer Yes or No.", ["Yes", "No"])
                    cprobs[c] = pr["Yes"]
                    calls += 1
            else:
                rotations = range(len(cand)) if decision == "listwise_debiased" else [0]
                cprobs = {c: 0.0 for c in cand}
                for r in rotations:
                    order = cand[r:] + cand[:r]
                    letters = [chr(65 + k) for k in range(len(order))]
                    menu = "\n".join(f"{l}) {describe(c)}" for l, c in zip(letters, order))
                    _, pr = llm.choose(system, body + f"\n\nCANDIDATE ROOT CAUSES:\n{menu}\n\nWhich root cause is best "
                                       "supported by the evidence? Answer with one letter.", letters)
                    for l, c in zip(letters, order):
                        cprobs[c] += pr[l] / len(rotations)
                    calls += 1
            choice = max(cand, key=lambda c: (cprobs[c], -cand.index(c)))
        ctx.emit("decision", method=decision, choice=choice, candidates=sorted(
            ({"cause": c, "title": kb[c]["title"], "score": round(float(cprobs.get(c, 0.0)), 3),
              "rule_top": c == rule_top, "chosen": c == choice} for c in cand), key=lambda d: -d["score"]))
        return {"block_id": entry["block_id"], "group_size": group_size, "steps": steps,
                "candidates": cand, "rule_top": rule_top, "choice": choice,
                "choice_probs": _round(cprobs), "decision": decision, "used_rule_ranking": "rule_ranking" in used,
                "used_text_match": "text_match" in used,
                "model_calls": calls, "seconds": round(time.time() - t0, 2)}

    def shortlist(self, entry, row, k=config.AGENT_SHORTLIST):
        """Candidate causes for the final menu: the rule engine's top hypotheses, padded with causes whose
        descriptions match the session's message text when the rules found little (e.g. new message templates)."""
        out = [h["cause"] for h in entry["hypotheses"]][:k]
        if len(out) < k:
            for key, _ in self.text_ranking(row.events):
                if key not in out:
                    out.append(key)
                if len(out) == k:
                    break
        return out

    # ── the tools (ordinary code; each returns a short text observation) ──
    def _tools(self, ctx, entry, row, per_session, correlations):
        profile = ctx.profile
        noun, nouns = profile.session_noun, profile.session_noun_plural

        def read_logs():
            if not has_raw_logs(ctx):
                return "No raw log lines are available for this input (it contains event sequences only)."
            lines, total = session_lines(ctx, entry["block_id"], limit=MAX_OBS_LINES)
            return (f"{len(lines)} of {total} lines (errors first):\n" + "\n".join("  " + _clip(l) for l in lines))

        def similar_cases():
            similar = entry.get("similar_logs") or self.res.pipeline(profile.name).neighbour_details(
                row.nb_ids, row.nb_dist)
            lines = describe_neighbours(entry["sequence"].split(), similar, profile.max_seq_len)
            kept = [_clip(l.strip(), 200) for l in lines if not l.strip().startswith("Sequence:")][:4]
            return "\n".join(kept) or "no similar past cases available."

        def event_evidence():
            ev = [e for e in entry["evidence"] if " ×" in e][:MAX_OBS_LINES]
            return "\n".join("  " + _clip(e) for e in ev) or "No error or warning events are present."

        def structural_check():
            _, ev = profile.structural_scores(Counter(row.events), set(row.events), row, self.stats)
            return "\n".join("  " + _clip(e) for e in ev) or "No dataset-specific findings (normal timings / lifecycle / length)."

        def node_correlation():
            hot = {n["node"]: n for n in correlations["nodes"]}
            mine = [hot[n] for n in entry["nodes"] if n in hot]
            out = [f"{profile.node_noun} {n['node']}: {n['anomalous_blocks']} of {n['total_blocks']} {nouns} on it are "
                   f"anomalous ({n['anomaly_share']:.0%})" for n in mine[:3]]
            if not out:
                out = [f"None of the {len(entry['nodes'])} {profile.node_noun}(s) of this {noun} is concentrated in "
                       "this run's anomalies." if len(entry["nodes"]) else "This input has no node information."]
            if entry["start"]:
                minute = str(pd.Timestamp(entry["start"]).floor("min"))
                burst = next((b for b in correlations["time_bursts"] if b["minute"] == minute), None)
                if burst:
                    out.append(f"It falls in a burst of {burst['anomalies']} anomalies at {minute}.")
            return "\n".join("  " + o for o in out)

        def repeat_pattern():
            n = sum(1 for p in per_session if p["sequence"] == entry["sequence"]) - 1
            return (f"{n} other anomalous {nouns} in this run have exactly this event pattern." if n
                    else f"No other anomaly in this run has this exact pattern (this {noun} is unique).")

        def text_match():
            top = self.text_ranking(row.events)[:3]
            if not top:
                return "No root cause description matches this session's messages."
            return "Knowledge-base text match: " + "; ".join(
                f"{n}. {profile.root_causes[k]['title']} (similarity {s:.2f})" for n, (k, s) in enumerate(top, 1))

        def rule_ranking():
            top = entry["hypotheses"][:3]
            return "Rule engine ranking: " + "; ".join(f"{i}. {h['title']} ({h['share']:.0%})"
                                                        for i, h in enumerate(top, 1))

        return {"read_logs": read_logs, "similar_cases": similar_cases, "event_evidence": event_evidence,
                "structural_check": structural_check, "node_correlation": node_correlation,
                "repeat_pattern": repeat_pattern, "rule_ranking": rule_ranking, "text_match": text_match}


def _round(probs):
    return {k: round(float(v), 3) for k, v in probs.items()}
