"""Root Cause Analysis Agent.

Combines four evidence sources per anomalous session:
  * anomaly evidence  - model probability, error/warning events and how strongly each event is
                        associated with anomalies in the training data (event_stats.json)
  * similar logs      - FAISS neighbours: their sequences and historical anomaly rates
  * contextual data   - dataset-specific structural checks (HDFS block lifecycle, OpenStack timings, ...),
                        length vs normal sessions, timing, nodes
  * correlation       - nodes / time windows / patterns shared across anomalous sessions
and ranks root-cause hypotheses from the dataset's knowledge base (log_agents/knowledge.py for HDFS, the
profile's own causes for BGL and OpenStack).

It also collects the same kind of evidence for a few representative *normal* sessions, so the Insight
agent can explain why those sessions were not flagged.
"""
from collections import Counter, defaultdict

import pandas as pd

from .base import Agent

MAX_NEIGHBOUR_DETAILS = 30
MAX_NORMAL_EXAMPLES = 3
# An event only counts as root-cause evidence if at least this share of training sessions containing it were
# anomalous (e.g. HDFS E4 'exception while serving' is 2.5% - it is routine in normal traffic).
MIN_EVENT_ANOMALY_RATE = 0.5


class RootCauseAnalysisAgent(Agent):
    name = "Root Cause Analysis Agent"
    task = "Determine probable root causes from evidence, similar logs, context and correlations"

    def __init__(self, resources):
        self.res = resources
        self.profile = None
        self.stats = {}
        self.ev_stats = {}
        self.normal_p05 = 13

    def _prepare(self, profile):
        """Bind the dataset profile and its training statistics."""
        self.profile = profile
        pipe = self.res.pipeline(profile.name)
        self.stats = pipe.event_stats
        self.ev_stats = self.stats.get("events", {})
        self.normal_p05 = self.stats.get("normal_length", {}).get("p05", 13)
        self.evidence_events = (profile.error_events | profile.warning_events
                                | set().union(*(rc["events"] for rc in profile.root_causes.values())))
        return pipe

    def run(self, ctx):
        sessions = ctx.data["detection"]["sessions"]
        pipe = self._prepare(ctx.profile)

        anomalies = sessions[sessions.is_anomaly == 1].sort_values("anomaly_proba", ascending=False)
        per_session, nb_cache = [], {}
        for _, s in anomalies.iterrows():
            hyps, evidence = self._hypotheses(s)
            if s.sequence not in nb_cache and len(nb_cache) < MAX_NEIGHBOUR_DETAILS:
                nb_cache[s.sequence] = pipe.neighbour_details(s.nb_ids, s.nb_dist)
            per_session.append({
                "block_id": s.block_id, "sequence": s.sequence, "n_events": int(s.n_events),
                "anomaly_proba": float(s.anomaly_proba), "decision_source": s.decision_source,
                "nb_anomaly_rate": None if pd.isna(s.nb_anomaly_rate) else float(s.nb_anomaly_rate),
                "primary_cause": hyps[0]["cause"], "hypotheses": hyps, "evidence": evidence,
                "error_events": s.error_events, "nodes": s.nodes,
                "duration_s": None if pd.isna(s.duration_s) else float(s.duration_s),
                "start": None if pd.isna(s.start) else str(s.start),
                "similar_logs": nb_cache.get(s.sequence),
            })

        correlations = self._correlate(sessions, per_session)
        causes = self._aggregate(per_session)
        # The chat model is shown the retrieved neighbours of each group's example session, so make sure
        # every example has them even when the neighbour-detail cap above was reached.
        nb_of = anomalies.set_index("block_id")
        for c in causes:
            ex = next(p for p in per_session if p["primary_cause"] == c["cause"])
            if ex["similar_logs"] is None:
                row = nb_of.loc[ex["block_id"]]
                ex["similar_logs"] = pipe.neighbour_details(row.nb_ids, row.nb_dist)
        normal_examples = self._normal_examples(sessions, pipe, ctx.data["detection"]["threshold"])
        ctx.data["rca"] = {"sessions": per_session, "causes": causes, "correlations": correlations,
                           "normal_examples": normal_examples}
        ctx.send(self.name, "Insight Agent", f"{len(causes)} root-cause group(s) and "
                                             f"{len(normal_examples)} normal example(s) with evidence")
        if not per_session:
            return f"No anomalies; evidence gathered for {len(normal_examples)} normal pattern(s)"
        top = causes[0]
        return (f"{len(per_session)} anomalies → {len(causes)} root-cause group(s); "
                f"top: {top['title']} ({top['count']} {ctx.profile.session_noun_plural})")

    # ── per-session hypotheses ────────────────────────────────────────────
    def _hypotheses(self, s):
        profile = self.profile
        kb = profile.root_causes
        events = s.events
        c = Counter(events)
        present = set(events)
        evidence = [f"Model anomaly probability {s.anomaly_proba:.1%} ({s.decision_source})"]
        if not pd.isna(s.nb_anomaly_rate):
            evidence.append(f"Retrieved similar historical {profile.session_noun_plural} are "
                            f"{s.nb_anomaly_rate:.0%} anomalous")
        for e in sorted(present & self.evidence_events, key=lambda x: int(x[1:]))[:12]:
            st = self.ev_stats.get(e)
            if st is None or st.get("anomaly_rate_when_present") is None:
                evidence.append(f"{e} ×{c[e]}: {profile.describe(e)} (no training statistics)")
            elif st["anomaly_rate_when_present"] >= MIN_EVENT_ANOMALY_RATE:
                evidence.append(f"{e} ×{c[e]}: {profile.describe(e)}; {st['anomaly_rate_when_present']:.0%} of "
                                f"{st['sessions']:,} training {profile.session_noun_plural} containing it were anomalous")
            else:
                evidence.append(f"{e} ×{c[e]}: {profile.describe(e)}; routine in normal traffic "
                                f"({st['anomaly_rate_when_present']:.1%} of {st['sessions']:,} training "
                                f"{profile.session_noun_plural} with it were anomalous) - not counted as evidence")

        scores = {}
        for key, cause in kb.items():
            hit = [e for e in present & cause["events"] if self._discriminative(e)]
            if hit:
                strength = max(self._rate(e) for e in hit)
                scores[key] = cause["weight"] * (0.5 + 0.5 * strength)

        struct_scores, struct_evidence = profile.structural_scores(c, present, s, self.stats)
        scores.update({k: v for k, v in struct_scores.items() if k in kb})
        evidence += struct_evidence

        if not pd.isna(s.max_gap_s) and s.max_gap_s and s.max_gap_s > 600:
            evidence.append(f"Longest gap between consecutive events: {s.max_gap_s:.0f}s")
        if len(s.nodes):
            evidence.append(f"{profile.node_noun}s involved: {', '.join(s.nodes[:6])}{' …' if len(s.nodes) > 6 else ''}")

        if not scores or max(scores.values()) < 0.5:
            nb = 0.5 if pd.isna(s.nb_anomaly_rate) else s.nb_anomaly_rate
            scores["rare_pattern"] = kb["rare_pattern"]["weight"] * (0.5 + nb)

        total = sum(scores.values())
        hyps = sorted(({"cause": k, "title": kb[k]["title"], "share": v / total}
                       for k, v in scores.items()), key=lambda h: -h["share"])
        return hyps, evidence

    def describe_block(self, ctx, block_id):
        """Label, evidence, root-cause hypotheses and FAISS neighbours for one session of a finished run, normal
        or anomalous (used by the follow-up chat). None if the run has no such session."""
        sessions = ctx.data["detection"]["sessions"]
        match = sessions[sessions.block_id == block_id]
        if match.empty:
            return None
        s = match.iloc[0]
        pipe = self._prepare(ctx.profile)
        known = next((p for p in (ctx.data.get("rca") or {}).get("sessions", []) if p["block_id"] == block_id), None)
        if s.is_anomaly and known:
            evidence, hyps, similar = known["evidence"], known["hypotheses"], known["similar_logs"]
        elif s.is_anomaly:
            (hyps, evidence), similar = self._hypotheses(s), None
        else:
            hyps, evidence, similar = [], self._normal_evidence(s, ctx.data["detection"]["threshold"]), None
        if similar is None:
            similar = pipe.neighbour_details(s.nb_ids, s.nb_dist)
        return {"block_id": block_id, "label": "Anomaly" if s.is_anomaly else "Normal",
                "anomaly_proba": float(s.anomaly_proba), "sequence": s.sequence,
                "evidence": evidence, "hypotheses": hyps, "similar_logs": similar}

    # ── evidence for normal sessions (why they were NOT flagged) ─────────────
    def _normal_examples(self, sessions, pipe, threshold):
        normal = sessions[sessions.is_anomaly == 0]
        if normal.empty:
            return []
        common = normal.sequence.value_counts()
        picks = [(normal[normal.sequence == common.index[0]].iloc[0],
                  f"most common normal pattern, {common.iloc[0]} of {len(normal)} normal sessions"
                  if common.iloc[0] > 1 else "a typical normal session; every normal pattern in this run is unique")]
        closest = normal.sort_values("anomaly_proba", ascending=False).iloc[0]
        picks.append((closest, "normal session with the highest anomaly probability"))
        risky = normal[normal.events.map(lambda e: bool(set(e) & self.evidence_events))]
        if len(risky):
            picks.append((risky.sort_values("anomaly_proba", ascending=False).iloc[0],
                          "normal session containing events that are often linked to anomalies"))
        out, seen = [], set()
        for s, why in picks:
            if s.sequence in seen:
                continue
            seen.add(s.sequence)
            out.append({"block_id": s.block_id, "sequence": s.sequence, "n_events": int(s.n_events),
                        "anomaly_proba": float(s.anomaly_proba), "why_picked": why,
                        "evidence": self._normal_evidence(s, threshold),
                        "similar_logs": pipe.neighbour_details(s.nb_ids, s.nb_dist)})
        return out[:MAX_NORMAL_EXAMPLES]

    def _normal_evidence(self, s, threshold):
        profile = self.profile
        c = Counter(s.events)
        evidence = [f"Model anomaly probability {s.anomaly_proba:.1%}, below the {threshold:.0%} threshold"]
        if not pd.isna(s.nb_anomaly_rate):
            evidence.append(f"Retrieved similar historical {profile.session_noun_plural} are "
                            f"{1 - s.nb_anomaly_rate:.0%} normal")
        evidence += profile.normal_structural_evidence(c, s, self.stats)
        flagged = sorted(set(s.events) & self.evidence_events, key=lambda x: int(x[1:]))
        if not flagged:
            evidence.append("No error or warning events")
        for e in flagged[:12]:
            st = self.ev_stats.get(e)
            if st is None or st.get("anomaly_rate_when_present") is None:
                evidence.append(f"{e} ×{c[e]}: {profile.describe(e)} (no training statistics)")
            else:
                evidence.append(f"{e} ×{c[e]}: {profile.describe(e)}; {st['anomaly_rate_when_present']:.1%} of "
                                f"{st['sessions']:,} training {profile.session_noun_plural} containing it were anomalous")
        if s.n_events >= self.normal_p05:
            evidence.append(f"Length {s.n_events} events, within the range of 95% of normal sessions "
                            f"(at least {int(self.normal_p05)})")
        return evidence

    def _rate(self, e):
        r = self.ev_stats.get(e, {}).get("anomaly_rate_when_present")
        return 0.5 if r is None else r

    def _discriminative(self, e):
        # without training statistics every listed event counts
        return not self.ev_stats or self._rate(e) >= MIN_EVENT_ANOMALY_RATE

    # ── cross-session correlation ─────────────────────────────────────────
    def _correlate(self, sessions, per_session):
        out = {"nodes": [], "time_bursts": [], "patterns": []}
        if not per_session:
            return out
        base_rate = sessions.is_anomaly.mean()

        node_total, node_anom = Counter(), Counter()
        for nodes, a in zip(sessions.nodes, sessions.is_anomaly):
            for n in nodes:
                node_total[n] += 1
                node_anom[n] += int(a)
        for n, a in node_anom.most_common():
            share = a / node_total[n]
            if a >= 2 and share >= max(2 * base_rate, 0.2):
                out["nodes"].append({"node": n, "anomalous_blocks": a, "total_blocks": node_total[n],
                                     "anomaly_share": share})
        out["nodes"] = out["nodes"][:10]

        times = pd.to_datetime(pd.Series([p["start"] for p in per_session]), errors="coerce").dropna()
        if len(times) >= 3:
            per_min = times.dt.floor("min").value_counts().sort_index()
            thr = max(3, per_min.mean() + 2 * per_min.std(ddof=0)) if len(per_min) > 1 else 3
            out["time_bursts"] = [{"minute": str(t), "anomalies": int(v)} for t, v in per_min.items() if v >= thr][:10]

        groups = defaultdict(list)
        for p in per_session:
            groups[p["sequence"]].append(p["block_id"])
        out["patterns"] = sorted(({"sequence": k, "blocks": len(v)} for k, v in groups.items() if len(v) > 1),
                                 key=lambda d: -d["blocks"])[:10]
        return out

    def _aggregate(self, per_session):
        kb_all = self.profile.root_causes
        sev = {"high": 0, "medium": 1, "low": 2}
        by = defaultdict(list)
        for p in per_session:
            by[p["primary_cause"]].append(p)
        causes = []
        for key, items in by.items():
            kb = kb_all[key]
            ev_counter = Counter(e for p in items for e in p["error_events"])
            causes.append({
                "cause": key, "title": kb["title"], "severity": kb["severity"], "count": len(items),
                "avg_share": sum(p["hypotheses"][0]["share"] for p in items) / len(items),
                "example_blocks": [p["block_id"] for p in items[:5]],
                "example_sequence": items[0]["sequence"],
                "key_events": [f"{e} ({self.profile.describe(e)}) ×{n}" for e, n in ev_counter.most_common(5)],
                "explanation": kb["explanation"], "impact": kb["impact"], "actions": kb["actions"],
            })
        return sorted(causes, key=lambda c: (sev[c["severity"]], -c["count"]))

    def validate(self, ctx):
        rca = ctx.data.get("rca")
        if rca is None:
            return ["no RCA output"]
        n_anom = int(ctx.data["detection"]["sessions"].is_anomaly.sum())
        if len(rca["sessions"]) != n_anom:
            return [f"RCA covered {len(rca['sessions'])} of {n_anom} anomalies"]
        for p in rca["sessions"]:
            if abs(sum(h["share"] for h in p["hypotheses"]) - 1) > 1e-6:
                return [f"hypothesis score shares for {p['block_id']} do not sum to 1"]
        return []
