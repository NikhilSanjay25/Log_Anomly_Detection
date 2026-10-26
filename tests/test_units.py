"""Unit tests for validation, fallbacks and agent policies (no language-model weights needed)."""
import pandas as pd

from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.agents.insight import describe_neighbours, deterministic_insight, validate_insight
from log_agents.agents.rca import RootCauseAnalysisAgent
from log_agents.context import RunContext, RunOptions
from log_agents.datasets import get_profile
from log_agents.ingestion import TextSource
from log_agents.resources import Resources

NORMAL = "E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21"


def _ctx(blocks=("blk_1",), nodes=(("10.0.0.1",),), anomalies=1, causes=None):
    ctx = RunContext("t", "")
    ctx.profile = get_profile("hdfs")
    ctx.data["detection"] = {"sessions": pd.DataFrame({
        "block_id": list(blocks), "nodes": [list(n) for n in nodes],
        "is_anomaly": [1] * anomalies + [0] * (len(blocks) - anomalies)})}
    ctx.data["rca"] = {"sessions": [], "causes": causes or [],
                       "correlations": {"nodes": [], "time_bursts": [], "patterns": []}}
    return ctx


GOOD = {"summary": "s", "explanation": "E7 broke blk_1", "recommendations": ["a"], "next_best_actions": ["b"],
        "impact": {"severity": "High", "description": "d", "affected": ["10.0.0.1"]}}


def test_validate_insight_accepts_grounded_answer():
    assert validate_insight(GOOD, _ctx()) == []


def test_validate_insight_rejects_hallucinations_and_bad_types():
    ctx = _ctx()
    bad = {**GOOD, "explanation": "blk_999 on 10.9.9.9 raised E77"}
    problems = " ".join(validate_insight(bad, ctx))
    assert "blk_999" in problems and "E77" in problems and "10.9.9.9" in problems
    assert validate_insight({**GOOD, "recommendations": [{"step": 1}]}, ctx)
    assert validate_insight({**GOOD, "impact": {**GOOD["impact"], "affected": "all nodes"}}, ctx)
    assert validate_insight({**GOOD, "impact": {**GOOD["impact"], "severity": "catastrophic"}}, ctx)


def test_deterministic_insight_without_rca_causes():
    ins = deterministic_insight(_ctx(causes=[]))   # previously raised IndexError
    assert validate_insight(ins, _ctx()) == []


def test_uncertain_sessions_are_flagged_but_keep_the_rf_label():
    ctx = CoordinatorAgent(Resources()).run(TextSource(NORMAL).read(), RunOptions(persist=False))
    s = ctx.data["detection"]["sessions"].iloc[0]
    assert s.is_anomaly == s.rf_label == 0
    assert "slm_label" not in ctx.data["detection"]["sessions"]       # no language model takes part in detection
    hi = CoordinatorAgent(Resources()).run(TextSource(NORMAL).read(), RunOptions(persist=False, threshold=0.0))
    assert hi.data["detection"]["sessions"].iloc[0].is_anomaly == 1  # the threshold alone decides the label


def test_rca_ignores_events_common_in_normal_traffic():
    rca = RootCauseAnalysisAgent(Resources())
    rca.profile = get_profile("hdfs")
    rca.ev_stats = {"E4": {"sessions": 129295, "anomaly_rate_when_present": 0.025},
                    "E13": {"sessions": 488, "anomaly_rate_when_present": 1.0}}
    rca.stats = {"events": rca.ev_stats, "normal_length": {"p05": 13}}
    rca.normal_p05 = 13
    rca.evidence_events = (rca.profile.error_events | rca.profile.warning_events
                           | set().union(*(rc["events"] for rc in rca.profile.root_causes.values())))
    seq = NORMAL.split()
    row = dict(anomaly_proba=0.9, decision_source="x", nb_anomaly_rate=float("nan"), max_gap_s=float("nan"), nodes=[])
    hyps, evidence = rca._hypotheses(pd.Series({**row, "events": seq + ["E4"], "n_events": len(seq) + 1}))
    assert hyps[0]["cause"] == "rare_pattern"                       # E4 alone explains nothing
    assert any("not counted as evidence" in e for e in evidence)
    hyps, _ = rca._hypotheses(pd.Series({**row, "events": seq + ["E13"], "n_events": len(seq) + 1}))
    assert hyps[0]["cause"] == "write_recovery"
    assert abs(sum(h["share"] for h in hyps) - 1) < 1e-9


def test_coordinator_survives_rca_crash():
    coord = CoordinatorAgent(Resources())

    def boom(ctx):
        raise RuntimeError("rca bug")
    coord.rca.run = boom
    ctx = coord.run(TextSource("E22 E5").read(), RunOptions(persist=False))
    assert ctx.data["status"] == "completed"
    assert [s.status for s in ctx.trace if s.agent == "Root Cause Analysis Agent"] == ["failed"]
    assert ctx.data["insight"]["summary"]


def test_validate_insight_rejects_invented_numbers():
    ctx = _ctx()
    ctx.data["insight_facts"] = "Block sessions analysed: 175; anomalous: 25 (14.3%) blk_1 E7 10.0.0.1"
    assert validate_insight({**GOOD, "summary": "25 of 175 sessions (14.3%) on 10.0.0.1 hit E7"}, ctx) == []
    assert "numbers" in " ".join(validate_insight({**GOOD, "summary": "40 sessions failed"}, ctx))


def test_describe_neighbours_groups_and_diffs():
    q = NORMAL.split()
    same = {"sequence": q, "label": "Normal", "occurrences": 10, "anomaly_rate": 0.0, "block_id": "blk_77"}
    shuffled = {**same, "sequence": q[1:] + q[:1], "occurrences": 3}
    extra = {**same, "sequence": q + ["E20"], "label": "Anomaly", "occurrences": 2, "anomaly_rate": 1.0}
    lines = describe_neighbours(q, [same, same, shuffled, extra, extra])
    text = "\n".join(lines)
    assert "3 distinct pattern(s)" in lines[0]
    assert "3 of 5 labelled Normal" in lines[0] and "2 of 5 labelled Anomaly" in lines[0]
    assert "identical to this session" in text and "same events in a different order" in text
    assert "has E20 ×1 that this session lacks" in text
    assert "blk_77" not in text               # neighbour block ids would make the validator reject the answer
    assert describe_neighbours(q, None) == []


def test_validate_insight_requires_normal_explanation_in_mixed_runs():
    ctx = _ctx(blocks=("blk_1", "blk_2"), nodes=(("10.0.0.1",), ()), anomalies=1)
    assert "normal_explanation" in " ".join(validate_insight(GOOD, ctx))
    assert validate_insight({**GOOD, "normal_explanation": "blk_2 completed its lifecycle"}, ctx) == []
    assert validate_insight({**GOOD, "normal_explanation": ["not text"]}, ctx)
    assert validate_insight(GOOD, _ctx()) == []                       # all anomalous: not required
