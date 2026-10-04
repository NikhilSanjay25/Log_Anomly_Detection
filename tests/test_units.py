"""Unit tests for validation, fallbacks and agent policies (no SLM weights needed)."""
import pandas as pd

from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.agents.insight import deterministic_insight, validate_insight
from log_agents.agents.rca import RootCauseAnalysisAgent
from log_agents.context import RunContext, RunOptions
from log_agents.ingestion import TextSource
from log_agents.resources import Resources
from log_agents.slm import extract_json

NORMAL = "E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21"


def _ctx(blocks=("blk_1",), nodes=(("10.0.0.1",),), anomalies=1, causes=None):
    ctx = RunContext("t", "")
    ctx.data["detection"] = {"sessions": pd.DataFrame({
        "block_id": list(blocks), "nodes": [list(n) for n in nodes],
        "is_anomaly": [1] * anomalies + [0] * (len(blocks) - anomalies)})}
    ctx.data["rca"] = {"sessions": [], "causes": causes or [],
                       "correlations": {"nodes": [], "time_bursts": [], "patterns": []}}
    return ctx


GOOD = {"summary": "s", "explanation": "E7 broke blk_1", "recommendations": ["a"], "next_best_actions": ["b"],
        "impact": {"severity": "High", "description": "d", "affected": ["10.0.0.1"]}}


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    assert extract_json("no json here") is None


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


def test_lora_is_advisory_and_never_overrides_rf():
    class AlwaysAnomaly:
        def classify(self, seqs):
            return ["Anomaly"] * len(seqs)

    res = Resources()
    res._lora, res._lora_checked = AlwaysAnomaly(), True
    ctx = CoordinatorAgent(res).run(TextSource(NORMAL).read(),
                                    RunOptions(use_slm_insight=False, lora_mode="all", persist=False))
    s = ctx.data["detection"]["sessions"].iloc[0]
    assert s.slm_label == "Anomaly" and s.slm_disagrees
    assert s.is_anomaly == 0 == s.rf_label


def test_rca_ignores_events_common_in_normal_traffic():
    rca = RootCauseAnalysisAgent(Resources())
    rca.ev_stats = {"E4": {"sessions": 129295, "anomaly_rate_when_present": 0.025},
                    "E13": {"sessions": 488, "anomaly_rate_when_present": 1.0}}
    rca.normal_p05 = 13
    seq = NORMAL.split()
    row = dict(anomaly_proba=0.9, decision_source="x", slm_label=None, slm_disagrees=False,
               nb_anomaly_rate=float("nan"), max_gap_s=float("nan"), nodes=[])
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
    ctx = coord.run(TextSource("E22 E5").read(), RunOptions(use_slm_insight=False, lora_mode="off", persist=False))
    assert ctx.data["status"] == "completed"
    assert [s.status for s in ctx.trace if s.agent == "Root Cause Analysis Agent"] == ["failed"]
    assert ctx.data["insight"]["summary"]


def test_validate_insight_rejects_invented_numbers():
    ctx = _ctx()
    ctx.data["insight_facts"] = "Block sessions analysed: 175; anomalous: 25 (14.3%) blk_1 E7 10.0.0.1"
    assert validate_insight({**GOOD, "summary": "25 of 175 sessions (14.3%) on 10.0.0.1 hit E7"}, ctx) == []
    assert "numbers" in " ".join(validate_insight({**GOOD, "summary": "40 sessions failed"}, ctx))
