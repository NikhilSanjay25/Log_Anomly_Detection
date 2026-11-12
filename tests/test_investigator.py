"""Tests of the Investigator agent (the language-model stage). Most tests use a scripted fake model so they run in
seconds without weights; one smoke test runs the real Qwen3-1.7B when it is available.

    python -m pytest tests/test_investigator.py -q
"""
import re
from pathlib import Path

import pytest

from log_agents import config
from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.agents.insight import build_facts
from log_agents.agents.investigator import ACTIONS, InvestigatorAgent
from log_agents.context import RunOptions
from log_agents.datasets import get_profile
from log_agents.ingestion import TextSource, UploadSource
from log_agents.report import to_markdown
from log_agents.resources import Resources
from log_agents.slm import model_available, parse_model_spec

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
FAKE = "fake/model"


class ScriptedLLM:
    """Stands in for LocalLLM. Records every prompt; answers actions in menu order (never 'decide' by itself, so the
    step budget ends the loop) and the final choice with the LAST option of the menu."""
    label = "Fake-LLM"

    def __init__(self, stop_after=None):
        self.calls, self.stop_after, self.turns = [], stop_after, 0

    def choose(self, system, user, labels):
        self.calls.append((user, list(labels)))
        if "ACTIONS YOU CAN STILL TAKE" in user:
            self.turns += 1
            pick = labels[-1] if (self.stop_after is not None and self.turns > self.stop_after
                                  and "I" in labels) else labels[0]
        elif "PROPOSED ROOT CAUSE" in user:                       # pointwise: Yes for the longest title
            pick = "Yes" if len(user) % 2 else "No"
        else:
            pick = labels[-1]
        return pick, {l: (1.0 if l == pick else 0.0) for l in labels}


def coordinator(llm):
    coord = CoordinatorAgent(Resources())
    coord.investigator = InvestigatorAgent(coord.res, llm_factory=lambda model: llm)
    return coord


def run(coord, name="hdfs_raw_sample.log", **opts):
    return coord.run(UploadSource(name, (SAMPLES / name).read_bytes()).read(),
                     RunOptions(persist=False, agent_model=FAKE, **opts))


def test_model_specs_pick_the_right_quantisation():
    assert parse_model_spec("Qwen/Qwen3-1.7B") == ("Qwen/Qwen3-1.7B", None)
    assert parse_model_spec("Qwen/Qwen3-8B") == ("Qwen/Qwen3-8B", "nf4")             # 8B only fits in 4-bit
    assert parse_model_spec("Qwen/Qwen3-1.7B:nf4") == ("Qwen/Qwen3-1.7B", "nf4")
    assert parse_model_spec("Qwen/Qwen3-4B:bf16") == ("Qwen/Qwen3-4B", None)


def test_the_loop_gathers_evidence_then_decides():
    llm = ScriptedLLM()
    ctx = run(coordinator(llm))
    inv = ctx.data["investigation"]
    assert ctx.data["status"] == "completed" and inv["status"] == "agent" and inv["model_label"] == "Fake-LLM"
    ep = inv["episodes"][0]
    acts = [s["action"] for s in ep["steps"]]
    assert len(acts) == config.AGENT_MAX_STEPS and len(set(acts)) == len(acts)       # budget respected, no repeats
    assert all(s["observation"] for s in ep["steps"])                               # every tool produced evidence
    assert ep["choice"] in ep["candidates"] and ep["choice"] in ctx.profile.root_causes
    assert ep["model_calls"] == len(ep["steps"]) + len(ep["candidates"])             # debiased: one call per rotation
    # the decision overrides the rule engine for every member of the case's group
    members = [p for p in ctx.data["rca"]["sessions"] if p["investigation"] == ep["case"]]
    assert members and all(p["primary_cause"] == ep["choice"] and p["decided_by"] == "Fake-LLM" for p in members)


def test_the_menu_never_offers_a_used_tool_or_deciding_before_any_evidence():
    llm = ScriptedLLM()
    run(coordinator(llm))
    menus = [(u, l) for u, l in llm.calls if "ACTIONS YOU CAN STILL TAKE" in u][:config.AGENT_MAX_STEPS]
    assert "I" not in menus[0][1]                                                   # must look before deciding
    for i, (user, labels) in enumerate(menus):
        taken = re.findall(r"^\[(\w+)\]", user.split("EVIDENCE SO FAR:")[1].split("ACTIONS YOU")[0], flags=re.M)
        assert len(taken) == i and not {ACTIONS[l][0] for l in labels} & set(taken)


def test_the_model_can_stop_early():
    ctx = run(coordinator(ScriptedLLM(stop_after=2)))
    ep = ctx.data["investigation"]["episodes"][0]
    assert [s["action"] for s in ep["steps"]][-1] == "decide" and len(ep["steps"]) == 3


def test_decision_methods_all_choose_from_the_shortlist():
    for decision, calls_per_candidate in (("pointwise", 1), ("listwise", 0), ("listwise_debiased", 1)):
        ctx = run(coordinator(ScriptedLLM()), agent_decision=decision)
        for ep in ctx.data["investigation"]["episodes"]:
            assert ep["choice"] in ep["candidates"] and ep["decision"] == decision
            final_calls = ep["model_calls"] - len(ep["steps"])
            if len(ep["candidates"]) > 1:
                assert final_calls == (len(ep["candidates"]) * calls_per_candidate or 1)


def test_the_candidate_order_does_not_follow_the_rule_ranking():
    ctx = run(coordinator(ScriptedLLM()))
    eps = ctx.data["investigation"]["episodes"]
    assert any(ep["candidates"][0] != ep["rule_top"] for ep in eps if len(ep["candidates"]) > 1)    # shuffled


def test_case_limit_keeps_the_rule_cause_for_the_rest():
    ctx = run(coordinator(ScriptedLLM()), agent_max_cases=1)
    inv = ctx.data["investigation"]
    assert inv["cases"] == 1 and inv["groups"] > 1
    others = [p for p in ctx.data["rca"]["sessions"] if p["decided_by"] != "Fake-LLM"]
    assert others and all(p["decided_by"] == "rules (case limit reached)" for p in others)
    assert any("investigated 1 of" in w for w in ctx.warnings)


def test_unavailable_model_falls_back_to_the_rules_and_says_so():
    def missing(model):
        raise OSError("weights not found")
    coord = CoordinatorAgent(Resources())
    coord.investigator = InvestigatorAgent(coord.res, llm_factory=missing)
    ctx = run(coord)
    assert ctx.data["status"] == "completed" and ctx.data["investigation"]["status"] == "unavailable"
    assert ctx.data["rca"]["causes"] and any("could not load" in w for w in ctx.warnings)
    assert all(p["decided_by"] == "rules" for p in ctx.data["rca"]["sessions"])
    assert "unavailable" in to_markdown(ctx.data["report"])


def test_no_anomalies_means_the_model_is_never_loaded():
    loaded = []
    coord = CoordinatorAgent(Resources())
    coord.investigator = InvestigatorAgent(coord.res, llm_factory=lambda m: loaded.append(m) or ScriptedLLM())
    normal = "E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21"
    ctx = coord.run(TextSource(normal).read(), RunOptions(persist=False, agent_model=FAKE))
    assert ctx.data["status"] == "completed" and not loaded


def test_rules_mode_is_the_old_rule_based_analysis():
    ctx = coordinator(ScriptedLLM()).run(UploadSource("s", (SAMPLES / "hdfs_raw_sample.log").read_bytes()).read(),
                                         RunOptions(persist=False, agent_model="rules"))
    assert ctx.data["investigation"]["status"] == "rules" and ctx.data["insight"]["source"] == "knowledge base"
    assert all(p["primary_cause"] == p["hypotheses"][0]["cause"] for p in ctx.data["rca"]["sessions"])


def test_report_and_chat_facts_describe_the_investigation():
    ctx = run(coordinator(ScriptedLLM()))
    assert "## Investigation" in to_markdown(ctx.data["report"])
    facts = build_facts(ctx)
    assert "INVESTIGATION: Fake-LLM investigated" in facts and "Investigation by Fake-LLM" in facts
    assert ctx.data["insight"]["source"].startswith("investigation by Fake-LLM")


@pytest.mark.skipif(not get_profile("bgl").available, reason="BGL model not trained")
def test_the_alert_tag_column_of_bgl_is_hidden_from_the_investigator():
    llm = ScriptedLLM()
    ctx = run(coordinator(llm), "bgl_raw_sample.log")
    tags = {"APPSEV", "KERNTERM", "KERNMC", "KERNREC", "KERNRTSP", "APPREAD", "APPRES", "KERNMNTF", "MASABNORM"}
    seen = " ".join(s["observation"] for ep in ctx.data["investigation"]["episodes"] for s in ep["steps"]
                    if s["action"] == "read_logs") + " ".join(u for u, _ in llm.calls)
    raw = (SAMPLES / "bgl_raw_sample.log").read_text()
    assert any(re.match(rf"^{t} ", ln) for t in tags for ln in raw.splitlines())       # the raw file has the tags...
    assert not any(re.search(rf"(^|\n)\s*{t} \d{{9,10}} ", seen) for t in tags)         # ...the model never sees them


@pytest.mark.skipif(not model_available("Qwen/Qwen3-1.7B"), reason="Qwen3-1.7B weights not on this machine")
def test_real_model_smoke():
    ctx = CoordinatorAgent(Resources()).run(
        UploadSource("s", (SAMPLES / "hdfs_raw_sample.log").read_bytes()).read(),
        RunOptions(persist=False, agent_model="Qwen/Qwen3-1.7B", agent_max_cases=3))
    inv = ctx.data["investigation"]
    assert ctx.data["status"] == "completed" and inv["status"] == "agent" and inv["cases"] == 3
    for ep in inv["episodes"]:
        assert ep["choice"] in ctx.profile.root_causes and all(abs(sum(s["probs"].values()) - 1) < 1e-2 for s in ep["steps"])


@pytest.mark.skipif(not model_available("Qwen/Qwen3-1.7B"), reason="Qwen3-1.7B weights not on this machine")
def test_chunked_prefill_gives_the_same_answer_as_one_pass(monkeypatch):
    """choose() prefills the prompt in chunks to bound GPU memory. Chunking changes the matmul shapes, so scores move
    a little (bf16 rounding on near-tied options); a position or cache bug would change the answer completely."""
    from log_agents import slm
    llm = Resources().llm("Qwen/Qwen3-1.7B")
    user = ("EVIDENCE SO FAR:\n" + "Similar historical sessions: 4 of 5 labelled Anomaly; has E7 x3 that this window lacks. " * 12
            + "\nWhich action next? Answer with one letter.")
    labels = list("ABCDEFG")
    monkeypatch.setattr(slm, "CHOOSE_CHUNK", 128)
    chunked_pick, chunked = llm.choose("You are an engineer.", user, labels)
    monkeypatch.setattr(slm, "CHOOSE_CHUNK", 100_000)
    whole_pick, whole = llm.choose("You are an engineer.", user, labels)
    assert chunked_pick == whole_pick and all(abs(chunked[l] - whole[l]) < 0.15 for l in labels)
