"""End-to-end tests of the agent workflow (no SLMs, so they run on CPU in seconds).

    python -m pytest tests -q
Requires samples/ (python scripts/make_sample_logs.py) and the model artifacts.
"""
from pathlib import Path

import pandas as pd
import pytest

from log_agents import config, hdfs
from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.agents.insight import build_facts, validate_insight
from log_agents.chat import RunChat
from log_agents.context import RunOptions
from log_agents.ingestion import KubernetesSource, TextSource, UploadSource
from log_agents.report import to_markdown
from log_agents.resources import Resources

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OPTS = dict(use_slm_insight=False, lora_mode="off", persist=False)


@pytest.fixture(scope="module")
def coord():
    return CoordinatorAgent(Resources())


@pytest.fixture(scope="module")
def truth():
    t = pd.read_csv(SAMPLES / "hdfs_traces_sample.csv")
    t["seq"] = t.Features.str.strip("[]").str.replace(",", " ")
    t["y"] = (t.Label == "Fail").astype(int)
    return t.set_index("BlockId")


def run_file(coord, name):
    path = SAMPLES / name
    return coord.run(UploadSource(name, path.read_bytes()).read(), RunOptions(**OPTS))


@pytest.mark.parametrize("name,fmt", [("hdfs_raw_sample.log", "hdfs_raw"),
                                      ("hdfs_sequences.txt", "hdfs_events"),
                                      ("hdfs_traces_sample.csv", "hdfs_trace_csv")])
def test_formats_end_to_end(coord, truth, name, fmt):
    ctx = run_file(coord, name)
    assert ctx.data["status"] == "completed", ctx.data.get("error")
    assert ctx.data["identification"]["format"] == fmt
    assert all(s.status == "ok" for s in ctx.trace), [(s.agent, s.status, s.issues) for s in ctx.trace]
    s = ctx.data["detection"]["sessions"].set_index("block_id")
    # parsing reproduces the exact training sequences, and the detector agrees with ground truth
    assert (s.sequence == truth.seq.reindex(s.index)).all()
    assert (s.is_anomaly == truth.y.reindex(s.index)).mean() > 0.95
    assert ctx.data["rca"]["causes"] and ctx.data["insight"]["recommendations"]
    assert "Root Cause Analysis" in to_markdown(ctx.data["report"])


def test_raw_cleaning_and_correlation(coord):
    ctx = run_file(coord, "hdfs_raw_sample.log")
    pre = ctx.data["preprocessing"]
    assert pre["duplicates_removed"] == 1 and pre["noise_removed"] == 1
    # the sample routes most anomalous blocks through one flaky DataNode; RCA must find it
    assert ctx.data["rca"]["correlations"]["nodes"]


def test_csv_ground_truth_metrics(coord):
    ctx = run_file(coord, "hdfs_traces_sample.csv")
    assert ctx.data["detection"]["metrics"]["f1"] > 0.9


def test_single_sequences(coord):
    ctx = coord.run(TextSource("E22 E5").read(), RunOptions(**OPTS))
    assert ctx.data["detection"]["sessions"].is_anomaly.iloc[0] == 1
    normal = "E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21"
    ctx = coord.run(TextSource(normal).read(), RunOptions(**OPTS))
    assert ctx.data["detection"]["sessions"].is_anomaly.iloc[0] == 0
    assert ctx.data["insight"]["source"] == "knowledge base"


def test_non_hdfs_rejected(coord):
    ctx = coord.run(TextSource('{"level":"info","msg":"pod started"}\n{"level":"error","msg":"oom"}').read(),
                    RunOptions(**OPTS))
    assert ctx.data["status"] == "failed"
    assert "Kubernetes" in ctx.data["error"]


def test_kubernetes_is_stub():
    with pytest.raises(NotImplementedError):
        KubernetesSource().read()


def test_insight_facts_include_similar_sessions(coord):
    ctx = run_file(coord, "hdfs_raw_sample.log")
    facts = build_facts(ctx)
    # every root-cause group's example and every normal example comes with its retrieved neighbours
    assert facts.count("Similar historical sessions") == (facts.count("ROOT CAUSE GROUP")
                                                          + facts.count("NORMAL EXAMPLE"))
    assert "NORMAL SESSIONS" in facts and ctx.data["rca"]["normal_examples"]
    # neighbour block ids are not in this run, so they must not reach the prompt
    assert set(hdfs.BLOCK_RE.findall(facts)) <= set(ctx.data["detection"]["sessions"].block_id)
    assert ctx.data["insight"]["normal_explanation"]                    # the fallback covers normal sessions too


def test_normal_only_run_gets_normal_evidence(coord):
    text = ("blk_1: E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21\n"
            "blk_2: E5 E5 E5 E22 E11 E9 E11 E9 E11 E9 E26 E26 E26 E3 E3 E4 E2 E3 E3 E4 E23 E23 E23 E21 E21 E21")
    ctx = coord.run(TextSource(text).read(), RunOptions(**OPTS))
    assert ctx.data["detection"]["sessions"].is_anomaly.sum() == 0
    examples = ctx.data["rca"]["normal_examples"]
    evidence = " ".join(e for p in examples for e in p["evidence"])
    assert "Lifecycle complete" in evidence and "E4" in evidence    # E4 is explained as routine, not hidden
    facts = build_facts(ctx)
    assert "ROOT CAUSE GROUP" not in facts and "NORMAL EXAMPLE" in facts
    ctx.data["insight_facts"] = facts
    assert validate_insight(ctx.data["insight"], ctx) == []          # fallback numbers all come from the facts


def test_chat_grounds_questions_on_the_run(coord):
    ctx = run_file(coord, "hdfs_raw_sample.log")
    chat = RunChat(ctx, coord.rca)
    s = ctx.data["detection"]["sessions"]
    anom, norm = s[s.is_anomaly == 1].block_id.iloc[0], s[s.is_anomaly == 0].block_id.iloc[0]
    msgs, content = chat.messages_for(f"Why is {anom} anomalous but {norm} not? What about blk_123?")
    assert msgs[0]["role"] == "system"
    assert "FACTS:" in msgs[0]["content"] and "YOUR EARLIER ANALYSIS" in msgs[0]["content"]
    # named sessions of this run get their own evidence and neighbours; unknown ids are ignored
    assert f"BLOCK {anom}: classified Anomaly" in content and f"BLOCK {norm}: classified Normal" in content
    assert "BLOCK blk_123" not in content and content.count("Similar historical sessions") == 2

    class FakeSLM:
        def chat(self, messages, max_new_tokens):
            return f"<think></think>{anom} failed. Compare blk_999, which retried 4242 times."
    answer, issues = chat.ask(FakeSLM(), "why?")
    assert answer.startswith(anom)
    assert any("blk_999" in i for i in issues) and any("4242" in i for i in issues)
    for i in range(10):
        chat.ask(FakeSLM(), f"question {i}")
    msgs, _ = chat.messages_for("one more")
    assert len(msgs) == 1 + 2 * config.CHAT_MAX_TURNS + 1          # history is bounded
    assert [m["role"] for m in msgs[:3]] == ["system", "user", "assistant"]


def test_chat_finds_sessions_without_block_ids(coord):
    ctx = coord.run(TextSource("E22 E5\n" + "E22 E5 E5 E5 E11 E9 E11 E9 E11 E9 E26 E26 E26 E23 E23 E23 E21 E21 E21")
                    .read(), RunOptions(**OPTS))
    chat = RunChat(ctx, coord.rca)
    assert chat.mentioned_blocks("compare seq_0001 with seq_0002 and seq_0099") == ["seq_0001", "seq_0002"]
