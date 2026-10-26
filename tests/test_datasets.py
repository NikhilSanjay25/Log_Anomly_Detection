"""Tests of the multi-dataset pipeline: Drain template mining, dataset identification, BGL and OpenStack end to end,
and the log-fed chat. No language-model weights are needed (the chat model is replaced by a fake).

    python -m pytest tests -q
Requires samples/ (scripts/make_sample_logs.py, scripts/make_dataset_samples.py) and the trained artifacts
(artifacts/bgl, artifacts/openstack from scripts/train_dataset.py).
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from log_agents import config
from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.chat import RunChat, has_raw_logs, log_context
from log_agents.context import RunOptions
from log_agents.datasets import PROFILES, detect_dataset, get_profile
from log_agents.datasets.bgl import BGLProfile
from log_agents.drain import DrainParser
from log_agents.ingestion import TextSource, UploadSource
from log_agents.ml import AnomalyPipeline
from log_agents.resources import Resources

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
needs_bgl = pytest.mark.skipif(not get_profile("bgl").available, reason="BGL model not trained")
needs_openstack = pytest.mark.skipif(not get_profile("openstack").available, reason="OpenStack model not trained")


@pytest.fixture(scope="module")
def coord():
    return CoordinatorAgent(Resources())


def run_sample(coord, name, **opts):
    return coord.run(UploadSource(name, (SAMPLES / name).read_bytes()).read(), RunOptions(persist=False, **opts))


# ── Drain ─────────────────────────────────────────────────────────────────
def test_drain_keeps_different_events_apart_and_masks_variables():
    p = DrainParser()
    for line in ["Instance spawned successfully.", "Instance destroyed successfully.",
                 "Took 12.5 seconds to spawn the instance on the hypervisor.",
                 "Took 30.1 seconds to spawn the instance on the hypervisor.",
                 "Took 2.00 seconds to destroy the instance on the hypervisor."]:
        p.add(line)
    assert len(p.templates) == 4                       # 'spawned' / 'destroyed' are different events
    assert p.match("Took 99 seconds to spawn the instance on the hypervisor.") == \
        p.match("Took 12.5 seconds to spawn the instance on the hypervisor.")
    assert p.match("Instance spawned successfully.") != p.match("Instance destroyed successfully.")
    assert p.match("a message that was never seen in training") is None
    clone = DrainParser.from_dict(p.to_dict())         # templates survive saving and loading
    assert clone.match("Took 5 seconds to destroy the instance on the hypervisor.") == \
        p.match("Took 5 seconds to destroy the instance on the hypervisor.")
    n = len(p.templates)
    p.match("yet another unseen line")
    assert len(p.templates) == n                       # matching never learns


# ── identification ────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,dataset,fmt", [
    ("hdfs_raw_sample.log", "hdfs", "hdfs_raw"), ("hdfs_sequences.txt", "hdfs", "hdfs_events"),
    ("hdfs_traces_sample.csv", "hdfs", "hdfs_trace_csv"), ("bgl_raw_sample.log", "bgl", "bgl_raw"),
    ("openstack_raw_sample.log", "openstack", "openstack_raw")])
def test_each_sample_is_recognised_as_its_dataset(name, dataset, fmt):
    lines = [ln for ln in (SAMPLES / name).read_text(encoding="utf-8").splitlines() if ln.strip()][:2000]
    profile, found, conf, scores = detect_dataset(lines)
    assert (profile.name, found) == (dataset, fmt) and conf > 0.9
    others = [s for n, per in scores.items() if n != dataset for s in per.values()]
    assert max(others) < 0.5                            # no other dataset claims the input


def test_forcing_the_wrong_dataset_is_rejected(coord):
    ctx = run_sample(coord, "hdfs_raw_sample.log", dataset="bgl")
    assert ctx.data["status"] == "failed" and "does not look like" in ctx.data["error"]


def test_unsupported_logs_name_what_they_are(coord):
    ctx = coord.run(TextSource("Oct 11 22:14:15 mymachine su: 'su root' failed on /dev/pts/8\n" * 5).read(),
                    RunOptions(persist=False))
    assert ctx.data["status"] == "failed" and "Syslog" in ctx.data["error"] and "BGL" in ctx.data["error"]


def test_missing_model_artifacts_give_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_dataset.py --dataset bgl"):
        AnomalyPipeline(BGLProfile(directory=tmp_path))


# ── BGL ───────────────────────────────────────────────────────────────────
@needs_bgl
def test_bgl_end_to_end(coord):
    ctx = run_sample(coord, "bgl_raw_sample.log")
    assert ctx.data["status"] == "completed", ctx.data.get("error")
    assert all(s.status == "ok" for s in ctx.trace), [(s.agent, s.status, s.issues) for s in ctx.trace]
    assert ctx.profile.name == "bgl" and ctx.data["identification"]["session"] == "window"
    s = ctx.data["detection"]["sessions"]
    assert len(s) == 45 and s.block_id.str.startswith("win_").all() and (s.n_events >= 95).all()
    m = ctx.data["detection"]["metrics"]                 # the label column of the log is ground truth, never an input
    assert m["n"] == 45 and m["accuracy"] >= 0.8
    rca = ctx.data["rca"]
    assert rca["causes"] and all(c["cause"] in ctx.profile.root_causes for c in rca["causes"])
    assert "windows" in ctx.data["insight"]["summary"]
    assert ctx.data["insight"]["source"] == "knowledge base"


@needs_bgl
def test_bgl_labels_are_not_a_model_input(coord):
    """Hiding every alert tag (replacing it by '-') must not change the predictions."""
    text = (SAMPLES / "bgl_raw_sample.log").read_text(encoding="utf-8")
    hidden = "\n".join("- " + ln.split(" ", 1)[1] for ln in text.splitlines())
    a = coord.run(TextSource(text).read(), RunOptions(persist=False)).data["detection"]["sessions"]
    b = coord.run(TextSource(hidden).read(), RunOptions(persist=False)).data["detection"]["sessions"]
    assert np.allclose(a.anomaly_proba.values, b.anomaly_proba.values, atol=1e-9)       # equal up to float rounding
    assert (a.is_anomaly.values == b.is_anomaly.values).all()


# ── OpenStack ─────────────────────────────────────────────────────────────
@needs_openstack
def test_openstack_end_to_end_finds_the_injected_delays(coord):
    ctx = run_sample(coord, "openstack_raw_sample.log")
    assert ctx.data["status"] == "completed", ctx.data.get("error")
    assert all(s.status == "ok" for s in ctx.trace), [(s.agent, s.status, s.issues) for s in ctx.trace]
    s = ctx.data["detection"]["sessions"]
    labelled = set(json.loads((SAMPLES / "openstack_raw_sample.labels.json").read_text())["anomalous_instances"])
    flagged = set(s[s.is_anomaly == 1].block_id)
    assert labelled <= flagged and len(flagged) <= 4
    assert ctx.data["rca"]["causes"][0]["cause"] == "slow_spawn"
    evidence = " ".join(e for p in ctx.data["rca"]["sessions"] for e in p["evidence"])
    assert "Spawn took" in evidence and "maximum" in evidence
    # lines without an instance id stay in Bronze but do not form sessions
    assert ctx.data["preprocessing"]["lines_without_instance"] > 0


def slow_down(text, instance, extra=25.0):
    """Make one VM spawn `extra` seconds slower: later log lines of that instance move later in time and the
    reported 'Took N seconds' grow, exactly what the injected anomalies of the dataset look like."""
    def shift(line):
        m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)", line)
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") + timedelta(seconds=extra)
        return line.replace(m.group(1), t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    out, started = [], False
    for ln in text.splitlines():
        if f"[instance: {instance}]" in ln:
            if "Creating image" in ln:
                started = True
            elif started:
                ln = shift(ln)
            ln = re.sub(r"Took ([\d.]+) seconds to (spawn|build)",
                        lambda m: f"Took {float(m.group(1)) + extra:.2f} seconds to {m.group(2)}", ln)
        out.append(ln)
    return chr(10).join(out)


@needs_openstack
def test_openstack_timing_alone_makes_an_instance_anomalous(coord):
    """The injected anomalies have normal event order; only the timing differs. Slow down one normal instance
    in the log and it is flagged, with the same event sequence, while every other instance is unchanged."""
    text = (SAMPLES / "openstack_raw_sample.log").read_text(encoding="utf-8")
    base = coord.run(TextSource(text).read(), RunOptions(persist=False)).data["detection"]["sessions"]
    victim = base[(base.is_anomaly == 0)].block_id.iloc[0]
    after = coord.run(TextSource(slow_down(text, victim)).read(), RunOptions(persist=False)).data["detection"]["sessions"]
    b, a = base.set_index("block_id"), after.set_index("block_id")
    assert b.loc[victim, "is_anomaly"] == 0 and a.loc[victim, "is_anomaly"] == 1
    assert a.loc[victim, "sequence"] == b.loc[victim, "sequence"]
    others = [i for i in b.index if i != victim]
    assert (a.loc[others, "is_anomaly"] == b.loc[others, "is_anomaly"]).all()


# ── chat is fed the relevant raw logs ─────────────────────────────────────
@pytest.fixture(scope="module")
def os_ctx(coord):
    return run_sample(coord, "openstack_raw_sample.log")


@needs_openstack
def test_chat_gets_the_raw_lines_of_a_named_session(coord, os_ctx):
    s = os_ctx.data["detection"]["sessions"]
    anomalous = s[s.is_anomaly == 1].block_id.iloc[0]
    chat = RunChat(os_ctx, coord.rca)
    msgs, content = chat.messages_for(f"Why is {anomalous} anomalous?")
    section = content.split(f"RAW LOG LINES of instance {anomalous}")[1]
    lines = [ln for ln in section.split("\n\n")[0].splitlines()[1:] if ln.strip()]
    assert lines and all(f"[instance: {anomalous}]" in ln for ln in lines)
    assert any("Took" in ln and "seconds to spawn" in ln for ln in lines)       # the evidence is in what it sees
    assert "FACTS:" in msgs[0]["content"] and "OpenStack" in msgs[0]["content"]


@needs_openstack
def test_chat_finds_logs_for_events_keywords_and_general_questions(coord, os_ctx):
    chat = RunChat(os_ctx, coord.rca)
    sv = os_ctx.data["medallion"]["silver"]
    eid = sv.event_id.iloc[0]
    _, by_event = chat.messages_for(f"What does {eid} mean here?")
    assert f"RAW LOG LINES of event {eid}" in by_event
    _, by_word = chat.messages_for("Which lines mention libvirt?")
    section = by_word.split("matching the question's keywords:")[1].strip().splitlines()
    assert section and all("libvirt" in ln.lower() for ln in section)
    _, general = chat.messages_for("Why are things anomalous in this run?")
    assert "RAW LOG LINES of the most anomalous instance" in general
    _, unrelated = chat.messages_for("Thanks!")
    assert "RAW LOG LINES" not in unrelated
    assert len(log_context(os_ctx, "Why are things anomalous?", [])) <= config.CHAT_MAX_LOG_CHARS + 500


@needs_openstack
def test_numbers_from_the_fed_logs_are_not_flagged(coord, os_ctx):
    s = os_ctx.data["detection"]["sessions"]
    anomalous = s[s.is_anomaly == 1].block_id.iloc[0]
    chat = RunChat(os_ctx, coord.rca)
    msgs, content = chat.messages_for(f"What happened to {anomalous}?")
    pid = re.search(r"\b\d{4} INFO\b", content).group(0).split()[0]             # a number only the raw lines contain
    assert pid not in os_ctx.data["insight_facts"]

    class FakeSLM:
        def chat(self, messages, max_new_tokens):
            return f"The logs show process {pid} and process 987654."
    answer, issues = chat.ask(FakeSLM(), f"What happened to {anomalous}?")
    assert any("987654" in i for i in issues) and not any(pid in i for i in issues)


def test_inputs_without_raw_lines_get_no_log_section(coord):
    ctx = coord.run(UploadSource("t.csv", (SAMPLES / "hdfs_traces_sample.csv").read_bytes()).read(),
                    RunOptions(persist=False))
    assert not has_raw_logs(ctx)
    s = ctx.data["detection"]["sessions"]
    _, content = RunChat(ctx, coord.rca).messages_for(f"Why is {s.block_id.iloc[0]} anomalous?")
    assert "RAW LOG LINES" not in content


@needs_bgl
def test_bgl_chat_shows_error_lines_first(coord):
    ctx = run_sample(coord, "bgl_raw_sample.log")
    s = ctx.data["detection"]["sessions"]
    win = s[s.is_anomaly == 1].sort_values("anomaly_proba", ascending=False).block_id.iloc[0]
    _, content = RunChat(ctx, coord.rca).messages_for(f"Explain {win}")
    assert f"RAW LOG LINES of window {win}" in content and "lines shown" in content       # 100-line window, capped
    assert re.search(r"RAS (KERNEL|APP) (FATAL|ERROR|SEVERE|FAILURE)", content)
