"""
Agentic HDFS Log Anomaly Detection - Streamlit UI
--------------------------------------------------
    streamlit run streamlit_app.py

Log Ingestion → Log Identification Agent → Preprocessing & Cleaning Agent → Medallion (Bronze/Silver/Gold)
→ Anomaly Detection Agent (Transformer → FAISS → RAG → Random Forest [+ Qwen3-0.6B LoRA second opinion])
→ Root Cause Analysis Agent → Insight Agent (Qwen3-1.7B SLM) → this UI, all orchestrated by the Coordinator Agent.
"""
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from log_agents import config
from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.chat import RunChat
from log_agents.context import RunMemory, RunOptions
from log_agents.hdfs import EVENT_INFO, TEMPLATE_TEXT
from log_agents.ingestion import KubernetesSource, TextSource, UploadSource
from log_agents.report import to_markdown
from log_agents.resources import Resources

SAMPLES = Path(__file__).parent / "samples"
SEV_ICON = {"critical": "🟥", "high": "🔴", "medium": "🟠", "low": "🟢"}
STATUS_ICON = {"ok": "✅", "fallback": "🟡", "failed": "❌", "skipped": "⏭️", "running": "⏳"}

st.set_page_config(page_title="HDFS Log Anomaly Agents", page_icon="🔍", layout="wide")

# Base colours/fonts live in .streamlit/config.toml; this adds what the theme config cannot express:
# the radial background, frosted-glass panels, the gradient title and the agent terminal.
st.markdown("""
<style>
.stApp {
  background: radial-gradient(ellipse at top, #0f172a 0%, #090d16 55%, #020617 100%) fixed;
}
[data-testid="stHeader"] {
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] > div:first-child {
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}
/* glass cards: KPI tiles, expanders, alerts */
[data-testid="stMetric"], [data-testid="stExpander"] details, [data-testid="stAlertContainer"] {
  background: rgba(30, 41, 59, 0.45);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 0.9rem;
  transition: border-color 0.2s ease;
}
[data-testid="stMetric"] { padding: 0.9rem 1.1rem; }
[data-testid="stMetric"]:hover, [data-testid="stExpander"] details:hover { border-color: #38bdf8; }
[data-testid="stMetricValue"] { color: #f8fafc; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: #94a3b8; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #94a3b8; }
/* status panels: anomaly / alert = rose, healthy = emerald */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  background: rgba(136, 19, 55, 0.3); border-color: rgba(244, 63, 94, 0.4); color: #fb7185;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #34d399;
}
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid rgba(255, 255, 255, 0.08); }
.stTabs [aria-selected="true"] { color: #38bdf8; }

.hero-title {
  display: inline-block; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;
  margin: 0.2rem 0 0 0; padding-bottom: 0.1rem;
  background: linear-gradient(90deg, #38bdf8 0%, #818cf8 35%, #c084fc 65%, #f472b6 100%);
  -webkit-background-clip: text; background-clip: text;
  color: transparent !important; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #94a3b8; margin: 0.4rem 0 0.9rem 0; font-size: 0.98rem; }
.pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem; margin: 0 0 1.4rem 0; }
.pipeline .chip {
  padding: 0.22rem 0.6rem; border-radius: 999px; font-size: 0.74rem; font-weight: 500; color: #cbd5e1;
  white-space: nowrap;
  background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);
}
.pipeline .chip.ml { color: #38bdf8; border-color: rgba(56, 189, 248, 0.35); }
.pipeline .chip.slm { color: #c084fc; border-color: rgba(192, 132, 252, 0.35); }
.pipeline .arrow { color: #64748b; font-size: 0.8rem; }

/* inline code: slate terminal look instead of Streamlit's default green */
.stMarkdown code:not(pre code) {
  color: #7dd3fc !important; background: rgba(2, 6, 23, 0.8) !important; border: 1px solid #1e293b;
  border-radius: 0.35rem; padding: 0.05rem 0.35rem;
}
/* primary action: title gradient */
.stButton button[kind="primary"]:not(:disabled) {
  background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 55%, #a855f7 100%);
  border: none; color: #f8fafc; font-weight: 600;
  box-shadow: 0 8px 24px -10px rgba(99, 102, 241, 0.6);
}
.stButton button[kind="primary"]:not(:disabled):hover { filter: brightness(1.12); }
/* sidebar: compact section headings */
[data-testid="stSidebar"] h2 {
  font-size: 0.78rem !important; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  color: #94a3b8 !important; padding: 1.1rem 0 0.2rem 0;
}
[data-testid="stSidebar"] h2:first-of-type { padding-top: 0; }
h3 { color: #f8fafc; }

/* empty state: agent cards */
.agent-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 0.8rem; margin-top: 0.4rem; }
.agent-card {
  background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.9rem;
  padding: 0.95rem 1.05rem; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  transition: border-color 0.2s ease, transform 0.2s ease;
}
.agent-card:hover { border-color: #38bdf8; transform: translateY(-2px); }
.agent-card .step { font-family: "JetBrains Mono", monospace; font-size: 0.72rem; color: #f472b6; }
.agent-card .name { color: #f8fafc; font-weight: 600; margin: 0.15rem 0 0.3rem 0; }
.agent-card .desc { color: #94a3b8; font-size: 0.85rem; line-height: 1.45; }

.agent-terminal {
  background: #020617; border: 1px solid #1e293b; border-radius: 0.75rem;
  padding: 0.9rem 1.1rem; max-height: 32rem; overflow: auto;
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.8rem; line-height: 1.65;
  color: #cbd5e1;
}
.agent-terminal .row { padding-left: 5.2em; text-indent: -5.2em; }
.agent-terminal .ts { color: #64748b; }
.agent-terminal .from { color: #38bdf8; }
.agent-terminal .arrow { color: #64748b; }
.agent-terminal .to { color: #c084fc; }
.agent-terminal .tag { color: #f472b6; }
.agent-terminal .ok { color: #34d399; }
.agent-terminal .bad { color: #fb7185; }
.agent-terminal .msg { color: #e2e8f0; }
.badge-anomaly, .badge-healthy {
  display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
}
.badge-anomaly { color: #fb7185; background: rgba(225, 29, 72, 0.15); border: 1px solid rgba(244, 63, 94, 0.4); }
.badge-healthy { color: #34d399; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; }
</style>
""", unsafe_allow_html=True)


def agent_terminal(messages):
    """Inter-agent messages as a terminal log. Content is escaped: it can contain text from the uploaded logs."""
    rows = []
    for m in messages:
        content = m.content
        if content.startswith("["):
            status, _, content = content[1:].partition("] ")
            tag = f'<span class="{"ok" if status == "ok" else "bad"}">[{html.escape(status.upper())}]</span>'
        elif content.startswith("Assigned task: "):
            tag, content = '<span class="tag">[TASK]</span>', content[len("Assigned task: "):]
        else:
            tag = '<span class="tag">[MSG]</span>'
        ts = datetime.fromtimestamp(m.ts).strftime("%H:%M:%S")
        rows.append(f'<div class="row"><span class="ts">{ts}</span> <span class="from">{html.escape(m.sender)}</span>'
                    f' <span class="arrow">→</span> <span class="to">{html.escape(m.recipient)}</span>'
                    f' {tag} <span class="msg">{html.escape(content)}</span></div>')
    return '<div class="agent-terminal">' + "".join(rows) + "</div>"


@st.cache_resource(show_spinner="Loading ML pipeline (Transformer, FAISS, Random Forest)…")
def get_resources():
    res = Resources()
    res.pipeline()
    return res


resources = get_resources()
coordinator = CoordinatorAgent(resources)

# ═══════════════════════════════ SIDEBAR ═══════════════════════════════
with st.sidebar:
    st.header("📥 Log Ingestion")
    source_kind = st.radio("Source", ["Upload file", "Paste logs", "Sample logs", "Kubernetes (real-time)"],
                           label_visibility="collapsed")
    batch_source = None
    if source_kind == "Upload file":
        up = st.file_uploader("HDFS log file", type=["log", "txt", "csv"],
                              help="Raw HDFS log lines, an Event_traces.csv, or event-ID sequences (one per line)")
        if up:
            batch_source = UploadSource(up.name, up.getvalue())
    elif source_kind == "Paste logs":
        txt = st.text_area("Raw HDFS log lines or event sequences", height=160,
                           placeholder="E22 E5 E5 E5 E11 E9 E11 E9 E26 E26\n"
                                       "or\n081109 203615 148 INFO dfs.DataNode$PacketResponder: …")
        if txt.strip():
            batch_source = TextSource(txt)
    elif source_kind == "Sample logs":
        samples = sorted(p.name for p in SAMPLES.glob("*")) if SAMPLES.exists() else []
        if samples:
            pick = st.selectbox("Sample", samples)
            batch_source = UploadSource(pick, (SAMPLES / pick).read_bytes())
        else:
            st.info("Run `python scripts/make_sample_logs.py` to create sample logs.")
    else:
        st.info("**Coming soon.** " + KubernetesSource.__doc__.split("\n\n")[1].strip().replace("\n", " "))
        st.text_input("Namespace", "default", disabled=True)
        st.text_input("Pod", "hdfs-datanode-0", disabled=True)

    st.header("⚙️ Agent settings")
    use_slm = st.toggle("Insight Agent uses SLM", value=True,
                        help=f"{config.INSIGHT_MODEL}. Off = deterministic knowledge-base insight (instant).")
    lora_available = resources.status()["lora_available"]
    lora_mode = st.selectbox("LoRA SLM second opinion (advisory)", ["auto", "all", "off"],
                             disabled=not lora_available,
                             help="auto: only uncertain sessions · all: every unique pattern (slower) · off. "
                                  "Shown as evidence only - the RF decision is final.",
                             index=0 if lora_available else 2)
    threshold = st.slider("Anomaly threshold (RF probability)", 0.1, 0.9, 0.5, 0.05)
    persist = st.toggle("Save medallion layers + run memory", value=True)

    st.header("🖥️ System status")
    ps = resources.pipeline().status()
    st.caption(f"Device **{ps['device']}** · FAISS **{ps['faiss_vectors']:,}** vectors · "
               f"RF **{ps['rf_estimators']}** trees · RAG metadata {'✅' if ps['rag_metadata'] else '⚠️ missing'}")
    st.caption(f"Insight SLM: `{config.INSIGHT_MODEL}` · LoRA adapter {'✅' if lora_available else '❌ not found'}")
    if ps["device"] == "cpu":
        st.caption("⚠️ CPU only: SLM steps take a few minutes. Install CUDA PyTorch to use your GPU.")

    recent = RunMemory().recent(5)
    if recent:
        st.header("🧠 Run memory")
        for r in recent:
            st.caption(f"{r['generated_at'][5:16]} · {r['source']} · {r.get('anomalies', '–')}/{r.get('sessions', '–')} anomalous")

# ═══════════════════════════════ HEADER ════════════════════════════════
PIPELINE = [("Ingestion", ""), ("Identification", ""), ("Preprocessing", ""), ("Medallion", ""),
            ("Transformer · FAISS · RAG · RF", "ml"), ("LoRA SLM", "slm"), ("Root Cause Analysis", ""),
            ("Insight SLM", "slm")]
st.markdown('<div class="hero-title">Agentic HDFS Log Anomaly Detection</div>'
            '<p class="hero-sub">A Coordinator Agent runs specialised agents end to end, from raw logs to root '
            'causes and recommendations.</p><div class="pipeline">'
            + '<span class="arrow">→</span>'.join(f'<span class="chip {cls}">{name}</span>' for name, cls in PIPELINE)
            + "</div>", unsafe_allow_html=True)

run_clicked = st.button("▶️ Run analysis", type="primary", width="stretch", disabled=batch_source is None)
# A fixed slot for the progress box: without it the result tabs below move up one position on the next rerun
# (e.g. when a chat question is sent), Streamlit treats them as new tabs and jumps back to the first one.
run_area = st.container()
if run_clicked:
    batch = batch_source.read()
    opts = RunOptions(use_slm_insight=use_slm, lora_mode=lora_mode if lora_available else "off",
                      threshold=threshold, persist=persist)
    with run_area, st.status("Coordinator Agent is running the workflow…", expanded=True) as status:
        def on_step(step):
            st.write(f"{STATUS_ICON.get(step.status, '•')} **{step.agent}** ({step.duration_s:.1f}s) — {step.summary}")
            for issue in step.issues:
                st.caption(f"↳ {issue}")
        ctx = coordinator.run(batch, opts, on_step=on_step)
        ok = ctx.data["status"] == "completed"
        status.update(label="Workflow completed" if ok else "Workflow stopped",
                      state="complete" if ok else "error", expanded=not ok)
    st.session_state["ctx"] = ctx

ctx = st.session_state.get("ctx")
if ctx is None:
    st.info("Choose a log source in the sidebar and press **Run analysis**. "
            "Try *Sample logs → hdfs_raw_sample.log* for a full demo.")
    AGENTS = [
        ("Log Identification Agent", "Recognises raw HDFS lines, Event_traces CSV or event sequences and picks the workflow."),
        ("Preprocessing & Cleaning Agent", "Parses, normalises, de-duplicates and removes noise."),
        ("Medallion Processing", "Bronze raw cleaned logs → Silver structured events → Gold block sessions."),
        ("Anomaly Detection Agent", "Transformer → FAISS → RAG → Random Forest, with an advisory LoRA SLM opinion."),
        ("Root Cause Analysis Agent", "Ranks causes from error events, similar logs, context and correlations."),
        ("Insight Agent", "Qwen3-1.7B writes explanation, impact and next-best actions, validated against the facts."),
    ]
    st.markdown('<div class="agent-grid">' + "".join(
        f'<div class="agent-card"><div class="step">0{i}</div><div class="name">{n}</div><div class="desc">{d}</div></div>'
        for i, (n, d) in enumerate(AGENTS, 1)) + "</div>", unsafe_allow_html=True)
    st.stop()

if ctx.data["status"] != "completed":
    st.error(ctx.data.get("error", "The workflow failed."))
    ident = ctx.data.get("identification") or {}
    if ident.get("family") == "unsupported":
        st.caption("Supported inputs: raw HDFS log lines · Event_traces.csv · event sequences like `E5 E22 E11 E9 E26`")
    st.stop()

d = ctx.data
report = d["report"]
sessions = d["detection"]["sessions"]
rca = d["rca"]
insight = d["insight"]
n_anom = int(sessions.is_anomaly.sum())

for w in ctx.warnings:
    st.warning(w)

tabs = st.tabs(["📊 Anomaly Summary", "🧭 Root Cause Analysis", "🤖 AI Insights", "🛠️ Recommendations",
                "🔬 Evidence", "📄 Reports", "🕸️ Agent Trace"])

# ── Anomaly Summary ───────────────────────────────────────────────────
with tabs[0]:
    ident = d["identification"]
    badge = (f'<span class="badge-anomaly">● {n_anom:,} anomalous session(s) detected</span>' if n_anom
             else '<span class="badge-healthy">● Healthy: no anomalies detected</span>')
    st.markdown(badge, unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Block sessions", f"{len(sessions):,}")
    c[1].metric("Anomalies", f"{n_anom:,}", f"{n_anom / len(sessions):.1%}", delta_color="inverse")
    c[2].metric("Uncertain", f"{int(sessions.uncertain.sum()):,}",
                help="RF probability near the threshold, or RF disagrees with similar historical sessions")
    c[3].metric("Root-cause groups", len(rca["causes"]))
    c[4].metric("Severity", insight["impact"]["severity"].upper())
    st.caption(f"Log family **{ident['family']}** · format `{ident['format']}` · confidence {ident['confidence']:.0%} · "
               f"workflow: {ident['workflow']}")
    st.caption(f"SLM second opinion: {d['detection']['slm_note']}")
    n_dis = int(sessions.slm_disagrees.sum())
    if n_dis:
        st.warning(f"The LoRA SLM disagrees with the RF on {n_dis} session(s). The RF label is kept "
                   "(it is far more accurate on held-out data); review these sessions manually.")
        dis = sessions[sessions.slm_disagrees]
        with st.expander("Sessions where the SLM disagrees"):
            st.dataframe(pd.DataFrame({"Block": dis.block_id, "RF label": dis.rf_label.map({1: "Anomaly", 0: "Normal"}),
                                       "RF prob.": dis.anomaly_proba.round(3), "SLM": dis.slm_label,
                                       "Sequence": dis.sequence}), hide_index=True, width="stretch")
    if d["detection"].get("metrics"):
        m = d["detection"]["metrics"]
        st.success(f"Ground-truth labels found in input — accuracy {m['accuracy']:.4f} · precision {m['precision']:.4f} · "
                   f"recall {m['recall']:.4f} · F1 {m['f1']:.4f} (TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']})")
    st.subheader("Anomalous sessions")
    view = sessions[sessions.is_anomaly == 1].sort_values("anomaly_proba", ascending=False)
    if view.empty:
        st.success("No anomalous sessions detected.")
    else:
        prim = {p["block_id"]: p["hypotheses"][0]["title"] for p in rca["sessions"]}
        st.dataframe(pd.DataFrame({
            "Block": view.block_id, "Probable cause": view.block_id.map(prim),
            "Anomaly prob.": view.anomaly_proba, "Events": view.n_events,
            "Error events": view.error_events.map(", ".join),
            "SLM opinion (advisory)": view.slm_label.fillna("–") + view.slm_disagrees.map({True: " ⚠️ disagrees", False: ""}),
            "Sequence": view.sequence}),
            hide_index=True, width="stretch",
            column_config={"Anomaly prob.": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ── Root Cause Analysis ───────────────────────────────────────────────
with tabs[1]:
    if not rca["causes"]:
        st.success("No anomalies, so no root causes to analyse.")
    else:
        st.bar_chart(pd.DataFrame({"sessions": [c["count"] for c in rca["causes"]]},
                                  index=[c["title"] for c in rca["causes"]]), horizontal=True)
        for c in rca["causes"]:
            with st.expander(f"{SEV_ICON[c['severity']]} **{c['title']}** — {c['count']} session(s) · "
                             f"avg rule-score share {c['avg_share']:.0%}", expanded=c is rca["causes"][0]):
                st.write(c["explanation"])
                st.write(f"**Impact:** {c['impact']}")
                if c["key_events"]:
                    st.write("**Key events:** " + " · ".join(c["key_events"]))
                st.write("**Example blocks:** " + ", ".join(f"`{b}`" for b in c["example_blocks"]))
                st.code(c["example_sequence"], language=None)
        corr = rca["correlations"]
        st.subheader("Correlations across anomalies")
        if not any(corr.values()):
            st.caption("No DataNode, time-window or repeated-pattern correlations found.")
        if corr["nodes"]:
            st.write("**DataNodes concentrated in anomalous blocks**")
            st.dataframe(pd.DataFrame(corr["nodes"]), hide_index=True, width="stretch",
                         column_config={"anomaly_share": st.column_config.NumberColumn(format="percent")})
        if corr["time_bursts"]:
            st.write("**Bursts of anomalies**")
            st.dataframe(pd.DataFrame(corr["time_bursts"]), hide_index=True)
        if corr["patterns"]:
            st.write("**Identical anomalous patterns in multiple blocks**")
            st.dataframe(pd.DataFrame(corr["patterns"]), hide_index=True, width="stretch")

# ── AI Insights ───────────────────────────────────────────────────────
with tabs[2]:
    st.caption(f"Generated by **{insight['source']}**")
    st.subheader("Summary")
    st.write(insight["summary"])
    st.subheader("Explanation")
    st.write(insight["explanation"])
    if insight.get("normal_explanation"):
        st.subheader("Why the other sessions look normal")
        st.write(insight["normal_explanation"])
    st.subheader(f"Impact analysis {SEV_ICON.get(insight['impact']['severity'], '')} "
                 f"{insight['impact']['severity'].upper()}")
    st.write(insight["impact"]["description"])
    if insight["impact"].get("affected"):
        st.write("**Affected:** " + ", ".join(f"`{a}`" for a in insight["impact"]["affected"]))
    if insight.get("slm_problems"):
        with st.expander("Why the SLM answer was rejected"):
            st.write(insight["slm_problems"])
            st.code(insight.get("slm_raw", ""), language=None)
    if insight.get("facts"):
        with st.expander("Grounding facts sent to the SLM"):
            st.code(insight["facts"], language=None)

    st.divider()
    st.subheader("💬 Ask about this analysis")
    st.caption(f"Answered by `{config.INSIGHT_MODEL}` from the same facts as this report. Name a block such as "
               "`blk_…` to pull in its own evidence and similar historical sessions. A small model can still reason "
               "wrongly; anything the run cannot back up is flagged under the answer. Each answer takes ~15-35 s.")
    chat = st.session_state.get("chat")
    if chat is None or chat.ctx is not ctx:
        chat = st.session_state["chat"] = RunChat(ctx, coordinator.rca)
    # new messages go into the same box as the history, so the input always stays below the conversation
    conversation = st.container()
    with conversation:
        for turn in chat.history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["display"])
                for issue in turn["issues"]:
                    st.caption(f"⚠️ Not backed by the facts: {issue}")
    question = st.chat_input("Ask a follow-up, e.g. Why is the top root cause the most likely one?",
                             key="insight_chat")
    if question:
        with conversation, st.chat_message("user"):
            st.markdown(question)
        with conversation, st.chat_message("assistant"):
            msgs, content = chat.messages_for(question)
            try:
                if not resources.status()["insight_slm_loaded"]:
                    with st.spinner(f"Loading {config.INSIGHT_MODEL}…"):
                        resources.insight_slm()
                raw = st.write_stream(resources.insight_slm().stream_chat(msgs))
            except Exception as e:
                st.error(f"The SLM could not answer: {type(e).__name__}: {e}")
            else:
                _, issues = chat.finish(question, content, raw)
                for issue in issues:
                    st.caption(f"⚠️ Not backed by the facts: {issue}")

# ── Recommendations ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🚀 Next best actions")
    for i, a in enumerate(insight["next_best_actions"], 1):
        st.markdown(f"**{i}.** {a}")
    st.subheader("Recommendations")
    for r in insight["recommendations"]:
        st.markdown(f"- {r}")
    if rca["causes"]:
        st.subheader("Runbook by root cause")
        for c in rca["causes"]:
            with st.expander(c["title"]):
                for a in c["actions"]:
                    st.markdown(f"- {a}")

# ── Evidence ──────────────────────────────────────────────────────────
with tabs[4]:
    if rca["sessions"]:
        by_block = {p["block_id"]: p for p in rca["sessions"]}
        pick = st.selectbox("Anomalous session", list(by_block),
                            format_func=lambda b: f"{b} — {by_block[b]['hypotheses'][0]['title']}")
        p = by_block[pick]
        left, right = st.columns([3, 2])
        with left:
            st.write("**Event sequence**")
            st.code(p["sequence"], language=None)
            st.write("**Evidence**")
            for e in p["evidence"]:
                st.markdown(f"- {e}")
        with right:
            st.write("**Root-cause hypotheses**")
            st.dataframe(pd.DataFrame(p["hypotheses"])[["title", "share"]], hide_index=True,
                         width="stretch",
                         column_config={"share": st.column_config.ProgressColumn(
                             "rule-score share", min_value=0, max_value=1, format="percent",
                             help="Heuristic: each hypothesis' share of the rule scores for this session. "
                                  "Not a calibrated probability.")})
        if p.get("similar_logs"):
            st.write("**Similar historical sessions (FAISS / RAG)**")
            st.dataframe(pd.DataFrame([{
                "historical block": n.get("block_id"), "label": n.get("label"), "distance": n["distance"],
                "pattern seen": n.get("occurrences"), "anomaly rate": n.get("anomaly_rate"),
                "sequence": " ".join(n.get("sequence", []))} for n in p["similar_logs"]]),
                hide_index=True, width="stretch")
        timeline = d["medallion"]["silver"]
        timeline = timeline[timeline.block_id == pick]
        st.write("**Block timeline (Silver layer)**")
        st.dataframe(timeline.assign(description=timeline.event_id.map(lambda e: EVENT_INFO[e][2]))
                     .drop(columns=["block_id"]), hide_index=True, width="stretch")
    else:
        st.caption("No anomalous sessions to show evidence for.")

    st.subheader("Medallion layers")
    med = d["medallion"]
    layer = st.radio("Layer", ["🥉 Bronze — raw cleaned logs", "🥈 Silver — structured & enriched",
                               "🥇 Gold — aggregated & correlated"], horizontal=True)
    df = {"🥉": med["bronze"], "🥈": med["silver"], "🥇": med["gold"]}[layer[:1]]
    st.caption(f"{len(df):,} rows" + (f" · saved to `{med['path']}`" if med.get("path") else ""))
    st.dataframe(df.head(500).astype(str), hide_index=True, width="stretch")
    with st.expander("Preprocessing & cleaning details"):
        st.json(d["preprocessing"])
    with st.expander("HDFS event templates"):
        st.dataframe(pd.DataFrame([{"event": e, "template": TEMPLATE_TEXT[e], "stage": s, "kind": k, "meaning": m}
                                   for e, (s, k, m) in EVENT_INFO.items()]), hide_index=True, width="stretch")

# ── Reports ───────────────────────────────────────────────────────────
with tabs[5]:
    md = to_markdown(report)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Markdown report", md, file_name=f"hdfs_report_{ctx.run_id}.md",
                       mime="text/markdown", width="stretch")
    c2.download_button("⬇️ JSON report", json.dumps(report, indent=1, default=str),
                       file_name=f"hdfs_report_{ctx.run_id}.json", mime="application/json", width="stretch")
    st.markdown(md)

# ── Agent trace ───────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Workflow plan")
    st.write(" → ".join(d["plan"]))
    st.subheader("Execution trace")
    st.dataframe(pd.DataFrame([{"agent": t.agent, "task": t.task, "status": f"{STATUS_ICON.get(t.status, '')} {t.status}",
                                "attempts": t.attempts, "seconds": round(t.duration_s, 2), "summary": t.summary}
                               for t in ctx.trace]), hide_index=True, width="stretch")
    st.subheader("Inter-agent messages")
    st.markdown(agent_terminal(ctx.messages), unsafe_allow_html=True)
