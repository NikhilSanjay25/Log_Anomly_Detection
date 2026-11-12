"""
Agentic Log Anomaly Detection (HDFS · BGL · OpenStack) - Streamlit UI
---------------------------------------------------------------------
    streamlit run streamlit_app.py

Log Ingestion → Log Identification Agent (which dataset?) → Preprocessing & Cleaning Agent → Medallion
(Bronze/Silver/Gold) → Anomaly Detection Agent (Transformer → FAISS → RAG → Random Forest, one model set per
dataset) → Root Cause Analysis Agent → Insight Agent (knowledge base) → this UI, all orchestrated by the
Coordinator Agent. The Qwen language model is used only by the follow-up chat, which is fed the relevant raw logs.
"""
import html
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from log_agents import config
from log_agents.agents.coordinator import CoordinatorAgent
from log_agents.chat import RunChat, has_raw_logs, session_lines
from log_agents.context import RunMemory, RunOptions
from log_agents.datasets import PROFILES
from log_agents.ingestion import KubernetesSource, TextSource, UploadSource
from log_agents.report import to_markdown
from log_agents.resources import Resources

SAMPLES = Path(__file__).parent / "samples"
SEV_ICON = {"critical": "🟥", "high": "🔴", "medium": "🟠", "low": "🟢"}
STATUS_ICON = {"ok": "✅", "fallback": "🟡", "failed": "❌", "skipped": "⏭️", "running": "⏳"}

st.set_page_config(page_title="Log Anomaly Agents", page_icon="🔍", layout="wide")

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


ACTION_ICONS = {"read_logs": "📜", "similar_cases": "🔎", "event_evidence": "📊", "structural_check": "🧪",
                "node_correlation": "🖧", "repeat_pattern": "🔁", "rule_ranking": "📏", "text_match": "🔤", "decide": "✅"}


def _md(text):
    """Cause titles contain template wildcards like <*>; keep Markdown from reading them as emphasis."""
    return text.replace("*", "∗")


class LiveAgentView:
    """Shows what the investigator is doing while it works, inside the progress box: the model loading, the case it is on,
    every tool it picks (with the score it gave each option) and what the tool returned, the final candidates with the
    model's scores, and a running list of the finished cases. `pace` pauses after each step so an audience can follow."""

    def __init__(self, pace=0.0):
        self.pace, self.box, self.finished = pace, None, []
        self.case, self.steps, self.decision, self.choosing = None, [], None, None
        self.total, self.calls_done, self.t0 = 1, 0, time.time()

    def __call__(self, kind, d):
        if self.box is None:
            self.box = st.container(border=True)
            with self.box:
                st.markdown("**🕵️ Investigator Agent, live**")
                self.head, self.bar, self.now, self.done = st.empty(), st.empty(), st.empty(), st.empty()
        handler = getattr(self, "_" + kind, None)
        if handler:
            handler(d)
            if self.pace and kind in ("step", "decision", "case_done"):
                time.sleep(self.pace)

    def _progress(self, done):
        calls = self.calls_done + len(self.steps)
        self.bar.progress(min(done / max(self.total, 1), 1.0),
                          text=f"Case {min(done + 1, self.total)} of {self.total} · {calls} model calls · {time.time() - self.t0:.0f}s")

    def _model_loading(self, d):
        self.head.info(f"🧠 Loading **{d['model']}** into GPU memory. The first run takes a while (about a minute for the 8B model).")

    def _model_loaded(self, d):
        self.head.success(f"🧠 **{d['model']}** is loaded ({d['seconds']:.0f}s)")

    def _model_failed(self, d):
        self.head.error(f"Could not load {d['model']}: {d['reason']}. Using the rule-based analysis instead.")

    def _investigation_start(self, d):
        self.total = d["cases"]
        self.head.success(f"🧠 **{d['model']}** is investigating {d['cases']} case(s): {d['anomalies']} anomalies in {d['groups']} groups")
        self.bar.progress(0.0, text="Starting…")

    def _case_start(self, d):
        self.case, self.steps, self.decision, self.choosing = d, [], None, None
        self._progress(d["index"])
        self._render()

    def _step(self, d):
        self.steps.append(d)
        self._progress(self.case["index"])
        self._render()

    def _deciding(self, d):
        self.choosing = d
        self._render()

    def _decision(self, d):
        self.decision = d
        self._render()

    def _case_done(self, d):
        self.calls_done += d["model_calls"]
        self.steps = []
        note = "" if d["agrees"] else f" (the rule engine said: {_md(d['rule_top_title'])})"
        self.finished.insert(0, f"{'✅' if d['agrees'] else '↔️'} `{d['block_id']}` → **{_md(d['title'])}**{note} · {d['seconds']:.0f}s")
        self.done.markdown("**Finished cases** (↔️ = a different cause than the rule engine's)\n\n"
                           + "\n".join("- " + line for line in self.finished[:12]))
        self._progress(d["index"] + 1)

    def _investigation_done(self, d):
        self.bar.progress(1.0, text=f"Done: {d['cases']} case(s), {d['model_calls']} model calls, {d['seconds']:.0f}s")
        self.now.empty()

    def _render(self):
        c = self.case
        with self.now.container():
            st.markdown(f"**Now investigating {c['noun']} `{c['block_id']}`** ({c['group_size']} identical anomalies; "
                        f"the detector gave it {c['probability']:.0%})")
            for i, s in enumerate(self.steps, 1):
                ranked = sorted(s["options"].items(), key=lambda kv: -kv[1])
                latest = i == len(self.steps) and not self.choosing
                with st.expander(f"{i}. {ACTION_ICONS.get(s['action'], '•')} {s['action']}: picked with {ranked[0][1]:.0%}",
                                 expanded=latest):
                    st.caption("The model's scores for the options it was offered: "
                               + ", ".join(f"{k} {v:.0%}" for k, v in ranked[:5]))
                    if s["observation"]:
                        st.code(s["observation"], language=None)
            if self.choosing and not self.decision:
                st.markdown("🤔 Enough evidence. **Choosing the root cause** from: " + "; ".join(_md(t) for t in self.choosing["candidates"]))
            if self.decision:
                st.markdown("**Final choice** (the model scored every candidate; the order it saw was shuffled)")
                st.dataframe(pd.DataFrame([{"root cause": x["title"], "model score": x["score"],
                                            "rule engine's top": "✓" if x["rule_top"] else "",
                                            "chosen": "✅" if x["chosen"] else ""} for x in self.decision["candidates"]]),
                             hide_index=True, width="stretch",
                             column_config={"model score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})


@st.cache_resource(show_spinner="Starting…")
def get_resources():
    return Resources()      # a dataset's model set is loaded the first time that dataset is analysed



resources = get_resources()
coordinator = CoordinatorAgent(resources)

# ═══════════════════════════════ SIDEBAR ═══════════════════════════════
with st.sidebar:
    st.header("📥 Log Ingestion")
    source_kind = st.radio("Source", ["Upload file", "Paste logs", "Sample logs", "Kubernetes (real-time)"],
                           label_visibility="collapsed")
    batch_source = None
    if source_kind == "Upload file":
        up = st.file_uploader("Log file", type=["log", "txt", "csv"],
                              help="Raw HDFS / BGL / OpenStack log lines, an HDFS Event_traces.csv, or HDFS "
                                   "event-ID sequences (one per line)")
        if up:
            batch_source = UploadSource(up.name, up.getvalue())
    elif source_kind == "Paste logs":
        txt = st.text_area("Raw log lines (HDFS, BGL, OpenStack) or HDFS event sequences", height=160,
                           placeholder="E22 E5 E5 E5 E11 E9 E11 E9 E26 E26\n"
                                       "or\n081109 203615 148 INFO dfs.DataNode$PacketResponder: …")
        if txt.strip():
            batch_source = TextSource(txt)
    elif source_kind == "Sample logs":
        samples = sorted(p.name for p in SAMPLES.glob("*") if p.is_file() and p.suffix in (".log", ".txt", ".csv")
                         ) if SAMPLES.exists() else []
        if samples:
            pick = st.selectbox("Sample", samples)
            batch_source = UploadSource(pick, (SAMPLES / pick).read_bytes())
        else:
            st.info("Run `python scripts/make_sample_logs.py` and `python scripts/make_dataset_samples.py`.")
    else:
        st.info("**Coming soon.** " + KubernetesSource.__doc__.split("\n\n")[1].strip().replace("\n", " "))
        st.text_input("Namespace", "default", disabled=True)
        st.text_input("Pod", "hdfs-datanode-0", disabled=True)

    st.header("⚙️ Agent settings")
    dataset_choice = st.selectbox("Dataset", ["Auto-detect"] + [p.title for p in PROFILES.values()],
                                  help="Auto-detect lets the Log Identification Agent decide from the log lines.")
    forced = next((n for n, p in PROFILES.items() if p.title == dataset_choice), None)
    rs = resources.status()
    MODEL_LABELS = {config.RULES: "Rule-based (no language model)", "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
                    "Qwen/Qwen3-4B": "Qwen3-4B (4-bit)", "Qwen/Qwen3-8B": "Qwen3-8B (4-bit)"}
    model_options = [config.RULES] + config.LLM_CHOICES
    agent_model = st.selectbox(
        "Investigator", model_options,
        index=model_options.index(config.AGENT_MODEL) if config.AGENT_MODEL in model_options else 1,
        format_func=lambda m: MODEL_LABELS[m] + ("" if m == config.RULES or rs["models_available"][m]
                                                 else " (not downloaded)"),
        help="The language model that investigates each anomaly: it picks the next evidence step from a menu and then "
             "the root cause from a shortlist. Rule-based = the old rule engine only. Models that are not downloaded "
             "fall back to the rules (python scripts/download_models.py).")
    threshold = st.slider("Anomaly threshold (RF probability)", 0.1, 0.9, 0.5, 0.05)
    persist = st.toggle("Save medallion layers + run memory", value=True)
    pace = st.slider("Slow down the live view (demo)", 0.0, 3.0, 0.0, 0.25,
                     help="Seconds to pause after each step of the investigation, so an audience can follow what the model does.")

    st.header("🖥️ System status")
    for name, prof in PROFILES.items():
        mark = "✅ ready" if rs["available"][name] else "❌ not trained"
        loaded = " · loaded" if name in rs["pipelines_loaded"] else ""
        st.caption(f"**{prof.title}** models {mark}{loaded}")
    st.caption("Language model: " + (f"`{rs['llm_loaded']}` loaded" if rs["llm_loaded"] else "loads when first needed"))
    for name in rs["pipelines_loaded"]:
        ps = resources.pipeline(name).status()
        st.caption(f"{PROFILES[name].title}: device **{ps['device']}** · FAISS **{ps['faiss_vectors']:,}** vectors · "
                   f"RF **{ps['rf_estimators']}** trees")


    recent = RunMemory().recent(5)
    if recent:
        st.header("🧠 Run memory")
        for r in recent:
            st.caption(f"{r['generated_at'][5:16]} · {r['source']} · {r.get('anomalies', '–')}/{r.get('sessions', '–')} anomalous")

# ═══════════════════════════════ HEADER ════════════════════════════════
PIPELINE = [("Ingestion", ""), ("Identification: HDFS · BGL · OpenStack", ""), ("Preprocessing", ""),
            ("Medallion", ""), ("Transformer · FAISS · RAG · RF", "ml"),
            ("Investigator (Qwen picks the steps)", "slm"), ("Insight", ""), ("Chat (Qwen)", "slm")]
st.markdown('<div class="hero-title">Agentic Log Anomaly Detection</div>'
            '<p class="hero-sub">A Coordinator Agent runs specialised agents end to end on HDFS, BGL and OpenStack '
            'logs, from raw lines to root causes, recommendations and a log-aware chat.</p><div class="pipeline">'
            + '<span class="arrow">→</span>'.join(f'<span class="chip {cls}">{name}</span>' for name, cls in PIPELINE)
            + "</div>", unsafe_allow_html=True)

run_clicked = st.button("▶️ Run analysis", type="primary", width="stretch", disabled=batch_source is None)
# A fixed slot for the progress box: without it the result tabs below move up one position on the next rerun
# (e.g. when a chat question is sent), Streamlit treats them as new tabs and jumps back to the first one.
run_area = st.container()
if run_clicked:
    batch = batch_source.read()
    opts = RunOptions(threshold=threshold, persist=persist, dataset=forced, agent_model=agent_model)
    with run_area, st.status("Coordinator Agent is running the workflow…", expanded=True) as status:
        def on_step(step):
            st.write(f"{STATUS_ICON.get(step.status, '•')} **{step.agent}** ({step.duration_s:.1f}s) — {step.summary}")
            for issue in step.issues:
                st.caption(f"↳ {issue}")
        ctx = coordinator.run(batch, opts, on_step=on_step, on_event=LiveAgentView(pace))
        ok = ctx.data["status"] == "completed"
        status.update(label="Workflow completed" if ok else "Workflow stopped",
                      state="complete" if ok else "error", expanded=not ok)
    st.session_state["ctx"] = ctx

ctx = st.session_state.get("ctx")
if ctx is None:
    st.info("Choose a log source in the sidebar and press **Run analysis**. "
            "Try *Sample logs → hdfs_raw_sample.log* for a full demo.")
    AGENTS = [
        ("Log Identification Agent", "Recognises HDFS, BGL or OpenStack logs (and the input format) and picks the workflow."),
        ("Preprocessing & Cleaning Agent", "Parses, normalises, de-duplicates and removes noise."),
        ("Medallion Processing", "Bronze raw cleaned logs → Silver structured events → Gold sessions (blocks, windows, instances)."),
        ("Anomaly Detection Agent", "Transformer → FAISS → RAG → Random Forest, trained separately for each dataset."),
        ("Root Cause Analysis Agent", "Ranks causes from error events, timings, similar logs, context and correlations."),
        ("Insight Agent", "Explanation, impact and next-best actions from the knowledge base; Qwen answers follow-up questions in the chat."),
    ]
    st.markdown('<div class="agent-grid">' + "".join(
        f'<div class="agent-card"><div class="step">0{i}</div><div class="name">{n}</div><div class="desc">{d}</div></div>'
        for i, (n, d) in enumerate(AGENTS, 1)) + "</div>", unsafe_allow_html=True)
    st.stop()

if ctx.data["status"] != "completed":
    st.error(ctx.data.get("error", "The workflow failed."))
    ident = ctx.data.get("identification") or {}
    if ident.get("family") == "unsupported":
        st.caption("Supported inputs: raw HDFS lines · HDFS Event_traces.csv · HDFS event sequences like `E5 E22 E11` · raw BGL RAS lines · raw OpenStack nova lines")
    st.stop()

d = ctx.data
report = d["report"]
sessions = d["detection"]["sessions"]
rca = d["rca"]
insight = d["insight"]
n_anom = int(sessions.is_anomaly.sum())
profile = ctx.profile
noun, noun_pl = profile.session_noun, profile.session_noun_plural

for w in ctx.warnings:
    st.warning(w)

tabs = st.tabs(["📊 Anomaly Summary", "🕵️ Investigation", "🧭 Root Cause Analysis", "🤖 AI Insights",
                "🛠️ Recommendations", "🔬 Evidence", "📄 Reports", "🕸️ Agent Trace"])

# ── Anomaly Summary ───────────────────────────────────────────────────
with tabs[0]:
    ident = d["identification"]
    badge = (f'<span class="badge-anomaly">● {n_anom:,} anomalous session(s) detected</span>' if n_anom
             else '<span class="badge-healthy">● Healthy: no anomalies detected</span>')
    st.markdown(badge, unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric(f"{noun_pl.capitalize()} analysed", f"{len(sessions):,}")
    c[1].metric("Anomalies", f"{n_anom:,}", f"{n_anom / len(sessions):.1%}", delta_color="inverse")
    c[2].metric("Flagged uncertain", f"{int(sessions.uncertain.sum()):,}",
                help="RF probability near the threshold, or RF disagrees with similar historical sessions. "
                     "The label is not changed; review these first.")
    c[3].metric("Root-cause groups", len(rca["causes"]))
    c[4].metric("Severity", insight["impact"]["severity"].upper())
    st.caption(f"Log family **{ident['family']}** · format `{ident['format']}` · confidence {ident['confidence']:.0%} · "
               f"workflow: {ident['workflow']}")
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
            noun.capitalize(): view.block_id, "Probable cause": view.block_id.map(prim),
            "Anomaly prob.": view.anomaly_proba, "Events": view.n_events,
            "Error events": view.error_events.map(", ".join),
            **{c: view[c] for c in profile.extra_features if c in view},
            "Uncertain": view.uncertain.map({True: "⚠️ review", False: ""}),
            "Sequence": view.sequence}),
            hide_index=True, width="stretch",
            column_config={"Anomaly prob.": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ── Investigation ─────────────────────────────────────────────────────
with tabs[1]:
    inv = d.get("investigation") or {}
    if inv.get("status") == "unavailable":
        st.warning(f"The language model could not be used ({inv.get('reason')}). The rule-based analysis was used instead.")
    elif inv.get("status") != "agent":
        st.info("Rule-based analysis: no language model was used. Choose a Qwen model under **Investigator** in the "
                "sidebar to have it investigate each anomaly.")
    else:
        st.caption(f"**{inv['model_label']}** investigates each group of identical anomalies. Every turn it picks ONE "
                   "action from a menu of tools (the tools are ordinary code); when it has looked enough it picks the "
                   "root cause from a shortlist. It does not decide what is anomalous, and it cannot invent a cause.")
        c = st.columns(5)
        c[0].metric("Cases investigated", f"{inv['cases']:,}", help=f"{inv['anomalies']} anomalies in {inv['groups']} groups")
        c[1].metric("Model calls", f"{inv['model_calls']:,}")
        c[2].metric("Evidence steps / case", f"{inv['mean_steps']:.1f}")
        c[3].metric("Asked the rule engine", f"{inv['rule_ranking_used']}/{inv['cases']}")
        c[4].metric("Same cause as rules", f"{inv['agrees_with_rule_top']}/{inv['cases']}",
                    help="How often the model chose the rule engine's own top cause")
        counts = pd.Series([s_["action"] for ep in inv["episodes"] for s_ in ep["steps"]]).value_counts()
        if len(counts):
            st.write("**Tools the model chose to use**")
            st.bar_chart(counts, horizontal=True)
        st.write("**Cases**")
        kb = profile.root_causes
        for ep in inv["episodes"][:40]:
            title = kb[ep["choice"]]["title"]
            with st.expander(f"`{ep['block_id']}` ({ep['group_size']} identical) → {title}"):
                for i, step in enumerate(ep["steps"], 1):
                    top = max(step["probs"], key=step["probs"].get)
                    st.markdown(f"**Step {i}: `{step['action']}`** (the model gave it {step['probs'][top]:.0%})")
                    if step.get("options"):
                        st.caption("Scores for the options it was offered: " + ", ".join(
                            f"{k} {v:.0%}" for k, v in sorted(step["options"].items(), key=lambda kv: -kv[1])[:5]))
                    if step["observation"]:
                        st.code(step["observation"], language=None)
                st.markdown("**Final choice among the shortlist** (the order shown to the model was shuffled)")
                st.dataframe(pd.DataFrame([{"cause": kb[k]["title"], "model score": ep["choice_probs"].get(k, 0.0),
                                            "rule engine's top": k == ep["rule_top"], "chosen": k == ep["choice"]}
                                           for k in ep["candidates"]]), hide_index=True, width="stretch")
                st.caption(f"{ep['model_calls']} model calls in {ep['seconds']:.1f}s · decision method: {ep['decision']}")
        if inv["cases"] > 40:
            st.caption(f"Showing 40 of {inv['cases']} cases.")

# ── Root Cause Analysis ───────────────────────────────────────────────
with tabs[2]:
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
                st.write(f"**Example {noun_pl}:** " + ", ".join(f"`{b}`" for b in c["example_blocks"]))
                st.code(c["example_sequence"], language=None)
        corr = rca["correlations"]
        st.subheader("Correlations across anomalies")
        if not any(corr.values()):
            st.caption(f"No {profile.node_noun}, time-window or repeated-pattern correlations found.")
        if corr["nodes"]:
            st.write(f"**{profile.node_noun}s concentrated in anomalous {noun_pl}**")
            st.dataframe(pd.DataFrame(corr["nodes"]), hide_index=True, width="stretch",
                         column_config={"anomaly_share": st.column_config.NumberColumn(format="percent")})
        if corr["time_bursts"]:
            st.write("**Bursts of anomalies**")
            st.dataframe(pd.DataFrame(corr["time_bursts"]), hide_index=True)
        if corr["patterns"]:
            st.write(f"**Identical anomalous patterns in multiple {noun_pl}**")
            st.dataframe(pd.DataFrame(corr["patterns"]), hide_index=True, width="stretch")

# ── AI Insights ───────────────────────────────────────────────────────
with tabs[3]:
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
    if insight.get("facts"):
        with st.expander("Facts the chat model is grounded on"):
            st.code(insight["facts"], language=None)

    st.divider()
    st.subheader("💬 Ask about this analysis")
    st.caption(f"Answered by `{ctx.options.agent_model if ctx.options.agent_model != config.RULES else config.CHAT_MODEL}` from the same facts as this report plus the raw log lines "
               f"relevant to your question. Name a {noun} id to pull in its own logs, evidence and similar historical "
               "sessions, or mention an event id (E12) or a keyword to see the matching log lines. A small model can "
               "still reason wrongly; anything the run cannot back up is flagged under the answer. Each answer takes "
               "~15-35 s.")
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
                chat_model = ctx.options.agent_model if ctx.options.agent_model != config.RULES else config.CHAT_MODEL
                if resources.status()["llm_loaded"] != chat_model:
                    with st.spinner(f"Loading {chat_model}…"):
                        resources.llm(chat_model)
                raw = st.write_stream(resources.llm(chat_model).stream_chat(msgs))
            except Exception as e:
                st.error(f"The chat model could not answer: {type(e).__name__}: {e}")
            else:
                _, issues = chat.finish(question, content, raw)
                for issue in issues:
                    st.caption(f"⚠️ Not backed by the facts: {issue}")

# ── Recommendations ───────────────────────────────────────────────────
with tabs[4]:
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
with tabs[5]:
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
                f"historical {noun}": n.get("block_id"), "label": n.get("label"), "distance": n["distance"],
                "pattern seen": n.get("occurrences"), "anomaly rate": n.get("anomaly_rate"),
                "sequence": " ".join(n.get("sequence", []))} for n in p["similar_logs"]]),
                hide_index=True, width="stretch")
        timeline = d["medallion"]["silver"]
        timeline = timeline[timeline.block_id == pick]
        st.write(f"**{noun.capitalize()} timeline (Silver layer)**")
        st.dataframe(timeline.assign(description=timeline.event_id.map(profile.describe))
                     .drop(columns=["block_id"]), hide_index=True, width="stretch")
        if has_raw_logs(ctx):
            lines, total = session_lines(ctx, pick, limit=200)
            st.write(f"**Raw log lines of this {noun}** ({len(lines)} of {total} shown, in file order)")
            st.code("\n".join(lines), language=None)
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
    with st.expander(f"{profile.title} event templates"):
        st.dataframe(pd.DataFrame([{"event": e, "template": profile.template_text[e], "stage": st_, "kind": k,
                                    "meaning": m} for e, (st_, k, m) in profile.event_info.items()]),
                     hide_index=True, width="stretch")

# ── Reports ───────────────────────────────────────────────────────────
with tabs[6]:
    md = to_markdown(report)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Markdown report", md, file_name=f"{profile.name}_report_{ctx.run_id}.md",
                       mime="text/markdown", width="stretch")
    c2.download_button("⬇️ JSON report", json.dumps(report, indent=1, default=str),
                       file_name=f"{profile.name}_report_{ctx.run_id}.json", mime="application/json", width="stretch")
    st.markdown(md)

# ── Agent trace ───────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Workflow plan")
    st.write(" → ".join(d["plan"]))
    st.subheader("Execution trace")
    st.dataframe(pd.DataFrame([{"agent": t.agent, "task": t.task, "status": f"{STATUS_ICON.get(t.status, '')} {t.status}",
                                "attempts": t.attempts, "seconds": round(t.duration_s, 2), "summary": t.summary}
                               for t in ctx.trace]), hide_index=True, width="stretch")
    st.subheader("Inter-agent messages")
    st.markdown(agent_terminal(ctx.messages), unsafe_allow_html=True)
