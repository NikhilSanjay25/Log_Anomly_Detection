"""
LogVerse AI Platform — Desktop Product Application UI
=====================================================
Super-cool, aesthetic dark-glassmorphism desktop dashboard built with Streamlit.
Features:
  - File uploader supporting .xlsx, .xls, .csv, .txt, .log
  - Live K8s/Docker stream simulator
  - Live AI Agent Workflow Canvas & Inter-Agent Communication Timeline
  - Interactive Root Cause Analysis (RCA) Causal Graph (Vis.js)
  - Medallion Data Pipeline Inspector (Bronze -> Silver -> Gold + AI Catalog)
  - PyTorch Transformer Anomaly Detection metrics & progress indicators
  - Local SLM Human-Understandable Diagnosis & Operational SOP Runbook
  - Human-in-the-Loop Incident Ticket Manager (SLM Auto-draft + Human Edit)
  - One-Click PDF Report Export Engine
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import streamlit.components.v1 as components

from logverse_pipeline import generate_sample_hdfs_log
from logverse_ml import MLAnomalyDetector
from logverse_rca_graph import RCAGraphBuilder, EVENT_CAUSALITY_MAP
from logverse_agents import MultiAgentOrchestrator
from logverse_pdf import generate_incident_pdf

def parse_uploaded_log_file(uploaded_file):
    """Parses uploaded file bytes into a clean text string (.xlsx, .xls, .csv, .txt, .log)."""
    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        lines = []
        content_col = next((c for c in df.columns if "content" in c.lower() or "log" in c.lower() or "text" in c.lower()), None)
        if content_col:
            lines = df[content_col].dropna().astype(str).tolist()
        else:
            for _, row in df.iterrows():
                row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
                lines.append(row_str)
        return "\n".join(lines)
    elif fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        content_col = next((c for c in df.columns if "content" in c.lower() or "log" in c.lower() or "text" in c.lower()), None)
        if content_col:
            lines = df[content_col].dropna().astype(str).tolist()
        else:
            lines = []
            for _, row in df.iterrows():
                row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
                lines.append(row_str)
        return "\n".join(lines)
    else:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

st.set_page_config(
    page_title="LogVerse AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
    color: #f8fafc;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.brand-subtitle { font-size: 0.95rem; color: #94a3b8; margin-bottom: 20px; }
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}
.glass-card-anomaly { background: rgba(127, 29, 29, 0.3); border: 1px solid rgba(239, 68, 68, 0.4); }
.glass-card-normal { background: rgba(6, 78, 59, 0.3); border: 1px solid rgba(16, 185, 129, 0.4); }
.agent-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.badge-completed { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

.comm-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.comm-time { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b; }
.comm-agent { font-weight: 700; color: #38bdf8; font-size: 0.9rem; }
.comm-target { font-weight: 700; color: #c084fc; font-size: 0.9rem; }
.comm-action { display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #f472b6; background: rgba(244, 114, 182, 0.12); border: 1px solid rgba(244, 114, 182, 0.3); padding: 2px 8px; border-radius: 4px; margin-left: 8px; }
.comm-text { font-size: 0.9rem; color: #e2e8f0; margin-top: 6px; line-height: 1.5; }

.event-tag { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 3px; }
.tag-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
.tag-info { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }

.kpi-val { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
.kpi-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator()

if "results" not in st.session_state:
    sample_text = generate_sample_hdfs_log()
    st.session_state.results = st.session_state.orchestrator.run_agentic_pipeline(sample_text, "sample_hdfs_default.log")

if "submitted_tickets" not in st.session_state:
    st.session_state.submitted_tickets = []

with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/data-configuration.png", width=64)
    st.markdown("### ⚡ Control Panel")
    st.caption("Agentic AI Log Intelligence Platform")

    st.markdown("---")
    st.markdown("#### 1. Log Data Source")
    data_source_mode = st.radio(
        "Ingestion Mode",
        ["Default HDFS Sample", "Upload Custom Log File (.xlsx, .txt, .log, .csv)", "Simulate Live K8s/Docker Stream"],
        index=0
    )

    log_input_text = ""
    source_filename = "upload.log"

    if data_source_mode == "Default HDFS Sample":
        log_input_text = generate_sample_hdfs_log()
        source_filename = "sample_hdfs_default.log"
    elif data_source_mode.startswith("Upload Custom Log"):
        uploaded_file = st.file_uploader("Upload Log File (.xlsx, .xls, .csv, .txt, .log)", type=["xlsx", "xls", "csv", "txt", "log"])
        if uploaded_file is not None:
            try:
                log_input_text = parse_uploaded_log_file(uploaded_file)
                source_filename = uploaded_file.name
                st.success(f"Loaded '{uploaded_file.name}'!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                log_input_text = generate_sample_hdfs_log()
        else:
            log_input_text = generate_sample_hdfs_log()
    else:
        st.info("🌐 K8s / Docker Ingestion Adapter Active")
        log_input_text = """2026-10-04T10:15:00Z pod/payment-service-789456 info: Receiving block blk_-1608999687919862906
2026-10-04T10:15:01Z pod/payment-service-789456 warn: Got exception while serving blk_-1608999687919862906 to container db-node-1
2026-10-04T10:15:02Z pod/payment-service-789456 error: writeBlock blk_-1608999687919862906 received exception java.io.IOException: Connection reset by peer
2026-10-04T10:15:03Z pod/payment-service-789456 warn: PendingReplicationMonitor timed out block blk_-1608999687919862906"""
        source_filename = "k8s_docker_simulated.log"

    st.markdown("---")
    st.markdown("#### 2. Local SLM Reasoning Engine")
    slm_choice = st.selectbox(
        "SLM Model Backbone",
        ["Qwen-2.5-0.5B (Local / PyTorch)", "Phi-3 Mini (Local SLM)", "Gemini 2.5 Flash Fallback"]
    )

    if st.button("🚀 Run Agentic Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Invoking Multi-Agent System & Medallion Pipeline..."):
            res = st.session_state.orchestrator.run_agentic_pipeline(log_input_text, source_filename)
            st.session_state.results = res
            st.success("Pipeline Run Completed!")

    st.markdown("---")
    st.caption("LogVerse AI Platform v2.6 | Medallion Lakehouse | Multi-Agent Architecture")

res = st.session_state.results
ai_cat = res["ai_catalog"]
target_blk = res["target_block"]
target_ml = res["ml_results"].get(target_blk, {})
is_anom = target_ml.get("is_anomaly", False)
slm_res = res["slm_explanation"]
rem_res = res["remediation"]

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="brand-title">LOGVERSE AI PLATFORM</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Autonomous Multi-Agent Log Intelligence & Root Cause Analysis Platform</p>', unsafe_allow_html=True)

with col_status:
    status_class = "glass-card-anomaly" if is_anom else "glass-card-normal"
    status_text = "🚨 CRITICAL ANOMALY DETECTED" if is_anom else "✅ HEALTHY SYSTEM PATTERN"
    st.markdown(f"""
    <div class="glass-card {status_class}" style="text-align: center; padding: 12px;">
        <div style="font-weight: 700; font-size: 0.9rem;">{status_text}</div>
        <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 4px;">Target: {target_blk}</div>
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("bronze_records_count", 0)}</div><div class="kpi-lbl">Bronze Raw Lines</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("silver_records_count", 0)}</div><div class="kpi-lbl">Silver Parsed Events</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("gold_sessions_count", 0)}</div><div class="kpi-lbl">Gold Session Blocks</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="glass-card"><div class="kpi-val">{target_ml.get("anomaly_probability", 0.0):.1%}</div><div class="kpi-lbl">Transformer Prob</div></div>', unsafe_allow_html=True)
with k5: st.markdown(f'<div class="glass-card"><div class="kpi-val">6 / 6</div><div class="kpi-lbl">Agents Active</div></div>', unsafe_allow_html=True)

tab_agents, tab_rca, tab_medallion, tab_ml, tab_slm, tab_ticket, tab_export = st.tabs([
    "🤖 Agent Workflow Canvas",
    "🕸️ Causal RCA Graph",
    "🏅 Medallion Data Pipeline",
    "🧠 PyTorch ML Backbone",
    "💡 Local SLM Diagnosis & Runbook",
    "🎫 Human-in-the-Loop Ticket Raising",
    "📄 PDF Report Export"
])

with tab_agents:
    st.markdown("### 🤖 Multi-Agent Orchestration Visualizer")
    ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
    agents_list = [
        ("Planner Agent", "Decomposes query into steps", ac1),
        ("Catalog Agent", "Registers Medallion schemas", ac2),
        ("ML Anomaly Agent", "Evaluates Transformer model", ac3),
        ("RCA Graph Agent", "Builds causal network graph", ac4),
        ("SLM Diagnostic Agent", "Generates plain-English RCA", ac5),
        ("Remediation Agent", "Outputs automated SOP runbook", ac6)
    ]
    for name, desc, col in agents_list:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; height: 160px;">
                <span class="agent-badge badge-completed">COMPLETED</span>
                <div style="font-weight: 700; font-size: 0.95rem; margin-top: 4px; color: #38bdf8;">{name}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 💬 Real-Time Inter-Agent Communication Stream")
    comm_log = res["communication_log"]

    for msg in comm_log:
        if isinstance(msg, dict):
            sender = msg.get("sender", "Agent")
            recipient = msg.get("recipient", "System")
            action = msg.get("action", "MESSAGE")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
        else:
            sender, recipient, action, content, timestamp = "System", "Agent", "INFO", str(msg), ""

        border_color = "#e11d48" if "ANOMALY" in action or "CRITICAL" in content.upper() else "#38bdf8"
        
        st.markdown(f"""
        <div class="comm-card" style="border-left-color: {border_color};">
            <div>
                <span class="comm-time">[{timestamp}]</span>
                <span class="comm-agent">{sender}</span>
                <span style="color: #64748b; font-weight: bold;"> ➔ </span>
                <span class="comm-target">{recipient}</span>
                <span class="comm-action">{action}</span>
            </div>
            <div class="comm-text">{content}</div>
        </div>
        """, unsafe_allow_html=True)

with tab_rca:
    st.markdown("### 🕸️ Causal Root Cause Analysis (RCA) Network Graph")
    rca_col_graph, rca_col_info = st.columns([3, 1])
    with rca_col_graph:
        components.html(res["rca_html"], height=520, scrolling=False)
    with rca_col_info:
        root_causes = res["root_causes"]
        if root_causes:
            for rc in root_causes: st.error(f"**Primary Root Cause:**\n{rc}")
        else: st.success("No critical error detected.")
        st.code(target_blk)

with tab_medallion:
    m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["1. Bronze Layer (Raw)", "2. Silver Layer (Parsed)", "3. Gold Layer (AI Features)", "4. Enterprise AI Catalog"])
    with m_tab1:
        st.markdown("##### 🥉 Bronze Layer — Immutable Ingested Log Records")
        bronze_records = res["bronze_records"]
        st.dataframe(pd.DataFrame(bronze_records) if isinstance(bronze_records, list) else bronze_records, use_container_width=True)
    with m_tab2: st.dataframe(res["silver_df"], use_container_width=True)
    with m_tab3: st.dataframe(res["gold_df"], use_container_width=True)
    with m_tab4:
        cat_data = res["ai_catalog"]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Source File", cat_data.get("source", "N/A"))
        with c2: st.metric("Source Type", cat_data.get("source_type", "HDFS"))
        with c3: st.metric("Ingest Latency", f"{cat_data.get('processing_time_sec', 0.0):.4f} s")
        with c4: st.metric("Anomalous Blocks", cat_data.get("total_anomalous_blocks", 0))

        st.markdown("##### 🏷️ Discovered Log Event Templates")
        unique_events = cat_data.get("unique_events_found", [])
        tag_htmls = []
        for eid in unique_events:
            meta = EVENT_CAUSALITY_MAP.get(eid, {"name": "Operation", "severity": "Info"})
            css_class = "tag-danger" if meta["severity"] in ["High", "Critical"] else "tag-info"
            tag_htmls.append(f'<span class="event-tag {css_class}">{eid}: {meta["name"]}</span>')
        st.markdown("".join(tag_htmls), unsafe_allow_html=True)

with tab_ml:
    ml1, ml2 = st.columns(2)
    with ml1:
        st.markdown("##### Transformer Anomaly Inference Dashboard")
        prob_val = target_ml.get("anomaly_probability", 0.0)
        status_lbl = target_ml.get("status_label", "NORMAL PATTERN")
        if is_anom: st.error(f"### 🚨 {status_lbl}")
        else: st.success(f"### ✅ {status_lbl}")

        st.markdown("**Anomaly Probability Score:**")
        st.progress(prob_val)
        st.caption(f"Model Anomaly Probability: **{prob_val:.2%}** | Confidence Score: **{target_ml.get('confidence', 0.0):.2%}**")

        st.markdown("##### Suspicious Event Codes Detected")
        sus_list = target_ml.get("suspicious_events", [])
        if sus_list:
            tags = [f'<span class="event-tag tag-danger">⚠️ {s}: {EVENT_CAUSALITY_MAP.get(s, {}).get("name", "Error")}</span>' for s in sus_list]
            st.markdown("<br>".join(tags), unsafe_allow_html=True)
        else: st.write("Zero anomalous event codes detected.")
    with ml2:
        feat_vec = target_ml.get("feature_vector", [])
        if feat_vec: st.line_chart(pd.DataFrame({"Dimension": range(len(feat_vec)), "Embedding Value": feat_vec}).set_index("Dimension"))

with tab_slm:
    st.markdown(f"""
    <div class="glass-card glass-card-anomaly" style="padding: 20px;">
        <h4 style="color: #38bdf8; margin-top: 0;">Executive Summary</h4>
        <p style="font-size: 1.05rem; line-height: 1.6;">{slm_res['summary']}</p>
        <h5 style="color: #a78bfa; margin-top: 16px;">Error Mechanics</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['mechanism']}</p>
        <h5 style="color: #f472b6; margin-top: 16px;">Operational Impact</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['impact']}</p>
    </div>
    """, unsafe_allow_html=True)
    for step in rem_res["sop_steps"]: st.code(step, language="bash")

with tab_ticket:
    st.markdown("### 🎫 Human-in-the-Loop Incident Ticket Manager")
    default_title = f"[INCIDENT-{target_blk[-6:]}] Critical Error in HDFS Block {target_blk}"
    default_desc = f"SLM Diagnostic Summary:\n{slm_res['summary']}\n\nMechanism:\n{slm_res['mechanism']}\n\nImpact:\n{slm_res['impact']}"
    with st.form("ticket_form_desktop"):
        t_title = st.text_input("Ticket Title", value=default_title)
        t_priority = st.selectbox("Priority Level", ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"], index=0 if is_anom else 2)
        t_assignee = st.selectbox("Assignee Team", ["DevOps / Infrastructure", "Data Engineering", "Security Operations", "Database Reliability"])
        t_block = st.text_input("Target Resource ID", value=target_blk)
        t_desc = st.text_area("SLM-Drafted Problem Description (Editable)", value=default_desc, height=140)
        t_user_notes = st.text_area("Operator Custom Notes", value="Verified anomalous log sequence by human operator. Approved for immediate remediation.", height=70)
        submit_ticket = st.form_submit_button("🚀 Approve & Raise Incident Ticket", type="primary")
        if submit_ticket:
            ticket_payload = {
                "ticket_id": f"TCK-2026-{np.random.randint(1000, 9999)}",
                "title": t_title,
                "priority": t_priority,
                "assignee": t_assignee,
                "target_block": t_block,
                "description": t_desc,
                "user_notes": t_user_notes,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.submitted_tickets.append(ticket_payload)
            st.success(f"✅ Ticket {ticket_payload['ticket_id']} successfully registered!")

    if st.session_state.submitted_tickets:
        st.dataframe(pd.DataFrame(st.session_state.submitted_tickets), use_container_width=True)

with tab_export:
    st.markdown("### 📄 Export Comprehensive Incident PDF Report")
    latest_ticket = st.session_state.submitted_tickets[-1] if st.session_state.submitted_tickets else None
    pdf_bytes = generate_incident_pdf(target_block=target_blk, ml_res=target_ml, slm_res=slm_res, rem_res=rem_res, ticket_data=latest_ticket)
    st.download_button(label="📥 Download PDF Incident Report", data=pdf_bytes, file_name=f"LogVerse_Incident_Report_{target_blk}.pdf", mime="application/pdf", type="primary")
