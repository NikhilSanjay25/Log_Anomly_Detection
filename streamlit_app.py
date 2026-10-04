"""
LogVerse AI Platform — Enterprise Log Intelligence Product
===========================================================
An Enterprise-Grade Agentic AI Data Platform powered by Medallion Architecture,
Multi-Agent Orchestration, PyTorch Deep Learning Transformer Backbone,
Interactive Root Cause Analysis (RCA) Causal Network Graphs, Local SLMs,
Human-in-the-Loop Ticket Raising, One-Click PDF Exports, and Multi-Format Log Ingestion (.xlsx, .xls, .csv, .txt, .log).

Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components

# Import modular pipeline engines & PDF generator
from logverse_pipeline import MedallionPipeline, generate_sample_hdfs_log
from logverse_ml import MLAnomalyDetector
from logverse_rca_graph import RCAGraphBuilder, EVENT_CAUSALITY_MAP
from logverse_agents import MultiAgentOrchestrator
from logverse_pdf import generate_incident_pdf

# ─────────────────────────────────────────────────────────────
# HELPER: FILE PARSER FOR MULTI-FORMAT UPLOADS (.xlsx, .xls, .csv, .txt, .log)
# ─────────────────────────────────────────────────────────────
def parse_uploaded_log_file(uploaded_file):
    """
    Parses uploaded file bytes into a clean text string.
    Supports .xlsx, .xls, .csv, .txt, .log formats.
    """
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
        
    else:  # .txt, .log
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────────────────────
# 1. STREAMLIT PAGE CONFIG & ENTERPRISE CYBERPUNK-GLASS CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogVerse AI — Enterprise Log Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ENTERPRISE_CSS = """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global Reset & Dark Theme */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #090d16 50%, #020617 100%);
    color: #f8fafc;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Header & Brand Banner */
.brand-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 12px 32px 0 rgba(0, 0, 0, 0.4);
}

.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 40%, #c084fc 80%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-subtitle {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Status Pill Indicators */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.pill-anomaly {
    background: rgba(225, 29, 72, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
}

.pill-healthy {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}

/* Glass Panels */
.glass-panel {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.glass-panel-alert {
    background: rgba(136, 19, 55, 0.3);
    border: 1px solid rgba(244, 63, 94, 0.4);
}

/* Agent Workflow Cards */
.agent-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.agent-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
}

.agent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-completed { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

/* Communication Stream Styling */
.comm-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.comm-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
}

.comm-agent {
    font-weight: 700;
    color: #38bdf8;
    font-size: 0.9rem;
}

.comm-target {
    font-weight: 700;
    color: #c084fc;
    font-size: 0.9rem;
}

.comm-action {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #f472b6;
    background: rgba(244, 114, 182, 0.12);
    border: 1px solid rgba(244, 114, 182, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
}

.comm-text {
    font-size: 0.9rem;
    color: #e2e8f0;
    margin-top: 6px;
    line-height: 1.5;
}

/* Event Tag Badges */
.event-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}

.tag-danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
.tag-info { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }

/* KPI Metrics */
.kpi-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
}

.kpi-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 4px;
}
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2. STATE MANAGEMENT & ORCHESTRATOR INITIALIZATION
# ─────────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator()

if "results" not in st.session_state:
    sample_text = generate_sample_hdfs_log()
    st.session_state.results = st.session_state.orchestrator.run_agentic_pipeline(sample_text, "sample_hdfs_default.log")

if "submitted_tickets" not in st.session_state:
    st.session_state.submitted_tickets = []

# ─────────────────────────────────────────────────────────────
# 3. ENTERPRISE SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Control Center")
    st.caption("LogVerse AI Platform v2.6 Enterprise Edition")

    st.markdown("---")
    st.markdown("#### 📥 1. Ingestion Adapter")
    ingest_mode = st.radio(
        "Source Type",
        ["Sample HDFS Logs", "Upload Custom Log File (.xlsx, .txt, .log, .csv)", "Live K8s / Docker Stream"],
        index=0
    )

    log_content_input = ""
    source_name = "log_source.log"

    if ingest_mode == "Sample HDFS Logs":
        log_content_input = generate_sample_hdfs_log()
        source_name = "sample_hdfs_benchmark.log"
    elif ingest_mode.startswith("Upload Custom Log"):
        up_file = st.file_uploader("Upload Log File (.xlsx, .xls, .csv, .txt, .log)", type=["xlsx", "xls", "csv", "txt", "log"])
        if up_file is not None:
            try:
                log_content_input = parse_uploaded_log_file(up_file)
                source_name = up_file.name
                st.success(f"Successfully loaded '{up_file.name}'!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                log_content_input = generate_sample_hdfs_log()
        else:
            log_content_input = generate_sample_hdfs_log()
    else:  # Kubernetes / Docker Stream Simulation
        st.info("🌐 K8s Pod Log Stream Listener (`kubectl logs -f`)")
        log_content_input = """2026-10-04T10:15:00Z pod/payment-service-789456 info: Receiving block blk_-1608999687919862906
2026-10-04T10:15:01Z pod/payment-service-789456 warn: Got exception while serving blk_-1608999687919862906 to container db-node-1
2026-10-04T10:15:02Z pod/payment-service-789456 error: writeBlock blk_-1608999687919862906 received exception java.io.IOException: Connection reset by peer
2026-10-04T10:15:03Z pod/payment-service-789456 warn: PendingReplicationMonitor timed out block blk_-1608999687919862906"""
        source_name = "k8s_payment_service_stream.log"

    st.markdown("---")
    st.markdown("#### 🧠 2. AI Reasoning Backbone")
    slm_model = st.selectbox(
        "Local SLM Architecture",
        ["Qwen-2.5-0.5B (Local / PyTorch)", "Phi-3 Mini (Local SLM)", "Gemini 2.5 Flash Fallback"]
    )

    st.markdown("---")
    if st.button("🚀 Run Agentic Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Processing Medallion Pipeline & Multi-Agent System..."):
            results = st.session_state.orchestrator.run_agentic_pipeline(log_content_input, source_name)
            st.session_state.results = results
            st.success("Pipeline Run Completed!")

    st.markdown("---")
    device_name = "CUDA GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU Execution"
    st.caption(f"Hardware Compute: {device_name}")

# ─────────────────────────────────────────────────────────────
# 4. BRAND HEADER & TOP METRICS BANNER
# ─────────────────────────────────────────────────────────────
results = st.session_state.results
ai_catalog = results["ai_catalog"]
target_blk = results["target_block"]
target_ml = results["ml_results"].get(target_blk, {})
is_anom = target_ml.get("is_anomaly", False)
slm_res = results["slm_explanation"]
rem_res = results["remediation"]

st.markdown(f"""
<div class="brand-header">
    <div>
        <p class="brand-title">LOGVERSE AI PLATFORM</p>
        <p class="brand-subtitle">Autonomous Multi-Agent Log Intelligence, Medallion Lakehouse & Causal RCA Platform</p>
    </div>
    <div>
        <span class="status-pill {'pill-anomaly' if is_anom else 'pill-healthy'}">
            {'🚨 CRITICAL ANOMALY DETECTED' if is_anom else '✅ HEALTHY SYSTEM PATTERN'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="glass-panel kpi-container"><div class="kpi-title">Bronze Raw Lines</div><div class="kpi-value">{ai_catalog.get("bronze_records_count", 0)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="glass-panel kpi-container"><div class="kpi-title">Silver Parsed Events</div><div class="kpi-value">{ai_catalog.get("silver_records_count", 0)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="glass-panel kpi-container"><div class="kpi-title">Gold Session Blocks</div><div class="kpi-value">{ai_catalog.get("gold_sessions_count", 0)}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="glass-panel kpi-container"><div class="kpi-title">Transformer Prob</div><div class="kpi-value">{target_ml.get("anomaly_probability", 0.0):.1%}</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="glass-panel kpi-container"><div class="kpi-title">Active AI Agents</div><div class="kpi-value">6 / 6</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5. MAIN TABBED ENTERPRISE WORKBENCH
# ─────────────────────────────────────────────────────────────
tab_canvas, tab_rca, tab_medallion, tab_ml, tab_slm, tab_ticket, tab_export = st.tabs([
    "🤖 Agent Workflow Canvas",
    "🕸️ Causal RCA Graph",
    "🏅 Medallion Data Pipeline",
    "🧠 PyTorch ML Backbone",
    "💡 Local SLM Diagnosis & SOP",
    "🎫 Human-in-the-Loop Ticket Raising",
    "📄 PDF Report Export"
])

# -------------------------------------------------------------
# TAB 1: AGENT WORKFLOW CANVAS & COMMUNICATION CONSOLE
# -------------------------------------------------------------
with tab_canvas:
    st.markdown("### 🤖 Multi-Agent Orchestration Canvas")
    st.caption("Real-time visual display showing agent states, subtask decomposition, and inter-agent communication messages.")

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    agents_info = [
        ("Planner Agent", "Formulates task plan", a1),
        ("Catalog Agent", "Indexes dataset metadata", a2),
        ("ML Anomaly Agent", "Evaluates Transformer model", a3),
        ("RCA Graph Agent", "Builds causal failure graph", a4),
        ("SLM Diagnostic Agent", "Generates plain-English root cause", a5),
        ("Remediation Agent", "Outputs automated SOP runbooks", a6)
    ]

    for name, desc, col in agents_info:
        with col:
            st.markdown(f"""
            <div class="agent-card">
                <span class="agent-badge badge-completed">COMPLETED</span>
                <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem; margin-top: 6px;">{name}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 💬 Real-Time Inter-Agent Communication Stream")
    comm_log = results["communication_log"]

    for msg in comm_log:
        if isinstance(msg, dict):
            sender = msg.get("sender", "Agent")
            recipient = msg.get("recipient", "System")
            action = msg.get("action", "MESSAGE")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
        else:
            sender = "System"
            recipient = "Agent"
            action = "INFO"
            content = str(msg)
            timestamp = ""

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

# -------------------------------------------------------------
# TAB 2: CAUSAL ROOT CAUSE ANALYSIS (RCA) GRAPH
# -------------------------------------------------------------
with tab_rca:
    st.markdown("### 🕸️ Causal Root Cause Analysis (RCA) Network Graph")
    st.caption("Interactive causal propagation tree connecting cluster components to error triggers and the primary root cause node.")

    c_graph, c_info = st.columns([3, 1])

    with c_graph:
        components.html(results["rca_html"], height=520, scrolling=False)

    with c_info:
        st.markdown("##### 📌 RCA Summary")
        r_causes = results["root_causes"]
        if r_causes:
            for rc in r_causes:
                st.error(f"**Root Cause Identified:**\n{rc}")
        else:
            st.success("No critical failure trigger detected.")

        st.markdown("**Target Session Block:**")
        st.code(target_blk)

        st.markdown("**Pinpointed Error Events:**")
        sus_events = target_ml.get("suspicious_events", [])
        if sus_events:
            for se in sus_events:
                meta = EVENT_CAUSALITY_MAP.get(se, {"name": "Exception Event"})
                st.warning(f"• Event {se}: {meta['name']}")
        else:
            st.write("None")

# -------------------------------------------------------------
# TAB 3: MEDALLION DATA PIPELINE & AI CATALOG
# -------------------------------------------------------------
with tab_medallion:
    st.markdown("### 🏅 Medallion Architecture Data Explorer")
    st.caption("Explore data transformations across Bronze (Raw Store), Silver (Parsed & Sessionized), and Gold (AI Features) tiers.")

    m1, m2, m3, m4 = st.tabs(["1. Bronze Layer (Raw Store)", "2. Silver Layer (Parsed Events)", "3. Gold Layer (AI Feature Sets)", "4. Enterprise AI Catalog"])

    with m1:
        st.markdown("##### 🥉 Bronze Tier — Immutable Ingested Log Records")
        bronze_records = results["bronze_records"]
        if isinstance(bronze_records, list):
            df_bronze = pd.DataFrame(bronze_records)
            st.dataframe(df_bronze, use_container_width=True)
        else:
            st.dataframe(bronze_records, use_container_width=True)

    with m2:
        st.markdown("##### 🥈 Silver Tier — Regex Template Matched Event Records")
        st.dataframe(results["silver_df"], use_container_width=True)

    with m3:
        st.markdown("##### 🥇 Gold Tier — Sessionized AI Feature Datasets")
        st.dataframe(results["gold_df"], use_container_width=True)

    with m4:
        st.markdown("##### 📚 Enterprise AI Catalog Metadata Registry")
        cat_data = results["ai_catalog"]
        
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

# -------------------------------------------------------------
# TAB 4: PYTORCH ML BACKBONE INSIGHTS
# -------------------------------------------------------------
with tab_ml:
    st.markdown("### 🧠 PyTorch Transformer Model Backbone")
    st.caption("Sequence evaluation and deep feature embedding extraction using `transformer_backbone.pth`.")

    ml1, ml2 = st.columns(2)

    with ml1:
        st.markdown("##### Transformer Anomaly Inference Dashboard")
        prob_val = target_ml.get("anomaly_probability", 0.0)
        status_lbl = target_ml.get("status_label", "NORMAL PATTERN")
        
        if is_anom:
            st.error(f"### 🚨 {status_lbl}")
        else:
            st.success(f"### ✅ {status_lbl}")

        st.markdown("**Anomaly Probability Score:**")
        st.progress(prob_val)
        st.caption(f"Model Anomaly Probability: **{prob_val:.2%}** | Confidence Score: **{target_ml.get('confidence', 0.0):.2%}**")

        st.markdown("##### Suspicious Event Codes Detected")
        sus_list = target_ml.get("suspicious_events", [])
        if sus_list:
            tags = []
            for s_eid in sus_list:
                meta = EVENT_CAUSALITY_MAP.get(s_eid, {"name": "Error Code", "cause": "Exception"})
                tags.append(f'<span class="event-tag tag-danger">⚠️ {s_eid}: {meta["name"]} ({meta["cause"]})</span>')
            st.markdown("<br>".join(tags), unsafe_allow_html=True)
        else:
            st.write("Zero anomalous event codes detected in this block.")

    with ml2:
        st.markdown("##### 64-Dimensional Sequence Feature Embeddings")
        feat_vals = target_ml.get("feature_vector", [])
        if feat_vals:
            df_feat = pd.DataFrame({"Dimension": range(len(feat_vals)), "Embedding Value": feat_vals})
            st.line_chart(df_feat.set_index("Dimension"))

# -------------------------------------------------------------
# TAB 5: LOCAL SLM DIAGNOSIS & REMEDIATION RUNBOOK
# -------------------------------------------------------------
with tab_slm:
    st.markdown("### 💡 Local SLM Natural Language Diagnosis")
    st.caption("Human-understandable diagnostic output generated by local SLM (Qwen / Phi-3 / Gemini) reasoning.")

    st.markdown(f"""
    <div class="glass-panel glass-panel-alert">
        <h4 style="color: #38bdf8; margin-top: 0;">Executive Summary</h4>
        <p style="font-size: 1.05rem; line-height: 1.6;">{slm_res['summary']}</p>
        
        <h5 style="color: #c084fc; margin-top: 16px;">Error Mechanics & Failure Sequence</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['mechanism']}</p>
        
        <h5 style="color: #f472b6; margin-top: 16px;">Operational Impact</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['impact']}</p>
        
        <h5 style="color: #34d399; margin-top: 16px;">Evidence & Model Confidence</h5>
        <p style="color: #94a3b8; font-size: 0.85rem;">{slm_res['confidence_note']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🛠️ Operational Remediation SOP Runbook")
    st.warning(f"**Urgency Level:** {rem_res['urgency']}")

    for step in rem_res["sop_steps"]:
        st.code(step, language="bash")

# -------------------------------------------------------------
# TAB 6: HUMAN-IN-THE-LOOP INCIDENT TICKET RAISING
# -------------------------------------------------------------
with tab_ticket:
    st.markdown("### 🎫 Human-in-the-Loop Incident Ticket Manager")
    st.caption("The SLM automatically drafts ticket details based on log diagnostics. Review, edit, and approve before submitting to DevOps.")

    default_title = f"[INCIDENT-{target_blk[-6:]}] Critical Error in HDFS Block {target_blk}"
    default_desc = f"SLM Diagnostic Summary:\n{slm_res['summary']}\n\nMechanism:\n{slm_res['mechanism']}\n\nImpact:\n{slm_res['impact']}"

    st.markdown("##### ✍️ SLM-Drafted Ticket (Editable by Human Operator)")

    with st.form("ticket_form"):
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            t_title = st.text_input("Ticket Title", value=default_title)
        with col_t2:
            t_priority = st.selectbox("Priority Level", ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"], index=0 if is_anom else 2)

        col_t3, col_t4 = st.columns([1, 1])
        with col_t3:
            t_assignee = st.selectbox("Assignee Team", ["DevOps / Infrastructure", "Data Engineering", "Security Operations", "Database Reliability"])
        with col_t4:
            t_block = st.text_input("Target Resource ID", value=target_blk)

        t_desc = st.text_area("SLM-Drafted Problem Description (Editable)", value=default_desc, height=140)
        t_user_notes = st.text_area("Operator Custom Notes & Overrides", value="Verified anomalous log sequence by human operator. Approved for immediate remediation.", height=70)

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
            st.success(f"✅ Ticket {ticket_payload['ticket_id']} successfully registered and routed to {t_assignee}!")

    if st.session_state.submitted_tickets:
        st.markdown("##### 📋 Registered Incident Tickets")
        st.dataframe(pd.DataFrame(st.session_state.submitted_tickets), use_container_width=True)

# -------------------------------------------------------------
# TAB 7: ONE-CLICK PDF REPORT EXPORT
# -------------------------------------------------------------
with tab_export:
    st.markdown("### 📄 Export Comprehensive Incident PDF Report")
    st.caption("Generate and download a professional, audit-ready PDF document containing Medallion stats, PyTorch confidence, RCA graphs, SLM diagnostics, and SOP steps.")

    col_pdf_info, col_pdf_btn = st.columns([2, 1])

    with col_pdf_info:
        st.markdown("""
        **The generated PDF report contains:**
        - 📌 **Executive Header & Severity Status**
        - 📊 **PyTorch ML Transformer Anomaly Metrics & Confidence**
        - 🎫 **Human-Approved Incident Ticket Payload** (if submitted)
        - 💡 **Local SLM Natural Language Diagnostic Reasoning**
        - 🛠️ **Step-by-Step Operational Remediation SOP Runbook**
        """)

    with col_pdf_btn:
        latest_ticket = st.session_state.submitted_tickets[-1] if st.session_state.submitted_tickets else None
        
        pdf_bytes = generate_incident_pdf(
            target_block=target_blk,
            ml_res=target_ml,
            slm_res=slm_res,
            rem_res=rem_res,
            ticket_data=latest_ticket
        )

        st.download_button(
            label="📥 Download PDF Incident Report",
            data=pdf_bytes,
            file_name=f"LogVerse_Incident_Report_{target_blk}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )