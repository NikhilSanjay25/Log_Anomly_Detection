"""
LogVerse AI Platform — Enterprise Log Intelligence Product
===========================================================
An Enterprise-Grade Agentic AI Data Platform powered by Medallion Architecture,
Multi-Agent Orchestration, PyTorch Deep Learning Transformer Backbone,
Interactive Root Cause Analysis (RCA) Causal Network Graphs, Local SLMs,
Human-in-the-Loop Ticket Raising, and One-Click PDF Report Exporting.

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
from logverse_rca_graph import RCAGraphBuilder
from logverse_agents import MultiAgentOrchestrator
from logverse_pdf import generate_incident_pdf

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

/* Console Terminal for Inter-Agent Communications */
.agent-terminal {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    background: #020617;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #1e293b;
    max-height: 320px;
    overflow-y: auto;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
}

.log-entry {
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    line-height: 1.5;
}

.log-time { color: #64748b; }
.log-sender { color: #38bdf8; font-weight: 700; }
.log-arrow { color: #64748b; }
.log-recipient { color: #c084fc; font-weight: 700; }
.log-action { color: #f472b6; font-weight: 600; padding: 1px 6px; background: rgba(244, 114, 182, 0.1); border-radius: 4px; }
.log-content { color: #e2e8f0; margin-top: 2px; }

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
    st.caption("LogVerse AI Platform v2.5 Enterprise Edition")

    st.markdown("---")
    st.markdown("#### 📥 1. Ingestion Adapter")
    ingest_mode = st.radio(
        "Source Type",
        ["Sample HDFS Logs", "Upload Custom Log File", "Live K8s / Docker Stream"],
        index=0
    )

    log_content_input = ""
    source_name = "log_source.log"

    if ingest_mode == "Sample HDFS Logs":
        log_content_input = generate_sample_hdfs_log()
        source_name = "sample_hdfs_benchmark.log"
    elif ingest_mode == "Upload Custom Log File":
        up_file = st.file_uploader("Upload Raw Log (.txt, .log)", type=["txt", "log"])
        if up_file is not None:
            log_content_input = up_file.getvalue().decode("utf-8", errors="ignore")
            source_name = up_file.name
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

    st.markdown("#### 💬 Real-Time Inter-Agent Communication Feed")
    comm_log = results["communication_log"]

    log_entries_html = []
    for msg in comm_log:
        entry = f"""
        <div class="log-entry">
            <div>
                <span class="log-time">[{msg['timestamp']}]</span>
                <span class="log-sender">{msg['sender']}</span>
                <span class="log-arrow">➔</span>
                <span class="log-recipient">{msg['recipient']}</span>
                <span class="log-action">{msg['action']}</span>
            </div>
            <div class="log-content">{msg['content']}</div>
        </div>
        """
        log_entries_html.append(entry)

    full_console_html = f'<div class="agent-terminal">{"".join(log_entries_html)}</div>'
    st.markdown(full_console_html, unsafe_allow_html=True)

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
                st.warning(f"• Event {se}: Socket/Write Exception")
        else:
            st.write("None")

# -------------------------------------------------------------
# TAB 3: MEDALLION DATA PIPELINE & AI CATALOG
# -------------------------------------------------------------
with tab_medallion:
    st.markdown("### 🏅 Medallion Architecture Data Explorer")
    st.caption("Explore data transformations across Bronze (Raw Store), Silver (Parsed & Sessionized), and Gold (AI Features) tiers.")

    m1, m2, m3, m4 = st.tabs(["1. Bronze Layer (Raw)", "2. Silver Layer (Parsed)", "3. Gold Layer (AI Features)", "4. Enterprise AI Catalog"])

    with m1:
        st.markdown("##### 🥉 Bronze Layer — Immutable Raw Ingestion Data")
        st.json(results["bronze_records"][:5])

    with m2:
        st.markdown("##### 🥈 Silver Layer — Regex Template Matched Event Records")
        st.dataframe(results["silver_df"], use_container_width=True)

    with m3:
        st.markdown("##### 🥇 Gold Layer — Sessionized AI Feature Datasets")
        st.dataframe(results["gold_df"], use_container_width=True)

    with m4:
        st.markdown("##### 📚 Enterprise AI Catalog Metadata Registry")
        st.json(results["ai_catalog"])

# -------------------------------------------------------------
# TAB 4: PYTORCH ML BACKBONE INSIGHTS
# -------------------------------------------------------------
with tab_ml:
    st.markdown("### 🧠 PyTorch Transformer Model Backbone")
    st.caption("Sequence evaluation and deep feature embedding extraction using `transformer_backbone.pth`.")

    ml1, ml2 = st.columns(2)

    with ml1:
        st.markdown("##### Transformer Inference Metrics")
        st.json({
            "target_block": target_blk,
            "status_label": target_ml.get("status_label"),
            "anomaly_probability": target_ml.get("anomaly_probability"),
            "confidence_score": target_ml.get("confidence"),
            "suspicious_events": target_ml.get("suspicious_events"),
            "total_events_evaluated": target_ml.get("total_events_evaluated")
        })

    with ml2:
        st.markdown("##### 64-Dimensional Sequence Feature Embeddings")
        feat_vals = target_ml.get("feature_vector", [])
        if feat_vals:
            df_feat = pd.DataFrame({"Dimension": range(len(feat_vals)), "Value": feat_vals})
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

    # SLM Draft Generation
    default_title = f"[INCIDENT-{target_blk[-6:]}] Critical Error in HDFS Block {target_blk}"
    default_desc = f"SLM Diagnostic Summary:\n{slm_res['summary']}\n\nMechanism:\n{slm_res['mechanism']}\n\nImpact:\n{slm_res['impact']}"
    default_priority = "P1 - Critical" if is_anom else "P3 - Normal"

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