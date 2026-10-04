"""
LogVerse AI Platform — Desktop Product Application UI
=====================================================
Super-cool, aesthetic dark-glassmorphism desktop dashboard built with Streamlit.
Features:
  - File uploader & Live K8s/Docker stream simulator
  - Live AI Agent Workflow Canvas & Inter-Agent Communication Timeline
  - Interactive Root Cause Analysis (RCA) Causal Graph (Vis.js)
  - Medallion Data Pipeline Inspector (Bronze -> Silver -> Gold + AI Catalog)
  - PyTorch Transformer Anomaly Detection metrics
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
from logverse_agents import MultiAgentOrchestrator
from logverse_pdf import generate_incident_pdf

# ─────────────────────────────────────────────────────────────
# STREAMLIT CONFIG & CUSTOM GLASSMORPHISM CSS THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogVerse AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Main Dark Theme Background */
.stApp {
    background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
    color: #f8fafc;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Header Styling */
.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.brand-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.glass-card-anomaly {
    background: rgba(127, 29, 29, 0.3);
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.glass-card-normal {
    background: rgba(6, 78, 59, 0.3);
    border: 1px solid rgba(16, 185, 129, 0.4);
}

/* Agent Status Badges */
.agent-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.badge-completed { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

/* Communication Log Stream */
.msg-container {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    background: #020617;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #1e293b;
    max-height: 280px;
    overflow-y: auto;
}

.msg-line {
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #0f172a;
}
.msg-time { color: #64748b; }
.msg-sender { color: #38bdf8; font-weight: bold; }
.msg-recipient { color: #a78bfa; font-weight: bold; }
.msg-action { color: #f472b6; font-weight: 600; }
.msg-content { color: #e2e8f0; }

/* Metric KPI Numbers */
.kpi-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
}
.kpi-lbl {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INITIALIZE ORCHESTRATOR & SESSION STATE
# ─────────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator()

if "results" not in st.session_state:
    sample_text = generate_sample_hdfs_log()
    st.session_state.results = st.session_state.orchestrator.run_agentic_pipeline(sample_text, "sample_hdfs_default.log")

if "submitted_tickets" not in st.session_state:
    st.session_state.submitted_tickets = []

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS & LOG INGESTION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/data-configuration.png", width=64)
    st.markdown("### ⚡ Control Panel")
    st.caption("Agentic AI Log Intelligence Platform")

    st.markdown("---")
    st.markdown("#### 1. Log Data Source")
    data_source_mode = st.radio(
        "Ingestion Mode",
        ["Default HDFS Sample", "Upload Custom Log File", "Simulate Live K8s/Docker Stream"],
        index=0
    )

    log_input_text = ""
    source_filename = "upload.log"

    if data_source_mode == "Default HDFS Sample":
        log_input_text = generate_sample_hdfs_log()
        source_filename = "sample_hdfs_default.log"
    elif data_source_mode == "Upload Custom Log File":
        uploaded_file = st.file_uploader("Upload Log File (.txt, .log)", type=["txt", "log"])
        if uploaded_file is not None:
            log_input_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            source_filename = uploaded_file.name
        else:
            log_input_text = generate_sample_hdfs_log()
    else:  # K8s / Docker Stream Simulation
        st.info("🌐 K8s / Docker Ingestion Adapter Active")
        log_input_text = """2026-10-04T10:15:00Z pod/payment-service-789456 info: Receiving request block blk_-1608999687919862906
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
    st.caption("LogVerse AI Platform v2.5 | Medallion Lakehouse | Multi-Agent Architecture")

# ─────────────────────────────────────────────────────────────
# HEADER & SYSTEM METRICS
# ─────────────────────────────────────────────────────────────
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

# Top KPI Metric Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("bronze_records_count", 0)}</div><div class="kpi-lbl">Bronze Raw Lines</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("silver_records_count", 0)}</div><div class="kpi-lbl">Silver Parsed Events</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="glass-card"><div class="kpi-val">{ai_cat.get("gold_sessions_count", 0)}</div><div class="kpi-lbl">Gold Session Blocks</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="glass-card"><div class="kpi-val">{target_ml.get("anomaly_probability", 0.0):.1%}</div><div class="kpi-lbl">Transformer Prob</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="glass-card"><div class="kpi-val">6 / 6</div><div class="kpi-lbl">Agents Active</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN TABBED INTERFACE
# ─────────────────────────────────────────────────────────────
tab_agents, tab_rca, tab_medallion, tab_ml, tab_slm, tab_ticket, tab_export = st.tabs([
    "🤖 Agent Workflow Canvas",
    "🕸️ Causal RCA Graph",
    "🏅 Medallion Data Pipeline",
    "🧠 PyTorch ML Backbone",
    "💡 Local SLM Diagnosis & Runbook",
    "🎫 Human-in-the-Loop Ticket Raising",
    "📄 PDF Report Export"
])

# -------------------------------------------------------------
# TAB 1: AGENT WORKFLOW CANVAS & LIVE COMMUNICATION
# -------------------------------------------------------------
with tab_agents:
    st.markdown("### 🤖 Multi-Agent Orchestration Visualizer")
    st.caption("Real-time step-by-step workflow execution showing agent reasoning, communication, and handoffs.")

    ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
    agents_list = [
        ("Planner Agent", "Decomposes query into execution steps", ac1),
        ("Catalog Agent", "Registers Medallion dataset schemas", ac2),
        ("ML Anomaly Agent", "Evaluates Transformer backbone model", ac3),
        ("RCA Graph Agent", "Builds causal failure network graph", ac4),
        ("SLM Diagnostic Agent", "Generates plain-English root cause", ac5),
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

    st.markdown("#### 💬 Inter-Agent Live Communication Stream")
    comm_logs = res["communication_log"]

    log_html_lines = []
    for msg in comm_logs:
        line = f"""
        <div class="msg-line">
            <span class="msg-time">[{msg['timestamp']}]</span>
            <span class="msg-sender">{msg['sender']}</span> ➔
            <span class="msg-recipient">{msg['recipient']}</span>:
            <span class="msg-action">[{msg['action']}]</span>
            <span class="msg-content">{msg['content']}</span>
        </div>
        """
        log_html_lines.append(line)

    full_log_html = f'<div class="msg-container">{"".join(log_html_lines)}</div>'
    st.markdown(full_log_html, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: CAUSAL ROOT CAUSE ANALYSIS (RCA) GRAPH
# -------------------------------------------------------------
with tab_rca:
    st.markdown("### 🕸️ Causal Root Cause Analysis (RCA) Network Graph")
    st.caption("Interactive causal propagation path from cluster components to error triggers and primary root cause node.")

    rca_col_graph, rca_col_info = st.columns([3, 1])

    with rca_col_graph:
        st.markdown("##### Interactive Network Node Graph")
        components.html(res["rca_html"], height=520, scrolling=False)

    with rca_col_info:
        st.markdown("##### 📌 RCA Summary")
        root_causes = res["root_causes"]
        if root_causes:
            for rc in root_causes:
                st.error(f"**Primary Root Cause Identified:**\n{rc}")
        else:
            st.success("No critical failure trigger detected.")

        st.markdown("**Evaluated Block:**")
        st.code(target_blk)

        st.markdown("**Pinpointed Suspicious Events:**")
        sus_events = target_ml.get("suspicious_events", [])
        if sus_events:
            for se in sus_events:
                st.warning(f"• Event {se}: Critical Exception/Timeout")
        else:
            st.write("None")

# -------------------------------------------------------------
# TAB 3: MEDALLION DATA PIPELINE & AI CATALOG
# -------------------------------------------------------------
with tab_medallion:
    st.markdown("### 🏅 Medallion Architecture Data Explorer")
    st.caption("Data transformation lifecycle across Bronze (Raw), Silver (Parsed & Structured), and Gold (AI Feature Sets) tiers.")

    m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["1. Bronze Layer (Raw)", "2. Silver Layer (Parsed)", "3. Gold Layer (AI Features)", "4. Enterprise AI Catalog"])

    with m_tab1:
        st.markdown("##### 🥉 Bronze Tier — Immutable Raw Log Ingestion")
        st.json(res["bronze_records"][:5])

    with m_tab2:
        st.markdown("##### 🥈 Silver Tier — Regex Parsed & Template Matched Event Stream")
        st.dataframe(res["silver_df"], use_container_width=True)

    with m_tab3:
        st.markdown("##### 🥇 Gold Tier — Sessionized Aggregations & ML Feature Vectors")
        st.dataframe(res["gold_df"], use_container_width=True)

    with m_tab4:
        st.markdown("##### 📚 Enterprise AI Catalog Metadata & Lineage")
        st.json(res["ai_catalog"])

# -------------------------------------------------------------
# TAB 4: PYTORCH ML BACKBONE INSIGHTS
# -------------------------------------------------------------
with tab_ml:
    st.markdown("### 🧠 PyTorch Transformer Model Backbone")
    st.caption("Evaluation details from `transformer_backbone.pth` and sequence feature extraction.")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        st.markdown("##### Model Prediction Output")
        st.json({
            "target_block": target_blk,
            "status": target_ml.get("status_label"),
            "anomaly_probability": target_ml.get("anomaly_probability"),
            "confidence_score": target_ml.get("confidence"),
            "suspicious_events_detected": target_ml.get("suspicious_events"),
            "events_evaluated_count": target_ml.get("total_events_evaluated")
        })

    with ml_col2:
        st.markdown("##### Extracted Deep Feature Embeddings (64-dim)")
        feat_vec = target_ml.get("feature_vector", [])
        if feat_vec:
            df_feat = pd.DataFrame({"Embedding Dimension": range(len(feat_vec)), "Value": feat_vec})
            st.line_chart(df_feat.set_index("Embedding Dimension"))

# -------------------------------------------------------------
# TAB 5: LOCAL SLM DIAGNOSIS & REMEDIATION RUNBOOK
# -------------------------------------------------------------
with tab_slm:
    st.markdown("### 💡 Local SLM Human-Understandable Diagnosis")
    st.caption("Small Language Model (Qwen/Phi-3/Gemini) natural language reasoning and operational SOP runbooks.")

    st.markdown(f"""
    <div class="glass-card glass-card-anomaly" style="padding: 20px;">
        <h4 style="color: #38bdf8; margin-top: 0;">Executive Summary</h4>
        <p style="font-size: 1.05rem; line-height: 1.6;">{slm_res['summary']}</p>
        
        <h5 style="color: #a78bfa; margin-top: 16px;">Error Mechanics & Sequence Analysis</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['mechanism']}</p>
        
        <h5 style="color: #f472b6; margin-top: 16px;">Operational Impact</h5>
        <p style="color: #cbd5e1; font-size: 0.95rem;">{slm_res['impact']}</p>
        
        <h5 style="color: #34d399; margin-top: 16px;">Grounding Confidence & Evidence</h5>
        <p style="color: #94a3b8; font-size: 0.85rem;">{slm_res['confidence_note']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🛠️ Operational Remediation SOP Runbook")
    st.warning(f"**Urgency Level:** {rem_res['urgency']}")

    for step in rem_res["sop_steps"]:
        st.code(step, language="bash")

# -------------------------------------------------------------
# TAB 6: HUMAN-IN-THE-LOOP INCIDENT TICKET MANAGER
# -------------------------------------------------------------
with tab_ticket:
    st.markdown("### 🎫 Human-in-the-Loop Incident Ticket Manager")
    st.caption("The SLM automatically drafts ticket details based on log diagnostics. Review, edit, and approve before submitting to DevOps.")

    default_title = f"[INCIDENT-{target_blk[-6:]}] Critical Error in HDFS Block {target_blk}"
    default_desc = f"SLM Diagnostic Summary:\n{slm_res['summary']}\n\nMechanism:\n{slm_res['mechanism']}\n\nImpact:\n{slm_res['impact']}"

    st.markdown("##### ✍️ SLM-Drafted Ticket (Editable by Human Operator)")

    with st.form("ticket_form_desktop"):
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
