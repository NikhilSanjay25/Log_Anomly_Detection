"""
LogVerse AI Platform — PDF Report & Ticket Export Engine
=========================================================
Generates professional enterprise PDF incident reports using ReportLab.
Includes Human-in-the-Loop ticket drafting and full RCA diagnostics.
"""

import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_incident_pdf(target_block, ml_res, slm_res, rem_res, ticket_data=None):
    """
    Generates an enterprise-grade PDF report and returns it as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")
    c_accent = colors.HexColor("#0284c7")
    c_danger = colors.HexColor("#e11d48")
    c_dark = colors.HexColor("#1e293b")
    c_text = colors.HexColor("#334155")
    c_light = colors.HexColor("#f8fafc")

    # Custom Typography Styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        alignment=TA_LEFT
    )

    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_accent,
        alignment=TA_LEFT
    )

    style_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=6
    )

    style_body = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text
    )

    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=4
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("LOGVERSE AI PLATFORM", style_subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("INCIDENT DIAGNOSIS & ROOT CAUSE ANALYSIS REPORT", style_title))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceBefore=4, spaceAfter=12))

    # 2. Executive Metadata Summary Table
    is_anom = ml_res.get("is_anomaly", False)
    prob = ml_res.get("anomaly_probability", 0.0)
    severity_label = "CRITICAL ANOMALY" if is_anom else "NORMAL HEALTHY"
    severity_color = c_danger if is_anom else colors.HexColor("#10b981")

    meta_data = [
        [Paragraph("<b>Report Generated:</b>", style_body), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style_body),
         Paragraph("<b>Severity Status:</b>", style_body), Paragraph(f"<font color='{severity_color.hexval()}'><b>{severity_label}</b></font>", style_body)],
        [Paragraph("<b>Target Session Block:</b>", style_body), Paragraph(f"<code>{target_block}</code>", style_body),
         Paragraph("<b>ML Anomaly Prob:</b>", style_body), Paragraph(f"{prob:.1%}", style_body)],
        [Paragraph("<b>Evaluated By:</b>", style_body), Paragraph("PyTorch Transformer Backbone + Local SLM", style_body),
         Paragraph("<b>Pipeline Engine:</b>", style_body), Paragraph("Medallion Architecture (Bronze->Silver->Gold)", style_body)]
    ]

    t_meta = Table(meta_data, colWidths=[120, 150, 110, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # 3. Human-in-the-Loop Incident Ticket Section (If Raised)
    if ticket_data:
        story.append(Paragraph("1. Incident Ticket Details (Human-in-the-Loop Approved)", style_heading))
        ticket_table_data = [
            [Paragraph("<b>Ticket ID:</b>", style_body), Paragraph(ticket_data.get("ticket_id", "TCK-1001"), style_body)],
            [Paragraph("<b>Ticket Title:</b>", style_body), Paragraph(ticket_data.get("title", ""), style_body)],
            [Paragraph("<b>Priority:</b>", style_body), Paragraph(f"<b>{ticket_data.get('priority', 'High')}</b>", style_body)],
            [Paragraph("<b>Assignee Group:</b>", style_body), Paragraph(ticket_data.get("assignee", "DevOps / Reliability Team"), style_body)],
            [Paragraph("<b>Problem Description:</b>", style_body), Paragraph(ticket_data.get("description", ""), style_body)],
            [Paragraph("<b>Operator Notes:</b>", style_body), Paragraph(ticket_data.get("user_notes", "Verified by operator."), style_body)]
        ]
        t_ticket = Table(ticket_table_data, colWidths=[130, 400])
        t_ticket.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_ticket)
        story.append(Spacer(1, 14))

    # 4. SLM Natural Language Diagnosis
    story.append(Paragraph("2. Local SLM Diagnostic Reasoning & Error Analysis", style_heading))
    story.append(Paragraph(f"<b>Executive Summary:</b> {slm_res.get('summary', '')}", style_body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Error Mechanics & Breakdown:</b> {slm_res.get('mechanism', '')}", style_body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Operational Impact:</b> {slm_res.get('impact', '')}", style_body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Grounding Evidence:</b> {slm_res.get('confidence_note', '')}", style_body))
    story.append(Spacer(1, 14))

    # 5. Operational Remediation SOP Runbook
    story.append(Paragraph("3. Recommended Operational Remediation SOP Runbook", style_heading))
    story.append(Paragraph(f"<b>Urgency:</b> {rem_res.get('urgency', 'CRITICAL')}", style_body))
    story.append(Spacer(1, 6))

    for step in rem_res.get("sop_steps", []):
        story.append(Paragraph(step, style_code))
        story.append(Spacer(1, 4))

    # Build Document
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
