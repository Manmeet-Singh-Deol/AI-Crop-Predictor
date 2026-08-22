"""
PDF Diagnosis Report Generator
Uses ReportLab to compile clean, professional, exportable Agronomy Diagnosis & Advisory Certificates.
"""

import io
import base64
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from typing import Dict, Any, Optional

def generate_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Compile agronomy diagnosis data into a multi-page PDF document.
    Data dictionary expects diagnosis, gradcam, severity, advisory, and weather sections.
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
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#065f46')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4b5563')
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#047857'),
        spaceBefore=8,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1f2937')
    )
    
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#111827')
    )

    story = []
    
    # --- Header Section ---
    report_id = f"AGR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    timestamp_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
    
    header_data = [
        [
            Paragraph("🌿 <b>AGRO-AI DIAGNOSTIC CERTIFICATE</b>", title_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Date:</b> {timestamp_str}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#059669'), spaceAfter=12))
    
    # --- Diagnosis Overview Table ---
    top_pred = data.get("top_prediction", {})
    advisory = data.get("advisory", {})
    severity = data.get("severity", {})
    weather = data.get("weather", {}).get("epidemiological_risk", {})
    
    crop_name = top_pred.get("crop", advisory.get("crop", "Crop"))
    disease_name = top_pred.get("disease", advisory.get("disease", "Diagnosis"))
    confidence = top_pred.get("confidence", 0.0)
    scientific_name = advisory.get("scientific_name", "N/A")
    pathogen_type = advisory.get("pathogen_type", "N/A")
    sev_pct = severity.get("severity_percentage", 0.0)
    sev_stage = severity.get("severity_stage", "N/A")
    urgency = severity.get("urgency", "Moderate")
    
    overview_data = [
        [
            Paragraph("<b>Target Crop:</b>", bold_body),
            Paragraph(str(crop_name), body_style),
            Paragraph("<b>Diagnostic Finding:</b>", bold_body),
            Paragraph(f"<b>{disease_name}</b>", ParagraphStyle('Diag', parent=body_style, textColor=colors.HexColor('#b91c1c'), fontName='Helvetica-Bold'))
        ],
        [
            Paragraph("<b>AI Confidence:</b>", bold_body),
            Paragraph(f"{confidence}%", body_style),
            Paragraph("<b>Pathogen Type:</b>", bold_body),
            Paragraph(str(pathogen_type), body_style)
        ],
        [
            Paragraph("<b>Scientific Taxonomy:</b>", bold_body),
            Paragraph(f"<i>{scientific_name}</i>", body_style),
            Paragraph("<b>Infection Severity:</b>", bold_body),
            Paragraph(f"<b>{sev_pct}% ({sev_stage})</b>", body_style)
        ],
        [
            Paragraph("<b>Intervention Urgency:</b>", bold_body),
            Paragraph(f"<b>{urgency}</b>", ParagraphStyle('Urg', parent=body_style, textColor=colors.HexColor('#c2410c'))),
            Paragraph("<b>Environmental Outbreak:</b>", bold_body),
            Paragraph(f"{weather.get('threat_level', 'Moderate Risk')} ({weather.get('overall_outbreak_risk', 50)}%)", body_style)
        ]
    ]
    
    overview_table = Table(overview_data, colWidths=[120, 150, 120, 150])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#86efac')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dcfce7')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 12))
    
    # --- Visual Analysis Section (Original vs Grad-CAM vs Severity Mask) ---
    story.append(Paragraph("📸 Visual Explainability & Lesion Segmentation", section_heading))
    
    def b64_to_rl_image(b64_str: Optional[str], width=165, height=140):
        if not b64_str:
            return Paragraph("<i>No Image Available</i>", body_style)
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_data = base64.b64decode(b64_str)
            img_buf = io.BytesIO(img_data)
            return RLImage(img_buf, width=width, height=height)
        except Exception:
            return Paragraph("<i>Image Load Error</i>", body_style)

    gradcam = data.get("gradcam", {})
    orig_img_flow = b64_to_rl_image(gradcam.get("original_image"))
    blended_img_flow = b64_to_rl_image(gradcam.get("blended_image"))
    severity_mask_flow = b64_to_rl_image(severity.get("severity_mask_image"))
    
    visual_table_data = [
        [orig_img_flow, blended_img_flow, severity_mask_flow],
        [
            Paragraph("<b>1. Original Leaf Sample</b>", ParagraphStyle('Cap1', parent=body_style, alignment=1)),
            Paragraph("<b>2. Grad-CAM Activation Heatmap</b>", ParagraphStyle('Cap2', parent=body_style, alignment=1)),
            Paragraph("<b>3. HSV Lesion Segmentation Mask</b>", ParagraphStyle('Cap3', parent=body_style, alignment=1))
        ]
    ]
    visual_table = Table(visual_table_data, colWidths=[180, 180, 180])
    visual_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(visual_table)
    story.append(Spacer(1, 10))
    
    # --- Key Diagnostic Symptoms ---
    symptoms = advisory.get("symptoms", [])
    if symptoms:
        story.append(Paragraph("🔍 Key Diagnostic Symptoms Identified", section_heading))
        for sym in symptoms:
            story.append(Paragraph(f"• {sym}", body_style))
        story.append(Spacer(1, 8))
        
    # --- Actionable Treatment Protocol ---
    story.append(Paragraph("🛡️ Integrated Agronomic Action Plan", section_heading))
    
    # Organic Controls
    organic = advisory.get("organic_controls", [])
    if organic:
        story.append(Paragraph("<b>🌿 Biological & Organic Solutions:</b>", bold_body))
        for org in organic:
            story.append(Paragraph(f"• {org}", body_style))
        story.append(Spacer(1, 6))
        
    # Chemical Controls
    chemicals = advisory.get("chemical_controls", [])
    if chemicals:
        story.append(Paragraph("<b>🧪 Commercial Brand Chemical & Insecticide Guide (Market Best Sellers):</b>", bold_body))
        chem_rows = [[
            Paragraph("<b>Commercial Brand & Best Seller</b>", ParagraphStyle('H1', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold', fontSize=8)),
            Paragraph("<b>Active Chemical Composition</b>", ParagraphStyle('H2', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold', fontSize=8)),
            Paragraph("<b>Dosage / L</b>", ParagraphStyle('H3', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold', fontSize=8)),
            Paragraph("<b>Timing & Pre-Harvest Interval (PHI)</b>", ParagraphStyle('H4', parent=body_style, textColor=colors.white, fontName='Helvetica-Bold', fontSize=8))
        ]]
        for c in chemicals:
            brand_display = f"<b>{c.get('product', 'Commercial Product')}</b>"
            active_comp = f"<i>{c.get('active_ingredient', c.get('product', ''))}</i><br/><font color='#059669'>[{c.get('type', 'Fungicide / Insecticide')}]</font>"
            timing_display = f"{c.get('timing', 'At disease onset')}<br/><font color='#b45309'><b>PHI:</b> {c.get('interval', '14 days')}</font>"
            
            chem_rows.append([
                Paragraph(brand_display, ParagraphStyle('B1', parent=body_style, fontSize=7.5, leading=9)),
                Paragraph(active_comp, ParagraphStyle('B2', parent=body_style, fontSize=7.5, leading=9)),
                Paragraph(f"<b>{c.get('dosage', 'Standard')}</b>", ParagraphStyle('B3', parent=body_style, fontSize=8, leading=9, textColor=colors.HexColor('#047857'))),
                Paragraph(timing_display, ParagraphStyle('B4', parent=body_style, fontSize=7.5, leading=9))
            ])
            
        chem_table = Table(chem_rows, colWidths=[150, 140, 75, 175])
        chem_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#065f46')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#9ca3af')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(chem_table)
        story.append(Spacer(1, 8))
        
    # Cultural Practices
    cultural = advisory.get("cultural_practices", [])
    if cultural:
        story.append(Paragraph("<b>🚜 Cultural & Field Management Best Practices:</b>", bold_body))
        for cul in cultural:
            story.append(Paragraph(f"• {cul}", body_style))
            
    # --- Footer Disclaimer ---
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#9ca3af'), spaceAfter=6))
    disclaimer_text = (
        "<b>Disclaimer:</b> This AI-generated diagnostic report is intended for informational and advisory guidance. "
        "Field agronomist inspection and local agricultural extension guidelines should be consulted before large-scale chemical spraying."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#6b7280'), alignment=1)))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
