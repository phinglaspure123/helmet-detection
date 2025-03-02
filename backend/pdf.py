from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os
from datetime import datetime
import random

def generate_challan(violation):
    """
    Generate a PDF challan for a traffic violation
    """
    # Create output directory if it doesn't exist
    os.makedirs("challans", exist_ok=True)
    
    # Define output path
    pdf_path = f"challans/challan_{violation['id']}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=1,
        spaceAfter=6
    )
    
    normal_style = styles["Normal"]
    
    # Prepare content elements
    elements = []
    
    # Add title
    elements.append(Paragraph("TRAFFIC VIOLATION CHALLAN", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add challan number and date
    elements.append(Paragraph(f"Challan No: CHN-{random.randint(100000, 999999)}", subtitle_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", subtitle_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add violation image if available
    if 'image_path' in violation and os.path.exists(violation['image_path']):
        img = Image(violation['image_path'], width=4*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
    
    # Add violation details
    elements.append(Paragraph("VIOLATION DETAILS", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create table for violation details
    data = [
        ["Violation Type:", violation['violation_type']],
        ["Date & Time:", violation['timestamp']],
        ["License Plate:", violation['license_plate']],
        ["Vehicle Type:", violation['vehicle_type']]
    ]
    
    table = Table(data, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Calculate fine amount (random for demo)
    fine_amount = random.randint(500, 2000)
    
    # Add fine details
    elements.append(Paragraph("FINE DETAILS", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create table for fine details
    fine_data = [
        ["Violation", "Fine Amount (INR)"],
        [violation['violation_type'], f"₹{fine_amount}"],
        ["Processing Fee", "₹50"],
        ["Total Amount Due", f"₹{fine_amount + 50}"]
    ]
    
    fine_table = Table(fine_data, colWidths=[2.5*inch, 2.5*inch])
    fine_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(fine_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add payment instructions
    elements.append(Paragraph("PAYMENT INSTRUCTIONS", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    payment_instructions = """
    1. Payment must be made within 30 days of challan issuance.
    2. Payment can be made online at www.trafficfines.gov.in
    3. You can also pay at any authorized traffic police station.
    4. Quote your challan number during payment.
    5. Non-payment will result in additional penalties.
    """
    elements.append(Paragraph(payment_instructions, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add legal notice
    elements.append(Paragraph("LEGAL NOTICE", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    legal_notice = """
    This challan has been issued under the provisions of the Motor Vehicles Act. 
    If you wish to contest this challan, you must appear in the designated traffic court 
    within 45 days of issuance with all relevant documentation.
    
    Failure to pay the fine or contest the challan within the specified period will result in 
    additional penalties and may lead to suspension of your driving license.
    """
    elements.append(Paragraph(legal_notice, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add footer
    footer_text = """
    This is a computer-generated document and does not require a physical signature.
    For any queries, contact the Traffic Control Department at helpdesk@traffic.gov.in or call 1800-XXX-XXXX.
    """
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    elements.append(Paragraph(footer_text, footer_style))
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Generated challan PDF: {pdf_path}")
    return pdf_path

def generate_bulk_challans(violations):
    """
    Generate multiple challans in one go
    """
    pdf_paths = []
    for violation in violations:
        pdf_path = generate_challan(violation)
        pdf_paths.append(pdf_path)
    
    return pdf_paths

def generate_summary_report(violations, output_path="challans/summary_report.pdf"):
    """
    Generate a summary report of all violations
    """
    # Create output directory if it doesn't exist
    os.makedirs("challans", exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=1,
        spaceAfter=6
    )
    
    normal_style = styles["Normal"]
    
    # Prepare content elements
    elements = []
    
    # Add title
    elements.append(Paragraph("TRAFFIC VIOLATIONS SUMMARY REPORT", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", subtitle_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Summary statistics
    elements.append(Paragraph("SUMMARY STATISTICS", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Count violations by type
    violation_types = {}
    for v in violations:
        v_type = v['violation_type']
        if v_type in violation_types:
            violation_types[v_type] += 1
        else:
            violation_types[v_type] = 1
    
    # Create table for summary statistics
    stats_data = [["Violation Type", "Count"]]
    for v_type, count in violation_types.items():
        stats_data.append([v_type, str(count)])
    
    stats_data.append(["Total Violations", str(len(violations))])
    
    stats_table = Table(stats_data, colWidths=[3*inch, 1.5*inch])
    stats_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # List of all violations
    elements.append(Paragraph("DETAILED VIOLATIONS LIST", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create table for violation details
    detail_data = [["Date & Time", "Violation Type", "License Plate", "Fine Amount"]]
    
    for v in violations:
        fine_amount = random.randint(500, 2000)
        detail_data.append([
            v['timestamp'],
            v['violation_type'],
            v['license_plate'],
            f"₹{fine_amount}"
        ])
    
    detail_table = Table(detail_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1*inch])
    detail_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(detail_table)
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Generated summary report: {output_path}")
    return output_path

def generate_bulk_challans(violations):
    """
    Generate multiple challans in one go
    """
    pdf_paths = []
    for violation in violations:
        pdf_path = generate_challan(violation)
        pdf_paths.append(pdf_path)
    
    return pdf_paths

def generate_summary_report(violations, output_path="challans/summary_report.pdf"):
    """
    Generate a summary report of all violations
    """
    # Create output directory if it doesn't exist
    os.makedirs("challans", exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=1,
        spaceAfter=6
    )
    
    normal_style = styles["Normal"]
    
    # Prepare content elements
    elements = []
    
    # Add title
    elements.append(Paragraph("TRAFFIC VIOLATIONS SUMMARY REPORT", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", subtitle_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Summary statistics
    elements.append(Paragraph("SUMMARY STATISTICS", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Count violations by type
    violation_types = {}
    for v in violations:
        v_type = v['violation_type']
        if v_type in violation_types:
            violation_types[v_type] += 1
        else:
            violation_types[v_type] = 1
    
    # Create table for summary statistics
    stats_data = [["Violation Type", "Count"]]
    for v_type, count in violation_types.items():
        stats_data.append([v_type, str(count)])
    
    stats_data.append(["Total Violations", str(len(violations))])
    
    stats_table = Table(stats_data, colWidths=[3*inch, 1.5*inch])
    stats_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # List of all violations
    elements.append(Paragraph("DETAILED VIOLATIONS LIST", subtitle_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create table for violation details
    detail_data = [["Date & Time", "Violation Type", "License Plate", "Fine Amount"]]
    
    for v in violations:
        fine_amount = random.randint(500, 2000)
        detail_data.append([
            v['timestamp'],
            v['violation_type'],
            v['license_plate'],
            f"₹{fine_amount}"
        ])
    
    detail_table = Table(detail_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1*inch])
    detail_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(detail_table)
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Generated summary report: {output_path}")
    return output_path