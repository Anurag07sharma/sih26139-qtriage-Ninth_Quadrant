from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    doc = SimpleDocTemplate("Data_Reduction_Report.pdf", pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1E293B"))
    heading_style = ParagraphStyle(name='HeadingStyle', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor("#0F172A"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle(name='BodyStyle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#334155"))

    # Title
    story.append(Paragraph("SIH26139: Data Processing & Reduction Audit Report", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This document outlines the exact data reduction metrics and structural differences between Classical and Quantum dataset preparation tracks.", body_style))
    story.append(Spacer(1, 15))

    # Section 1: Data Reduction Table
    story.append(Paragraph("1. Data Reduction & Filtering Summary", heading_style))
    
    table_data = [
        ["Track", "Raw Rows", "Final Rows", "Row Reduction", "Raw Feats", "Final Feats", "Feat Reduction", "Train/Test Split"],
        ["heart", "1,190", "918", "-22.8% (Dedup)", "11", "6", "-45.5%", "734 / 184"],
        ["heart_fhs", "4,240", "4,240", "0%", "15", "6", "-60.0%", "3,392 / 848"],
        ["diabetes", "253,680", "100,000", "-60.6% (Subsample)", "21", "8", "-61.9%", "80,000 / 20,000"],
        ["diabetes_pima", "768", "768", "0%", "8", "8", "0%", "614 / 154"]
    ]
    
    t1 = Table(table_data, colWidths=[65, 55, 60, 95, 55, 60, 75, 75])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    # Section 2: Classical vs Quantum Comparison Table
    story.append(Paragraph("2. Structural Differences: Classical vs. Quantum Data", heading_style))
    
    q_diff_data = [
        ["Dimension / Property", "Classical Model Data", "Quantum Model Data"],
        ["File Format", "2D Tabular CSV (.csv)", "Binary Tensors (.npy)"],
        ["Normalization", "StandardScaler (Mean=0, Std=1)", "Angle: MinMax [0, pi] | Amplitude: L2 Norm [0, 1]"],
        ["Padding Constraints", "None (Native feature count)", "Padded to nearest 2^n dimension for qubits"],
        ["Sample Size", "Full train split (up to 80,000 rows)", "Stratified subsample (300 rows) for VQC limits"],
        ["Mathematical Role", "Used directly in linear/tree equations", "Mapped to Bloch Sphere rotations or state vectors"]
    ]

    t2 = Table(q_diff_data, colWidths=[110, 200, 230])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#FFFFFF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t2)
    
    doc.build(story)
    print("Report generated successfully as 'Data_Reduction_Report.pdf'")

if __name__ == "__main__":
    generate_pdf()