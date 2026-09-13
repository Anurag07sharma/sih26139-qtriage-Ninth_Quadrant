import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf():
    # Save the PDF in the exact same folder where this script is located
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_Preprocessing_Explanation.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#0F172A"), spaceAfter=15)
    heading_style = ParagraphStyle(name='Heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#1E3A8A"), spaceBefore=15, spaceAfter=8)
    sub_heading = ParagraphStyle(name='SubHeading', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor("#0F766E"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle(name='Body', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)
    bold_body = ParagraphStyle(name='BoldBody', parent=body_style, fontName='Helvetica-Bold')

    story = []

    # Title
    story.append(Paragraph("SIH26139: Detailed Data Preprocessing & Pipeline Architecture", title_style))
    story.append(Paragraph("This document provides a block-by-block explanation of the prepare_data.py script, detailing how raw medical data is transformed into quantum-ready matrices.", body_style))
    story.append(Spacer(1, 10))

    # Module 1
    story.append(Paragraph("Module 1: Secure Data Acquisition (download_all)", heading_style))
    story.append(Paragraph("<b>What it does:</b> Fetches raw CSV files directly from verified public mirrors.", body_style))
    story.append(Paragraph("<b>How it works:</b> Validates the integrity of the downloaded data by comparing the resulting DataFrame's shape (rows/cols) against the officially documented shape. It then generates a SHA-256 digital hash.", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'We automated data ingestion with SHA-256 hashing to prove to the jury that zero synthetic or tampered data was used.'", body_style))

    # Module 2
    story.append(Paragraph("Module 2: Data Cleaning & Hygiene", heading_style))
    story.append(Paragraph("<b>What it does:</b> Removes logical errors and prevents data leakage.", body_style))
    story.append(Paragraph("<b>How it works:</b> For the Heart track, it drops 272 duplicate rows caused by a historical database merge. For the Pima dataset, it replaces biologically impossible zeros (e.g., blood pressure = 0) with NaN. For the BRFSS dataset, it stratifies and subsamples 253,680 rows down to 100,000 to optimize training.", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'We applied strict medical data hygiene—dropping duplicates and handling biological impossibilities—followed by strict train/test splitting to guarantee zero data leakage.'", body_style))

    # Module 3
    story.append(Paragraph("Module 3: Ensemble Feature Selection", heading_style))
    story.append(Paragraph("<b>What it does:</b> Mathematically selects the most predictive clinical features while dropping redundant ones.", body_style))
    story.append(Paragraph("<b>How it works:</b> Runs a '6-Council' test (Mutual Info, RF Importance, XGBoost Gain, Permutation, SHAP, and CV Stability). It averages the ranks. It then calculates the Spearman correlation matrix and drops features that are >90% correlated with stronger features (Redundancy Pruning).", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'Rather than guessing, we mathematically ranked features using 6 different algorithms, pruning redundant data to isolate the strongest clinical signals.'", body_style))

    # Module 4
    story.append(Paragraph("Module 4: Classical Scaling & Imputation", heading_style))
    story.append(Paragraph("<b>What it does:</b> Standardizes the numbers so classical machine learning models treat all features equally.", body_style))
    story.append(Paragraph("<b>How it works:</b> Uses <i>SimpleImputer</i> (median strategy) to fill missing values, and <i>StandardScaler</i> to map data to a mean of 0 and std-dev of 1. Crucially, these are fitted ONLY on the training split.", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'We applied median imputation and Z-score scaling fitted strictly on training data, ensuring the model treats Age and Cholesterol with equal mathematical weight without peaking at the test set.'", body_style))

    # Module 5
    story.append(Paragraph("Module 5: Quantum Encoding Matrix Generation", heading_style))
    story.append(Paragraph("<b>What it does:</b> Translates standard 2D data into quantum state formats (physical angles and unit vectors).", body_style))
    story.append(Paragraph("<b>How it works:</b>", bold_body))
    story.append(Paragraph("1. <b>Angle Encoding:</b> Uses MinMaxScaler to map features into the [0, π] range to act as physical qubit rotation angles.", body_style))
    story.append(Paragraph("2. <b>Amplitude Encoding:</b> Because amplitude vectors require dimensions to be powers of 2, the pipeline zero-pads the data (e.g., 6 features padded to 8 dims) to fit on 3 qubits (2^3). It then applies L2-normalization so the vector magnitude equals 1.0 (a law of quantum mechanics).", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'Quantum simulators cannot read CSVs. Our pipeline automatically zero-pads and L2-normalizes the clinical data into a Hilbert space representation for PennyLane VQC ingestion.'", body_style))

    # Module 6
    story.append(Paragraph("Module 6: The Audit Trail (metadata.json)", heading_style))
    story.append(Paragraph("<b>What it does:</b> Generates a massive JSON receipt of the entire pipeline execution.", body_style))
    story.append(Paragraph("<b>How it works:</b> Logs original row counts, split ratios, exact dropped features, and SHA-256 hashes.", body_style))
    story.append(Paragraph("<b>Presentation Pitch:</b> 'We built an automated audit trail to guarantee complete transparency and reproducibility for our medical pipeline.'", body_style))

    doc.build(story)
    print(f"Success! Detailed report generated at: {pdf_path}")

if __name__ == "__main__":
    create_pdf()