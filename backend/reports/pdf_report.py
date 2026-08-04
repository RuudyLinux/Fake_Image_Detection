"""
PDF report generation for blind-detection results.
Summarizes verdict, class probabilities, signal breakdown, and the
uploaded image side-by-side with its Grad-CAM/heuristic heatmap (if any).
"""

import io
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether,
)

STATIC_DIR = Path(__file__).parent.parent / "static"

VERDICT_COLORS = {
    "REAL": colors.HexColor("#22c55e"),
    "AI_GENERATED": colors.HexColor("#ef4444"),
    "EDITED": colors.HexColor("#f97316"),
}

SIGNAL_LABELS = {
    "spectral_anomaly": "Spectral 1/f Deviation",
    "gan_upsampling": "GAN Upsampling Artifact",
    "noise_inconsistency": "Noise Inconsistency",
    "texture_boundary": "Texture Boundary Anomaly",
    "edge_sharpness": "Edge/Gradient Anomaly",
    "none": "None",
}


def generate_blind_report(image_bytes: bytes, result: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleFG", parent=styles["Title"], textColor=colors.HexColor("#7c3aed"),
    )
    verdict_color = VERDICT_COLORS.get(result["prediction"], colors.black)
    verdict_style = ParagraphStyle(
        "Verdict", parent=styles["Heading1"], textColor=verdict_color, fontSize=22,
    )

    story = []
    story.append(Paragraph("FakeGuard AI — Detection Report", title_style))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated {timestamp}", styles["Normal"]))
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph(f"Verdict: {result['prediction'].replace('_', ' ')}", verdict_style))
    story.append(Paragraph(
        f"Confidence: {result['confidence'] * 100:.1f}%  ·  "
        f"Primary signal: {SIGNAL_LABELS.get(result['forgery_type'], result['forgery_type'])}",
        styles["Normal"],
    ))
    if result.get("signal_agreement") is not None:
        agreement = "agree" if result["signal_agreement"] else "disagree"
        story.append(Paragraph(f"CNN and heuristic signals {agreement} on this verdict.", styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    # class probabilities table
    probs = result["class_probabilities"]
    prob_rows = [["Class", "Probability"]] + [
        [cls.replace("_", " "), f"{p * 100:.1f}%"] for cls, p in probs.items()
    ]
    prob_table = Table(prob_rows, colWidths=[80 * mm, 40 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("Class Probabilities", styles["Heading3"]))
    story.append(prob_table)
    story.append(Spacer(1, 8 * mm))

    # signal analysis table
    analysis = result["analysis"]
    signal_rows = [["Signal", "Score"]] + [
        [k.replace("_", " ").title(), f"{v:.3f}"] for k, v in analysis.items()
    ]
    signal_table = Table(signal_rows, colWidths=[80 * mm, 40 * mm])
    signal_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("Signal Analysis Breakdown", styles["Heading3"]))
    story.append(signal_table)
    story.append(Spacer(1, 8 * mm))

    # images: uploaded + heatmap side by side
    img_row = [RLImage(io.BytesIO(image_bytes), width=75 * mm, height=75 * mm, kind="proportional")]
    labels = ["Uploaded Image"]
    heatmap_url = result.get("heatmap_url")
    if heatmap_url:
        heatmap_path = STATIC_DIR / Path(heatmap_url).name
        if heatmap_path.exists():
            img_row.append(RLImage(str(heatmap_path), width=75 * mm, height=75 * mm, kind="proportional"))
            labels.append("Anomaly Heatmap")

    story.append(KeepTogether([
        Paragraph("Visual Evidence", styles["Heading3"]),
        Table([img_row, labels], colWidths=[80 * mm] * len(img_row)),
    ]))

    doc.build(story)
    return buf.getvalue()


def generate_comparative_report(original_bytes: bytes, suspected_bytes: bytes, result: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleFG", parent=styles["Title"], textColor=colors.HexColor("#7c3aed"),
    )
    is_manipulated = result["is_manipulated"]
    verdict_color = colors.HexColor("#ef4444") if is_manipulated else colors.HexColor("#22c55e")
    verdict_style = ParagraphStyle(
        "Verdict", parent=styles["Heading1"], textColor=verdict_color, fontSize=22,
    )

    story = []
    story.append(Paragraph("FakeGuard AI — Comparative Detection Report", title_style))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated {timestamp}", styles["Normal"]))
    story.append(Spacer(1, 10 * mm))

    verdict_label = "MANIPULATED" if is_manipulated else "AUTHENTIC"
    story.append(Paragraph(f"Verdict: {verdict_label}", verdict_style))
    story.append(Paragraph(
        f"Similarity (SSIM): {result['similarity_score'] * 100:.1f}%  ·  "
        f"Altered regions detected: {result['regions_count']}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 8 * mm))

    metric_rows = [
        ["Metric", "Value"],
        ["Structural Similarity (SSIM)", f"{result['similarity_score'] * 100:.1f}%"],
        ["Perceptual hash distance (pHash)", f"{result['phash_distance']} / 64"],
        ["Difference hash distance (dHash)", f"{result['dhash_distance']} / 64"],
        ["Altered regions", str(result["regions_count"])],
    ]
    metric_table = Table(metric_rows, colWidths=[100 * mm, 40 * mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1b4b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph("Comparison Metrics", styles["Heading3"]))
    story.append(metric_table)
    story.append(Spacer(1, 8 * mm))

    img_row = [
        RLImage(io.BytesIO(original_bytes), width=38 * mm, height=38 * mm, kind="proportional"),
        RLImage(io.BytesIO(suspected_bytes), width=38 * mm, height=38 * mm, kind="proportional"),
    ]
    labels = ["Original Image", "Suspected Image"]

    diff_url = result.get("diff_map_url")
    if diff_url:
        diff_path = STATIC_DIR / Path(diff_url).name
        if diff_path.exists():
            img_row.append(RLImage(str(diff_path), width=38 * mm, height=38 * mm, kind="proportional"))
            labels.append("Difference Map")

    highlighted_url = result.get("highlighted_url")
    if highlighted_url:
        highlighted_path = STATIC_DIR / Path(highlighted_url).name
        if highlighted_path.exists():
            img_row.append(RLImage(str(highlighted_path), width=38 * mm, height=38 * mm, kind="proportional"))
            labels.append("Highlighted Regions")

    col_width = min(40 * mm, 170 * mm / len(img_row))
    story.append(KeepTogether([
        Paragraph("Visual Evidence", styles["Heading3"]),
        Table([img_row, labels], colWidths=[col_width] * len(img_row)),
    ]))

    doc.build(story)
    return buf.getvalue()
