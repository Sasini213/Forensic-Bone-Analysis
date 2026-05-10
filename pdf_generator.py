"""
pdf_generator.py
Place this in project root: D:\Project\Bone_Gender_Age_Project\pdf_generator.py
(same folder as app.py)
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

DARK_BLUE   = colors.HexColor("#1a3558")
MID_BLUE    = colors.HexColor("#2563a8")
LIGHT_BLUE  = colors.HexColor("#dbeafe")
GREEN       = colors.HexColor("#16a34a")
LIGHT_GREEN = colors.HexColor("#dcfce7")
LIGHT_GREY  = colors.HexColor("#f1f5f9")
MID_GREY    = colors.HexColor("#94a3b8")
TEXT_DARK   = colors.HexColor("#1e293b")
TEXT_LIGHT  = colors.HexColor("#64748b")
WHITE       = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CW     = PAGE_W - 2 * MARGIN
HALF   = (CW - 8 * mm) / 2


def _s(name, **kw):
    return ParagraphStyle(name, **kw)


STYLES = {
    "title":            _s("title",    fontName="Helvetica-Bold", fontSize=20, textColor=WHITE, alignment=TA_CENTER, spaceAfter=4),
    "subtitle":         _s("subtitle", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#bfdbfe"), alignment=TA_CENTER, spaceAfter=2),
    "case_ref":         _s("case_ref", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER),
    "section":          _s("section",  fontName="Helvetica-Bold", fontSize=11, textColor=DARK_BLUE, spaceBefore=10, spaceAfter=4),
    "body":             _s("body",     fontName="Helvetica", fontSize=9, textColor=TEXT_DARK, spaceAfter=3, leading=14),
    "label":            _s("label",    fontName="Helvetica-Bold", fontSize=9, textColor=TEXT_DARK),
    "cell":             _s("cell",     fontName="Helvetica", fontSize=9, textColor=TEXT_DARK),
    "th":               _s("th",       fontName="Helvetica-Bold", fontSize=9, textColor=WHITE),
    "result_val_blue":  _s("rvb",      fontName="Helvetica-Bold", fontSize=18, textColor=MID_BLUE, alignment=TA_CENTER),
    "result_val_green": _s("rvg",      fontName="Helvetica-Bold", fontSize=18, textColor=GREEN, alignment=TA_CENTER),
    "conf":             _s("conf",     fontName="Helvetica", fontSize=9, textColor=TEXT_LIGHT, alignment=TA_CENTER),
    "result_label":     _s("rl",       fontName="Helvetica-Bold", fontSize=10, textColor=TEXT_DARK),
    "disclaimer":       _s("disc",     fontName="Helvetica-Oblique", fontSize=8, textColor=TEXT_LIGHT, leading=12, spaceBefore=4),
    "footer":           _s("footer",   fontName="Helvetica", fontSize=8, textColor=TEXT_LIGHT, alignment=TA_CENTER, spaceBefore=4),
}


def _fmt_conf(c):
    try:
        return f"{float(str(c).replace('%','')):.1f}%"
    except (ValueError, TypeError):
        return str(c)


def _header(prediction):
    data = [
        [Paragraph("FORENSIC BONE ANALYSIS SYSTEM", STYLES["title"])],
        [Paragraph("Sri Lanka Police / Government Analyst's Department", STYLES["subtitle"])],
        [Paragraph(f"Case Reference: {prediction.get('case_reference', 'N/A')}", STYLES["case_ref"])],
    ]
    t = Table(data, colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 1), DARK_BLUE),
        ("BACKGROUND",    (0, 2), (-1, 2), MID_BLUE),
        ("TOPPADDING",    (0, 0), (-1, 0), 14),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING",    (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("TOPPADDING",    (0, 2), (-1, 2), 6),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 6),
    ]))
    return t


def _meta_table(prediction, generated_by):
    ts = prediction.get("timestamp", "")
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y-%m-%d %H:%M:%S")

    left_rows = [
        ("Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Prediction Date:",  str(ts)),
        ("Bone Type:",        str(prediction.get("bone_type", "N/A")).replace("_", " ").title()),
        ("Analysis ID:",      str(prediction.get("id", "N/A"))),
    ]
    right_rows = [
        ("Analyst:",  str(generated_by)),
        ("System:",   "Forensic Bone Analysis v1.0"),
        ("Models:",   "Random Forest (scikit-learn)"),
        ("Dataset:",  "4,501 augmented records"),
    ]

    def build(rows, col_w, label_w):
        return Table(
            [[Paragraph(k, STYLES["label"]), Paragraph(v, STYLES["cell"])] for k, v in rows],
            colWidths=[label_w, col_w - label_w],
        )

    tbl = Table(
        [[build(left_rows, HALF, 36 * mm), build(right_rows, HALF, 28 * mm)]],
        colWidths=[HALF, HALF],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GREY),
    ]))
    return tbl


def _result_box(label, value, confidence, bg, val_style):
    border_color = MID_BLUE if bg == LIGHT_BLUE else GREEN
    inner = Table(
        [
            [Paragraph(label, STYLES["result_label"])],
            [Paragraph(str(value), val_style)],
            [Paragraph(f"Confidence: {confidence}", STYLES["conf"])],
        ],
        colWidths=[HALF],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 1.5, border_color),
    ]))
    return inner


def _results_section(prediction):
    sex   = str(prediction.get("predicted_sex",  prediction.get("gender",             "N/A"))).title()
    age   = str(prediction.get("predicted_age",  prediction.get("age_group",          "N/A")))
    sex_c = _fmt_conf(prediction.get("sex_confidence", prediction.get("gender_confidence", "N/A")))
    age_c = _fmt_conf(prediction.get("age_confidence", "N/A"))

    tbl = Table(
        [[
            _result_box("BIOLOGICAL SEX", sex, sex_c, LIGHT_BLUE,  STYLES["result_val_blue"]),
            _result_box("AGE GROUP",      age, age_c, LIGHT_GREEN, STYLES["result_val_green"]),
        ]],
        colWidths=[HALF, HALF],
    )
    tbl.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def _measurements_table(measurements):
    if not measurements:
        return Paragraph("No measurement data recorded.", STYLES["body"])

    rows = [[Paragraph("Measurement", STYLES["th"]), Paragraph("Value (mm)", STYLES["th"])]]
    for key, val in measurements.items():
        label = str(key).replace("_", " ").upper()
        value = str(val) if val not in (None, "", "None", 0, "0") else "N/A"
        rows.append([Paragraph(label, STYLES["cell"]), Paragraph(value, STYLES["cell"])])

    tbl = Table(rows, colWidths=[CW * 0.65, CW * 0.35])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  DARK_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("ALIGN",          (1, 0), (1, -1),  "CENTER"),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("BOX",            (0, 0), (-1, -1), 0.5, MID_GREY),
        ("LINEBELOW",      (0, 0), (-1, -1), 0.3, MID_GREY),
    ]))
    return tbl


def generate_prediction_pdf(prediction: dict, generated_by: str = "System") -> bytes:
    """
    Build a forensic case PDF in memory and return the raw bytes.

    Accepts these dict keys (flexible naming for compatibility):
        id, case_reference, bone_type, timestamp,
        predicted_sex  OR gender
        predicted_age  OR age_group
        sex_confidence OR gender_confidence
        age_confidence
        measurements   (dict  {field: value})
        notes          (str, optional)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"Forensic Report - {prediction.get('case_reference', '')}",
        author="Forensic Bone Analysis System",
    )

    story = []

    story.append(_header(prediction))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("REPORT INFORMATION", STYLES["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=4))
    story.append(_meta_table(prediction, generated_by))
    story.append(Spacer(1, 5 * mm))

    story.append(KeepTogether([
        Paragraph("PREDICTION RESULTS", STYLES["section"]),
        HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=4),
        _results_section(prediction),
    ]))
    story.append(Spacer(1, 5 * mm))

    story.append(KeepTogether([
        Paragraph("BONE MEASUREMENTS ENTERED", STYLES["section"]),
        HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=4),
        _measurements_table(prediction.get("measurements", {})),
    ]))
    story.append(Spacer(1, 5 * mm))

    notes = str(prediction.get("notes", "") or "").strip()
    if notes:
        story.append(KeepTogether([
            Paragraph("ANALYST NOTES", STYLES["section"]),
            HRFlowable(width="100%", thickness=1, color=MID_BLUE, spaceAfter=4),
            Paragraph(notes, STYLES["body"]),
        ]))
        story.append(Spacer(1, 5 * mm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-assisted decision support system. "
        "Predictions are based on statistical models trained on skeletal morphology data and are "
        "intended to assist, not replace, the judgement of a qualified forensic expert. "
        "All conclusions must be verified by a certified forensic anthropologist before use "
        "in any legal or investigative proceeding.",
        STYLES["disclaimer"]
    ))
    story.append(Paragraph(
        f"Forensic Bone Analysis System  |  {datetime.now().strftime('%Y-%m-%d')}",
        STYLES["footer"]
    ))

    doc.build(story)
    return buffer.getvalue()
