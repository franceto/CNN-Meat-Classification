from pathlib import Path
from uuid import uuid4
import shutil
import subprocess
import zipfile
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import ImageDraw, ImageFont
from app.services.model_service import APP_DIR, OUTPUT_DIR, get_image_info, predict_image


BATCH_DIR = OUTPUT_DIR / "batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCHIVE_EXTS = {".zip", ".rar"}
MAX_BATCH_IMAGES = 80
def setup_pdf_fonts():
    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf"
    ]

    bold_candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf"
    ]

    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    bold_path = next((p for p in bold_candidates if Path(p).exists()), None)

    if font_path:
        pdfmetrics.registerFont(TTFont("VNFont", font_path))
        normal_font = "VNFont"
    else:
        normal_font = "Helvetica"

    if bold_path:
        pdfmetrics.registerFont(TTFont("VNFontBold", bold_path))
        bold_font = "VNFontBold"
    else:
        bold_font = normal_font

    return normal_font, bold_font


PDF_FONT, PDF_FONT_BOLD = setup_pdf_fonts()


PDF_FONT, PDF_FONT_BOLD = setup_pdf_fonts()
def pdf_text(text):
    return escape(str(text))


def make_pdf_styles():
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName=PDF_FONT_BOLD,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#5E1212"),
            spaceAfter=14
        ),
        "heading": ParagraphStyle(
            "HeadingCustom",
            parent=styles["Heading2"],
            fontName=PDF_FONT_BOLD,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#8B1E1E"),
            spaceBefore=8,
            spaceAfter=8
        ),
        "normal": ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontName=PDF_FONT,
            fontSize=10,
            leading=14
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=styles["BodyText"],
            fontName=PDF_FONT,
            fontSize=9,
            leading=12
        ),
        "table_bold": ParagraphStyle(
            "TableBold",
            parent=styles["BodyText"],
            fontName=PDF_FONT_BOLD,
            fontSize=9,
            leading=12
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontName=PDF_FONT_BOLD,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#5E1212")
        )
    }



def P(text, style):
    return Paragraph(pdf_text(text), style)


def PH(text, style):
    return Paragraph(str(text), style)
def make_prob_chart_image(probabilities, out_dir):
    w, h = 920, 240
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
        bold = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
        bold = ImageFont.load_default()

    items = [
        ("Fresh", probabilities.get("Fresh", 0)),
        ("Spoiled", probabilities.get("Spoiled", 0))
    ]

    bar_colors = {
        "Fresh": (139, 30, 30),
        "Spoiled": (255, 179, 92)
    }

    draw.text((30, 20), "Biểu đồ phân bố xác suất 2 class", fill=(94, 18, 18), font=bold)

    x0, y0 = 180, 85
    bw, bh = 560, 32

    for i, (label, prob) in enumerate(items):
        y = y0 + i * 65
        percent = prob * 100

        draw.text((35, y), label, fill=(36, 19, 19), font=font)
        draw.rounded_rectangle((x0, y, x0 + bw, y + bh), radius=13, fill=(245, 235, 230))
        draw.rounded_rectangle((x0, y, x0 + int(bw * prob), y + bh), radius=13, fill=bar_colors[label])
        draw.text((x0 + bw + 30, y), f"{percent:.2f}%", fill=(36, 19, 19), font=font)

    chart_path = out_dir / f"prob_chart_{uuid4().hex}.jpg"
    img.save(chart_path, quality=92, optimize=True)
    return chart_path
def find_7z():
    candidates = [
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe"
    ]

    for p in candidates:
        if p and Path(p).exists():
            return str(p)

    return None


def safe_name(name):
    return Path(name).name.replace("/", "_").replace("\\", "_")


def collect_images(root):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def extract_archive(archive_path, out_dir):
    suffix = archive_path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(out_dir)
        return

    if suffix == ".rar":
        seven_zip = find_7z()

        if not seven_zip:
            raise RuntimeError("File .rar cần cài 7-Zip và đảm bảo lệnh 7z dùng được trong PATH.")

        res = subprocess.run(
            [seven_zip, "x", str(archive_path), f"-o{out_dir}", "-y"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        if res.returncode != 0:
            raise RuntimeError(res.stderr or "Không giải nén được file .rar.")

        return

    raise RuntimeError("Chỉ hỗ trợ .zip hoặc .rar.")


def image_to_reportlab(path, max_w, max_h):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    w, h = img.size

    scale = min(max_w / w, max_h / h)
    return RLImage(str(path), width=w * scale, height=h * scale)


def make_prob_bar(probabilities):
    fresh = probabilities.get("Fresh", 0) * 100
    spoiled = probabilities.get("Spoiled", 0) * 100

    data = [
        ["Fresh", f"{fresh:.2f}%"],
        ["Spoiled", f"{spoiled:.2f}%"]
    ]

    table = Table(data, colWidths=[4 * cm, 4 * cm])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E3C9C1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E3C9C1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFF3EC")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF9F4")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#5E1212")),
        ("FONTNAME", (0, 0), (-1, -1), PDF_FONT_BOLD),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    return table


def make_pdf_report(results, report_path):
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    s = make_pdf_styles()

    story = []
    story.append(P("Báo cáo dự đoán hàng loạt - Meat Freshness Classifier", s["title"]))
    story.append(PH(f"Tổng số ảnh xử lý: <b>{len(results)}</b>", s["normal"]))
    story.append(Spacer(1, 10))

    summary_rows = [
        [P("Ảnh", s["table_bold"]), P("Dự đoán", s["table_bold"]), P("Độ tin cậy", s["table_bold"])]
    ]

    for r in results:
        summary_rows.append([
            P(r["info"]["name"], s["table"]),
            P(r["class"], s["table"]),
            P(f"{r['confidence'] * 100:.2f}%", s["table"])
        ])

    summary_table = Table(summary_rows, colWidths=[8 * cm, 4 * cm, 4 * cm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B1E1E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E3C9C1")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(summary_table)
    story.append(PageBreak())

    for i, r in enumerate(results, 1):
        info = r["info"]

        story.append(P(f"{i}. {info['name']}", s["heading"]))

        info_table = Table([
            [P("Tên ảnh", s["table_bold"]), P(info["name"], s["table"])],
            [P("Kích thước", s["table_bold"]), P(f"{info['size_kb']} KB", s["table"])],
            [P("Định dạng", s["table_bold"]), P(info["format"], s["table"])],
            [P("Độ phân giải", s["table_bold"]), P(f"{info['width']} × {info['height']} px", s["table"])],
        ], colWidths=[4 * cm, 12 * cm])

        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FFF3EC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E3C9C1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E3C9C1")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 8))

        img_table = Table([
            [
                image_to_reportlab(r["image_path"], 8 * cm, 6 * cm),
                image_to_reportlab(r["overlay_path"], 8 * cm, 6 * cm)
            ],
            [
                P("Ảnh gốc", s["caption"]),
                P("Grad-CAM overlay", s["caption"])
            ]
        ], colWidths=[8.2 * cm, 8.2 * cm])

        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))

        story.append(img_table)
        story.append(Spacer(1, 8))

        story.append(PH(
            f"Kết quả: <b>{pdf_text(r['class'])}</b> - Độ tin cậy: <b>{r['confidence'] * 100:.2f}%</b>",
            s["normal"]
        ))

        story.append(Spacer(1, 8))
        story.append(P("Biểu đồ phân bố xác suất 2 class:", s["normal"]))

        chart_path = make_prob_chart_image(r["probabilities"], report_path.parent)
        story.append(image_to_reportlab(chart_path, 16 * cm, 5 * cm))

        if i < len(results):
            story.append(PageBreak())

    doc.build(story)


def process_batch_from_paths(image_paths):
    if len(image_paths) == 0:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ.")

    if len(image_paths) > MAX_BATCH_IMAGES:
        raise RuntimeError(f"Số ảnh tối đa cho mỗi lần dự đoán là {MAX_BATCH_IMAGES} ảnh.")

    results = []

    for img_path in image_paths:
        info = get_image_info(img_path)
        pred = predict_image(img_path)

        overlay_rel = pred["overlay_url"].replace("/static/", "")
        overlay_path = APP_DIR / "static" / overlay_rel

        results.append({
            "info": info,
            "image_path": img_path,
            "overlay_path": overlay_path,
            "class": pred["class"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "overlay_url": pred["overlay_url"]
        })

    report_name = f"batch_report_{uuid4().hex}.pdf"
    report_path = OUTPUT_DIR / report_name
    make_pdf_report(results, report_path)

    return {
        "count": len(results),
        "report_url": f"/download_report/{report_name}",
        "results": [
            {
                "name": r["info"]["name"],
                "format": r["info"]["format"],
                "size_kb": r["info"]["size_kb"],
                "resolution": f"{r['info']['width']} × {r['info']['height']}",
                "class": r["class"],
                "confidence": r["confidence"],
                "probabilities": r["probabilities"],
                "overlay_url": r["overlay_url"]
            }
            for r in results
        ]
    }


def process_batch_uploads(saved_files):
    work_dir = BATCH_DIR / f"batch_{uuid4().hex}"
    input_dir = work_dir / "inputs"
    extract_dir = work_dir / "extract"
    input_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for file_path in saved_files:
        suffix = file_path.suffix.lower()

        if suffix in IMAGE_EXTS:
            image_paths.append(file_path)

        elif suffix in ARCHIVE_EXTS:
            extract_archive(file_path, extract_dir)
            image_paths.extend(collect_images(extract_dir))

    image_paths = sorted(set(image_paths), key=lambda p: str(p).lower())
    return process_batch_from_paths(image_paths)
