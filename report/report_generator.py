from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from datetime import datetime

def generate_crispdm_pdf(
    pdf_path: str,
    label_stats: dict,
    project_title: str,
    description: str
):
    """
    使用 ReportLab 生成簡易 CRISP-DM 報告。
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    margin = 20 * mm
    x = margin
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, project_title)
    y -= 15 * mm

    # Meta
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 10 * mm

    # Description
    c.setFont("Helvetica", 12)
    text_obj = c.beginText()
    text_obj.setTextOrigin(x, y)
    text_obj.setLeading(16)
    text_obj.textLines(description)
    c.drawText(text_obj)
    y = text_obj.getY() - 10 * mm

    crispdm_text = """
    CRISP-DM 流程摘要：
    1. Business Understanding
    2. Data Understanding
    3. Data Preparation
    4. Modeling
    5. Evaluation
    6. Deployment
    """
    text_obj = c.beginText()
    text_obj.setTextOrigin(x, y)
    text_obj.setLeading(15)
    text_obj.textLines(crispdm_text.strip())
    c.drawText(text_obj)
    y = text_obj.getY() - 10 * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "資料切割與分類結果統計：")
    y -= 8 * mm

    c.setFont("Helvetica", 12)
    if label_stats:
        for label, count in sorted(label_stats.items(), key=lambda kv: kv[0]):
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 12)
            c.drawString(x + 5 * mm, y, f"Label {label}: {count} samples")
            y -= 7 * mm
    else:
        c.drawString(x + 5 * mm, y, "目前沒有統計資料（output 目錄為空）。")
        y -= 7 * mm

    c.showPage()
    c.save()
