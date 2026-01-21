import sqlite3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

DB_PATH = "data/packaging.db"

def load_logs():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT material_name, predicted_cost, predicted_co2, created_at
        FROM recommendation_logs
    """).fetchall()
    conn.close()
    return rows

def generate_pdf():
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/sustainability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    data = load_logs()

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "EcoPack AI – Sustainability Report")

    y -= 40
    c.setFont("Helvetica", 10)

    for row in data:
        line = f"{row[0]} | Cost: {row[1]} | CO₂: {row[2]} | {row[3]}"
        c.drawString(50, y, line)
        y -= 15

        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return filename
