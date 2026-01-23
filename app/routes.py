from flask import Blueprint, render_template, request, jsonify
from analytics.recommend_material import recommend_material
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from flask import send_file
import matplotlib
matplotlib.use("Agg")  # important for Flask
import matplotlib.pyplot as plt
import os
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from openpyxl import Workbook
import io

main = Blueprint("main", __name__)


# ---------------------------------------
# Home Page (Input Form)
# ---------------------------------------
@main.route("/")
def home():
    return render_template("index.html")

@main.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("data/materials.db")
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    # Absolute path to static folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(base_dir, "static")

    co2_chart_path = os.path.join(static_path, "co2_chart.png")
    cost_chart_path = os.path.join(static_path, "cost_chart.png")

    # CO2 chart
    co2_avg = df.groupby("material")["predicted_co2"].mean()
    plt.figure()
    co2_avg.plot(kind="bar")
    plt.title("Average CO2 Score by Material")
    plt.tight_layout()
    plt.savefig(co2_chart_path)
    plt.close()

    # Cost chart
    cost_avg = df.groupby("material")["predicted_cost"].mean()
    plt.figure()
    cost_avg.plot(kind="bar")
    plt.title("Average Cost Score by Material")
    plt.tight_layout()
    plt.savefig(cost_chart_path)
    plt.close()

        # Material usage trend (frequency)
    usage_count = df["material"].value_counts()

    usage_chart_path = os.path.join(static_path, "usage_chart.png")

    plt.figure()
    usage_count.plot(kind="bar")
    plt.title("Material Usage Frequency")
    plt.xlabel("Material")
    plt.ylabel("Number of Times Recommended")
    plt.tight_layout()
    plt.savefig(usage_chart_path)
    plt.close()

    return render_template("dashboard.html")


# ---------------------------------------
# Form Submission → Show Result Page
# ---------------------------------------
@main.route("/recommend", methods=["POST"])
def recommend():
    industry = request.form.get("industry")
    weight = float(request.form.get("weight"))
    fragility = float(request.form.get("fragility"))
    eco_priority = float(request.form.get("eco_priority"))

    result = recommend_material(fragility, weight, eco_priority, industry)

    return render_template("index.html", result=result)

# ---------------------------------------
# REST API Endpoint (JSON)
# ---------------------------------------
@main.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()

    fragility = int(data.get("fragility"))
    weight = float(data.get("weight"))
    eco_priority = float(data.get("eco_priority"))
    industry = data.get("industry")

    result = recommend_material(
        fragility=fragility,
        weight=weight,
        eco_priority=eco_priority,
        industry=industry,
    )

    return jsonify(result)
@main.route("/export/excel")
def export_excel():
    conn = sqlite3.connect("data/materials.db")
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Sustainability Report"

    ws.append(list(df.columns))

    for row in df.itertuples(index=False):
        ws.append(list(row))

    file_stream = io.BytesIO()
    wb.save(file_stream)
    file_stream.seek(0)

    return send_file(
        file_stream,
        as_attachment=True,
        download_name="sustainability_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
@main.route("/export/pdf")
def export_pdf():
    conn = sqlite3.connect("data/materials.db")
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    file_stream = io.BytesIO()
    doc = SimpleDocTemplate(file_stream, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("EcoPack AI - Sustainability Report", styles['Title']))
    elements.append(Spacer(1, 20))

    for _, row in df.iterrows():
        text = f"""
        Industry: {row['industry']} <br/>
        Material: {row['material']} <br/>
        Predicted Cost: {row['predicted_cost']} <br/>
        Predicted CO2: {row['predicted_co2']} <br/><br/>
        """
        elements.append(Paragraph(text, styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    file_stream.seek(0)

    return send_file(
        file_stream,
        as_attachment=True,
        download_name="sustainability_report.pdf",
        mimetype="application/pdf",
    )
