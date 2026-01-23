from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from analytics.recommend_material import recommend_material
from analytics.cloud_db import get_connection

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from openpyxl import Workbook
import io

main = Blueprint("main", __name__)


@main.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------
# Dashboard (Charts)
# ---------------------------------------
@main.route("/dashboard")
def dashboard():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    # Prevent crash when no data
    if df.empty:
        return render_template(
            "dashboard.html",
            message="No recommendations yet. Please run a recommendation first."
        )

    df["predicted_co2"] = pd.to_numeric(df["predicted_co2"], errors="coerce")
    df["predicted_cost"] = pd.to_numeric(df["predicted_cost"], errors="coerce")

    static_path = current_app.static_folder
    os.makedirs(static_path, exist_ok=True)

    co2_chart_path = os.path.join(static_path, "co2_chart.png")
    cost_chart_path = os.path.join(static_path, "cost_chart.png")
    usage_chart_path = os.path.join(static_path, "usage_chart.png")

    # CO2 chart
    plt.figure()
    df.groupby("material")["predicted_co2"].mean().plot(kind="bar")
    plt.title("Average CO2 Score by Material")
    plt.tight_layout()
    plt.savefig(co2_chart_path)
    plt.close()

    # Cost chart
    plt.figure()
    df.groupby("material")["predicted_cost"].mean().plot(kind="bar")
    plt.title("Average Cost Score by Material")
    plt.tight_layout()
    plt.savefig(cost_chart_path)
    plt.close()

    # Usage chart
    plt.figure()
    df["material"].value_counts().plot(kind="bar")
    plt.title("Material Usage Frequency")
    plt.tight_layout()
    plt.savefig(usage_chart_path)
    plt.close()

    return render_template("dashboard.html")


# ---------------------------------------
# Recommendation
# ---------------------------------------
@main.route("/recommend", methods=["POST"])
def recommend():
    result = recommend_material(
        fragility=float(request.form.get("fragility")),
        weight=float(request.form.get("weight")),
        eco_priority=float(request.form.get("eco_priority")),
        industry=request.form.get("industry"),
    )
    return render_template("index.html", result=result)


# ---------------------------------------
# API Endpoint
# ---------------------------------------
@main.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()

    result = recommend_material(
        fragility=float(data.get("fragility")),
        weight=float(data.get("weight")),
        eco_priority=float(data.get("eco_priority")),
        industry=data.get("industry"),
    )

    return jsonify(result)


# ---------------------------------------
# Export Excel
# ---------------------------------------
@main.route("/export/excel")
def export_excel():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.append(list(df.columns))

    for row in df.itertuples(index=False):
        ws.append(list(row))

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        download_name="sustainability_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------
# Export PDF
# ---------------------------------------
@main.route("/export/pdf")
def export_pdf():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    stream = io.BytesIO()
    doc = SimpleDocTemplate(stream, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("EcoPack AI - Sustainability Report", styles["Title"]),
        Spacer(1, 20),
    ]

    for _, row in df.iterrows():
        text = f"""
        Industry: {row['industry']} <br/>
        Material: {row['material']} <br/>
        Predicted Cost: {row['predicted_cost']} <br/>
        Predicted CO2: {row['predicted_co2']} <br/><br/>
        """
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        download_name="sustainability_report.pdf",
        mimetype="application/pdf",
    )
