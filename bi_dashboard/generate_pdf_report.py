import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os

# Load data
DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "packaging.db"
)

def load_logs():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()
    return df

def generate_co2_chart(df, path):
    df["created_at"] = pd.to_datetime(df["created_at"])
    daily_avg = df.groupby(df["created_at"].dt.date)["predicted_co2"].mean()

    plt.figure()
    daily_avg.plot()
    plt.xlabel("Date")
    plt.ylabel("Average CO₂")
    plt.title("Average CO₂ Impact Over Time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def generate_pdf(df):
    os.makedirs("bi_dashboard/reports", exist_ok=True)
    os.makedirs("bi_dashboard/charts", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"bi_dashboard/reports/sustainability_report_{timestamp}.pdf"
    chart_path = "bi_dashboard/charts/co2_trend.png"

    generate_co2_chart(df, chart_path)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Packaging Sustainability Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary KPIs
    elements.append(Paragraph(
        f"""
        <b>Generated:</b> {datetime.now().strftime('%d %B %Y')}<br/>
        <b>Total Recommendations:</b> {len(df)}<br/>
        <b>Average Cost:</b> ${df['predicted_cost'].mean():.2f}<br/>
        <b>Average CO₂:</b> {df['predicted_co2'].mean():.2f}
        """,
        styles["Normal"]
    ))

    elements.append(Spacer(1, 15))

    # Chart
    elements.append(Paragraph("<b>CO₂ Impact Trend</b>", styles["Heading2"]))
    elements.append(Image(chart_path, width=400, height=250))

    elements.append(Spacer(1, 15))

    # Top materials table
    top_materials = df[["material_name", "predicted_cost", "predicted_co2"]].head(5)
    table_data = [["Material", "Cost", "CO₂"]] + top_materials.values.tolist()

    elements.append(Paragraph("<b>Top Recommended Materials</b>", styles["Heading2"]))
    elements.append(Table(table_data))

    # Executive summary
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "This report summarizes sustainable packaging recommendations based on cost efficiency and environmental impact. "
        "The selected materials prioritize low carbon footprint while maintaining economic feasibility.",
        styles["Normal"]
    ))

    doc.build(elements)
    print(f"✅ PDF report generated: {pdf_path}")

if __name__ == "__main__":
    df = load_logs()
    generate_pdf(df)
