import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ✅ IMPORTANT (no GUI)
import matplotlib.pyplot as plt
import os

DB_PATH = "data/packaging.db"
CHART_DIR = "static/charts"

os.makedirs(CHART_DIR, exist_ok=True)

def generate_charts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    if df.empty:
        return

    df["created_at"] = pd.to_datetime(df["created_at"])

    # ---------------- Material Usage ----------------
    material_usage = df["material_name"].value_counts()

    plt.figure(figsize=(6, 4))
    material_usage.plot(kind="bar")
    plt.title("Material Usage Trend")
    plt.xlabel("Material")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/material_usage.png")
    plt.close()

    # ---------------- CO2 Trend ----------------
    co2_trend = df.groupby(df["created_at"].dt.date)["predicted_co2"].mean()

    plt.figure(figsize=(6, 4))
    co2_trend.plot(marker="o")
    plt.title("Average CO₂ Impact Over Time")
    plt.xlabel("Date")
    plt.ylabel("Predicted CO₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/co2_trend.png")
    plt.close()

    # ---------------- Cost Distribution ----------------
    plt.figure(figsize=(6, 4))
    plt.hist(df["predicted_cost"], bins=10)
    plt.title("Cost Distribution")
    plt.xlabel("Predicted Cost")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/cost_distribution.png")
    plt.close()
