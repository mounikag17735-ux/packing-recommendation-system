import os
import sqlite3
import pandas as pd
import matplotlib

# 🔥 Headless backend (required)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- PATH FIX (CRITICAL) --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def p(*paths):
    return os.path.join(BASE_DIR, *paths)


DB_PATH = p("data", "packaging.db")
CHART_DIR = p("static", "charts")


# -------------------- MAIN FUNCTION --------------------
def generate_charts():
    os.makedirs(CHART_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    if df.empty:
        print("⚠️ No data available for charts")
        return

    df["created_at"] = pd.to_datetime(df["created_at"])

    # ---------------- Material Usage ----------------
    material_usage = df["material_name"].dropna().value_counts()

    if not material_usage.empty:
        plt.figure()
        material_usage.plot(kind="bar")
        plt.title("Material Usage Trend")
        plt.xlabel("Material")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(p("static", "charts", "material_usage.png"))
        plt.close()

    # ---------------- CO₂ Trend ----------------
    co2_trend = (
        df.dropna(subset=["predicted_co2"])
          .groupby(df["created_at"].dt.date)["predicted_co2"]
          .mean()
    )

    if not co2_trend.empty:
        plt.figure()
        co2_trend.plot(marker="o")
        plt.title("Average CO₂ Impact Over Time")
        plt.xlabel("Date")
        plt.ylabel("Predicted CO₂")
        plt.tight_layout()
        plt.savefig(p("static", "charts", "co2_trend.png"))
        plt.close()

    # ---------------- Cost Distribution ----------------
    costs = df["predicted_cost"].dropna()

    if not costs.empty:
        plt.figure()
        plt.hist(costs, bins=10)
        plt.title("Cost Distribution")
        plt.xlabel("Predicted Cost")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(p("static", "charts", "cost_distribution.png"))
        plt.close()

    print("✅ Charts generated successfully at:", CHART_DIR)
