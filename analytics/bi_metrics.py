import sqlite3
import pandas as pd

DB_PATH = "data/packaging.db"


def get_bi_metrics():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
    conn.close()

    if df.empty:
        return {
            "total_recommendations": 0,
            "average_cost": 0,
            "average_co2": 0,
            "baseline_cost": 0,
            "baseline_co2": 0,
            "cost_savings_percent": 0,
            "co2_reduction_percent": 0,
        }

    avg_cost = df["predicted_cost"].mean()
    avg_co2 = df["predicted_co2"].mean()

    baseline_cost = df["predicted_cost"].max()
    baseline_co2 = df["predicted_co2"].max()

    cost_savings_pct = ((baseline_cost - avg_cost) / baseline_cost) * 100
    co2_reduction_pct = ((baseline_co2 - avg_co2) / baseline_co2) * 100

    return {
        "total_recommendations": len(df),
        "average_cost": round(avg_cost, 2),
        "average_co2": round(avg_co2, 2),
        "baseline_cost": round(baseline_cost, 2),
        "baseline_co2": round(baseline_co2, 2),
        "cost_savings_percent": round(cost_savings_pct, 2),
        "co2_reduction_percent": round(co2_reduction_pct, 2),
    }
