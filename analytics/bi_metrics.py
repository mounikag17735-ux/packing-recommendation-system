import sqlite3
import pandas as pd

conn = sqlite3.connect("data/packaging.db")
df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
conn.close()

print("Total records:", len(df))

if df.empty:
    print("No recommendation logs found.")
    exit()

material_usage = df["material_name"].value_counts()

print("\nMaterial Usage Trends:")
print(material_usage)

avg_cost = df["predicted_cost"].mean()
avg_co2 = df["predicted_co2"].mean()

print("\nAverage Predicted Cost:", round(avg_cost, 2))
print("Average Predicted CO₂:", round(avg_co2, 2))

baseline_cost = df["predicted_cost"].max()
baseline_co2 = df["predicted_co2"].max()

cost_savings_pct = ((baseline_cost - avg_cost) / baseline_cost) * 100
co2_reduction_pct = ((baseline_co2 - avg_co2) / baseline_co2) * 100

print("\nBaseline Cost:", round(baseline_cost, 2))
print("Baseline CO₂:", round(baseline_co2, 2))

print("\nEstimated Cost Savings (%):", round(cost_savings_pct, 2))
print("Estimated CO₂ Reduction (%):", round(co2_reduction_pct, 2))

bi_summary = {
    "total_recommendations": len(df),
    "average_cost": round(avg_cost, 2),
    "average_co2": round(avg_co2, 2),
    "baseline_cost": round(baseline_cost, 2),
    "baseline_co2": round(baseline_co2, 2),
    "cost_savings_percent": round(cost_savings_pct, 2),
    "co2_reduction_percent": round(co2_reduction_pct, 2)
}

print("\nBI Summary:")
for k, v in bi_summary.items():
    print(f"{k}: {v}")
