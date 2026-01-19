import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/packaging.db")
df = pd.read_sql("SELECT * FROM recommendation_logs", conn)
conn.close()

df["created_at"] = pd.to_datetime(df["created_at"])

material_usage = df["material_name"].value_counts()

plt.figure()
material_usage.plot(kind="bar")
plt.title("Material Usage Trend")
plt.xlabel("Material")
plt.ylabel("Number of Recommendations")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

co2_trend = df.groupby(df["created_at"].dt.date)["predicted_co2"].mean()

plt.figure()
co2_trend.plot(marker="o")
plt.title("Average CO₂ Impact Over Time")
plt.xlabel("Date")
plt.ylabel("Predicted CO₂")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(df["predicted_cost"], bins=10)
plt.title("Cost Distribution of Recommended Materials")
plt.xlabel("Predicted Cost")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
