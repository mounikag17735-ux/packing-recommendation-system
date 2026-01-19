import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "data/packaging.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def load_logs():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT
            material_name,
            predicted_cost,
            predicted_co2,
            created_at
        FROM recommendation_logs
        ORDER BY created_at ASC
    """, conn)
    conn.close()
    return df

def export_excel_report(df):
    os.makedirs("reports", exist_ok=True)

    filename = f"reports/sustainability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Recommendations")

        # Summary sheet
        summary = pd.DataFrame({
            "Metric": [
                "Average Predicted Cost",
                "Average Predicted CO₂",
                "Total Recommendations"
            ],
            "Value": [
                round(df["predicted_cost"].mean(), 2),
                round(df["predicted_co2"].mean(), 2),
                len(df)
            ]
        })

        summary.to_excel(writer, index=False, sheet_name="Summary")

    print(f"✅ Excel report generated: {filename}")


if __name__ == "__main__":
    df = load_logs()

    if df.empty:
        print("⚠️ No recommendation data found.")
    else:
        export_excel_report(df)
