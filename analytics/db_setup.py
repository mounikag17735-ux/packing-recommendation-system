import pandas as pd
from analytics.cloud_db import get_connection


def setup_database():
    print("🗄️ Setting up Cloud database...")

    conn = get_connection()
    cursor = conn.cursor()

    # Create table only if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_type TEXT,
            industry_category TEXT,
            strength REAL,
            weight_capacity REAL,
            biodegradability_score REAL,
            co2_emission_score REAL,
            recyclability_percent REAL
        )
    """)

    # Check if already seeded
    count = cursor.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
    if count > 0:
        print("✅ Cloud DB already seeded")
        conn.close()
        return

    print("🌱 Seeding Cloud DB from local CSV...")

    df = pd.read_csv("data/materials_data.csv")

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO materials (
                material_type,
                industry_category,
                strength,
                weight_capacity,
                biodegradability_score,
                co2_emission_score,
                recyclability_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["material_type"],
            row["industry_category"],
            row["strength"],
            row["weight_capacity"],
            row["biodegradability_score"],
            row["co2_emission_score"],
            row["recyclability_percent"],
        ))

    conn.commit()
    conn.close()
    print("✅ Cloud DB ready!")
