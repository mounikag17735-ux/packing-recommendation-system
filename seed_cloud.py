import sqlite3
import pandas as pd
from analytics.cloud_db import get_connection

print("🌱 Seeding SQLiteCloud from local materials.db...")

# 1. Read local DB
local_conn = sqlite3.connect("data/materials.db")
df = pd.read_sql("SELECT * FROM materials", local_conn)
local_conn.close()

print(f"Local rows: {len(df)}")

# 2. Connect to cloud
cloud_conn = get_connection()
cursor = cloud_conn.cursor()

# 3. Clear cloud table FIRST (very important)
cursor.execute("DELETE FROM materials")
cloud_conn.commit()

# 4. Insert rows manually (not using to_sql)
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        row["material_type"],
        row["industry_category"],
        row["strength"],
        row["weight_capacity"],
        row["biodegradability_score"],
        row["co2_emission_score"],
        row["recyclability_percent"],
    ))

cloud_conn.commit()
cloud_conn.close()

print("✅ Cloud DB seeded successfully!")
