import sqlite3
import pandas as pd
import os

DB_PATH = "data/packaging.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---------- CREATE TABLES ----------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            MATERIAL_ID INTEGER,
            MATERIAL_TYPE TEXT,
            STRENGTH REAL,
            WEIGHT_CAPACITY REAL,
            BIODEGRADABILITY_SCORE REAL,
            Co2_EMISSION_SCORE REAL,
            RECYCLABILITY_PERCENTAGE REAL,
            INDUSTRY_CATEGORY TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_name TEXT,
            predicted_cost REAL,
            predicted_co2 REAL,
            eco_priority REAL,
            fragility_level TEXT,
            industry TEXT,
            product_weight REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---------- SEED MATERIALS ----------
    count = cursor.execute("SELECT COUNT(*) FROM materials").fetchone()[0]

    if count == 0:
        df = pd.read_csv("data/materials_cleaned.csv")

        df[[
            "MATERIAL_ID",
            "MATERIAL_TYPE",
            "STRENGTH",
            "WEIGHT_CAPACITY",
            "BIODEGRADABILITY_SCORE",
            "Co2_EMISSION_SCORE",
            "RECYCLABILITY_PERCENTAGE",
            "INDUSTRY_CATEGORY"
        ]].to_sql("materials", conn, if_exists="append", index=False)

        print(f"✅ Seeded {len(df)} materials into DB")

    conn.commit()
    conn.close()
