import sqlite3
import pandas as pd
import os

DB_PATH = "data/packaging.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create materials table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS materials (
        MATERIAL_TYPE TEXT,
        STRENGTH INTEGER,
        WEIGHT_CAPACITY REAL,
        INDUSTRY_CATEGORY TEXT
    )
    """)

    # Create logs table
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

    # 🔥 INSERT DEFAULT MATERIALS IF EMPTY
    count = cursor.execute("SELECT COUNT(*) FROM materials").fetchone()[0]

    if count == 0:
        df = pd.read_csv("data/materials_data.csv")
        REQUIRED_COLUMNS = [
            "MATERIAL_TYPE",
            "STRENGTH",
            "WEIGHT_CAPACITY",
            "INDUSTRY_CATEGORY"
        ]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        if missing:
            print("❌ Missing columns in materials CSV:", missing)
            print("Available columns:", list(df.columns))
            return  # DO NOT CRASH APP

        df[REQUIRED_COLUMNS].to_sql(
            "materials",
            conn,
            if_exists="append",
            index=False
        )

        print("✅ Materials table initialized")


    conn.commit()
    conn.close()
