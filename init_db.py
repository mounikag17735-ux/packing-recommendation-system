import sqlite3
import os

DB_PATH = "data/packaging.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        MATERIAL_TYPE TEXT,
        STRENGTH REAL,
        WEIGHT_CAPACITY REAL,
        INDUSTRY_CATEGORY TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        predicted_cost REAL,
        predicted_co2 REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
