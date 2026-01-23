import sqlite3

DB_PATH = "data/materials.db"

def create_logs_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        industry TEXT,
        material TEXT,
        predicted_cost REAL,
        predicted_co2 REAL
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_logs_table()
