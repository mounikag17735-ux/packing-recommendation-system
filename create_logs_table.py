import sqlite3

conn = sqlite3.connect("data/packaging.db")
cursor = conn.cursor()

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

conn.commit()
conn.close()
print("Table 'recommendation_logs' created successfully!")
