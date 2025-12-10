import sqlite3
from pathlib import Path

# Path to the DB file
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "packaging.db"


def print_schema():
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print(" No tables found in the database.")
        return

    # Loop through each table and print its schema
    for table in tables:
        table_name = table[0]
        print(f"\n Table: {table_name}\n")

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        if not columns:
            print("  (No columns found)")
            continue

        for col in columns:
            col_name = col[1]
            col_type = col[2]
            print(f"  - {col_name}: {col_type}")

    conn.close()
    
if __name__ == "__main__":
    print_schema()
