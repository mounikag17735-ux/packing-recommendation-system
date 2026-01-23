# analytics/cloud_db.py
import os
import sqlitecloud


def get_connection():
    url = os.getenv("SQLITECLOUD_URL")

    if not url:
        raise RuntimeError("❌ SQLITECLOUD_URL not set in environment variables")

    conn = sqlitecloud.connect(url)
    return conn
