import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'database', 'forensic.db')

print("=== Database Tables ===")

try:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        for table in tables:
            print(f"Table: {table[0]}")
            columns = cursor.execute(
                f"PRAGMA table_info({table[0]})"
            ).fetchall()
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            print()

except sqlite3.Error as e:
    print(f"Database error: {e}")