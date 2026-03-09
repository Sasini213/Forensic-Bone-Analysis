import sqlite3

conn = sqlite3.connect('database/forensic.db')
cursor = conn.cursor()

print("=== Database Tables ===")
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for table in tables:
    print(f"Table: {table[0]}")
    columns = cursor.execute(f"PRAGMA table_info({table[0]})").fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    print()

conn.close()