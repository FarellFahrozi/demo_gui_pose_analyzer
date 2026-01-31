import sqlite3
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
db_path = os.path.join(base_dir, 'kuro_posture.db')

if not os.path.exists(db_path):
    print("Database file does not exist.")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(patients)")
    columns = cursor.fetchall()
    print("Columns in patients table:")
    for col in columns:
        print(col)
    conn.close()
