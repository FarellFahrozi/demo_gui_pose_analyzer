import sqlite3
import os

db_path = 'test/kuro_posture.db'

if os.path.exists(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\nTables found: {[t[0] for t in tables]}")
    
    # Delete all data from each table
    for table in tables:
        table_name = table[0]
        if table_name != 'sqlite_sequence':
            cursor.execute(f"DELETE FROM {table_name}")
            print(f"✓ Deleted all data from table: {table_name}")
    
    conn.commit()
    
    # Verify deletion
    print("\n--- Verification ---")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for table in cursor.fetchall():
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count} rows")
    
    conn.close()
    print("\n✅ Database cleaned successfully!")
else:
    print(f"❌ Database file not found: {db_path}")
