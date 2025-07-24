import sqlite3
import os

def init_metadata_db(db_path: str):
    """create SQLite db and metadata table if they don't already exist"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            start_time REAL NOT NULL,
            duration REAL NOT NULL,
            domain TEXT,
            cluster INTEGER,
            caption TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_metadata_rows(db_path: str, rows: list[dict]):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tuples = [
        (
            row["id"],
            row["file_path"],
            row["start_time"],
            row["duration"],
            row.get("domain"),
            row.get("cluster"),
            row.get("caption")
        )
        for row in rows
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO metadata
        (id, file_path, start_time, duration, domain, cluster, caption)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, tuples)

    conn.commit()
    conn.close()

def get_metadata_by_id(db_path: str, index: int) -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM metadata WHERE id = ?
    """, (index,))
    
    result = cursor.fetchone()
    conn.close()

    return {
        "id": result[0],
        "file_path": result[1],
        "start_time": result[2],
        "duration": result[3],
        "domain": result[4],
        "cluster": result[5],
        "caption": result[6]
    }