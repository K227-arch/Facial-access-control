import sqlite3
import os
from datetime import datetime, timedelta

# Ensure instance folder exists
os.makedirs("instance", exist_ok=True)
DB_PATH = "instance/face_recognition.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create table if not exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='FaceRecords'")
    if not cur.fetchone():
        cur.execute("""
            CREATE TABLE FaceRecords (
                RecordID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Confidence REAL,
                Timestamp TEXT,
                Status TEXT,
                BoundingBox TEXT,
                Embedding BLOB
            )
        """
        )
        conn.commit()
    conn.close()


def save_face_record(name, confidence, bbox, embedding, timestamp=None):
    timestamp = timestamp or datetime.now().isoformat(sep=' ', timespec='seconds')
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO FaceRecords 
        (Name, Confidence, Timestamp, Status, BoundingBox, Embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            confidence,
            timestamp,
            'recognized',
            str(bbox),
            embedding.tobytes()
        )
    )
    conn.commit()
    conn.close()


def get_records(time_range='all'):
    conn = get_db_connection()
    cur = conn.cursor()
    query = "SELECT Name, Confidence, BoundingBox, Status, Timestamp FROM FaceRecords"
    params = []
    now = datetime.now()
    if time_range == 'today':
        today = now.strftime('%Y-%m-%d')
        query += " WHERE Timestamp LIKE ?"
        params.append(f"{today}%")
    elif time_range == 'week':
        start_week = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        query += " WHERE Timestamp >= ?"
        params.append(f"{start_week} 00:00:00")
    elif time_range == 'month':
        ym = now.strftime('%Y-%m')
        query += " WHERE Timestamp LIKE ?"
        params.append(f"{ym}%")
    query += " ORDER BY Timestamp DESC"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    records = []
    for row in rows:
        rec = dict(row)
        rec['Confidence'] = round(rec['Confidence'],2)
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(rec['Timestamp'])
            rec['Timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
        records.append(rec)
    return records


def get_kpi_counts(records):
    total = len(records)
    statuses = [r['Status'].lower() for r in records]
    return {
        'recognized': statuses.count('recognized'),
        'unknown': statuses.count('unknown'),
        'entering': statuses.count('entering'),
        'exiting': statuses.count('exiting'),
        'total': total
    }


def get_peak_hours(time_range='all'):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT substr(Timestamp,12,2) AS hour, COUNT(*) AS cnt "
        "FROM FaceRecords GROUP BY hour ORDER BY cnt DESC LIMIT 4"
    )
    peaks = [{'hour': f"{row['hour']}:00", 'count': row['cnt']} for row in cur.fetchall()]
    conn.close()
    return peaks


def get_recent_incidents(limit=3):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT Status, Name, Description, Timestamp FROM Incidents  "
        "WHERE Status != 'recognized' ORDER BY Timestamp DESC LIMIT ?",
        (limit,)
    )
    incs = [{'status': row['Status'], 'name': row['Name'], 'timestamp': row['Timestamp'], 'description': row['Description']} for row in cur.fetchall()]
    conn.close()
    return incs

def save_incident(name, status, description=None, timestamp=None):
    timestamp = timestamp or datetime.now().isoformat(sep=' ', timespec='seconds')
    conn = get_db_connection()
    cur = conn.cursor()

    # Create Incidents table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Incidents (
            IncidentID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Status TEXT NOT NULL,
            Description TEXT,
            Timestamp TEXT
        )
    """)

    # Insert the incident record
    cur.execute(
        """
        INSERT INTO Incidents (Name, Status, Description, Timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (name, status, description, timestamp)
    )

    conn.commit()
    conn.close()


def get_all_incidents():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, status, description, timestamp FROM incidents ORDER BY timestamp DESC")
    results = c.fetchall()
    conn.close()

    return [{
        'name': row[0],
        'status': row[1],
        'description': row[2],
        'timestamp': row[3]
    } for row in results]