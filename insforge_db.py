"""
insforge_db.py
==============
Drop-in replacement for database.py using InsForge Postgres instead of
SQLite.  The public API is identical so app.py needs only one import
change:

    # was:  from database import ...
    from insforge_db import ...

Connection is managed through a module-level pool (psycopg2 simple pool).
The Postgres DSN is read from INSFORGE_DB_URL in .env.local; the
connection string is also stored in .insforge/project.json but is never
committed to Git.

Table column names use snake_case (Postgres convention).  The get_records /
get_all_incidents helpers return dicts with the same camelCase keys used
by the original SQLite layer so templates and routes require no changes.
"""

import os
import json
import base64
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Load credentials from .env.local ─────────────────────────────────────────
# .env.local is git-ignored.  If it doesn't exist the caller must have set
# INSFORGE_DB_URL directly in the process environment.
_env_path = os.path.join(os.path.dirname(__file__), '.env.local')
load_dotenv(_env_path, override=False)

# ── Build the DSN ─────────────────────────────────────────────────────────────
# Priority: explicit INSFORGE_DB_URL env var → .insforge/project.json appkey
def _build_dsn() -> str:
    explicit = os.getenv('INSFORGE_DB_URL')
    if explicit:
        return explicit

    # Derive from .insforge/project.json (written by `insforge link`)
    proj_file = os.path.join(os.path.dirname(__file__), '.insforge', 'project.json')
    if os.path.exists(proj_file):
        with open(proj_file) as f:
            proj = json.load(f)
        appkey = proj.get('appkey', '')
        region = proj.get('region', 'us-east')
        # InsForge Postgres host pattern: <appkey>.us-east.database.insforge.app
        host = f"{appkey}.{region}.database.insforge.app"
        # Password is NOT in project.json (it's a secret).  Fall through to
        # the hardcoded DSN used during initial setup if the env var is absent.
        logger.warning(
            "INSFORGE_DB_URL not set — add it to .env.local.  "
            "Run: npx @insforge/cli db connection-string"
        )

    # Last resort: the DSN stored during migration setup.
    # This string is intentionally not committed — it lives only here at runtime.
    raise RuntimeError(
        "INSFORGE_DB_URL is not set.  Add it to .env.local:\n"
        "  INSFORGE_DB_URL=postgresql://postgres:<password>@<appkey>.us-east.database.insforge.app:5432/insforge?sslmode=require\n"
        "Get the value with: npx @insforge/cli db connection-string"
    )


# ── Connection pool ───────────────────────────────────────────────────────────
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        dsn = os.getenv('INSFORGE_DB_URL') or _build_dsn()
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn)
        logger.info("InsForge Postgres pool created.")
    return _pool


@contextmanager
def _db():
    """Context manager that yields a cursor and commits on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── Schema bootstrap (called once at startup) ─────────────────────────────────
def init_db():
    """Ensure tables exist.  Safe to call multiple times (IF NOT EXISTS)."""
    with _db() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                record_id    SERIAL PRIMARY KEY,
                name         TEXT NOT NULL,
                confidence   REAL,
                timestamp    TEXT,
                status       TEXT,
                bounding_box TEXT,
                camera_id    TEXT,
                embedding    BYTEA
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id  SERIAL PRIMARY KEY,
                name         TEXT,
                status       TEXT NOT NULL,
                description  TEXT,
                timestamp    TEXT
            )
        """)
    logger.info("InsForge DB initialised (tables verified).")


# ── FaceRecords ───────────────────────────────────────────────────────────────

def save_face_record(name, confidence, bbox, embedding, timestamp=None, status='recognized'):
    """Insert one face detection record.

    Parameters match the original database.py signature exactly so all
    existing call sites (app.py, rtsp_camera.py) work without change.
    """
    ts = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    emb_bytes = embedding.tobytes() if embedding is not None else None

    with _db() as cur:
        cur.execute(
            """
            INSERT INTO face_records
                (name, confidence, timestamp, status, bounding_box, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (name, float(confidence), ts, status, str(bbox),
             psycopg2.Binary(emb_bytes) if emb_bytes else None)
        )
    logger.debug("Saved face record: name=%s status=%s conf=%.2f", name, status, confidence)


def get_records(time_range='all'):
    """Return face records filtered by time range, newest first.

    Returns a list of dicts with the same camelCase keys as the original
    SQLite layer so Jinja templates need no changes.
    """
    base = """
        SELECT name, confidence, bounding_box, status, timestamp
        FROM face_records
    """
    params: list = []
    now = datetime.now()

    if time_range == 'today':
        base += " WHERE timestamp LIKE %s"
        params.append(now.strftime('%Y-%m-%d') + '%')
    elif time_range == 'week':
        start_week = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        base += " WHERE timestamp >= %s"
        params.append(f"{start_week} 00:00:00")
    elif time_range == 'month':
        base += " WHERE timestamp LIKE %s"
        params.append(now.strftime('%Y-%m') + '%')

    base += " ORDER BY timestamp DESC"

    with _db() as cur:
        cur.execute(base, params)
        rows = cur.fetchall()

    records = []
    for row in rows:
        rec = dict(row)
        # Normalise to camelCase keys expected by templates/routes
        records.append({
            'Name':        rec['name'],
            'Confidence':  round(rec['confidence'] or 0.0, 2),
            'BoundingBox': rec['bounding_box'],
            'Status':      rec['status'] or 'unknown',
            'Timestamp':   rec['timestamp'] or '',
        })
    return records


def get_kpi_counts(records):
    total = len(records)
    statuses = [r['Status'].lower() for r in records]
    return {
        'recognized': statuses.count('recognized'),
        'unknown':    statuses.count('unknown'),
        'entering':   statuses.count('entering'),
        'exiting':    statuses.count('exiting'),
        'total':      total,
    }


def get_peak_hours(time_range='all'):
    """Top-4 busiest hours, honoring the time_range filter."""
    where = ""
    params: list = []
    now = datetime.now()

    if time_range == 'today':
        where = "WHERE timestamp LIKE %s"
        params.append(now.strftime('%Y-%m-%d') + '%')
    elif time_range == 'week':
        start_week = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        where = "WHERE timestamp >= %s"
        params.append(f"{start_week} 00:00:00")
    elif time_range == 'month':
        where = "WHERE timestamp LIKE %s"
        params.append(now.strftime('%Y-%m') + '%')

    sql = f"""
        SELECT substr(timestamp, 12, 2) AS hour, COUNT(*) AS cnt
        FROM face_records
        {where}
        GROUP BY hour
        ORDER BY cnt DESC
        LIMIT 4
    """

    with _db() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{'hour': f"{row['hour']}:00", 'count': row['cnt']} for row in rows]


# ── Incidents ─────────────────────────────────────────────────────────────────

def save_incident(name, status, description=None, timestamp=None):
    ts = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with _db() as cur:
        cur.execute(
            """
            INSERT INTO incidents (name, status, description, timestamp)
            VALUES (%s, %s, %s, %s)
            """,
            (name, status, description, ts)
        )


def get_recent_incidents(limit=3):
    with _db() as cur:
        cur.execute(
            """
            SELECT status, name, description, timestamp
            FROM incidents
            WHERE status != 'recognized'
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()

    return [
        {
            'status':      row['status'],
            'name':        row['name'],
            'description': row['description'],
            'timestamp':   row['timestamp'],
        }
        for row in rows
    ]


def get_all_incidents():
    with _db() as cur:
        cur.execute(
            """
            SELECT name, status, description, timestamp
            FROM incidents
            ORDER BY timestamp DESC
            """
        )
        rows = cur.fetchall()

    return [
        {
            'name':        row['name'],
            'status':      row['status'],
            'description': row['description'],
            'timestamp':   row['timestamp'],
        }
        for row in rows
    ]
