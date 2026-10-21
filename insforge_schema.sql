-- InsForge schema for Facial-access-control
-- Tables use snake_case to match Postgres conventions.
-- The Python db layer maps them to the camelCase names used in the
-- original SQLite schema so the rest of app.py stays unchanged.

CREATE TABLE IF NOT EXISTS face_records (
    record_id    SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    confidence   REAL,
    timestamp    TEXT,
    status       TEXT,
    bounding_box TEXT,
    camera_id    TEXT,
    embedding    BYTEA
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id  SERIAL PRIMARY KEY,
    name         TEXT,
    status       TEXT NOT NULL,
    description  TEXT,
    timestamp    TEXT
);
