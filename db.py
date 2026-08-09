"""SQLite access layer. One connection per request, read-only at runtime."""
import os
import sqlite3

DB_PATH = os.environ.get("NEARMISS_DB", os.path.join(os.path.dirname(__file__), "data", "nearmiss.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY,
    report_id  TEXT UNIQUE,
    ym         TEXT NOT NULL,          -- 'YYYY-MM'
    airport    TEXT NOT NULL,          -- ICAO, e.g. KSFO
    aircraft   TEXT,
    phase      TEXT,                   -- approach / taxi / climbout ...
    narrative  TEXT NOT NULL,
    cluster_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_reports_apt_ym ON reports(airport, ym);
CREATE INDEX IF NOT EXISTS ix_reports_cluster ON reports(cluster_id, airport, ym);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id TEXT PRIMARY KEY,
    label      TEXT,
    keywords   TEXT,
    n          INTEGER
);

CREATE TABLE IF NOT EXISTS airports (
    icao      TEXT PRIMARY KEY,
    name      TEXT,
    runways   TEXT,
    ops_year  INTEGER,
    notes     TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS reports_fts
USING fts5(narrative, content='reports', content_rowid='id', tokenize='porter');
"""


def connect(write: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if not write:
        con.execute("PRAGMA query_only = ON")
    return con


def init(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()

