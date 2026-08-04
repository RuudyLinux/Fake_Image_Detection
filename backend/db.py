"""
SQLite persistence for the detection history dashboard.
No external DB needed — a lightweight local store of past results.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT NOT NULL,
                mode          TEXT NOT NULL,
                prediction    TEXT,
                confidence    REAL,
                forgery_type  TEXT,
                thumbnail_url TEXT,
                heatmap_url   TEXT,
                result_json   TEXT NOT NULL
            )
        """)


def insert_detection(mode: str, result: dict, thumbnail_url: str | None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO detections
               (created_at, mode, prediction, confidence, forgery_type, thumbnail_url, heatmap_url, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                mode,
                result.get("prediction"),
                result.get("confidence"),
                result.get("forgery_type"),
                thumbnail_url,
                result.get("heatmap_url"),
                json.dumps(result),
            ),
        )
        return cur.lastrowid


def list_detections(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, created_at, mode, prediction, confidence, forgery_type, thumbnail_url, heatmap_url
               FROM detections ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def clear_detections() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM detections")
