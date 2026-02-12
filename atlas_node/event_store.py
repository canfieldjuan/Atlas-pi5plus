"""Local SQLite event store for security and recognition events.

WAL mode for concurrent read/write, with automatic 30-day rotation.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class EventStore:
    """SQLite-backed event logging for the edge node."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or config.EVENT_DB_PATH
        self._conn: sqlite3.Connection | None = None

    def init(self):
        """Open database and create tables."""
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS recognition_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epoch REAL NOT NULL,
                person_name TEXT NOT NULL,
                recognition_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                track_id INTEGER,
                camera_source TEXT DEFAULT 'cam1',
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epoch REAL NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                track_id INTEGER,
                camera_source TEXT DEFAULT 'cam1',
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_recog_timestamp
                ON recognition_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_recog_person
                ON recognition_events(person_name);
            CREATE INDEX IF NOT EXISTS idx_sec_timestamp
                ON security_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sec_type
                ON security_events(event_type);
        """)
        self._conn.commit()

        self._rotate()
        log.info("EventStore ready: %s", self._db_path)

    def log_recognition(
        self,
        person_name: str,
        recognition_type: str,
        confidence: float,
        track_id: int | None = None,
        metadata: dict | None = None,
    ):
        """Log a recognition event (face, gait, face+gait, speaker)."""
        if not self._conn:
            return
        now = datetime.now()
        self._conn.execute(
            """INSERT INTO recognition_events
               (timestamp, epoch, person_name, recognition_type, confidence, track_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(),
                time.time(),
                person_name,
                recognition_type,
                confidence,
                track_id,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def log_security(
        self,
        event_type: str,
        confidence: float = 0.0,
        track_id: int | None = None,
        metadata: dict | None = None,
    ):
        """Log a security event (motion_detected, person_entered, person_left, unknown_face)."""
        if not self._conn:
            return
        now = datetime.now()
        self._conn.execute(
            """INSERT INTO security_events
               (timestamp, epoch, event_type, confidence, track_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now.isoformat(),
                time.time(),
                event_type,
                confidence,
                track_id,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def query_recent(
        self,
        hours: float = 1.0,
        person: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query recent events across both tables."""
        if not self._conn:
            return []

        cutoff = time.time() - hours * 3600
        results = []

        # Recognition events
        q = "SELECT timestamp, person_name, recognition_type, confidence, track_id, metadata FROM recognition_events WHERE epoch >= ?"
        params: list = [cutoff]
        if person:
            q += " AND person_name = ?"
            params.append(person)
        q += " ORDER BY epoch DESC LIMIT ?"
        params.append(limit)

        for row in self._conn.execute(q, params).fetchall():
            results.append({
                "table": "recognition",
                "timestamp": row[0],
                "person_name": row[1],
                "recognition_type": row[2],
                "confidence": row[3],
                "track_id": row[4],
                "metadata": json.loads(row[5]) if row[5] else None,
            })

        # Security events
        q = "SELECT timestamp, event_type, confidence, track_id, metadata FROM security_events WHERE epoch >= ?"
        params = [cutoff]
        if event_type:
            q += " AND event_type = ?"
            params.append(event_type)
        q += " ORDER BY epoch DESC LIMIT ?"
        params.append(limit)

        for row in self._conn.execute(q, params).fetchall():
            results.append({
                "table": "security",
                "timestamp": row[0],
                "event_type": row[1],
                "confidence": row[2],
                "track_id": row[3],
                "metadata": json.loads(row[4]) if row[4] else None,
            })

        results.sort(key=lambda r: r["timestamp"], reverse=True)
        return results[:limit]

    def _rotate(self):
        """Delete events older than retention period."""
        if not self._conn:
            return
        cutoff = (datetime.now() - timedelta(days=config.EVENT_RETENTION_DAYS)).isoformat()
        r1 = self._conn.execute(
            "DELETE FROM recognition_events WHERE timestamp < ?", (cutoff,)
        )
        r2 = self._conn.execute(
            "DELETE FROM security_events WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        total = r1.rowcount + r2.rowcount
        if total > 0:
            log.info("Rotated %d old events (>%d days)", total, config.EVENT_RETENTION_DAYS)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("EventStore closed")
