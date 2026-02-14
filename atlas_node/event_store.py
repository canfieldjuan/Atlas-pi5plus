"""Offline event buffer for critical events during Brain disconnections.

Persists security and recognition events to a local SQLite queue when Brain
is unreachable.  Events are drained to Brain on reconnect via the WS client.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class OfflineEventBuffer:
    """SQLite-backed FIFO queue for critical events that must survive Brain outages."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or config.OFFLINE_BUFFER_DB_PATH
        self._conn: sqlite3.Connection | None = None

    def init(self):
        """Open database and create table."""
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                msg_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        self._conn.commit()

        # Cleanup events older than retention period
        cutoff = time.time() - config.OFFLINE_BUFFER_RETENTION_DAYS * 86400
        cur = self._conn.execute(
            "DELETE FROM pending_events WHERE created_at < ?", (cutoff,)
        )
        if cur.rowcount > 0:
            self._conn.commit()
            log.info("Cleaned up %d expired buffered events", cur.rowcount)

        count = self.count()
        log.info("OfflineEventBuffer ready: %s (%d pending)", self._db_path, count)

    def enqueue(self, msg_type: str, payload: dict) -> None:
        """Persist a critical event for later delivery."""
        if not self._conn:
            return
        self._conn.execute(
            "INSERT INTO pending_events (created_at, msg_type, payload) VALUES (?, ?, ?)",
            (time.time(), msg_type, json.dumps(payload)),
        )
        self._conn.commit()

    def dequeue_batch(self, limit: int = 50) -> list[tuple[int, dict]]:
        """Fetch the oldest pending events (FIFO order).

        Returns list of (row_id, payload_dict) tuples.
        """
        if not self._conn:
            return []
        rows = self._conn.execute(
            "SELECT id, payload FROM pending_events ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row_id, payload_str in rows:
            try:
                result.append((row_id, json.loads(payload_str)))
            except (json.JSONDecodeError, TypeError):
                log.warning("Corrupt buffered event id=%d, skipping", row_id)
                # Remove corrupt entry
                self._conn.execute("DELETE FROM pending_events WHERE id = ?", (row_id,))
        return result

    def remove(self, ids: list[int]) -> None:
        """Delete events by ID after successful send."""
        if not self._conn or not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM pending_events WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()

    def count(self) -> int:
        """Number of pending events in the buffer."""
        if not self._conn:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM pending_events").fetchone()
        return row[0] if row else 0

    def close(self):
        """Checkpoint WAL and close the database."""
        if self._conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                log.debug("WAL checkpoint on close failed", exc_info=True)
            self._conn.close()
            self._conn = None
            log.info("OfflineEventBuffer closed")
