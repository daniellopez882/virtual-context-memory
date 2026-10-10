"""One SQLite file per agent holds recall messages, archival passages and core memory.

The BM25 and cosine indexes are not stored: they are rebuilt in memory from these rows
when a store is opened.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY,
    id TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    session INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS message_vectors (
    id TEXT PRIMARY KEY,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS archival (
    row INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sources TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS core_blocks (
    name TEXT PRIMARY KEY,
    limit_tokens INTEGER NOT NULL,
    lines TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper that owns the connection and serialises writes across threads."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        if self.path != ":memory:":
            # WAL with synchronous=NORMAL keeps commits cheap: the log is fsynced at
            # checkpoints, not on every appended message.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def size_bytes(self) -> int:
        """Logical database size (pages in use), independent of WAL checkpoint timing."""
        with self.lock:
            pages = self.conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
            freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        return int((pages - freelist) * page_size)

    def get_kv(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_kv(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()
