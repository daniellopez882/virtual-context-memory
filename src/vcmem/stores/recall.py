"""Recall storage: the complete message log, persisted in SQLite and searchable with BM25.

Every ingested message lands here before it enters main context, so eviction never loses
anything: an evicted message is still one ``conversation_search`` away. The BM25 index
lives in memory and is rebuilt from the stored messages when the store is opened.

A message's embedding is stored next to it when a policy needs one, so a page fault
reads the vector instead of encoding the message again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vcmem.messages import Message
from vcmem.stores.bm25 import BM25Index

if TYPE_CHECKING:
    from vcmem.stores.sqlite import Database


@dataclass(frozen=True, slots=True)
class RecallHit:
    message: Message
    score: float


def index_text(message: Message) -> str:
    # Speaker names help ("what did Caroline ..."); the session date would add the same
    # tokens to every message of a session, so it is left out of the index.
    return f"{message.speaker} {message.text}"


class RecallStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._messages: list[Message] = []
        self._by_id: dict[str, int] = {}
        self._index = BM25Index()
        with db.lock:
            rows = db.conn.execute(
                "SELECT seq, id, role, speaker, text, timestamp, session FROM messages ORDER BY seq"
            ).fetchall()
        for seq, mid, role, speaker, text, timestamp, session in rows:
            self._remember(Message(mid, text, speaker, role, timestamp, session, seq))

    def _remember(self, message: Message) -> None:
        self._by_id[message.id] = len(self._messages)
        self._messages.append(message)
        self._index.add(index_text(message))

    def __len__(self) -> int:
        return len(self._messages)

    def __contains__(self, message_id: object) -> bool:
        return message_id in self._by_id

    def append(self, message: Message, vector: np.ndarray | None = None) -> Message:
        if message.id in self._by_id:
            raise ValueError(f"duplicate message id {message.id!r}")
        stored = message.with_seq(len(self._messages))
        with self._db.lock:
            self._db.conn.execute(
                "INSERT INTO messages(seq, id, role, speaker, text, timestamp, session) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    stored.seq,
                    stored.id,
                    stored.role,
                    stored.speaker,
                    stored.text,
                    stored.timestamp,
                    stored.session,
                ),
            )
            if vector is not None:
                self._write_vector(stored.id, vector)
            self._db.conn.commit()
        self._remember(stored)
        return stored

    def _write_vector(self, message_id: str, vector: np.ndarray) -> None:
        self._db.conn.execute(
            "INSERT OR REPLACE INTO message_vectors(id, vector) VALUES (?, ?)",
            (message_id, np.asarray(vector, dtype=np.float32).tobytes()),
        )

    def set_vector(self, message_id: str, vector: np.ndarray) -> None:
        with self._db.lock:
            self._write_vector(message_id, vector)
            self._db.conn.commit()

    def vector(self, message_id: str) -> np.ndarray | None:
        with self._db.lock:
            row = self._db.conn.execute(
                "SELECT vector FROM message_vectors WHERE id = ?", (message_id,)
            ).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def get(self, message_id: str) -> Message | None:
        idx = self._by_id.get(message_id)
        return None if idx is None else self._messages[idx]

    def all(self) -> list[Message]:
        return list(self._messages)

    def search(self, query: str, k: int, offset: int = 0) -> tuple[list[RecallHit], int]:
        """Return hits ``offset .. offset + k`` and the total number of matching messages."""
        scores = self._index.scores(query)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        window = ranked[offset : offset + k]
        return [RecallHit(self._messages[d], s) for d, s in window], len(ranked)
