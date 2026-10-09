"""Recall storage: the complete message log, persisted in SQLite and searchable with BM25.

Every ingested message lands here before it enters main context, so eviction never loses
anything: an evicted message is still one ``conversation_search`` away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    def append(self, message: Message) -> Message:
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
            self._db.conn.commit()
        self._remember(stored)
        return stored

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
