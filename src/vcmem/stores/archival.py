"""Archival storage: free-text passages the agent chose to keep, searched by cosine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vcmem.stores.cosine import CosineIndex

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vcmem.embed import Embedder
    from vcmem.stores.sqlite import Database


@dataclass(frozen=True, slots=True)
class Passage:
    row: int
    text: str
    timestamp: str
    sources: frozenset[str]


@dataclass(frozen=True, slots=True)
class ArchivalHit:
    passage: Passage
    score: float


class ArchivalStore:
    def __init__(self, db: Database, embedder: Embedder) -> None:
        self._db = db
        self._embedder = embedder
        self._index = CosineIndex(embedder.dim)
        self._passages: list[Passage] = []
        with db.lock:
            rows = db.conn.execute(
                "SELECT row, text, timestamp, sources, vector FROM archival ORDER BY row"
            ).fetchall()
        for row, text, timestamp, sources, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != embedder.dim:
                raise ValueError(
                    f"archival vectors have dim {vec.shape[0]}, embedder has {embedder.dim}; "
                    "open the database with the embedder that wrote it"
                )
            self._passages.append(Passage(row, text, timestamp, frozenset(json.loads(sources))))
            self._index.add(vec)

    def __len__(self) -> int:
        return len(self._passages)

    def insert(self, text: str, sources: Iterable[str] = (), timestamp: str = "") -> Passage:
        vec = self._embedder.embed([text])[0].astype(np.float32)
        passage = Passage(len(self._passages), text, timestamp, frozenset(sources))
        with self._db.lock:
            self._db.conn.execute(
                "INSERT INTO archival(row, text, timestamp, sources, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    passage.row,
                    passage.text,
                    passage.timestamp,
                    json.dumps(sorted(passage.sources)),
                    vec.tobytes(),
                ),
            )
            self._db.conn.commit()
        self._passages.append(passage)
        self._index.add(vec)
        return passage

    def search(self, query: str, k: int, offset: int = 0) -> tuple[list[ArchivalHit], int]:
        """Return hits ``offset .. offset + k`` by cosine similarity and the store size."""
        if not self._passages:
            return [], 0
        vec = self._embedder.embed([query])[0]
        top = self._index.search(vec, offset + k)[offset:]
        return [ArchivalHit(self._passages[i], s) for i, s in top], len(self._passages)
