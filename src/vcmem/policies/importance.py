"""Generative-Agents retrieval score used as an eviction score.

Park et al. (2023) rank memories by recency + importance + relevance, each min-max
normalised over the candidates, with exponential recency decay. Here the lowest-scoring
resident page is evicted; relevance is cosine similarity to the incoming text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from vcmem.policies.base import EvictionPolicy, PageInfo

if TYPE_CHECKING:
    from collections.abc import Container


@dataclass(slots=True)
class _Page:
    importance: float
    embedding: np.ndarray | None
    last_access: int


@dataclass(slots=True)
class _Features:
    """Raw recency, importance and relevance of the pages resident when it was built."""

    keys: list[str]
    index: dict[str, int]
    raw: np.ndarray  # shape (3, len(keys))
    alive: np.ndarray  # False once a page has been evicted


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class ImportancePolicy(EvictionPolicy):
    name = "importance"
    uses_features = True

    def __init__(
        self,
        decay: float = 0.995,
        w_recency: float = 1.0,
        w_importance: float = 1.0,
        w_relevance: float = 1.0,
    ) -> None:
        self.decay = decay
        self.weights = np.array([w_recency, w_importance, w_relevance], dtype=np.float64)
        self._pages: dict[str, _Page] = {}
        self._query: np.ndarray | None = None
        self._now = 0
        # A batch flush asks for many victims while nothing but the candidate set changes,
        # so the raw features are computed once and evictions only clear a mask. Any admit,
        # touch or new query drops them.
        self._features: _Features | None = None

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        # Embeddings are never mutated, so a checkpoint shares them instead of copying.
        clone = type(self).__new__(type(self))
        clone.decay = self.decay
        clone.weights = self.weights
        clone._pages = {
            k: _Page(p.importance, p.embedding, p.last_access) for k, p in self._pages.items()
        }
        clone._query = self._query
        clone._now = self._now
        clone._features = None
        return clone

    def admit(self, key: str, info: PageInfo, now: int) -> None:
        self._now = max(self._now, now)
        self._pages[key] = _Page(info.importance, info.embedding, now)
        self._features = None

    def touch(self, key: str, now: int) -> None:
        self._now = max(self._now, now)
        page = self._pages.get(key)
        if page is not None:
            page.last_access = now
        self._features = None

    def observe_query(self, embedding: np.ndarray | None) -> None:
        self._query = embedding
        self._features = None

    def _build_features(self) -> _Features:
        keys = list(self._pages)
        pages = list(self._pages.values())
        age = np.array([self._now - p.last_access for p in pages], dtype=np.float64)
        query = self._query
        # One dot product per page rather than a matrix product: the float32 sums then
        # round exactly as they did when relevance was computed page by page.
        relevance = np.array(
            [
                0.0 if query is None or p.embedding is None else float(p.embedding @ query)
                for p in pages
            ],
            dtype=np.float64,
        )
        raw = np.stack(
            [
                np.power(self.decay, age),
                np.array([p.importance for p in pages], dtype=np.float64),
                relevance,
            ]
        )
        return _Features(keys, {k: i for i, k in enumerate(keys)}, raw, np.ones(len(keys), bool))

    def victim(self, pinned: Container[str]) -> str | None:
        if self._features is None:
            self._features = self._build_features()
        f = self._features
        candidates = np.flatnonzero(
            f.alive & np.fromiter((k not in pinned for k in f.keys), bool, len(f.keys))
        )
        if candidates.size == 0:
            return None
        parts = np.stack([_minmax(row) for row in f.raw[:, candidates]])
        # argmin returns the first minimum; keys are in admission order, so ties go to the
        # page admitted earliest.
        return f.keys[int(candidates[int(np.argmin(self.weights @ parts))])]

    def evict(self, key: str) -> None:
        del self._pages[key]
        if self._features is not None:
            self._features.alive[self._features.index[key]] = False

    def resident(self) -> list[str]:
        return list(self._pages)

    def __contains__(self, key: object) -> bool:
        return key in self._pages

    def __len__(self) -> int:
        return len(self._pages)
