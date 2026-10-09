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
        return clone

    def admit(self, key: str, info: PageInfo, now: int) -> None:
        self._now = max(self._now, now)
        self._pages[key] = _Page(info.importance, info.embedding, now)

    def touch(self, key: str, now: int) -> None:
        self._now = max(self._now, now)
        page = self._pages.get(key)
        if page is not None:
            page.last_access = now

    def observe_query(self, embedding: np.ndarray | None) -> None:
        self._query = embedding

    def scores(self, keys: list[str]) -> np.ndarray:
        pages = [self._pages[k] for k in keys]
        age = np.array([self._now - p.last_access for p in pages], dtype=np.float64)
        recency = np.power(self.decay, age)
        importance = np.array([p.importance for p in pages], dtype=np.float64)
        relevance = np.zeros(len(pages), dtype=np.float64)
        if self._query is not None:
            for i, p in enumerate(pages):
                if p.embedding is not None:
                    relevance[i] = float(p.embedding @ self._query)
        parts = np.stack([_minmax(recency), _minmax(importance), _minmax(relevance)])
        return self.weights @ parts

    def victim(self, pinned: Container[str]) -> str | None:
        keys = [k for k in self._pages if k not in pinned]
        if not keys:
            return None
        # argmin returns the first minimum; dict order is admission order, so ties go to
        # the page admitted earliest.
        return keys[int(np.argmin(self.scores(keys)))]

    def evict(self, key: str) -> None:
        del self._pages[key]

    def resident(self) -> list[str]:
        return list(self._pages)

    def __contains__(self, key: object) -> bool:
        return key in self._pages

    def __len__(self) -> int:
        return len(self._pages)
