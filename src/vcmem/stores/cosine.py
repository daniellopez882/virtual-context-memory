"""Brute-force cosine index over L2-normalised float32 vectors.

Archival storage per agent is thousands of rows, not millions, so an exact scan with a
single matrix-vector product beats maintaining an approximate index.
"""

from __future__ import annotations

import numpy as np


class CosineIndex:
    def __init__(self, dim: int, initial_capacity: int = 256) -> None:
        self.dim = dim
        self._matrix = np.zeros((initial_capacity, dim), dtype=np.float32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, vector: np.ndarray) -> int:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vec.shape[0]}")
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        if self._n == self._matrix.shape[0]:
            grown = np.zeros((self._matrix.shape[0] * 2, self.dim), dtype=np.float32)
            grown[: self._n] = self._matrix[: self._n]
            self._matrix = grown
        self._matrix[self._n] = vec
        self._n += 1
        return self._n - 1

    def scores(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        return self._matrix[: self._n] @ (q / norm if norm > 0 else q)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._n == 0 or k <= 0:
            return []
        sims = self.scores(query)
        k = min(k, self._n)
        top = np.argpartition(-sims, k - 1)[:k]
        # Stable order: score descending, then insertion order.
        order = sorted(top.tolist(), key=lambda i: (-float(sims[i]), i))
        return [(i, float(sims[i])) for i in order]
