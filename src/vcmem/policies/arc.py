"""Adaptive Replacement Cache (Megiddo and Modha, FAST 2003), weighted by page size.

Classic ARC counts pages. Context pages are messages of very different lengths, so every
list here is measured in tokens: ``c`` is the queue capacity in tokens and ``p`` is the
target token share of T1.

T1 holds pages referenced once since admission, T2 pages referenced at least twice.
B1 and B2 are ghost lists: keys of recently evicted T1/T2 pages, without content. A page
fault on a B1 ghost means recency was undervalued (grow ``p``); a fault on a B2 ghost
means frequency was undervalued (shrink ``p``).

The paper's REPLACE step runs before the requested page enters the cache. Here the
manager admits first and asks for victims afterwards, so the policy remembers which
pages arrived in the current step and leaves them out of the ``|T1| > p`` test and out of
the candidates (unless nothing else is left). Ghost trimming for a miss (case IV of the
paper) also happens on admission, before the new page counts. With unit sizes and one
eviction per miss the lists and ``p`` then match the paper's algorithm step for step;
``tests/test_policies.py`` checks that against a reference implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from vcmem.policies.base import EvictionPolicy, PageInfo

if TYPE_CHECKING:
    from collections.abc import Container


class ARCPolicy(EvictionPolicy):
    name = "arc"

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ARC capacity must be positive")
        self.c = capacity
        self.p = 0.0
        self.t1: OrderedDict[str, int] = OrderedDict()
        self.t2: OrderedDict[str, int] = OrderedDict()
        self.b1: OrderedDict[str, int] = OrderedDict()
        self.b2: OrderedDict[str, int] = OrderedDict()
        self._sz = {"t1": 0, "t2": 0, "b1": 0, "b2": 0}
        self._last_fault_from_b2 = False
        self._step: int | None = None
        self._incoming: set[str] = set()
        self.hits = 0
        self.ghost_hits_b1 = 0
        self.ghost_hits_b2 = 0

    def size_of(self, which: str) -> int:
        return self._sz[which]

    def _list(self, which: str) -> OrderedDict[str, int]:
        return {"t1": self.t1, "t2": self.t2, "b1": self.b1, "b2": self.b2}[which]

    def _push(self, which: str, key: str, size: int) -> None:
        self._list(which)[key] = size
        self._sz[which] += size

    def _pop(self, which: str, key: str) -> int:
        size = self._list(which).pop(key)
        self._sz[which] -= size
        return size

    def _drop_lru(self, which: str) -> None:
        self._pop(which, next(iter(self._list(which))))

    def _total(self) -> int:
        return sum(self._sz.values())

    def admit(self, key: str, info: PageInfo, now: int) -> None:
        if key in self.t1 or key in self.t2:
            self.touch(key, now)
            return
        if now != self._step:
            self._step = now
            self._incoming = set()
        size = max(1, info.size)
        self._last_fault_from_b2 = False
        if key in self.b1:
            ratio = self._sz["b2"] / self._sz["b1"]
            self.p = min(float(self.c), self.p + max(1.0, ratio) * size)
            self._pop("b1", key)
            self._push("t2", key, size)
            self.ghost_hits_b1 += 1
        elif key in self.b2:
            ratio = self._sz["b1"] / self._sz["b2"]
            self.p = max(0.0, self.p - max(1.0, ratio) * size)
            self._pop("b2", key)
            self._push("t2", key, size)
            self._last_fault_from_b2 = True
            self.ghost_hits_b2 += 1
        else:
            self._make_room(size)
            self._push("t1", key, size)
        self._incoming.add(key)

    def _make_room(self, size: int) -> None:
        """Case IV of the paper: trim ghosts for a missed page before it is inserted."""
        sz = self._sz
        if sz["t1"] + sz["b1"] + size > self.c and not self.b1:
            # Case IV(a) with an empty B1: T1 alone fills L1. The paper drops T1's LRU page
            # without keeping a ghost; evict() gets the same result by trimming B1.
            return
        while self.b1 and sz["t1"] + sz["b1"] + size > self.c:
            self._drop_lru("b1")
        while self.b2 and self._total() + size > 2 * self.c:
            self._drop_lru("b2")

    def touch(self, key: str, now: int) -> None:
        if key in self.t1:
            self._push("t2", key, self._pop("t1", key))
        elif key in self.t2:
            self.t2.move_to_end(key)
        else:
            return
        self.hits += 1

    def victim(self, pinned: Container[str]) -> str | None:
        incoming = self._incoming
        from_t1 = next((k for k in self.t1 if k not in pinned and k not in incoming), None)
        from_t2 = next((k for k in self.t2 if k not in pinned and k not in incoming), None)
        if from_t1 is None and from_t2 is None:
            # Only this step's own pages are evictable; give up the oldest of them.
            return next((k for k in (*self.t1, *self.t2) if k not in pinned), None)
        t1_size = self._sz["t1"] - sum(self.t1[k] for k in incoming if k in self.t1)
        prefer_t1 = t1_size > self.p or (self._last_fault_from_b2 and t1_size == self.p)
        if from_t1 is not None and (prefer_t1 or from_t2 is None):
            return from_t1
        return from_t2

    def evict(self, key: str) -> None:
        if key in self.t1:
            self._push("b1", key, self._pop("t1", key))
        elif key in self.t2:
            self._push("b2", key, self._pop("t2", key))
        else:
            raise KeyError(key)
        self._incoming.discard(key)
        self._trim_ghosts()

    def _trim_ghosts(self) -> None:
        sz = self._sz
        while self.b1 and sz["t1"] + sz["b1"] > self.c:
            self._drop_lru("b1")
        while self.b2 and self._total() > 2 * self.c:
            self._drop_lru("b2")
        while self.b1 and self._total() > 2 * self.c:
            self._drop_lru("b1")

    def resident(self) -> list[str]:
        return [*self.t1, *self.t2]

    def __contains__(self, key: object) -> bool:
        return key in self.t1 or key in self.t2

    def __len__(self) -> int:
        return len(self.t1) + len(self.t2)

    def stats(self) -> dict[str, float]:
        return {
            "p": self.p,
            "hits": float(self.hits),
            "ghost_hits_b1": float(self.ghost_hits_b1),
            "ghost_hits_b2": float(self.ghost_hits_b2),
        }
