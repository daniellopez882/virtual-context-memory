from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from vcmem.policies.base import EvictionPolicy, PageInfo

if TYPE_CHECKING:
    from collections.abc import Container


class FIFOPolicy(EvictionPolicy):
    """Evict in admission order. A page fault re-admits at the tail; hits change nothing."""

    name = "fifo"

    def __init__(self) -> None:
        self._order: OrderedDict[str, None] = OrderedDict()

    def admit(self, key: str, info: PageInfo, now: int) -> None:
        self._order.pop(key, None)
        self._order[key] = None

    def touch(self, key: str, now: int) -> None:
        return None

    def victim(self, pinned: Container[str]) -> str | None:
        return next((k for k in self._order if k not in pinned), None)

    def evict(self, key: str) -> None:
        del self._order[key]

    def resident(self) -> list[str]:
        return list(self._order)

    def __contains__(self, key: object) -> bool:
        return key in self._order

    def __len__(self) -> int:
        return len(self._order)
