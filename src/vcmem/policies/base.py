"""The eviction-policy API.

The manager owns the token accounting and calls the policy only to learn *which* resident
page to evict. A policy sees four events:

* ``admit``  a page enters main context (a new message, or a page fault that pulls an
  evicted message back in);
* ``touch``  a resident page was referenced again (a search hit on something in context);
* ``victim`` the manager is over capacity and asks for the next page to evict;
* ``evict``  the manager removed that page.

Sizes are in tokens because context pages are variable sized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Container

    import numpy as np


@dataclass(frozen=True, slots=True)
class PageInfo:
    """What a policy may know about a page beyond its key."""

    size: int
    importance: float = 0.0
    embedding: np.ndarray | None = None


class EvictionPolicy(ABC):
    name: str = "base"
    # When true the manager computes an embedding and an importance score for every
    # admitted page and reports the embedding of each step's input via observe_query.
    uses_features: bool = False

    @abstractmethod
    def admit(self, key: str, info: PageInfo, now: int) -> None: ...

    @abstractmethod
    def touch(self, key: str, now: int) -> None: ...

    @abstractmethod
    def victim(self, pinned: Container[str]) -> str | None:
        """Return the resident key to evict next, never one in ``pinned``."""

    @abstractmethod
    def evict(self, key: str) -> None: ...

    @abstractmethod
    def resident(self) -> list[str]: ...

    def observe_query(self, embedding: np.ndarray | None) -> None:
        """Called once per step with the embedding of the incoming text (default: ignored)."""
        del embedding

    def __contains__(self, key: object) -> bool:
        return key in set(self.resident())

    def __len__(self) -> int:
        return len(self.resident())

    def stats(self) -> dict[str, float]:
        return {}
