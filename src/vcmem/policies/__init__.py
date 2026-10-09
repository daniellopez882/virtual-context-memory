"""Eviction policies and a small registry so callers can plug in their own."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vcmem.policies.arc import ARCPolicy
from vcmem.policies.base import EvictionPolicy, PageInfo
from vcmem.policies.fifo import FIFOPolicy
from vcmem.policies.importance import ImportancePolicy
from vcmem.policies.lru import LRUPolicy

if TYPE_CHECKING:
    from collections.abc import Callable


def _fifo(capacity: int) -> EvictionPolicy:
    return FIFOPolicy()


def _lru(capacity: int) -> EvictionPolicy:
    return LRUPolicy()


def _importance(capacity: int) -> EvictionPolicy:
    return ImportancePolicy()


_REGISTRY: dict[str, Callable[[int], EvictionPolicy]] = {
    "fifo": _fifo,
    "lru": _lru,
    "arc": ARCPolicy,
    "importance": _importance,
}


def register_policy(name: str, factory: Callable[[int], EvictionPolicy]) -> None:
    """Make ``make_policy(name, capacity)`` build a custom policy (capacity is in tokens)."""
    _REGISTRY[name] = factory


def make_policy(name: str, capacity: int) -> EvictionPolicy:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(_REGISTRY)}") from None
    return factory(capacity)


def policy_names() -> list[str]:
    return list(_REGISTRY)


__all__ = [
    "ARCPolicy",
    "EvictionPolicy",
    "FIFOPolicy",
    "ImportancePolicy",
    "LRUPolicy",
    "PageInfo",
    "make_policy",
    "policy_names",
    "register_policy",
]
