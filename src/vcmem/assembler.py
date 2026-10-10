"""Token-budgeted context assembly with priority classes.

Every prompt the library produces goes through :class:`ContextAssembler`, which is the one
place that enforces the budget. Items are kept by priority class first, then by
``keep_rank`` inside a class; pinned items are never dropped (they are truncated if they
alone overflow the budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import TYPE_CHECKING

from vcmem.providers.base import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vcmem.tokens import TokenCounter


class Priority(IntEnum):
    """Lower value survives longer when the budget is tight."""

    SYSTEM = 0
    INPUT = 1
    CORE = 2
    SUMMARY = 3
    ARCHIVAL = 4
    RECALL = 5
    QUEUE = 6


# Rendering order differs from drop order: the incoming message always goes last, and
# retrieved messages interleave chronologically with the queue.
_RENDER_RANK = {
    Priority.SYSTEM: 0,
    Priority.CORE: 1,
    Priority.SUMMARY: 2,
    Priority.ARCHIVAL: 3,
    Priority.RECALL: 4,
    Priority.QUEUE: 4,
    Priority.INPUT: 5,
}


@dataclass(frozen=True, slots=True)
class ContextItem:
    text: str
    priority: Priority
    position: float = 0.0
    keep_rank: float = 0.0
    pinned: bool = False
    role: str = "system"
    full_sources: frozenset[str] = field(default_factory=frozenset)
    partial_sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class AssembledContext:
    items: tuple[ContextItem, ...]
    dropped: tuple[ContextItem, ...]
    item_tokens: tuple[int, ...]
    budget: int

    @property
    def tokens(self) -> int:
        return sum(self.item_tokens)

    def render(self) -> str:
        return "\n".join(item.text for item in self.items if item.text)

    def full_sources(self) -> frozenset[str]:
        out: set[str] = set()
        for item in self.items:
            out |= item.full_sources
        return frozenset(out)

    def partial_sources(self) -> frozenset[str]:
        out: set[str] = set()
        for item in self.items:
            out |= item.partial_sources
        return frozenset(out)

    def to_chat_messages(self) -> list[ChatMessage]:
        """Chat messages: one system message for the fixed regions, then the queue."""
        system_parts: list[str] = []
        chat: list[ChatMessage] = []
        for item in self.items:
            if not item.text:
                continue
            if item.priority in (Priority.QUEUE, Priority.RECALL, Priority.INPUT):
                role = "assistant" if item.role == "assistant" else "user"
                chat.append(ChatMessage(role, item.text))
            else:
                system_parts.append(item.text)
        return [ChatMessage("system", "\n\n".join(system_parts)), *chat]


class ContextAssembler:
    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    @property
    def counter(self) -> TokenCounter:
        return self._counter

    def assemble(self, items: Iterable[ContextItem], budget: int) -> AssembledContext:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        pinned: list[tuple[ContextItem, int]] = []
        optional: list[tuple[ContextItem, int]] = []
        for item in items:
            n = self._counter.count(item.text)
            (pinned if item.pinned else optional).append((item, n))

        pinned = self._fit_pinned(pinned, budget)
        remaining = budget - sum(n for _, n in pinned)

        kept = list(pinned)
        dropped: list[ContextItem] = []
        optional.sort(key=lambda pair: (pair[0].priority, -pair[0].keep_rank))
        for item, n in optional:
            if n <= remaining:
                kept.append((item, n))
                remaining -= n
            else:
                dropped.append(item)

        kept.sort(key=lambda pair: (_RENDER_RANK[pair[0].priority], pair[0].position))
        return AssembledContext(
            items=tuple(item for item, _ in kept),
            dropped=tuple(dropped),
            item_tokens=tuple(n for _, n in kept),
            budget=budget,
        )

    def _fit_pinned(
        self, pinned: list[tuple[ContextItem, int]], budget: int
    ) -> list[tuple[ContextItem, int]]:
        total = sum(n for _, n in pinned)
        while total > budget:
            # Shorten the largest pinned item first; the system prompt is usually the
            # longest, but an oversized user input should not wipe it out entirely.
            idx = max(range(len(pinned)), key=lambda i: pinned[i][1])
            item, n = pinned[idx]
            target = max(0, n - (total - budget))
            text = self._counter.truncate(item.text, target)
            new_n = self._counter.count(text)
            if new_n >= n:
                text, new_n = "", 0
            # A truncated message no longer carries its full text.
            shortened = replace(
                item,
                text=text,
                full_sources=frozenset(),
                partial_sources=item.partial_sources | item.full_sources,
            )
            pinned[idx] = (shortened, new_n)
            total += new_n - n
        return pinned
