"""A deterministic provider for tests and offline runs: canned responses, recorded calls."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from vcmem.providers.base import ChatMessage, ChatResponse, ProviderError

Responder = Callable[[Sequence[ChatMessage]], ChatResponse | str]


class FakeProvider:
    """Answers from a fixed script (in order) or from a function of the messages."""

    def __init__(self, script: Iterable[ChatResponse | str] | Responder) -> None:
        if callable(script):
            self._responder: Responder | None = script
            self._script: list[ChatResponse | str] = []
        else:
            self._responder = None
            self._script = list(script)
        self.calls: list[tuple[list[ChatMessage], list[dict[str, Any]] | None]] = []

    def complete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatResponse:
        self.calls.append((list(messages), list(tools) if tools is not None else None))
        if self._responder is not None:
            out = self._responder(messages)
        elif self._script:
            out = self._script.pop(0)
        else:
            raise ProviderError("fake provider script exhausted")
        return ChatResponse(content=out) if isinstance(out, str) else out
