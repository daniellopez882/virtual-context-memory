"""The provider interface every LLM-backed step goes through.

Plain dataclasses rather than pydantic models: the heartbeat loop builds these on every
step of a replay, and validation there would only cost time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

ChatRole = Literal["system", "user", "assistant", "tool"]


class ProviderError(RuntimeError):
    """The provider could not produce a usable response."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Set when the model emitted arguments that are not valid JSON; the dispatcher turns
    # that into an error the model can see and correct.
    raw_arguments: str | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class ChatProvider(Protocol):
    def complete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatResponse: ...


def text_of(provider: ChatProvider, system: str, user: str) -> str:
    """One-shot completion without tools; raises ``ProviderError`` on an empty answer."""
    response = provider.complete(
        [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    )
    if not response.content or not response.content.strip():
        raise ProviderError("provider returned no text")
    return response.content.strip()
