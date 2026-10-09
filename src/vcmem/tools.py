"""The memory tools an agent calls, with JSON schemas for function calling.

The same methods back the heartbeat loop (through ``dispatch``) and the MCP server.
``conversation_search`` is the page-fault path: the hits it returns are paged into the
queue, where the eviction policy sees them as accesses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from vcmem.stores.core import CoreMemoryError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.manager import MemoryManager
    from vcmem.providers.base import ToolCall

HEARTBEAT_HELP = "Set true to get control back after this call (to chain more calls)."


class CoreMemoryAppendArgs(BaseModel):
    """Append a line to a core memory block (always visible in context)."""

    block: str = Field(description="Name of the core memory block.")
    content: str = Field(description="Text to append.")
    source_ids: list[str] = Field(default_factory=list, description="Ids of source messages.")
    request_heartbeat: bool = Field(default=False, description=HEARTBEAT_HELP)


class CoreMemoryReplaceArgs(BaseModel):
    """Replace text inside a core memory block; an empty new_content deletes the line."""

    block: str = Field(description="Name of the core memory block.")
    old_content: str = Field(description="Exact text to replace.")
    new_content: str = Field(description="Replacement text.")
    source_ids: list[str] = Field(default_factory=list, description="Ids of source messages.")
    request_heartbeat: bool = Field(default=False, description=HEARTBEAT_HELP)


class ArchivalInsertArgs(BaseModel):
    """Save a passage to archival storage (unbounded, searched by meaning)."""

    content: str = Field(description="Text to store.")
    source_ids: list[str] = Field(default_factory=list, description="Ids of source messages.")
    request_heartbeat: bool = Field(default=False, description=HEARTBEAT_HELP)


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for.")
    page: int = Field(default=0, ge=0, description="Result page, starting at 0.")
    request_heartbeat: bool = Field(default=False, description=HEARTBEAT_HELP)


class ArchivalSearchArgs(SearchArgs):
    """Search archival storage by semantic similarity; results go to the archival region."""


class ConversationSearchArgs(SearchArgs):
    """Keyword search over the full message history; hits are paged into context."""


class SendMessageArgs(BaseModel):
    """Send a visible reply to the user and end the step."""

    message: str = Field(description="The reply.")


class ConversationHit(BaseModel):
    id: str
    speaker: str
    timestamp: str
    text: str
    score: float


class ConversationPage(BaseModel):
    query: str
    page: int
    total: int
    has_more: bool
    results: list[ConversationHit]
    paged_in: list[str]
    paged_in_tokens: int


class ArchivalHitModel(BaseModel):
    row: int
    text: str
    timestamp: str
    score: float
    shown: bool


class ArchivalPage(BaseModel):
    query: str
    page: int
    total: int
    has_more: bool
    results: list[ArchivalHitModel]


class ToolResult(BaseModel):
    ok: bool
    content: str


ERROR_PREFIX = "error: "


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=ERROR_PREFIX + message)


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "core_memory_append": CoreMemoryAppendArgs,
    "core_memory_replace": CoreMemoryReplaceArgs,
    "archival_insert": ArchivalInsertArgs,
    "archival_search": ArchivalSearchArgs,
    "conversation_search": ConversationSearchArgs,
    "send_message": SendMessageArgs,
}


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI function-calling schemas for every tool."""
    specs = []
    for name, model in TOOL_ARGS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        description = schema.pop("description", "") or (model.__doc__ or "").strip()
        specs.append(
            {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": schema},
            }
        )
    return specs


class MemoryTools:
    def __init__(self, manager: MemoryManager, page_size: int = 5) -> None:
        self.manager = manager
        self.page_size = page_size
        self.replies: list[str] = []

    def core_memory_append(self, block: str, content: str, source_ids: Sequence[str] = ()) -> str:
        self.manager.core.append(block, content, source_ids)
        return f"appended to {block}"

    def core_memory_replace(
        self, block: str, old_content: str, new_content: str, source_ids: Sequence[str] = ()
    ) -> str:
        self.manager.core.replace(block, old_content, new_content, source_ids)
        return f"replaced in {block}"

    def archival_insert(self, content: str, source_ids: Sequence[str] = ()) -> int:
        stamp = ""
        if self.manager.step_input is not None and self.manager.step_input.message is not None:
            stamp = self.manager.step_input.message.timestamp
        return self.manager.archival.insert(content, source_ids, stamp).row

    def archival_search(self, query: str, page: int = 0) -> ArchivalPage:
        hits, total = self.manager.archival.search(
            query, self.page_size, offset=page * self.page_size
        )
        shown = {h.passage.row for h in self.manager.show_archival(hits)}
        return ArchivalPage(
            query=query,
            page=page,
            total=total,
            has_more=(page + 1) * self.page_size < total,
            results=[
                ArchivalHitModel(
                    row=h.passage.row,
                    text=h.passage.text,
                    timestamp=h.passage.timestamp,
                    score=round(h.score, 4),
                    shown=h.passage.row in shown,
                )
                for h in hits
            ],
        )

    def conversation_search(self, query: str, page: int = 0) -> ConversationPage:
        hits, total = self.manager.recall.search(
            query, self.page_size, offset=page * self.page_size
        )
        paged = self.manager.page_in([h.message for h in hits])
        return ConversationPage(
            query=query,
            page=page,
            total=total,
            has_more=(page + 1) * self.page_size < total,
            results=[
                ConversationHit(
                    id=h.message.id,
                    speaker=h.message.speaker,
                    timestamp=h.message.timestamp,
                    text=h.message.text,
                    score=round(h.score, 4),
                )
                for h in hits
            ],
            paged_in=list(paged.admitted),
            paged_in_tokens=paged.tokens,
        )

    def send_message(self, message: str) -> str:
        self.replies.append(message)
        return "sent"

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Validate and run one tool call; errors come back as results the model can read."""
        model = TOOL_ARGS.get(call.name)
        if model is None:
            return _error(f"unknown tool {call.name!r}")
        if call.raw_arguments is not None:
            return _error(f"arguments are not valid JSON: {call.raw_arguments}")
        try:
            args = model.model_validate(call.arguments)
        except ValidationError as exc:
            return _error(f"invalid arguments: {exc.errors(include_url=False)}")
        params = args.model_dump(exclude={"request_heartbeat"})
        try:
            out = getattr(self, call.name)(**params)
        except CoreMemoryError as exc:
            return _error(str(exc))
        if isinstance(out, BaseModel):
            return ToolResult(ok=True, content=out.model_dump_json())
        return ToolResult(ok=True, content=json.dumps(out))
