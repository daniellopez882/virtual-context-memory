"""MCP server exposing the memory tools (stdio, or streamable HTTP with a health route)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from starlette.responses import JSONResponse

from vcmem.messages import Message
from vcmem.tools import ArchivalPage, ConversationPage, MemoryTools

if TYPE_CHECKING:
    from starlette.requests import Request

    from vcmem.manager import MemoryManager

INSTRUCTIONS = (
    "Paged memory for a long conversation. Call append_message for every turn. "
    "conversation_search pages old messages back into context; archival_insert and "
    "archival_search keep facts outside the context window; get_context returns the "
    "token-budgeted prompt. Pass a question to get_context to start a new step for it, "
    "run searches, then call get_context without a question to see the paged result."
)


class AppendResult(BaseModel):
    id: str
    queue_tokens: int
    evictions_total: int


class ContextView(BaseModel):
    tokens: int
    budget: int
    resident_messages: int
    text: str


def create_server(manager: MemoryManager, page_size: int = 5) -> MCPServer:
    tools = MemoryTools(manager, page_size)
    server = MCPServer(name="virtual-context-memory", instructions=INSTRUCTIONS)
    # Tool handlers may run on worker threads; the manager is not thread-safe.
    lock = threading.Lock()

    @server.tool()
    def append_message(
        text: str,
        speaker: str = "user",
        role: Literal["user", "assistant"] = "user",
        timestamp: str = "",
    ) -> AppendResult:
        """Store a conversation turn in recall storage and admit it to the context queue."""
        with lock:
            stored = manager.ingest(
                Message(f"m{len(manager.recall)}", text, speaker, role, timestamp)
            )
            return AppendResult(
                id=stored.id,
                queue_tokens=manager.queue_tokens,
                evictions_total=manager.stats.evictions,
            )

    @server.tool()
    def core_memory_append(block: str, content: str) -> str:
        """Append a line to a core memory block (always in context)."""
        with lock:
            return tools.core_memory_append(block, content)

    @server.tool()
    def core_memory_replace(block: str, old_content: str, new_content: str) -> str:
        """Replace text in a core memory block; empty new_content deletes the line."""
        with lock:
            return tools.core_memory_replace(block, old_content, new_content)

    @server.tool()
    def archival_insert(content: str) -> int:
        """Save a passage to archival storage; returns its row number."""
        with lock:
            return tools.archival_insert(content)

    @server.tool()
    def archival_search(query: str, page: int = 0) -> ArchivalPage:
        """Semantic search over archival storage; hits are shown in the archival region."""
        with lock:
            return tools.archival_search(query, page)

    @server.tool()
    def conversation_search(query: str, page: int = 0) -> ConversationPage:
        """Keyword (BM25) search over every stored message; hits are paged into context."""
        with lock:
            return tools.conversation_search(query, page)

    @server.tool()
    def get_context(question: str = "") -> ContextView:
        """Return the assembled prompt. A non-empty question starts a new step for it."""
        with lock:
            if question:
                manager.set_question(question)
            context = manager.assemble()
            return ContextView(
                tokens=context.tokens,
                budget=context.budget,
                resident_messages=len(manager.resident_ids()),
                text=context.render(),
            )

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "messages": len(manager.recall),
                "archival_passages": len(manager.archival),
                "embedder": type(manager.embedder).__name__,
                "policy": manager.policy.name,
                "budget": manager.plan.total,
            }
        )

    return server
