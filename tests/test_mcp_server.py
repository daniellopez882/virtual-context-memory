from __future__ import annotations

from typing import Any

import anyio
from mcp import Client

from vcmem.build import build_manager
from vcmem.mcp_server import create_server


def _run(scenario: Any) -> Any:
    manager = build_manager(budget=400, policy="arc")
    server = create_server(manager)

    async def main() -> Any:
        # legacy mode drives the real JSON-RPC stream loop over in-memory streams
        async with Client(server, mode="legacy", raise_exceptions=True) as client:
            return await scenario(client)

    return manager, anyio.run(main)


def test_tools_are_listed() -> None:
    async def scenario(client: Client) -> set[str]:
        listed = await client.list_tools()
        return {t.name for t in listed.tools}

    _, names = _run(scenario)
    assert names == {
        "append_message",
        "core_memory_append",
        "core_memory_replace",
        "archival_insert",
        "archival_search",
        "conversation_search",
        "get_context",
    }


def test_memory_round_trip_over_mcp() -> None:
    async def scenario(client: Client) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for i in range(30):
            text = f"Message {i}: I practised the cello, note tag{i}."
            await client.call_tool("append_message", {"text": text, "speaker": "Ana"})
        await client.call_tool("core_memory_append", {"block": "human", "content": "plays cello"})
        replaced = await client.call_tool(
            "core_memory_replace",
            {"block": "human", "old_content": "cello", "new_content": "viola"},
        )
        out["replaced"] = replaced.is_error
        missing = await client.call_tool(
            "core_memory_replace", {"block": "human", "old_content": "drums", "new_content": "x"}
        )
        out["missing_is_error"] = missing.is_error
        row = await client.call_tool("archival_insert", {"content": "Ana's teacher is Rosa"})
        out["row"] = row.structured_content
        ctx = await client.call_tool("get_context", {"question": "Who teaches Ana? tag2"})
        out["before"] = ctx.structured_content
        search = await client.call_tool("conversation_search", {"query": "tag2"})
        out["search"] = search.structured_content
        arch = await client.call_tool("archival_search", {"query": "teacher Rosa"})
        out["archival"] = arch.structured_content
        ctx = await client.call_tool("get_context", {})
        out["after"] = ctx.structured_content
        return out

    manager, out = _run(scenario)
    assert out["replaced"] is False
    assert out["missing_is_error"] is True
    assert out["row"] == {"result": 0}
    assert out["search"]["results"][0]["id"] == "m2"
    assert out["search"]["paged_in"] == ["m2"]
    assert out["archival"]["results"][0]["shown"] is True
    after = out["after"]
    assert "tag2" in after["text"] and "Rosa" in after["text"] and "viola" in after["text"]
    assert after["tokens"] <= after["budget"] == 400
    assert len(manager.recall) == 30
