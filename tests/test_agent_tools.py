from __future__ import annotations

import json
from typing import TYPE_CHECKING

from conftest import chat
from vcmem.agent import HeartbeatAgent
from vcmem.build import build_manager
from vcmem.messages import Message
from vcmem.providers import ChatResponse, FakeProvider, ToolCall
from vcmem.scoring import HeuristicImportance
from vcmem.scripted import ScriptedMemoryAgent
from vcmem.tools import ERROR_PREFIX, MemoryTools, tool_specs

if TYPE_CHECKING:
    from vcmem.manager import MemoryManager


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=dict(args))


def test_tool_specs_are_openai_function_schemas() -> None:
    specs = {s["function"]["name"]: s for s in tool_specs()}
    assert set(specs) == {
        "core_memory_append",
        "core_memory_replace",
        "archival_insert",
        "archival_search",
        "conversation_search",
        "send_message",
    }
    params = specs["conversation_search"]["function"]["parameters"]
    assert params["required"] == ["query"]
    assert "request_heartbeat" in params["properties"]
    assert specs["conversation_search"]["function"]["description"]


def test_dispatch_validates_and_reports_errors(small_manager: MemoryManager) -> None:
    tools = MemoryTools(small_manager)
    assert tools.dispatch(call("nope")).content.startswith(ERROR_PREFIX)
    bad = tools.dispatch(call("archival_insert"))
    assert not bad.ok and "invalid arguments" in bad.content
    raw = tools.dispatch(ToolCall("c", "archival_insert", {}, raw_arguments="{oops"))
    assert not raw.ok and "not valid JSON" in raw.content
    full = tools.dispatch(call("core_memory_append", block="Ana", content="word " * 200))
    assert not full.ok and "limit" in full.content
    ok = tools.dispatch(call("archival_insert", content="Ana likes figs", request_heartbeat=True))
    assert ok.ok and json.loads(ok.content) == 0


def test_conversation_search_paginates_and_pages_in(small_manager: MemoryManager) -> None:
    for m in chat(40):
        small_manager.ingest(m)
    tools = MemoryTools(small_manager, page_size=2)
    small_manager.set_question("garden")
    first = tools.conversation_search("garden", page=0)
    second = tools.conversation_search("garden", page=1)
    assert first.total == second.total == 5
    assert first.has_more and second.has_more
    assert len(first.results) == len(second.results) == 2
    assert not {r.id for r in first.results} & {r.id for r in second.results}
    for r in first.results + second.results:
        assert r.id in small_manager.resident_ids()
    last = tools.conversation_search("garden", page=2)
    assert not last.has_more and len(last.results) == 1


def test_heartbeat_chains_until_no_heartbeat(small_manager: MemoryManager) -> None:
    provider = FakeProvider(
        [
            ChatResponse(
                tool_calls=(
                    call(
                        "archival_insert",
                        content="Ben's sister lives in Porto",
                        request_heartbeat=True,
                    ),
                )
            ),
            ChatResponse(tool_calls=(call("archival_search", query="sister Porto"),)),
            ChatResponse(content="never reached"),
        ]
    )
    agent = HeartbeatAgent(small_manager, provider)
    trace = agent.observe(Message("u1", "Where does my sister live?", "Ben"))
    assert [c.name for c in trace.calls] == ["archival_insert", "archival_search"]
    assert trace.provider_calls == 2
    # the second request carried the first tool result back to the provider
    second_request = provider.calls[1][0]
    assert second_request[-1].role == "tool"
    assert provider.calls[1][1] is not None


def test_failed_call_forces_a_heartbeat_and_send_message_replies(
    small_manager: MemoryManager,
) -> None:
    provider = FakeProvider(
        [
            ChatResponse(tool_calls=(call("core_memory_append", block="nobody", content="x"),)),
            ChatResponse(tool_calls=(call("send_message", message="Noted."),)),
        ]
    )
    trace = HeartbeatAgent(small_manager, provider).ask("hello?")
    assert trace.reply == "Noted."
    assert [r.ok for r in trace.results] == [False, True]


def test_step_limit_is_reported(small_manager: MemoryManager) -> None:
    looping = FakeProvider(
        lambda _m: ChatResponse(
            tool_calls=(call("archival_search", query="x", request_heartbeat=True),)
        )
    )
    trace = HeartbeatAgent(small_manager, looping, max_heartbeats=3).ask("loop")
    assert trace.hit_step_limit and trace.provider_calls == 3


def _run_scripted(policy: str) -> tuple[list[str], list[list[str]], str]:
    manager = build_manager(budget=360, policy=policy, core_blocks=("Ana", "Ben"))
    agent = HeartbeatAgent(manager, ScriptedMemoryAgent(manager, HeuristicImportance()))
    log: list[list[str]] = []
    texts = [
        ("Ana", "I adopted a greyhound called Pixel last week."),
        ("Ben", "I started training for the Lisbon half marathon."),
        ("Ana", "My sister Clara moved to Porto for a nursing job."),
        ("Ben", "Do you remember the greyhound you mentioned? How is Pixel?"),
    ] * 6
    for i, (speaker, text) in enumerate(texts):
        trace = agent.observe(Message(f"t{i}", text, speaker, timestamp="1 May 2023"))
        log.append([c.name for c in trace.calls])
    trace = agent.ask("What is the name of Ana's dog?")
    log.append([c.name for c in trace.calls])
    return manager.resident_ids(), log, manager.assemble().render()


def test_scripted_agent_is_deterministic_and_uses_all_tools() -> None:
    first = _run_scripted("arc")
    second = _run_scripted("arc")
    assert first == second
    calls = {name for step in first[1] for name in step}
    assert {
        "conversation_search",
        "core_memory_append",
        "core_memory_replace",
        "archival_insert",
        "archival_search",
    } <= calls
    assert "Pixel" in first[2]
