from __future__ import annotations

import json

import httpx
import pytest

from vcmem.messages import Message
from vcmem.providers import (
    ChatMessage,
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ToolCall,
)
from vcmem.qa import token_f1
from vcmem.scoring import HeuristicImportance, ProviderImportance


def _provider(handler: httpx.MockTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "http://llm.test/v1/", "test-model", api_key="k", transport=handler
    )


def test_openai_adapter_sends_tools_and_parses_tool_calls() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "archival_search", "arguments": '{"query": "dog"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "archival_search", "arguments": "{broken"},
                },
            ],
        }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    provider = _provider(httpx.MockTransport(handle))
    history = [
        ChatMessage("user", "hi"),
        ChatMessage("assistant", None, (ToolCall("c0", "send_message", {"message": "x"}),)),
        ChatMessage("tool", "sent", tool_call_id="c0"),
    ]
    tools = [{"type": "function", "function": {"name": "archival_search", "parameters": {}}}]
    response = provider.complete(history, tools)
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model" and body["tool_choice"] == "auto"
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"message": "x"}'
    assert body["messages"][2]["tool_call_id"] == "c0"
    assert response.tool_calls[0] == ToolCall("call_1", "archival_search", {"query": "dog"})
    assert response.tool_calls[1].raw_arguments == "{broken"


def test_openai_adapter_raises_provider_error_on_http_failure() -> None:
    provider = _provider(httpx.MockTransport(lambda r: httpx.Response(503, text="busy")))
    with pytest.raises(ProviderError, match="503"):
        provider.complete([ChatMessage("user", "hi")])
    garbage = _provider(httpx.MockTransport(lambda r: httpx.Response(200, json={"x": 1})))
    with pytest.raises(ProviderError, match="unexpected"):
        garbage.complete([ChatMessage("user", "hi")])


def test_provider_importance_parses_rating() -> None:
    scorer = ProviderImportance(FakeProvider(["Rating: 7"]))
    assert scorer.score(Message("a", "I got into college")) == pytest.approx(6 / 9)
    with pytest.raises(ProviderError):
        ProviderImportance(FakeProvider(["no idea"])).score(Message("b", "x"))


def test_heuristic_importance_ranks_facts_above_chit_chat() -> None:
    h = HeuristicImportance()
    fact = h.score(Message("a", "I moved to Porto in May 2023 to work at the Lisbon Clinic."))
    chat = h.score(Message("b", "Haha yes, totally."))
    assert 0.0 <= chat < fact <= 1.0


def test_token_f1() -> None:
    assert token_f1("The Lisbon marathon", "lisbon marathon") == 1.0
    assert token_f1("Porto", "Lisbon") == 0.0
    assert token_f1("in May 2023", "7 May 2023") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
