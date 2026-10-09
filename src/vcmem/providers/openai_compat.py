"""Adapter for any server that speaks the OpenAI ``/chat/completions`` protocol.

That covers OpenAI itself, Ollama (``http://localhost:11434/v1``), vLLM and LM Studio.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import httpx

from vcmem.providers.base import ChatResponse, ProviderError, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.providers.base import ChatMessage

ENV_API_KEY = "VCMEM_LLM_API_KEY"


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def complete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"request to {self.base_url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {response.text[:300]}") from exc
        return ChatResponse(
            content=message.get("content"),
            tool_calls=tuple(_parse_call(c) for c in message.get("tool_calls") or ()),
        )

    def close(self) -> None:
        self._client.close()


def _parse_call(raw: dict[str, Any]) -> ToolCall:
    fn = raw.get("function", {})
    args_text = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_text)
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
    except ValueError:
        return ToolCall(raw.get("id", ""), fn.get("name", ""), {}, raw_arguments=args_text)
    return ToolCall(raw.get("id", ""), fn.get("name", ""), args)
