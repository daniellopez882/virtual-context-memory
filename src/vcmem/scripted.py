"""A deterministic agent that drives the memory tools by fixed rules.

It implements the provider interface, so it runs through the same heartbeat loop, tool
schemas and dispatcher as an LLM would. The replay benchmark uses it to measure paging
policies without an LLM in the loop. Its rules:

On a conversation message
  1. if the message contains a recall cue ("remember", "last time", "how's the ..."),
     ``conversation_search`` with the message text (one page);
  2. if it contains a first-person statement, ``core_memory_append`` that sentence to the
     speaker's block; when the block is full, ``core_memory_replace`` its oldest line;
  3. if its heuristic importance is at least ``archive_threshold``, ``archival_insert`` it.

On a question
  1. ``conversation_search`` with the question, following pages until
     ``question_fill`` of the queue capacity has been paged in or ``max_pages`` is reached;
  2. ``archival_search`` with the question.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from vcmem.messages import split_sentences
from vcmem.providers.base import ChatResponse, ToolCall
from vcmem.scoring import DISCLOSURE
from vcmem.stores.bm25 import tokenize
from vcmem.tools import ERROR_PREFIX

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.manager import MemoryManager
    from vcmem.messages import Message
    from vcmem.providers.base import ChatMessage
    from vcmem.scoring import ImportanceScorer

RECALL_CUE = re.compile(
    r"\b(remember|recall|last time|you (?:said|mentioned|told)|mentioned|again|still|"
    r"any (?:news|updates?)|how(?:'s| is| was| did| are| were) (?:the|your|that|it|things)|"
    r"ago|back then)\b",
    re.IGNORECASE,
)

_Call = tuple[str, dict[str, Any]]


class ScriptedMemoryAgent:
    def __init__(
        self,
        manager: MemoryManager,
        importance: ImportanceScorer,
        *,
        archive_threshold: float = 0.6,
        question_fill: float = 0.5,
        max_pages: int = 4,
        core_line_tokens: int = 40,
    ) -> None:
        self.manager = manager
        self.importance = importance
        self.archive_threshold = archive_threshold
        self.question_fill = question_fill
        self.max_pages = max_pages
        self.core_line_tokens = core_line_tokens
        self._pending: list[_Call] = []
        self._last: _Call | None = None
        self._paged_tokens = 0
        self._calls = 0

    # ------------------------------------------------------------- planning
    def _disclosure(self, message: Message) -> str | None:
        for sentence in split_sentences(message.text):
            if DISCLOSURE.search(sentence) and len(set(tokenize(sentence))) >= 4:
                line = f"{message.timestamp}: {sentence}" if message.timestamp else sentence
                return self.manager.counter.truncate(line, self.core_line_tokens)
        return None

    def _plan(self) -> list[_Call]:
        step = self.manager.step_input
        if step is None:
            return []
        self._paged_tokens = 0
        if step.message is None:
            return [
                ("conversation_search", {"query": step.text, "page": 0}),
                ("archival_search", {"query": step.text, "page": 0}),
            ]
        message = step.message
        plan: list[_Call] = []
        if RECALL_CUE.search(message.text):
            plan.append(("conversation_search", {"query": message.text, "page": 0}))
        line = self._disclosure(message)
        if line is not None and message.speaker in self.manager.core.blocks:
            plan.append(
                (
                    "core_memory_append",
                    {"block": message.speaker, "content": line, "source_ids": [message.id]},
                )
            )
        if self.importance.score(message) >= self.archive_threshold:
            plan.append(
                ("archival_insert", {"content": message.render(), "source_ids": [message.id]})
            )
        return plan

    def _react(self, result: str) -> None:
        if self._last is None:
            return
        name, args = self._last
        if name == "core_memory_append" and result.startswith(ERROR_PREFIX):
            block = self.manager.core.blocks.get(args["block"])
            if block is not None and block.lines:
                self._pending.insert(
                    0,
                    (
                        "core_memory_replace",
                        {
                            "block": args["block"],
                            "old_content": block.lines[0].text,
                            "new_content": args["content"],
                            "source_ids": args["source_ids"],
                        },
                    ),
                )
        elif name == "conversation_search" and not result.startswith(ERROR_PREFIX):
            step = self.manager.step_input
            if step is None or not step.is_question:
                return
            page = json.loads(result)
            self._paged_tokens += int(page["paged_in_tokens"])
            next_page = int(page["page"]) + 1
            budget = self.question_fill * self.manager.plan.queue_capacity
            if page["has_more"] and next_page < self.max_pages and self._paged_tokens < budget:
                self._pending.insert(0, ("conversation_search", {**args, "page": next_page}))

    # ------------------------------------------------------------- provider
    def complete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]] | None = None
    ) -> ChatResponse:
        if messages and messages[-1].role == "tool":
            self._react(messages[-1].content or "")
        else:
            self._pending = self._plan()
        if not self._pending:
            self._last = None
            return ChatResponse()
        name, args = self._pending.pop(0)
        self._last = (name, args)
        self._calls += 1
        call = ToolCall(f"call-{self._calls}", name, {**args, "request_heartbeat": True})
        return ChatResponse(tool_calls=(call,))
