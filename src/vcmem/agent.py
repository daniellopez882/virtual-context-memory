"""The heartbeat loop: MemGPT-style tool chaining over the provider interface.

Each step assembles the context, asks the provider for a response and executes its tool
calls. A call with ``request_heartbeat=true`` (or a failed call, so the model can recover)
hands control back to the provider; otherwise the step ends. The tool calls and results
of a step are shown to the provider within that step only, and are not kept in the queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vcmem.providers.base import ChatMessage, ToolCall
from vcmem.tools import MemoryTools, ToolResult, tool_specs

if TYPE_CHECKING:
    from vcmem.manager import MemoryManager
    from vcmem.messages import Message
    from vcmem.providers.base import ChatProvider


@dataclass(slots=True)
class StepTrace:
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    reply: str | None = None
    provider_calls: int = 0
    hit_step_limit: bool = False


class HeartbeatAgent:
    def __init__(
        self,
        manager: MemoryManager,
        provider: ChatProvider,
        tools: MemoryTools | None = None,
        max_heartbeats: int = 8,
    ) -> None:
        self.manager = manager
        self.provider = provider
        self.tools = tools or MemoryTools(manager)
        self.max_heartbeats = max_heartbeats
        self._specs = tool_specs()

    def observe(self, message: Message) -> StepTrace:
        """Ingest a conversation message and let the agent manage memory around it."""
        self.manager.ingest(message)
        return self._run()

    def ask(self, question: str) -> StepTrace:
        """Run a step whose input is a question (not stored in recall)."""
        self.manager.set_question(question)
        return self._run()

    def _run(self) -> StepTrace:
        trace = StepTrace()
        transcript: list[ChatMessage] = []
        for _ in range(self.max_heartbeats):
            context = self.manager.assemble()
            prompt = context.to_chat_messages()
            response = self.provider.complete([*prompt, *transcript], self._specs)
            trace.provider_calls += 1
            if not response.tool_calls:
                trace.reply = response.content
                return trace
            transcript.append(ChatMessage("assistant", response.content, response.tool_calls))
            again = False
            for call in response.tool_calls:
                result = self.tools.dispatch(call)
                trace.calls.append(call)
                trace.results.append(result)
                transcript.append(ChatMessage("tool", result.content, tool_call_id=call.id))
                if call.name == "send_message" and result.ok:
                    trace.reply = str(call.arguments["message"])
                elif not result.ok or call.arguments.get("request_heartbeat"):
                    again = True
            if not again:
                return trace
        trace.hit_step_limit = True
        return trace
