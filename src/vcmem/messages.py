"""The unit the memory manager pages in and out: one conversation message."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


@dataclass(frozen=True, slots=True)
class Message:
    """A single message. ``seq`` is assigned by the manager on ingest when left at -1."""

    id: str
    text: str
    speaker: str = "user"
    role: Role = "user"
    timestamp: str = ""
    session: int = 0
    seq: int = -1

    def render(self) -> str:
        if self.timestamp:
            return f"{self.speaker} ({self.timestamp}): {self.text}"
        return f"{self.speaker}: {self.text}"

    def with_seq(self, seq: int) -> Message:
        return replace(self, seq=seq)


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation followed by whitespace; good enough for chat turns."""
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]
