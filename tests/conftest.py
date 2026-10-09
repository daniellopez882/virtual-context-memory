from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vcmem.build import build_manager
from vcmem.messages import Message

if TYPE_CHECKING:
    from vcmem.manager import MemoryManager

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_locomo() -> Path:
    return FIXTURES / "mini_locomo.json"


def chat(n: int, speakers: tuple[str, str] = ("Ana", "Ben")) -> list[Message]:
    """Distinct messages, each carrying a unique keyword ``tag<i>`` so it can be found."""
    topics = ["garden", "guitar", "Lisbon", "pottery", "marathon", "kitten", "museum", "recipe"]
    return [
        Message(
            id=f"m{i}",
            text=f"I talked about {topics[i % len(topics)]} today, note tag{i}.",
            speaker=speakers[i % 2],
            timestamp="1 May 2023",
        )
        for i in range(n)
    ]


@pytest.fixture
def small_manager() -> MemoryManager:
    return build_manager(budget=400, policy="lru", core_blocks=("Ana", "Ben"))
