from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import chat
from vcmem.build import build_manager
from vcmem.embed import HashingEmbedder
from vcmem.manager import BudgetPlan, MemoryManager
from vcmem.messages import Message
from vcmem.policies import ARCPolicy
from vcmem.stores.sqlite import Database
from vcmem.tools import MemoryTools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np


def test_budget_plan_reserves_regions() -> None:
    plan = BudgetPlan.split(1000, system_tokens=50)
    assert plan.queue_capacity == 1000 - 50 - 120 - 100 - 100 - 64
    with pytest.raises(ValueError, match="leaves"):
        BudgetPlan.split(200, system_tokens=50)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=10, max_value=80),
    st.sampled_from(["fifo", "lru", "arc", "importance"]),
    st.integers(min_value=330, max_value=700),
)
def test_evicted_messages_stay_retrievable(n: int, policy: str, budget: int) -> None:
    manager = build_manager(budget=budget, policy=policy)
    messages = chat(n)
    for m in messages:
        manager.ingest(m)
        assert manager.queue_tokens <= manager.plan.queue_capacity
        assert manager.assemble().tokens <= budget
    resident = set(manager.resident_ids())
    evicted = [m for m in messages if m.id not in resident]
    if n > 25:
        assert evicted, "the budget should force evictions"
    for m in evicted:
        assert manager.recall.get(m.id) is not None
        hits, _ = manager.recall.search(f"tag{m.id[1:]}", k=1)
        assert hits[0].message.id == m.id
        assert m.id in manager.summary.covers


def test_page_in_admits_hits_and_touches_resident(small_manager: MemoryManager) -> None:
    messages = chat(40)
    for m in messages:
        small_manager.ingest(m)
    old = small_manager.recall.get("m0")
    newest = small_manager.recall.get("m39")
    assert old is not None and newest is not None
    assert "m0" not in small_manager.resident_ids()
    result = small_manager.page_in([old, newest])
    assert result.admitted == ("m0",)
    assert result.hits == ("m39",)
    assert "m0" in small_manager.resident_ids()
    assert small_manager.stats.page_ins == 1


def test_page_in_never_pins_more_than_capacity() -> None:
    manager = build_manager(budget=330, policy="fifo")
    messages = chat(60)
    for m in messages:
        manager.ingest(m)
    manager.set_question("anything")
    stored = [manager.recall.get(m.id) for m in messages[:40]]
    result = manager.page_in([m for m in stored if m is not None])
    assert result.skipped
    assert manager.queue_tokens <= manager.plan.queue_capacity


def test_checkpoint_restore_rewinds_in_context_state() -> None:
    manager = build_manager(budget=400, policy="arc")
    for m in chat(30):
        manager.ingest(m)
    state = manager.checkpoint()
    before = (manager.resident_ids(), manager.summary, manager.now)
    manager.set_question("tag3 tag4")
    old = [manager.recall.get(f"m{i}") for i in range(5)]
    manager.page_in([m for m in old if m is not None])
    assert manager.resident_ids() != before[0]
    manager.restore(state)
    assert (manager.resident_ids(), manager.summary, manager.now) == before
    assert isinstance(manager.policy, ARCPolicy)
    assert sorted(manager.policy.resident()) == sorted(before[0])
    # a checkpoint can be restored more than once
    manager.page_in([m for m in old if m is not None])
    manager.restore(state)
    assert manager.resident_ids() == before[0]


def test_question_is_rendered_last_and_not_stored(small_manager: MemoryManager) -> None:
    for m in chat(5):
        small_manager.ingest(m)
    small_manager.set_question("What did Ana talk about?")
    ctx = small_manager.assemble()
    assert ctx.items[-1].text == "What did Ana talk about?"
    assert len(small_manager.recall) == 5


def test_reopening_a_database_restores_summary_and_recent_queue(tmp_path: Path) -> None:
    path = tmp_path / "agent.sqlite"

    def open_manager() -> MemoryManager:
        return MemoryManager(
            Database(path), budget=400, policy="lru", embedder=HashingEmbedder(dim=32)
        )

    first = open_manager()
    for m in chat(40):
        first.ingest(m)
    summary = first.summary
    assert summary.units
    first.db.close()

    second = open_manager()
    assert second.summary == summary
    assert second.resident_ids()
    assert second.resident_ids()[-1] == "m39"
    assert second.queue_tokens <= second.plan.queue_capacity


def test_coverage_counts_full_text_only_when_present(small_manager: MemoryManager) -> None:
    small_manager.ingest(Message("x1", "I adopted a greyhound called Pixel.", "Ana"))
    small_manager.core.append("Ana", "adopted a greyhound", sources=["x1"])
    small_manager.archival.insert("Ana: I adopted a greyhound called Pixel.", sources=["x1"])
    hits, _ = small_manager.archival.search("greyhound", k=1)
    small_manager.show_archival(hits)
    ctx = small_manager.assemble()
    core_item = next(i for i in ctx.items if i.text.startswith("[Ana]"))
    assert core_item.partial_sources == frozenset({"x1"})
    archival_item = next(i for i in ctx.items if i.text.startswith("[archival"))
    assert archival_item.full_sources == frozenset({"x1"})


class _CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=32)
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.texts.extend(texts)
        return super().embed(texts)


def test_importance_encodes_each_text_once() -> None:
    embedder = _CountingEmbedder()
    manager = MemoryManager(Database(), budget=400, policy="importance", embedder=embedder)
    tools = MemoryTools(manager)
    messages = chat(40)
    for m in messages:
        manager.ingest(m)
        # the archival passage is the rendered message, already encoded for this step
        tools.archival_insert(m.render(), [m.id])
    assert len(embedder.texts) == len(set(embedder.texts)) == 40
    assert "m0" not in manager.resident_ids()
    manager.set_question("tag0 tag1")
    tools.conversation_search("tag0 tag1")
    tools.archival_search("tag0 tag1")
    assert "m0" in manager.resident_ids()
    # the page fault reads the stored vector; only the question is new
    assert embedder.texts[40:] == ["tag0 tag1"]
