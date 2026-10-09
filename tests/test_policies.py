from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vcmem.policies import (
    ARCPolicy,
    EvictionPolicy,
    FIFOPolicy,
    ImportancePolicy,
    LRUPolicy,
    PageInfo,
    make_policy,
    register_policy,
)


def info(size: int = 1, importance: float = 0.0, emb: list[float] | None = None) -> PageInfo:
    vec = None if emb is None else np.asarray(emb, dtype=np.float32)
    return PageInfo(size, importance, vec)


def test_fifo_ignores_hits() -> None:
    p = FIFOPolicy()
    for i, k in enumerate("abc"):
        p.admit(k, info(), i)
    p.touch("a", 5)
    assert p.victim(set()) == "a"
    assert p.victim({"a"}) == "b"


def test_lru_evicts_least_recently_used() -> None:
    p = LRUPolicy()
    for i, k in enumerate("abc"):
        p.admit(k, info(), i)
    p.touch("a", 5)
    assert p.victim(set()) == "b"
    p.evict("b")
    assert p.resident() == ["c", "a"]


def test_importance_policy_evicts_lowest_combined_score() -> None:
    p = ImportancePolicy()
    p.admit("old_dull", info(importance=0.1, emb=[1, 0]), now=0)
    p.admit("old_key", info(importance=0.9, emb=[1, 0]), now=0)
    p.admit("new_dull", info(importance=0.1, emb=[0, 1]), now=10)
    p.observe_query(np.asarray([0, 1], dtype=np.float32))
    assert p.victim(set()) == "old_dull"
    # without old_dull, recency and relevance outweigh old_key's importance
    assert p.victim({"old_dull"}) == "old_key"


def test_registry_builds_and_accepts_custom_policies() -> None:
    assert isinstance(make_policy("arc", 100), ARCPolicy)
    register_policy("lru-alias", lambda capacity: LRUPolicy())
    assert isinstance(make_policy("lru-alias", 10), LRUPolicy)
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("nope", 10)


def test_arc_promotes_on_second_reference_and_adapts_on_ghost_hits() -> None:
    arc = ARCPolicy(capacity=4)
    for i, k in enumerate("abcd"):
        arc.admit(k, info(), i)
    arc.touch("a", 5)
    assert list(arc.t2) == ["a"]
    victim = arc.victim(set())
    assert victim == "b"
    arc.evict(victim)
    assert "b" in arc.b1
    arc.admit("b", info(), 6)  # page fault on a B1 ghost: recency was undervalued
    assert arc.p > 0
    assert "b" in arc.t2
    assert arc.ghost_hits_b1 == 1


def test_arc_resists_a_scan() -> None:
    """Pages referenced twice survive a long run of one-off pages."""
    arc = ARCPolicy(capacity=6)
    resident: dict[str, int] = {}

    def admit(key: str, now: int) -> None:
        arc.admit(key, info(), now)
        resident[key] = 1
        while sum(resident.values()) > arc.c:
            v = arc.victim(set())
            assert v is not None
            arc.evict(v)
            del resident[v]

    for i, k in enumerate("hot1 hot2".split()):
        admit(k, i)
        arc.touch(k, i)
    for i in range(30):
        admit(f"scan{i}", 10 + i)
    assert {"hot1", "hot2"} <= set(arc.resident())


ops = st.lists(
    st.tuples(
        st.sampled_from(["admit", "touch"]),
        st.integers(min_value=0, max_value=25),
        st.integers(min_value=1, max_value=12),
    ),
    max_size=200,
)


@settings(max_examples=200)
@given(ops, st.integers(min_value=12, max_value=60), st.sets(st.integers(0, 25), max_size=3))
def test_arc_invariants_hold_under_manager_style_eviction(
    sequence: list[tuple[str, int, int]], capacity: int, pinned_ids: set[int]
) -> None:
    arc = ARCPolicy(capacity)
    sizes: dict[str, int] = {}
    pinned = {f"k{i}" for i in pinned_ids}
    for now, (op, key_id, size) in enumerate(sequence):
        key = f"k{key_id}"
        if op == "touch":
            arc.touch(key, now)
            continue
        if key in arc:
            arc.touch(key, now)
            continue
        arc.admit(key, info(size), now)
        sizes[key] = size
        # the manager evicts until the resident set fits again
        while arc.size_of("t1") + arc.size_of("t2") > capacity:
            victim = arc.victim(pinned)
            if victim is None:
                break
            assert victim not in pinned
            arc.evict(victim)
        pinned_resident = sum(sizes[k] for k in pinned if k in arc)
        if pinned_resident <= capacity:
            assert arc.size_of("t1") + arc.size_of("t2") <= capacity
            assert arc.size_of("t1") + arc.size_of("b1") <= capacity
        total = sum(arc.size_of(x) for x in ("t1", "t2", "b1", "b2"))
        assert total <= 2 * capacity or not (arc.b1 or arc.b2)
        assert 0.0 <= arc.p <= capacity
        lists = [set(arc.t1), set(arc.t2), set(arc.b1), set(arc.b2)]
        assert sum(len(x) for x in lists) == len(set().union(*lists))


@pytest.mark.parametrize("name", ["fifo", "lru", "arc", "importance"])
def test_every_policy_respects_pins(name: str) -> None:
    p: EvictionPolicy = make_policy(name, 100)
    for i, k in enumerate("abc"):
        p.admit(k, info(5, 0.5, [1.0, 0.0]), i)
    assert p.victim({"a", "b", "c"}) is None
    v = p.victim({"a", "b"})
    assert v == "c"
