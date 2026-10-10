from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
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


class ReferenceARC:
    """Fig. 4 of Megiddo and Modha (2003), unit-size pages, written as close to the paper
    as possible. Lists are ordered LRU first."""

    def __init__(self, c: int) -> None:
        self.c = c
        self.p = 0.0
        self.t1: list[str] = []
        self.t2: list[str] = []
        self.b1: list[str] = []
        self.b2: list[str] = []

    def _replace(self, x: str) -> None:
        if self.t1 and ((x in self.b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            self.b1.append(self.t1.pop(0))
        else:
            self.b2.append(self.t2.pop(0))

    def request(self, x: str) -> None:
        if x in self.t1 or x in self.t2:  # case I: hit
            (self.t1 if x in self.t1 else self.t2).remove(x)
            self.t2.append(x)
        elif x in self.b1:  # case II
            self.p = min(float(self.c), self.p + max(1.0, len(self.b2) / len(self.b1)))
            self._replace(x)
            self.b1.remove(x)
            self.t2.append(x)
        elif x in self.b2:  # case III
            self.p = max(0.0, self.p - max(1.0, len(self.b1) / len(self.b2)))
            self._replace(x)
            self.b2.remove(x)
            self.t2.append(x)
        else:  # case IV
            l1 = len(self.t1) + len(self.b1)
            total = l1 + len(self.t2) + len(self.b2)
            if l1 == self.c:
                if len(self.t1) < self.c:
                    self.b1.pop(0)
                    self._replace(x)
                else:
                    self.t1.pop(0)
            elif total >= self.c:
                if total == 2 * self.c:
                    self.b2.pop(0)
                self._replace(x)
            self.t1.append(x)


def drive_like_the_manager(arc: ARCPolicy, x: str, now: int) -> None:
    """A hit touches; a miss admits, then victims are evicted until the pages fit."""
    if x in arc:
        arc.touch(x, now)
        return
    arc.admit(x, info(1), now)
    while arc.size_of("t1") + arc.size_of("t2") > arc.c:
        victim = arc.victim(set())
        assert victim is not None
        assert victim != x
        arc.evict(victim)


@st.composite
def unit_traces(draw: st.DrawFn) -> tuple[int, list[str]]:
    c = draw(st.integers(min_value=1, max_value=8))
    keys = draw(st.integers(min_value=c + 1, max_value=4 * c + 2))
    trace = draw(st.lists(st.integers(0, keys - 1), max_size=150))
    return c, [f"k{k}" for k in trace]


@settings(max_examples=400, deadline=None)
@given(unit_traces())
@example((2, list("aabcbd")))  # |T1| must not count the page being admitted
@example((2, list("aabcbde")))  # B2 keeps its ghosts when T1's LRU is dropped
@example((5, [f"k{i}" for i in (0, 4, 8, 7, 6, 4, 7, 5, 9, 3, 8, 2, 4, 2, 1)]))
def test_arc_matches_the_paper_with_unit_sizes(case: tuple[int, list[str]]) -> None:
    c, trace = case
    arc = ARCPolicy(c)
    ref = ReferenceARC(c)
    for now, x in enumerate(trace):
        drive_like_the_manager(arc, x, now)
        ref.request(x)
        state = (list(arc.t1), list(arc.t2), list(arc.b1), list(arc.b2))
        assert state == (ref.t1, ref.t2, ref.b1, ref.b2), f"diverged at step {now} ({x})"
        assert arc.p == pytest.approx(ref.p)


@pytest.mark.parametrize("name", ["fifo", "lru", "arc", "importance"])
def test_every_policy_respects_pins(name: str) -> None:
    p: EvictionPolicy = make_policy(name, 100)
    for i, k in enumerate("abc"):
        p.admit(k, info(5, 0.5, [1.0, 0.0]), i)
    assert p.victim({"a", "b", "c"}) is None
    v = p.victim({"a", "b"})
    assert v == "c"
