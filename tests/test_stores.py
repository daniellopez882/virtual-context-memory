from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vcmem.embed import HashingEmbedder
from vcmem.messages import Message
from vcmem.stores import (
    ArchivalStore,
    BM25Index,
    CoreLine,
    CoreMemory,
    CoreMemoryError,
    CosineIndex,
    Database,
    RecallStore,
)
from vcmem.tokens import RegexTokenCounter

if TYPE_CHECKING:
    from pathlib import Path


def test_bm25_matches_hand_computed_score() -> None:
    index = BM25Index(k1=1.2, b=0.75)
    index.add("apple banana")
    index.add("apple apple cherry")
    index.add("durian")
    scores = index.scores("cherry")
    # df(cherry)=1, N=3 -> idf = ln(1 + 2.5/1.5); doc 1 has tf=1, len 3; avg len 6/3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2)
    assert scores == pytest.approx({1: idf * 1 * 2.2 / (1 + norm)})


def test_bm25_ignores_stopwords_and_unknown_terms() -> None:
    index = BM25Index()
    index.add("the dog barked")
    assert index.scores("the and of") == {}
    assert index.scores("zebra") == {}
    assert set(index.scores("dog")) == {0}


def test_cosine_index_ranks_by_similarity_and_grows() -> None:
    index = CosineIndex(dim=2, initial_capacity=1)
    for vec in ([1, 0], [0, 1], [1, 1]):
        index.add(np.asarray(vec, dtype=np.float32))
    top = index.search(np.asarray([1, 0.1], dtype=np.float32), k=2)
    assert [i for i, _ in top] == [0, 2]
    assert top[0][1] == pytest.approx(1 / math.sqrt(1.01), rel=1e-5)
    with pytest.raises(ValueError, match="dim"):
        index.add(np.zeros(3, dtype=np.float32))


def test_recall_store_round_trips_through_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    db = Database(path)
    store = RecallStore(db)
    a = store.append(Message("a", "I bought a red bicycle", "Ana", "user", "1 May 2023", 1))
    store.append(Message("b", "The bicycle has a basket", "Ben", "assistant", "", 1))
    assert a.seq == 0
    with pytest.raises(ValueError, match="duplicate"):
        store.append(Message("a", "again"))
    db.close()

    reopened = RecallStore(Database(path))
    assert [m.id for m in reopened.all()] == ["a", "b"]
    assert reopened.get("a") == a
    hits, total = reopened.search("bicycle basket", k=1)
    assert total == 2
    assert hits[0].message.id == "b"
    page2, _ = reopened.search("bicycle basket", k=1, offset=1)
    assert page2[0].message.id == "a"


def test_archival_store_round_trips_vectors_and_sources(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    embedder = HashingEmbedder(dim=64)
    store = ArchivalStore(Database(path), embedder)
    store.insert("Ana adopted a greyhound named Pixel", sources=["m3"], timestamp="2 May 2023")
    store.insert("Ben runs the Lisbon half marathon")
    reopened = ArchivalStore(Database(path), embedder)
    assert len(reopened) == 2
    hits, total = reopened.search("greyhound Pixel", k=1)
    assert total == 2
    assert hits[0].passage.sources == frozenset({"m3"})
    assert hits[0].passage.timestamp == "2 May 2023"
    with pytest.raises(ValueError, match="dim"):
        ArchivalStore(Database(path), HashingEmbedder(dim=32))


def test_core_memory_limits_replace_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    counter = RegexTokenCounter()
    core = CoreMemory(Database(path), counter, {"human": 12})
    core.append("human", "likes hiking", sources=["m1"])
    with pytest.raises(CoreMemoryError, match="limit"):
        core.append("human", "has a very long list of hobbies and pets")
    core.replace("human", "hiking", "climbing", sources=["m2"])
    with pytest.raises(CoreMemoryError, match="not found"):
        core.replace("human", "sailing", "x")
    with pytest.raises(CoreMemoryError, match="unknown block"):
        core.append("persona", "x")

    reopened = CoreMemory(Database(path), counter, {"human": 12})
    line = reopened.blocks["human"].lines[0]
    assert line.text == "likes climbing"
    assert line.sources == frozenset({"m1", "m2"})
    reopened.replace("human", "likes climbing", "")
    assert reopened.blocks["human"].lines == []


def test_core_replace_of_a_whole_line_drops_the_old_links() -> None:
    core = CoreMemory(Database(), RegexTokenCounter(), {"human": 12})
    core.append("human", "I adopted a dog", sources=["m1"])
    for i in range(2, 5):
        old = core.blocks["human"].lines[0].text
        core.replace("human", old, f"I moved to city number {i}", sources=[f"m{i}"])
    line = core.blocks["human"].lines[0]
    assert line.text == "I moved to city number 4"
    assert line.sources == frozenset({"m4"})
    # leaving only punctuation of the old text counts as replacing the whole line
    core.replace("human", "I moved to city number 4", "likes tea.", sources=["m5"])
    core.replace("human", "likes tea", "has a cat", sources=["m6"])
    assert core.blocks["human"].lines[0] == CoreLine("has a cat.", frozenset({"m6"}))
    # a partial edit keeps the links of the words that survive
    core.replace("human", "cat", "kitten", sources=["m7"])
    assert core.blocks["human"].lines[0].sources == frozenset({"m6", "m7"})


def test_core_restore_rewrites_the_persisted_blocks(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    counter = RegexTokenCounter()
    core = CoreMemory(Database(path), counter, {"human": 12})
    core.append("human", "likes hiking", sources=["m1"])
    snapshot = core.snapshot()
    core.replace("human", "likes hiking", "likes sailing", sources=["m2"])
    core.restore(snapshot)
    reopened = CoreMemory(Database(path), counter, {"human": 12})
    assert reopened.blocks == snapshot


def test_recall_store_keeps_message_vectors(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    store = RecallStore(Database(path))
    vec = np.arange(4, dtype=np.float32)
    store.append(Message("a", "first"), vec)
    store.append(Message("b", "second"))
    assert store.vector("b") is None
    store.set_vector("b", vec + 1)
    reopened = RecallStore(Database(path))
    np.testing.assert_array_equal(reopened.vector("a"), vec)
    np.testing.assert_array_equal(reopened.vector("b"), vec + 1)


def test_database_reports_size_and_kv(tmp_path: Path) -> None:
    db = Database(tmp_path / "mem.sqlite")
    db.set_kv("k", "v1")
    db.set_kv("k", "v2")
    assert db.get_kv("k") == "v2"
    assert db.get_kv("missing") is None
    assert db.size_bytes() > 0
