from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING

import httpx
import pytest

from vcmem.assembler import ContextAssembler
from vcmem.assets import Asset, ChecksumError, fetch, sha256_of
from vcmem.bench import RunConfig, run_benchmark, write_results
from vcmem.bench.baselines import BASELINE_SYSTEM, ConversationIndex, baseline_context
from vcmem.bench.report import render_report
from vcmem.bench.runner import (
    Replay,
    ReplayPlan,
    comparison_pairs,
    paired_bootstrap,
    percentile,
)
from vcmem.bench.runner import (
    RunConfig as Config,
)
from vcmem.embed import HashingEmbedder
from vcmem.locomo import ABSTAIN, load_locomo, parse_conversation, parse_evidence, short_date
from vcmem.messages import Message
from vcmem.providers import FakeProvider
from vcmem.tokens import CachedCounter, RegexTokenCounter

if TYPE_CHECKING:
    from pathlib import Path


def test_evidence_parsing_handles_release_quirks() -> None:
    assert parse_evidence(["D8:6; D9:17"]) == ("D8:6", "D9:17")
    assert parse_evidence(["D:11:26", "D", "D9:1 D4:4"]) == ("D11:26", "D9:1", "D4:4")
    assert parse_evidence(["D1:3", "D1:3"]) == ("D1:3",)


def test_short_date() -> None:
    assert short_date("1:56 pm on 8 May, 2023") == "8 May 2023"
    assert short_date("sometime") == "sometime"


def test_loader_reads_fixture(mini_locomo: Path) -> None:
    (conv,) = load_locomo(mini_locomo)
    assert conv.speakers == ("Iris", "Tom")
    assert [s.date for s in conv.sessions] == ["2 May 2023", "20 May 2023", "8 June 2023"]
    messages = conv.messages()
    assert len(messages) == 18
    assert messages[9].text.endswith("[shared image: a photo of a climbing wall]")
    by_q = {q.question: q for q in conv.qa}
    assert by_q["Where is Iris's climbing course and when did it start?"].evidence == (
        "D2:4",
        "D2:6",
    )
    assert by_q["Who is visiting Granada?"].evidence == ("D3:4",)
    assert by_q["What did Miso break?"].evidence == ("D3:1",)
    assert conv.dropped_evidence == 1


def test_adversarial_gold_answer_is_an_abstention(mini_locomo: Path) -> None:
    # The release stores only the trap answer for adversarial questions; scoring it as
    # gold would reward a model for falling for the trap.
    (conv,) = load_locomo(mini_locomo)
    adversarial = [q for q in conv.qa if q.category == 5]
    assert len(adversarial) == 2
    assert all(q.answer == ABSTAIN for q in adversarial)
    raw = json.loads(mini_locomo.read_text(encoding="utf-8"))[0]["qa"]
    traps = {q["adversarial_answer"] for q in raw if q["category"] == 5}
    assert traps.isdisjoint(q.answer for q in adversarial)


def test_an_explicit_answer_wins_over_the_trap() -> None:
    sample = json.loads(
        '{"sample_id": "s", "conversation": {"speaker_a": "A", "speaker_b": "B"}, "qa": ['
        '{"question": "Is Oscar a pet?", "adversarial_answer": "Yes", "answer": "No",'
        ' "evidence": [], "category": 5}]}'
    )
    assert parse_conversation(sample).qa[0].answer == "No"


def test_replay_plan_schedules_fresh_probes(mini_locomo: Path) -> None:
    (conv,) = load_locomo(mini_locomo)
    plan = ReplayPlan.of(conv)
    assert len(plan.questions) == 7
    assert sorted(plan.session_end.items()) == [(5, 1), (11, 2), (17, 3)]
    assert {s: len(qs) for s, qs in plan.due.items()} == {1: 3, 2: 2, 3: 2}
    assert [m.seq for m in plan.messages] == list(range(18))


def test_paired_bootstrap_resamples_whole_conversations() -> None:
    same = {"c1": [1.0, 0.0], "c2": [0.5]}
    assert paired_bootstrap(same, same, 200) == (0.0, 0.0, 0.0)
    better = {"c1": [1.0, 1.0], "c2": [1.0], "c3": [0.0]}
    worse = {"c1": [0.0, 1.0], "c2": [0.0], "c3": [0.0]}
    diff, lo, hi = paired_bootstrap(better, worse, 500)
    assert diff == pytest.approx(2 / 4)
    assert 0.0 <= lo <= diff <= hi <= 1.0
    with pytest.raises(ValueError, match="same questions"):
        paired_bootstrap(better, {"c1": [0.0]}, 10)


def test_comparison_pairs_cover_policies_and_baselines() -> None:
    pairs = comparison_pairs(Config().methods)
    assert ("memgpt-importance", "memgpt-fifo") in pairs
    assert ("memgpt-fifo", "window") in pairs and ("memgpt-arc", "hybrid") in pairs
    assert all(a != b for a, b in pairs)


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.95) == 5


def _bench(mini_locomo: Path, tmp_path: Path, **kw: object) -> dict:
    config = RunConfig(budgets=(330, 2000), **kw)  # type: ignore[arg-type]
    return run_benchmark(
        config,
        mini_locomo,
        HashingEmbedder(dim=64),
        RegexTokenCounter(),
        command="vcmem bench (test)",
        repo=tmp_path,
    )


def test_benchmark_harness_end_to_end(mini_locomo: Path, tmp_path: Path) -> None:
    results = _bench(mini_locomo, tmp_path)
    for key in ("machine", "date", "git_sha", "command", "results", "dataset", "not_run"):
        assert key in results
    assert results["dataset"]["questions_scored"] == 7
    assert results["dataset"]["sha256_matches_pinned"] is False
    assert set(results["not_run"]) == {"qa_f1", "llm_agent"}
    rows = {(r["probe"], r["method"], r["budget"]): r for r in results["results"]}
    assert len(rows) == 2 * 8 * 2
    for row in rows.values():
        assert 0.0 <= row["evidence_recall"] <= row["evidence_recall_any"] <= 1.0
        assert row["context_tokens_max"] <= row["budget"]
    # a budget larger than the whole chat keeps everything for the window baseline
    assert rows[("final", "window", 2000)]["evidence_recall"] == 1.0
    assert rows[("final", "memgpt-arc", 330)]["turns"] == 18
    assert set(rows[("fresh", "hybrid", 330)]["by_conversation"]) == {"fixture-1"}
    # importance embeds every admitted page and each step's input; FIFO does not
    fifo = rows[("final", "memgpt-fifo", 330)]["embed_texts_per_turn"]
    assert rows[("final", "memgpt-importance", 330)]["embed_texts_per_turn"] > fifo + 1
    turn_tokens = results["dataset"]["turn_tokens"]
    assert turn_tokens["rendered_max"] > turn_tokens["text_max"] > turn_tokens["text_mean"] > 0
    assert results["dataset"]["recall_cue_turns"] >= 1
    pairs = {(r["probe"], r["budget"], r["a"], r["b"]) for r in results["paired_bootstrap"]["rows"]}
    assert ("fresh", 2000, "memgpt-arc", "memgpt-fifo") in pairs
    assert ("final", 330, "memgpt-lru", "window") in pairs
    out = tmp_path / "r.json"
    write_results(results, out)
    report = render_report(json.loads(out.read_text(encoding="utf-8")))
    assert "| memgpt-importance |" in report and "fresh probe" in report


def test_window_baseline_stops_at_the_first_message_that_does_not_fit() -> None:
    counter = RegexTokenCounter()
    old = Message("old", "hi", seq=0)
    mid = Message("mid", " ".join(["word"] * 40), seq=1)
    new = Message("new", "see you soon", seq=2)
    embedder = HashingEmbedder(dim=16)
    index = ConversationIndex(embedder.dim)
    index.extend([old, mid, new], embedder.embed([m.render() for m in (old, mid, new)]))
    question = "When?"
    fixed = counter.count(BASELINE_SYSTEM) + counter.count(question)
    # room for "new" and "old" but not for "mid": a sliding window keeps only "new"
    budget = fixed + counter.count(new.render()) + counter.count(old.render()) + 2
    assert counter.count(mid.render()) > budget - fixed
    ctx = baseline_context(
        "window", index, question, embedder.embed([question])[0], budget, ContextAssembler(counter)
    )
    assert ctx.full_sources() == {"new"}
    assert ctx.tokens <= budget


class _TallyCounter(RegexTokenCounter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[str] = Counter()

    def count(self, text: str) -> int:
        self.calls[text] += 1
        return super().count(text)


def test_each_paging_replay_counts_its_own_turns(mini_locomo: Path) -> None:
    # Latency per turn is compared across policies, so no replay may inherit token
    # counts cached by the one before it.
    (conv,) = load_locomo(mini_locomo)
    tally = _TallyCounter()
    replay = Replay(
        RunConfig(budgets=(330,), methods=("memgpt-fifo", "memgpt-lru")),
        HashingEmbedder(dim=64),
        CachedCounter(tally),
    )
    replay.run_conversation(conv)
    assert all(tally.calls[m.render()] == 2 for m in conv.messages())


def test_benchmark_scores_answers_when_a_provider_is_given(
    mini_locomo: Path, tmp_path: Path
) -> None:
    results = _bench(
        mini_locomo, tmp_path, methods=("bm25",), qa_provider=FakeProvider(lambda _m: "Miso")
    )
    assert "qa_f1" not in results["not_run"]
    assert all("qa_f1" in r for r in results["results"])


def test_fetch_verifies_checksums(tmp_path: Path) -> None:
    payload = b"hello locomo"
    good = Asset("x", "d/x.json", "http://files.test/x.json", "0" * 64, len(payload), "test")
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))
    client = httpx.Client(transport=transport)
    with pytest.raises(ChecksumError):
        fetch(tmp_path, [good], client=client, log=lambda _s: None)
    assert not (tmp_path / "d" / "x.json").exists()

    digest = hashlib.sha256(payload).hexdigest()
    ok = Asset("x", "d/x.json", "http://files.test/x.json", digest, len(payload), "test")
    paths = fetch(tmp_path, [ok], client=client, log=lambda _s: None)
    assert sha256_of(paths["x"]) == digest
    lines: list[str] = []
    fetch(tmp_path, [ok], client=client, log=lines.append)
    assert lines[0].startswith("ok")
