from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vcmem.messages import Message
from vcmem.providers import ChatResponse, FakeProvider, ProviderError
from vcmem.summary import ProviderSummarizer, Summary, TextRankSummarizer, textrank_scores
from vcmem.tokens import RegexTokenCounter

counter = RegexTokenCounter()
VOCAB = ["garden", "tomato", "river", "boat", "guitar", "concert", "Lisbon", "cake", "dog"]

sentence = st.lists(st.sampled_from(VOCAB), min_size=3, max_size=8).map(
    lambda ws: " ".join(ws) + "."
)
batches = st.lists(st.lists(sentence, min_size=1, max_size=3), min_size=1, max_size=6)


def _messages(batch: list[list[str]], offset: int) -> list[Message]:
    return [
        Message(f"m{offset + i}", " ".join(sents), "Ana", seq=offset + i)
        for i, sents in enumerate(batch)
    ]


@settings(max_examples=100)
@given(st.lists(batches, min_size=1, max_size=4), st.integers(min_value=10, max_value=80))
def test_recursive_summary_keeps_source_links(rounds: list, limit: int) -> None:
    summarizer = TextRankSummarizer(counter)
    summary = Summary()
    by_id: dict[str, Message] = {}
    offset = 0
    for batch in rounds:
        evicted = _messages(batch, offset)
        offset += len(evicted)
        by_id.update({m.id: m for m in evicted})
        summary = summarizer.update(summary, evicted, limit)
        assert counter.count(summary.render()) <= limit
        assert summary.covers == frozenset(by_id)
        for unit in summary.units:
            assert unit.sources and unit.sources <= summary.covers
            # extractive: every unit is a sentence of one of its source messages
            body = unit.text.split(": ", 1)[1]
            assert any(body in by_id[s].text for s in unit.sources)
    assert summary.generation == len(rounds)


def test_textrank_prefers_central_sentences() -> None:
    sets = [
        frozenset({"garden", "tomato", "river"}),
        frozenset({"garden", "tomato", "boat"}),
        frozenset({"garden", "tomato", "cake"}),
        frozenset({"guitar", "concert", "lisbon"}),
    ]
    scores = textrank_scores(sets)
    assert scores[3] < min(scores[:3])


def test_summary_json_round_trip() -> None:
    summarizer = TextRankSummarizer(counter)
    s = summarizer.update(Summary(), _messages([["garden tomato river boat."]], 0), 40)
    assert Summary.from_json(s.to_json()) == s


def test_provider_summarizer_uses_provider_and_links_whole_chain() -> None:
    provider = FakeProvider(["Ana grows tomatoes by the river."])
    summarizer = ProviderSummarizer(provider, counter)
    out = summarizer.update(Summary(), _messages([["garden tomato river."], ["boat."]], 0), 40)
    assert out.units[0].text == "Ana grows tomatoes by the river."
    assert out.units[0].sources == frozenset({"m0", "m1"})
    prompt = provider.calls[0][0][1].content or ""
    assert "Messages leaving the context window" in prompt


def test_provider_summarizer_falls_back_to_textrank_on_failure() -> None:
    def broken(_messages: object) -> ChatResponse:
        raise ProviderError("down")

    summarizer = ProviderSummarizer(FakeProvider(broken), counter)
    out = summarizer.update(Summary(), _messages([["garden tomato river boat."]], 0), 40)
    assert summarizer.fallbacks == 1
    assert out.units and out.units[0].sources == frozenset({"m0"})
