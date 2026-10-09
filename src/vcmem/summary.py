"""Recursive summaries of evicted messages.

Each flush folds the evicted batch into the previous summary:
``summary_n = summarize(summary_{n-1}, evicted_n)``. Every unit of a summary carries the
ids of the messages it came from, so a summary can always be traced back to recall
storage.

``TextRankSummarizer`` is extractive (Mihalcea and Tarau, 2004): sentences are graph
nodes, edges are normalised word overlap, and PageRank picks the most central sentences
that fit the token limit. ``ProviderSummarizer`` asks an LLM for an abstractive summary and
falls back to TextRank when the provider fails.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from vcmem.messages import Message, split_sentences
from vcmem.providers.base import ProviderError, text_of
from vcmem.stores.bm25 import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.providers.base import ChatProvider
    from vcmem.tokens import TokenCounter

SUMMARY_HEADER = "[summary of evicted messages]"


@dataclass(frozen=True, slots=True)
class SummaryUnit:
    text: str
    sources: frozenset[str]
    position: int


@dataclass(frozen=True, slots=True)
class Summary:
    units: tuple[SummaryUnit, ...] = ()
    covers: frozenset[str] = field(default_factory=frozenset)
    generation: int = 0

    def render(self) -> str:
        if not self.units:
            return ""
        return "\n".join([SUMMARY_HEADER, *(u.text for u in self.units)])

    def to_json(self) -> str:
        return json.dumps(
            {
                "generation": self.generation,
                "covers": sorted(self.covers),
                "units": [[u.text, sorted(u.sources), u.position] for u in self.units],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Summary:
        data = json.loads(raw)
        units = tuple(SummaryUnit(t, frozenset(s), int(p)) for t, s, p in data["units"])
        return cls(units, frozenset(data["covers"]), int(data["generation"]))


class Summarizer(Protocol):
    def update(self, previous: Summary, evicted: Sequence[Message], max_tokens: int) -> Summary:
        """Fold ``evicted`` into ``previous`` within ``max_tokens`` (header included)."""
        ...


def textrank_scores(word_sets: Sequence[frozenset[str]], damping: float = 0.85) -> np.ndarray:
    n = len(word_sets)
    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(word_sets[i] & word_sets[j])
            if not overlap:
                continue
            denom = math.log(len(word_sets[i])) + math.log(len(word_sets[j]))
            if denom > 0:
                weights[i, j] = weights[j, i] = overlap / denom
    out_weight = weights.sum(axis=1)
    transition = np.divide(
        weights, out_weight[:, None], out=np.zeros_like(weights), where=out_weight[:, None] > 0
    )
    scores = np.ones(n, dtype=np.float64)
    for _ in range(100):
        updated = (1.0 - damping) + damping * (transition.T @ scores)
        if np.abs(updated - scores).max() < 1e-6:
            return updated
        scores = updated
    return scores


class TextRankSummarizer:
    def __init__(self, counter: TokenCounter, min_words: int = 3) -> None:
        self._counter = counter
        self.min_words = min_words

    def _units_of(self, message: Message) -> list[SummaryUnit]:
        prefix = (
            f"{message.speaker} ({message.timestamp}): "
            if message.timestamp
            else f"{message.speaker}: "
        )
        return [
            SummaryUnit(prefix + sentence, frozenset({message.id}), message.seq)
            for sentence in split_sentences(message.text)
            if len(set(tokenize(sentence))) >= self.min_words
        ]

    def update(self, previous: Summary, evicted: Sequence[Message], max_tokens: int) -> Summary:
        covers = previous.covers | {m.id for m in evicted}
        candidates = list(previous.units)
        seen = {u.text for u in candidates}
        for message in evicted:
            for unit in self._units_of(message):
                if unit.text not in seen:
                    seen.add(unit.text)
                    candidates.append(unit)
        remaining = max_tokens - self._counter.count(SUMMARY_HEADER)
        if not candidates or remaining <= 0:
            return Summary((), covers, previous.generation + 1)
        scores = textrank_scores([frozenset(tokenize(u.text)) for u in candidates])
        # Higher centrality first; among equals prefer the more recent sentence.
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], -candidates[i].position))
        chosen: list[SummaryUnit] = []
        for i in order:
            n = self._counter.count(candidates[i].text)
            if n <= remaining:
                chosen.append(candidates[i])
                remaining -= n
        chosen.sort(key=lambda u: u.position)
        return Summary(tuple(chosen), covers, previous.generation + 1)


SUMMARY_SYSTEM = (
    "You maintain a running summary of a conversation for an assistant whose context "
    "window is limited. Keep names, dates, plans and stated facts; drop greetings."
)


class ProviderSummarizer:
    """Abstractive recursive summary through the provider; TextRank when the call fails."""

    def __init__(
        self, provider: ChatProvider, counter: TokenCounter, fallback: Summarizer | None = None
    ) -> None:
        self._provider = provider
        self._counter = counter
        self._fallback = fallback or TextRankSummarizer(counter)
        self.fallbacks = 0

    def update(self, previous: Summary, evicted: Sequence[Message], max_tokens: int) -> Summary:
        limit = max_tokens - self._counter.count(SUMMARY_HEADER)
        if limit <= 0:
            return Summary((), previous.covers | {m.id for m in evicted}, previous.generation + 1)
        prior = "\n".join(u.text for u in previous.units) or "(empty)"
        batch = "\n".join(m.render() for m in evicted)
        prompt = (
            f"Current summary:\n{prior}\n\nMessages leaving the context window:\n{batch}\n\n"
            f"Write the updated summary in at most {limit} tokens."
        )
        try:
            text = text_of(self._provider, SUMMARY_SYSTEM, prompt)
        except ProviderError:
            self.fallbacks += 1
            return self._fallback.update(previous, evicted, max_tokens)
        text = self._counter.truncate(text, limit)
        covers = previous.covers | {m.id for m in evicted}
        # An abstractive summary cannot be traced sentence by sentence, so the single unit
        # links to everything folded into the chain so far.
        position = min((m.seq for m in evicted), default=0)
        return Summary((SummaryUnit(text, covers, position),), covers, previous.generation + 1)
