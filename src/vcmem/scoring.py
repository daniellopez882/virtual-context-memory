"""Importance scores in [0, 1] for the importance eviction policy and the scripted agent.

Generative Agents asks the LLM to rate each memory's poignancy from 1 to 10.
``ProviderImportance`` does exactly that through the provider interface.
``HeuristicImportance`` is the offline stand-in used by the benchmark: it rewards the
surface features that make a chat turn worth remembering (names, dates and numbers,
first-person statements of fact) and is labelled as a heuristic wherever it is reported.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Protocol

from vcmem.messages import split_sentences
from vcmem.providers.base import ProviderError, text_of
from vcmem.stores.bm25 import tokenize

if TYPE_CHECKING:
    from vcmem.messages import Message
    from vcmem.providers.base import ChatProvider


class ImportanceScorer(Protocol):
    def score(self, message: Message) -> float: ...


_TIME = re.compile(
    r"\b(\d+|yesterday|today|tomorrow|tonight|weekend|week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|"
    r"july|august|september|october|november|december|ago|last|next)\b",
    re.IGNORECASE,
)
DISCLOSURE = re.compile(
    r"\b(i am|i'm|i was|i've|i have|i had|my|we|our|i (?:just |recently |finally )?"
    r"(?:went|started|got|joined|moved|bought|adopted|finished|made|took|signed|visited|"
    r"began|decided|love|loved|work|live|tried|learned|won|lost|ran|painted|wrote))\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][\w'-]*")


def proper_noun_count(text: str) -> int:
    """Capitalised words that do not start a sentence (a cheap proper-noun signal)."""
    count = 0
    for sentence in split_sentences(text):
        words = _WORD.findall(sentence)
        count += sum(1 for w in words[1:] if w[0].isupper() and w not in {"I", "I'm", "I've"})
    return count


class HeuristicImportance:
    def score(self, message: Message) -> float:
        text = message.text
        proper = proper_noun_count(text)
        has_time = 1.0 if _TIME.search(text) else 0.0
        has_disclosure = 1.0 if DISCLOSURE.search(text) else 0.0
        content = len(tokenize(text))
        raw = 0.4 * proper + 0.6 * has_time + 0.6 * has_disclosure + 0.05 * content
        return 1.0 - math.exp(-raw)


IMPORTANCE_PROMPT = (
    "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) "
    "and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely "
    "poignancy of the following piece of memory. Answer with a single integer.\n"
    "Memory: {memory}\nRating:"
)


class ProviderImportance:
    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def score(self, message: Message) -> float:
        reply = text_of(
            self._provider,
            "You rate memories.",
            IMPORTANCE_PROMPT.format(memory=message.render()),
        )
        match = re.search(r"\b(10|[1-9])\b", reply)
        if match is None:
            raise ProviderError(f"no 1-10 rating in reply {reply[:80]!r}")
        return (int(match.group(1)) - 1) / 9.0
