"""Answering LoCoMo questions from an assembled context through the provider, and scoring.

The replay benchmark measures evidence recall without any LLM. When a provider is
configured (``vcmem bench --qa-base-url ...``) it also answers each question from the
same context and reports token F1 against the gold answer.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING

from vcmem.providers.base import text_of

if TYPE_CHECKING:
    from vcmem.assembler import AssembledContext
    from vcmem.providers.base import ChatProvider

QA_SYSTEM = (
    "Answer the final question using only the memory shown. Reply with a short phrase. "
    "If the memory does not contain the answer, reply: not mentioned."
)

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise_answer(text: str) -> list[str]:
    """LoCoMo-style normalisation: lowercase, strip punctuation and articles (no stemming)."""
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return text.split()


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalise_answer(prediction), normalise_answer(gold)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer(provider: ChatProvider, context: AssembledContext) -> str:
    """The context already ends with the question (the assembler's input region)."""
    return text_of(provider, QA_SYSTEM, context.render())
