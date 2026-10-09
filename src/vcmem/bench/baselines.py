"""Context builders for the comparison methods that do not page.

``window``  the most recent messages that fit (what a plain chat loop sends);
``bm25``, ``dense``, ``hybrid``  the question's top-ranked messages, added in rank order
until the budget is full, then shown chronologically. ``hybrid`` fuses the BM25 and
dense rankings with reciprocal rank fusion (k = 60).

All of them go through the same ``ContextAssembler`` as the memory manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vcmem.assembler import AssembledContext, ContextAssembler, ContextItem, Priority
from vcmem.stores.bm25 import BM25Index
from vcmem.stores.cosine import CosineIndex
from vcmem.stores.recall import index_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.messages import Message

BASELINE_SYSTEM = "Answer the final question using the conversation excerpts below."
BASELINES = ("window", "bm25", "dense", "hybrid")
RRF_K = 60


class ConversationIndex:
    """BM25 and dense indexes over a growing conversation prefix."""

    def __init__(self, dim: int) -> None:
        self.messages: list[Message] = []
        self.bm25 = BM25Index()
        self.dense = CosineIndex(dim)

    def extend(self, messages: Sequence[Message], vectors: np.ndarray) -> None:
        for message, vec in zip(messages, vectors, strict=True):
            self.messages.append(message)
            self.bm25.add(index_text(message))
            self.dense.add(vec)

    def bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.scores(query)
        return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]

    def dense_ranking(self, query_vec: np.ndarray) -> list[int]:
        sims = self.dense.scores(query_vec)
        return np.lexsort((np.arange(len(sims)), -sims)).tolist()

    def hybrid_ranking(self, query: str, query_vec: np.ndarray) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self.bm25_ranking(query), self.dense_ranking(query_vec)):
            for rank, doc in enumerate(ranking):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [d for d, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]


def baseline_context(
    method: str,
    index: ConversationIndex,
    question: str,
    query_vec: np.ndarray,
    budget: int,
    assembler: ContextAssembler,
) -> AssembledContext:
    items = [
        ContextItem(BASELINE_SYSTEM, Priority.SYSTEM, pinned=True),
        ContextItem(question, Priority.INPUT, pinned=True, role="user"),
    ]
    messages = index.messages
    if method == "window":
        items += [
            ContextItem(
                m.render(),
                Priority.QUEUE,
                position=m.seq,
                keep_rank=m.seq,
                role="user",
                full_sources=frozenset({m.id}),
            )
            for m in messages
        ]
    else:
        if method == "bm25":
            ranking = index.bm25_ranking(question)
        elif method == "dense":
            ranking = index.dense_ranking(query_vec)
        elif method == "hybrid":
            ranking = index.hybrid_ranking(question, query_vec)
        else:
            raise ValueError(f"unknown baseline {method!r}")
        items += [
            ContextItem(
                messages[doc].render(),
                Priority.RECALL,
                position=messages[doc].seq,
                keep_rank=-rank,
                role="user",
                full_sources=frozenset({messages[doc].id}),
            )
            for rank, doc in enumerate(ranking)
        ]
    return assembler.assemble(items, budget)
