"""Incremental Okapi BM25.

Documents are only ever appended (recall storage is a log), so the index keeps postings
as ``term -> {doc: tf}`` and recomputes IDF lazily from document frequencies at query
time. IDF uses the non-negative form ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

# Kept short on purpose: chat turns are short and pronouns such as "she" or "my" are
# noise for keyword recall, but dropping too much hurts names and numbers.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her
    hers him his how i m s t d ll ve re if in into is it its just me my of on or our she so
    than that the their them then there they this to too was we were what when where which
    who why will with would you your yours yeah oh really""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    def add(self, text: str) -> int:
        """Index ``text`` and return its document number."""
        doc = len(self._doc_len)
        terms = Counter(tokenize(text))
        for term, tf in terms.items():
            self._postings.setdefault(term, {})[doc] = tf
        length = sum(terms.values())
        self._doc_len.append(length)
        self._total_len += length
        return doc

    def idf(self, term: str) -> float:
        n = len(self._doc_len)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        n = len(self._doc_len)
        if n == 0:
            return {}
        avg_len = self._total_len / n or 1.0
        out: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc, tf in postings.items():
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[doc] / avg_len)
                out[doc] = out.get(doc, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out
