"""Token counting for budgets.

Both counters are additive across whitespace: ``count(a + "\n" + b) == count(a) + count(b)``.
The assembler relies on that to keep its per-item accounting exact for the joined prompt.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tokenizers import Tokenizer

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


_REGEX_TOKEN = re.compile(r"\w+|[^\w\s]")


class RegexTokenCounter:
    """One token per word run or punctuation mark. Deterministic and dependency free."""

    def count(self, text: str) -> int:
        return sum(1 for _ in _REGEX_TOKEN.finditer(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        end = 0
        for i, match in enumerate(_REGEX_TOKEN.finditer(text)):
            if i == max_tokens:
                return text[:end]
            end = match.end()
        return text


class WordPieceTokenCounter:
    """Counts tokens with a Hugging Face ``tokenizer.json`` (here: the MiniLM WordPiece vocab)."""

    def __init__(self, tokenizer_file: Path) -> None:
        self._tok = Tokenizer.from_file(str(tokenizer_file))
        self._tok.no_truncation()
        self._tok.no_padding()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        enc = self._tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= max_tokens:
            return text
        return text[: enc.offsets[max_tokens - 1][1]]


class CachedCounter:
    """Memoises ``count``: the same message is re-counted on every assembly otherwise."""

    def __init__(self, inner: TokenCounter, maxsize: int = 200_000) -> None:
        self._inner = inner
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._maxsize = maxsize

    def count(self, text: str) -> int:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        n = self._inner.count(text)
        self._cache[text] = n
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return n

    def truncate(self, text: str, max_tokens: int) -> str:
        return self._inner.truncate(text, max_tokens)
