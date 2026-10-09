"""Factories that wire a manager from sensible defaults."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

from vcmem.config import minilm_dir
from vcmem.embed import HashingEmbedder, MiniLMEmbedder
from vcmem.manager import MemoryManager
from vcmem.scoring import HeuristicImportance
from vcmem.stores.sqlite import Database
from vcmem.tokens import RegexTokenCounter, WordPieceTokenCounter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from vcmem.embed import Embedder
    from vcmem.policies import EvictionPolicy
    from vcmem.scoring import ImportanceScorer
    from vcmem.summary import Summarizer
    from vcmem.tokens import TokenCounter

EmbedderKind = Literal["auto", "minilm", "hashing"]
CounterKind = Literal["auto", "wordpiece", "regex"]


def _model_present() -> bool:
    d = minilm_dir()
    return (d / "model_quantized.onnx").exists() and (d / "tokenizer.json").exists()


def make_embedder(kind: EmbedderKind = "auto") -> Embedder:
    """``auto`` uses MiniLM when ``vcmem fetch`` has been run, else the hashing embedder."""
    if kind == "minilm" or (kind == "auto" and _model_present()):
        if not _model_present():
            raise FileNotFoundError(f"MiniLM files not found in {minilm_dir()}; run `vcmem fetch`")
        return MiniLMEmbedder(minilm_dir())
    if kind == "auto":
        print("vcmem: MiniLM model not found, using the hashing embedder", file=sys.stderr)
    return HashingEmbedder()


def make_counter(kind: CounterKind = "auto") -> TokenCounter:
    tokenizer = minilm_dir() / "tokenizer.json"
    if kind == "wordpiece" or (kind == "auto" and tokenizer.exists()):
        if not tokenizer.exists():
            raise FileNotFoundError(f"{tokenizer} not found; run `vcmem fetch`")
        return WordPieceTokenCounter(tokenizer)
    return RegexTokenCounter()


def build_manager(
    budget: int = 2048,
    policy: str | EvictionPolicy = "arc",
    *,
    db_path: str | Path = ":memory:",
    embedder: Embedder | None = None,
    counter: TokenCounter | None = None,
    summarizer: Summarizer | None = None,
    importance: ImportanceScorer | None = None,
    core_blocks: Sequence[str] = ("persona", "human"),
    flush_to: float = 0.75,
) -> MemoryManager:
    """A manager with offline defaults: regex token counts and the hashing embedder."""
    return MemoryManager(
        Database(db_path),
        budget=budget,
        policy=policy,
        embedder=embedder or HashingEmbedder(),
        counter=counter or RegexTokenCounter(),
        summarizer=summarizer,
        importance=importance or HeuristicImportance(),
        core_blocks=core_blocks,
        flush_to=flush_to,
    )
