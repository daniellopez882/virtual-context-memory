"""The replay benchmark behind ``vcmem bench``."""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from typing import TYPE_CHECKING, Any

from vcmem.assets import ASSETS, LOCOMO_COMMIT, MINILM_REVISION, sha256_of
from vcmem.bench.machine import git_state, machine_info
from vcmem.bench.runner import (
    BOOTSTRAP_RESAMPLES,
    METHODS,
    POLICIES,
    PROBES,
    Replay,
    RunConfig,
)
from vcmem.embed import MiniLMEmbedder
from vcmem.locomo import load_locomo
from vcmem.scripted import RECALL_CUE
from vcmem.tokens import WordPieceTokenCounter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vcmem.embed import Embedder
    from vcmem.tokens import TokenCounter

__all__ = ["METHODS", "POLICIES", "PROBES", "RunConfig", "run_benchmark", "write_results"]


def _describe_embedder(embedder: Embedder) -> str:
    if isinstance(embedder, MiniLMEmbedder):
        return "all-MiniLM-L6-v2 (Xenova quantised ONNX, onnxruntime CPU)"
    return f"{type(embedder).__name__} (dim {embedder.dim}, offline; no semantic model)"


def _describe_counter(counter: TokenCounter) -> str:
    if isinstance(counter, WordPieceTokenCounter):
        return "WordPiece, all-MiniLM-L6-v2 tokenizer.json (bert-base-uncased vocabulary)"
    return f"{type(counter).__name__}"


def run_benchmark(
    config: RunConfig,
    data_path: Path,
    embedder: Embedder,
    counter: TokenCounter,
    *,
    command: str,
    repo: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = time.perf_counter()
    conversations = load_locomo(data_path)[: config.max_conversations]
    replay = Replay(config, embedder, counter)
    for i, conv in enumerate(conversations, 1):
        t0 = time.perf_counter()
        replay.run_conversation(conv)
        log(
            f"[{i}/{len(conversations)}] {conv.sample_id}: {len(conv.messages())} turns, "
            f"{time.perf_counter() - t0:.1f}s"
        )
    sha, dirty = git_state(repo)
    locomo = next(a for a in ASSETS if a.name == "locomo10")
    model = next(a for a in ASSETS if a.name == "minilm-onnx-quantized")
    data_sha = sha256_of(data_path)
    turns = [m for c in conversations for m in c.messages()]
    text_tokens = [counter.count(m.text) for m in turns]
    page_tokens = [counter.count(m.render()) for m in turns]
    not_run = {
        "qa_f1": "not run: needs an LLM endpoint (vcmem bench --qa-base-url ... --qa-model ...)",
        "llm_agent": "not run: memgpt-* methods used the scripted deterministic agent",
    }
    if config.qa_provider is not None:
        not_run.pop("qa_f1")
    if config.agent_provider is not None:
        not_run.pop("llm_agent")
    return {
        "benchmark": "locomo-evidence-recall",
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "command": command,
        "machine": machine_info(),
        "config": {
            "budgets": list(config.budgets),
            "methods": list(config.methods),
            "max_conversations": config.max_conversations,
            "max_questions": config.max_questions,
            "flush_to": config.flush_to,
            "page_size": config.page_size,
            "archive_threshold": config.archive_threshold,
            "question_fill": config.question_fill,
            "agent": "llm" if config.agent_provider is not None else "scripted",
            "importance": "HeuristicImportance (offline heuristic, not an LLM rating)",
            "token_counter": _describe_counter(counter),
            "embedder": _describe_embedder(embedder),
        },
        "dataset": {
            "name": "LoCoMo (locomo10.json)",
            "source_commit": LOCOMO_COMMIT,
            "license": locomo.license,
            "sha256": data_sha,
            "sha256_matches_pinned": data_sha == locomo.sha256,
            "conversations": len(conversations),
            "turns": sum(len(c.messages()) for c in conversations),
            "questions_scored": replay.questions_scored,
            "questions_without_evidence": sum(
                1 for c in conversations for q in c.qa if not q.evidence
            ),
            "dropped_evidence_ids": sum(c.dropped_evidence for c in conversations),
            "turn_tokens": {
                "counter": _describe_counter(counter),
                "text_mean": round(statistics.fmean(text_tokens), 1),
                "text_max": max(text_tokens),
                "rendered_mean": round(statistics.fmean(page_tokens), 1),
                "rendered_max": max(page_tokens),
            },
            "recall_cue_turns": sum(1 for m in turns if RECALL_CUE.search(m.text)),
        },
        "model": {
            "name": "Xenova/all-MiniLM-L6-v2 model_quantized.onnx",
            "revision": MINILM_REVISION,
            "license": model.license,
            "used": isinstance(embedder, MiniLMEmbedder),
        },
        "embedding": {
            "texts": replay.embed_texts,
            "seconds": round(replay.embed_seconds, 2),
            "ms_per_text": round(1000 * replay.embed_seconds / max(1, replay.embed_texts), 3),
        },
        "results": replay.summary(),
        "paired_bootstrap": {
            "metric": "evidence_recall",
            "unit": "conversation",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": 0,
            "interval": "95% percentile",
            "rows": replay.comparisons(BOOTSTRAP_RESAMPLES),
        },
        "not_run": not_run,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }


def write_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
