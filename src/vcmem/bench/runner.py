"""Turn-by-turn LoCoMo replay: how much gold evidence each method keeps in context.

Every question is probed twice:

* ``final``  after the whole conversation, the standard LoCoMo setting;
* ``fresh``  right after the session that holds the question's latest evidence, when the
  evidence is recent enough that what stayed resident matters.

The paging methods (``memgpt-<policy>``) replay every turn through the heartbeat loop
with the scripted agent and answer each probe from a checkpoint, so probes never see
each other's page faults. The baselines build a context per probe over the conversation
prefix seen so far (their indexes grow session by session).

Evidence is *in context* when the full text of the evidence message is in the assembled
prompt (``evidence_recall``); ``evidence_recall_any`` also counts evidence that is only
linked through a summary, core-memory or archival line.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vcmem.agent import HeartbeatAgent
from vcmem.assembler import ContextAssembler
from vcmem.bench.baselines import BASELINES, ConversationIndex, baseline_context
from vcmem.embed import CachedEmbedder
from vcmem.locomo import CATEGORY_NAMES, Conversation, QAItem
from vcmem.manager import MemoryManager
from vcmem.qa import answer, token_f1
from vcmem.scoring import HeuristicImportance
from vcmem.scripted import ScriptedMemoryAgent
from vcmem.stores.sqlite import Database
from vcmem.tokens import CachedCounter
from vcmem.tools import MemoryTools

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vcmem.assembler import AssembledContext
    from vcmem.embed import Embedder
    from vcmem.messages import Message
    from vcmem.providers.base import ChatProvider
    from vcmem.tokens import TokenCounter

POLICIES = ("fifo", "lru", "arc", "importance")
METHODS = (*BASELINES, *(f"memgpt-{p}" for p in POLICIES))
PROBES = ("final", "fresh")


@dataclass(slots=True)
class RunConfig:
    budgets: Sequence[int] = (1024, 2048, 4096)
    methods: Sequence[str] = METHODS
    max_conversations: int | None = None
    max_questions: int | None = None
    flush_to: float = 0.75
    page_size: int = 5
    archive_threshold: float = 0.8
    question_fill: float = 0.5
    qa_provider: ChatProvider | None = None
    agent_provider: ChatProvider | None = None


@dataclass(slots=True)
class _Acc:
    recall: list[float] = field(default_factory=list)
    recall_any: list[float] = field(default_factory=list)
    all_full: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    by_category: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    question_ms: list[float] = field(default_factory=list)
    turn_ms: list[float] = field(default_factory=list)
    storage_bytes: list[int] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 1]."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Per-conversation schedule: where sessions end and which questions are due there."""

    messages: list[Message]
    questions: list[QAItem]
    due: dict[int, list[QAItem]]
    session_end: dict[int, int]

    @classmethod
    def of(cls, conv: Conversation, max_questions: int | None = None) -> ReplayPlan:
        messages = [m.with_seq(i) for i, m in enumerate(conv.messages())]
        session_of = {m.id: m.session for m in messages}
        questions = [q for q in conv.qa if q.evidence][:max_questions]
        due: dict[int, list[QAItem]] = defaultdict(list)
        for q in questions:
            due[max(session_of[e] for e in q.evidence)].append(q)
        # index of the last message of each session -> session number
        session_end = {
            i: m.session
            for i, m in enumerate(messages)
            if i + 1 == len(messages) or messages[i + 1].session != m.session
        }
        return cls(messages, questions, dict(due), session_end)


class Replay:
    def __init__(
        self,
        config: RunConfig,
        embedder: Embedder,
        counter: TokenCounter,
    ) -> None:
        unknown = set(config.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}; known: {list(METHODS)}")
        self.config = config
        self.embedder = (
            embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder)
        )
        self.counter = counter if isinstance(counter, CachedCounter) else CachedCounter(counter)
        self.assembler = ContextAssembler(self.counter)
        self.acc: dict[tuple[str, str, int], _Acc] = defaultdict(_Acc)
        self.embed_seconds = 0.0
        self.embed_texts = 0
        self.questions_scored = 0

    def _score(
        self, key: tuple[str, str, int], qa: QAItem, ctx: AssembledContext, ms: float
    ) -> None:
        acc = self.acc[key]
        full = ctx.full_sources()
        linked = full | ctx.partial_sources()
        n = len(qa.evidence)
        hit_full = sum(1 for e in qa.evidence if e in full) / n
        acc.recall.append(hit_full)
        acc.recall_any.append(sum(1 for e in qa.evidence if e in linked) / n)
        acc.all_full.append(float(hit_full == 1.0))
        acc.tokens.append(ctx.tokens)
        acc.by_category[qa.category].append(hit_full)
        acc.question_ms.append(ms)
        if self.config.qa_provider is not None:
            acc.f1.append(token_f1(answer(self.config.qa_provider, ctx), qa.answer))

    def run_conversation(self, conv: Conversation) -> None:
        plan = ReplayPlan.of(conv, self.config.max_questions)
        self.questions_scored += len(plan.questions)
        start = time.perf_counter()
        texts = [m.render() for m in plan.messages] + [q.question for q in plan.questions]
        self.embed_texts += self.embedder.prime(texts)
        self.embed_seconds += time.perf_counter() - start

        baselines = [m for m in self.config.methods if m in BASELINES]
        if baselines:
            self._run_baselines(plan, baselines)
        for budget in self.config.budgets:
            for method in self.config.methods:
                if method.startswith("memgpt-"):
                    self._replay(conv, plan, method, budget)

    def _run_baselines(self, plan: ReplayPlan, methods: Sequence[str]) -> None:
        index = ConversationIndex(self.embedder.dim)
        query_vecs = self.embedder.embed([q.question for q in plan.questions])
        vec_of = {q.index: v for q, v in zip(plan.questions, query_vecs, strict=True)}

        def probe(name: str, questions: Sequence[QAItem]) -> None:
            for qa in questions:
                for budget in self.config.budgets:
                    for method in methods:
                        t0 = time.perf_counter()
                        ctx = baseline_context(
                            method, index, qa.question, vec_of[qa.index], budget, self.assembler
                        )
                        ms = (time.perf_counter() - t0) * 1000
                        self._score((name, method, budget), qa, ctx, ms)

        start = 0
        for end, session in sorted(plan.session_end.items()):
            batch = plan.messages[start : end + 1]
            index.extend(batch, self.embedder.embed([m.render() for m in batch]))
            start = end + 1
            probe("fresh", plan.due.get(session, []))
        probe("final", plan.questions)

    def _replay(self, conv: Conversation, plan: ReplayPlan, method: str, budget: int) -> None:
        policy = method.removeprefix("memgpt-")
        acc = self.acc[("final", method, budget)]
        with tempfile.TemporaryDirectory(prefix="vcmem-") as tmp:
            db = Database(Path(tmp) / "memory.sqlite")
            try:
                manager = MemoryManager(
                    db,
                    budget=budget,
                    policy=policy,
                    embedder=self.embedder,
                    counter=self.counter,
                    importance=HeuristicImportance(),
                    core_blocks=conv.speakers,
                    flush_to=self.config.flush_to,
                )
                provider = self.config.agent_provider or ScriptedMemoryAgent(
                    manager,
                    HeuristicImportance(),
                    archive_threshold=self.config.archive_threshold,
                    question_fill=self.config.question_fill,
                )
                tools = MemoryTools(manager, self.config.page_size)
                agent = HeartbeatAgent(manager, provider, tools)

                def probe(name: str, questions: Sequence[QAItem]) -> None:
                    state = manager.checkpoint()
                    for qa in questions:
                        manager.restore(state)
                        t0 = time.perf_counter()
                        agent.ask(qa.question)
                        ctx = manager.assemble()
                        ms = (time.perf_counter() - t0) * 1000
                        self._score((name, method, budget), qa, ctx, ms)
                    manager.restore(state)

                for i, message in enumerate(plan.messages):
                    t0 = time.perf_counter()
                    agent.observe(message)
                    acc.turn_ms.append((time.perf_counter() - t0) * 1000)
                    session = plan.session_end.get(i)
                    if session is not None and plan.due.get(session):
                        probe("fresh", plan.due[session])
                acc.storage_bytes.append(db.size_bytes())
                for key, value in _manager_counts(manager).items():
                    acc.stats[key] += value
                probe("final", plan.questions)
            finally:
                db.close()

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for probe in PROBES:
            for method in self.config.methods:
                for budget in self.config.budgets:
                    acc = self.acc.get((probe, method, budget))
                    if acc is not None and acc.recall:
                        rows.append(_row(probe, method, budget, acc))
        return rows


def _row(probe: str, method: str, budget: int, acc: _Acc) -> dict[str, Any]:
    row: dict[str, Any] = {
        "probe": probe,
        "method": method,
        "budget": budget,
        "questions": len(acc.recall),
        "evidence_recall": round(statistics.fmean(acc.recall), 4),
        "evidence_recall_any": round(statistics.fmean(acc.recall_any), 4),
        "all_evidence_rate": round(statistics.fmean(acc.all_full), 4),
        "context_tokens_mean": round(statistics.fmean(acc.tokens), 1),
        "context_tokens_max": max(acc.tokens),
        "by_category": {
            CATEGORY_NAMES.get(c, str(c)): {
                "questions": len(v),
                "evidence_recall": round(statistics.fmean(v), 4),
            }
            for c, v in sorted(acc.by_category.items())
        },
        "question_ms_p50": round(percentile(acc.question_ms, 0.5), 3),
        "question_ms_p95": round(percentile(acc.question_ms, 0.95), 3),
    }
    if acc.turn_ms:
        row["turns"] = len(acc.turn_ms)
        row["turn_ms_p50"] = round(percentile(acc.turn_ms, 0.5), 3)
        row["turn_ms_p95"] = round(percentile(acc.turn_ms, 0.95), 3)
        row["storage_bytes_mean"] = round(statistics.fmean(acc.storage_bytes))
        row["storage_kib_mean"] = round(row["storage_bytes_mean"] / 1024, 1)
        row["manager_totals"] = {k: int(v) for k, v in sorted(acc.stats.items())}
    if acc.f1:
        row["qa_f1"] = round(statistics.fmean(acc.f1), 4)
    return row


def _manager_counts(manager: MemoryManager) -> dict[str, float]:
    s = manager.stats
    counts: dict[str, float] = {
        "evictions": s.evictions,
        "flushes": s.flushes,
        "page_ins": s.page_ins,
        "page_hits": s.page_hits,
        "page_skipped": s.page_skipped,
        "summary_updates": s.summary_updates,
        "archival_passages": len(manager.archival),
    }
    counts.update({k: v for k, v in manager.policy.stats().items() if k != "p"})
    return counts
