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

import numpy as np

from vcmem.agent import HeartbeatAgent
from vcmem.assembler import ContextAssembler
from vcmem.bench.baselines import BASELINES, ConversationIndex, baseline_context
from vcmem.embed import CachedEmbedder
from vcmem.locomo import CATEGORY_NAMES, Conversation, QAItem
from vcmem.manager import MemoryManager
from vcmem.qa import answer, token_f1
from vcmem.scoring import HeuristicImportance
from vcmem.scripted import ARCHIVE_THRESHOLD, QUESTION_FILL, ScriptedMemoryAgent
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
BOOTSTRAP_RESAMPLES = 2000
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
    archive_threshold: float = ARCHIVE_THRESHOLD
    question_fill: float = QUESTION_FILL
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
    # per-question recall grouped by conversation, for the conversation-level bootstrap
    by_conversation: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    turn_embeds: int = 0
    question_embeds: list[int] = field(default_factory=list)


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


def replay_manager(
    db: Database,
    conv: Conversation,
    policy: str,
    budget: int,
    config: RunConfig,
    embedder: Embedder,
    counter: TokenCounter,
) -> MemoryManager:
    """The memory manager exactly as a ``memgpt-<policy>`` replay builds it."""
    return MemoryManager(
        db,
        budget=budget,
        policy=policy,
        embedder=embedder,
        counter=counter,
        importance=HeuristicImportance(),
        core_blocks=conv.speakers,
        flush_to=config.flush_to,
    )


def replay_agent(manager: MemoryManager, config: RunConfig) -> HeartbeatAgent:
    """The heartbeat loop a replay drives: the scripted agent unless an LLM is configured."""
    provider = config.agent_provider or ScriptedMemoryAgent(
        manager,
        HeuristicImportance(),
        archive_threshold=config.archive_threshold,
        question_fill=config.question_fill,
    )
    return HeartbeatAgent(manager, provider, MemoryTools(manager, config.page_size))


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
        # The paging replays get the bare counter, so each MemoryManager builds its own
        # count cache: otherwise the first policy replayed pays for counting every turn and
        # the later ones read the cache, which skews the per-turn latency comparison.
        self.base_counter = counter.inner if isinstance(counter, CachedCounter) else counter
        self.counter = CachedCounter(self.base_counter)
        self.assembler = ContextAssembler(self.counter)
        self.acc: dict[tuple[str, str, int], _Acc] = defaultdict(_Acc)
        self.embed_seconds = 0.0
        self.embed_texts = 0
        self.questions_scored = 0

    def _score(
        self,
        key: tuple[str, str, int],
        conv: Conversation,
        qa: QAItem,
        ctx: AssembledContext,
        ms: float,
        embeds: int = 0,
    ) -> None:
        acc = self.acc[key]
        full = ctx.full_sources()
        linked = full | ctx.partial_sources()
        n = len(qa.evidence)
        hit_full = sum(1 for e in qa.evidence if e in full) / n
        acc.recall.append(hit_full)
        acc.by_conversation[conv.sample_id].append(hit_full)
        acc.question_embeds.append(embeds)
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
            # A retrieval index counts tokens when a message is indexed, the way the
            # manager counts at ingest; neither belongs in per-question latency.
            for text in texts:
                self.counter.count(text)
            self._run_baselines(conv, plan, baselines)
        for budget in self.config.budgets:
            for method in self.config.methods:
                if method.startswith("memgpt-"):
                    self._replay(conv, plan, method, budget)

    def _run_baselines(self, conv: Conversation, plan: ReplayPlan, methods: Sequence[str]) -> None:
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
                        self._score((name, method, budget), conv, qa, ctx, ms)

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
                manager = replay_manager(
                    db, conv, policy, budget, self.config, self.embedder, self.base_counter
                )
                agent = replay_agent(manager, self.config)
                embedder = self.embedder

                def probe(name: str, questions: Sequence[QAItem]) -> None:
                    state = manager.checkpoint()
                    for qa in questions:
                        manager.restore(state)
                        before = embedder.requested
                        t0 = time.perf_counter()
                        agent.ask(qa.question)
                        ctx = manager.assemble()
                        ms = (time.perf_counter() - t0) * 1000
                        embeds = embedder.requested - before
                        self._score((name, method, budget), conv, qa, ctx, ms, embeds)
                    manager.restore(state)

                for i, message in enumerate(plan.messages):
                    before = embedder.requested
                    t0 = time.perf_counter()
                    agent.observe(message)
                    acc.turn_ms.append((time.perf_counter() - t0) * 1000)
                    acc.turn_embeds += embedder.requested - before
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

    def comparisons(self, resamples: int = BOOTSTRAP_RESAMPLES) -> list[dict[str, Any]]:
        """Paired differences in evidence recall with conversation-level bootstrap CIs."""
        rows = []
        for probe in PROBES:
            for budget in self.config.budgets:
                for a, b in comparison_pairs(self.config.methods):
                    acc_a = self.acc.get((probe, a, budget))
                    acc_b = self.acc.get((probe, b, budget))
                    if acc_a is None or acc_b is None or not acc_a.recall:
                        continue
                    diff, lo, hi = paired_bootstrap(
                        acc_a.by_conversation, acc_b.by_conversation, resamples
                    )
                    rows.append(
                        {
                            "probe": probe,
                            "budget": budget,
                            "a": a,
                            "b": b,
                            "diff": round(diff, 4),
                            "ci95": [round(lo, 4), round(hi, 4)],
                        }
                    )
        return rows


def comparison_pairs(methods: Sequence[str]) -> list[tuple[str, str]]:
    """Every paging policy against FIFO paging, and every paging method against the
    window and hybrid baselines."""
    paging = [m for m in methods if m.startswith("memgpt-")]
    pairs = []
    if "memgpt-fifo" in paging:
        pairs += [(m, "memgpt-fifo") for m in paging if m != "memgpt-fifo"]
    pairs += [(m, base) for base in ("window", "hybrid") if base in methods for m in paging]
    return pairs


def paired_bootstrap(
    a: dict[str, list[float]], b: dict[str, list[float]], resamples: int, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` over questions and a 95% percentile interval from resampling
    whole conversations, since questions about one conversation are not independent.

    Both methods must have scored the same questions, conversation by conversation.
    """
    convs = sorted(a)
    if sorted(b) != convs or any(len(a[c]) != len(b[c]) for c in convs):
        raise ValueError("a paired bootstrap needs the same questions for both methods")
    diff_sum = np.array([sum(a[c]) - sum(b[c]) for c in convs], dtype=np.float64)
    counts = np.array([len(a[c]) for c in convs], dtype=np.float64)
    picks = np.random.default_rng(seed).integers(0, len(convs), size=(resamples, len(convs)))
    stats = diff_sum[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(diff_sum.sum() / counts.sum()), float(lo), float(hi)


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
        "by_conversation": {
            sid: {"questions": len(v), "evidence_recall": round(statistics.fmean(v), 4)}
            for sid, v in sorted(acc.by_conversation.items())
        },
        "question_ms_p50": round(percentile(acc.question_ms, 0.5), 3),
        "question_ms_p95": round(percentile(acc.question_ms, 0.95), 3),
    }
    if acc.turn_ms:
        row["turns"] = len(acc.turn_ms)
        row["embed_texts_per_turn"] = round(acc.turn_embeds / len(acc.turn_ms), 3)
        row["embed_texts_per_question"] = round(statistics.fmean(acc.question_embeds), 3)
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
