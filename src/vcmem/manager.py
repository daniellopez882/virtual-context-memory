"""The virtual context manager.

Main context (the prompt) is treated like RAM with fixed regions:

=========  =============================================  ==================
region     contents                                       size
=========  =============================================  ==================
system     fixed instructions                             measured
core       named blocks the agent edits                   ``core`` share
summary    recursive summary of evicted messages          ``summary`` share
archival   results of the last ``archival_search``        ``archival`` share
queue      resident message pages, managed by a policy    the rest
input      the question being answered                    ``input_reserve``
=========  =============================================  ==================

Every message is written to recall storage first. When the queue goes over capacity the
eviction policy picks victims until the queue is back under ``flush_to * capacity``
(batch eviction, as in MemGPT's queue flush), and the evicted messages are folded into
the recursive summary. ``page_in`` is the page-fault path: a search hit that is not
resident is admitted again.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vcmem.assembler import AssembledContext, ContextAssembler, ContextItem, Priority
from vcmem.embed import RecentEmbedder
from vcmem.policies import EvictionPolicy, PageInfo, make_policy
from vcmem.stores.archival import ArchivalHit, ArchivalStore
from vcmem.stores.core import CoreBlock, CoreMemory
from vcmem.stores.recall import RecallStore
from vcmem.summary import Summary, TextRankSummarizer
from vcmem.tokens import CachedCounter, RegexTokenCounter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from vcmem.embed import Embedder
    from vcmem.messages import Message
    from vcmem.scoring import ImportanceScorer
    from vcmem.stores.sqlite import Database
    from vcmem.summary import Summarizer
    from vcmem.tokens import TokenCounter

SYSTEM_PROMPT = (
    "You are an assistant with paged memory. Your context holds core memory blocks, a "
    "summary of evicted messages, archival search results and recent or recalled messages. "
    "Older messages stay searchable with conversation_search; facts you save with "
    "archival_insert are searchable with archival_search. Set request_heartbeat to keep "
    "working after a tool call."
)
ARCHIVAL_HEADER = "[archival search results]"
SUMMARY_KV = "summary"


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    total: int
    system: int
    core: int
    summary: int
    archival: int
    input_reserve: int

    @property
    def queue_capacity(self) -> int:
        return (
            self.total - self.system - self.core - self.summary - self.archival - self.input_reserve
        )

    @classmethod
    def split(
        cls,
        total: int,
        system_tokens: int,
        *,
        core: float = 0.12,
        summary: float = 0.10,
        archival: float = 0.10,
        input_reserve: int = 64,
    ) -> BudgetPlan:
        plan = cls(
            total=total,
            system=system_tokens,
            core=int(total * core),
            summary=int(total * summary),
            archival=int(total * archival),
            input_reserve=input_reserve,
        )
        if plan.queue_capacity < 32:
            raise ValueError(
                f"a {total}-token budget leaves {plan.queue_capacity} tokens for the queue"
            )
        return plan


@dataclass(frozen=True, slots=True)
class QueueEntry:
    message: Message
    tokens: int
    paged_in: bool


@dataclass(slots=True)
class ManagerStats:
    ingested: int = 0
    evictions: int = 0
    flushes: int = 0
    page_ins: int = 0
    page_hits: int = 0
    page_skipped: int = 0
    summary_updates: int = 0


@dataclass(frozen=True, slots=True)
class PageInResult:
    admitted: tuple[str, ...]
    hits: tuple[str, ...]
    skipped: tuple[str, ...]
    tokens: int


@dataclass(frozen=True, slots=True)
class StepInput:
    """What the current step is about: an ingested message or a question."""

    text: str
    message: Message | None = None

    @property
    def is_question(self) -> bool:
        return self.message is None


@dataclass(slots=True)
class _State:
    queue: dict[str, QueueEntry]
    queue_tokens: int
    policy: EvictionPolicy
    summary: Summary
    core: dict[str, CoreBlock]
    now: int
    stats: ManagerStats


class MemoryManager:
    def __init__(
        self,
        db: Database,
        *,
        budget: int | BudgetPlan,
        policy: str | EvictionPolicy = "fifo",
        embedder: Embedder,
        counter: TokenCounter | None = None,
        summarizer: Summarizer | None = None,
        importance: ImportanceScorer | None = None,
        core_blocks: Sequence[str] = ("persona", "human"),
        flush_to: float = 0.75,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if not 0.0 < flush_to <= 1.0:
            raise ValueError("flush_to must be in (0, 1]")
        base_counter = counter or RegexTokenCounter()
        self.counter: TokenCounter = (
            base_counter if isinstance(base_counter, CachedCounter) else CachedCounter(base_counter)
        )
        self.system_prompt = system_prompt
        self.plan = (
            budget
            if isinstance(budget, BudgetPlan)
            else BudgetPlan.split(budget, self.counter.count(system_prompt))
        )
        self.db = db
        self.embedder = embedder
        # Everything the manager and its stores encode goes through one short memo, so a
        # text needed twice in a step (query, page, archival passage) is encoded once.
        self._encoder = RecentEmbedder(embedder)
        self.recall = RecallStore(db)
        self.archival = ArchivalStore(db, self._encoder)
        per_block = self.plan.core // max(1, len(core_blocks))
        self.core = CoreMemory(db, self.counter, {name: per_block for name in core_blocks})
        self.summarizer: Summarizer = summarizer or TextRankSummarizer(self.counter)
        self.importance = importance
        self.policy = (
            policy
            if isinstance(policy, EvictionPolicy)
            else make_policy(policy, self.plan.queue_capacity)
        )
        self.flush_to = flush_to
        self.stats = ManagerStats()
        self._assembler = ContextAssembler(self.counter)
        self._queue: dict[str, QueueEntry] = {}
        self._queue_tokens = 0
        self._pinned: set[str] = set()
        self._archival_view: list[ArchivalHit] = []
        self._input: StepInput | None = None
        self.now = 0
        saved = db.get_kv(SUMMARY_KV)
        self.summary = Summary.from_json(saved) if saved else Summary()
        self._reload_recent()

    # ----------------------------------------------------------------- state
    @property
    def queue_tokens(self) -> int:
        return self._queue_tokens

    @property
    def step_input(self) -> StepInput | None:
        return self._input

    def resident_ids(self) -> list[str]:
        return [e.message.id for e in sorted(self._queue.values(), key=lambda e: e.message.seq)]

    def _reload_recent(self) -> None:
        """On reopening a database, refill the queue with the newest messages that fit."""
        target = int(self.plan.queue_capacity * self.flush_to)
        chosen: list[Message] = []
        used = 0
        for message in reversed(self.recall.all()):
            n = self.counter.count(message.render())
            if used + n > target:
                break
            chosen.append(message)
            used += n
        for message in reversed(chosen):
            self._admit(message, paged_in=False)

    # ----------------------------------------------------------------- steps
    def _begin_step(self, step: StepInput) -> np.ndarray | None:
        """Start a step; return the input's embedding when the policy uses one."""
        self.now += 1
        self._pinned.clear()
        self._archival_view = []
        self._input = step
        if not self.policy.uses_features:
            return None
        vector = self._encoder.embed([step.text])[0]
        self.policy.observe_query(vector)
        return vector

    def ingest(self, message: Message) -> Message:
        """Persist ``message`` to recall storage and admit it to the queue (pinned)."""
        vector = self._begin_step(StepInput(message.render(), message))
        stored = self.recall.append(message, vector)
        self._input = StepInput(stored.render(), stored)
        self.stats.ingested += 1
        self._admit(stored, paged_in=False, vector=vector)
        self._pinned.add(stored.id)
        self._enforce_capacity()
        return stored

    def set_question(self, text: str) -> None:
        """Start a step whose input is a question that is answered, not stored."""
        self._begin_step(StepInput(text))

    def page_in(self, messages: Iterable[Message]) -> PageInResult:
        """Bring search hits into main context; resident hits only refresh the policy."""
        admitted: list[str] = []
        hits: list[str] = []
        skipped: list[str] = []
        cap = self.plan.queue_capacity
        pinned_tokens = sum(self._queue[k].tokens for k in self._pinned if k in self._queue)
        for message in messages:
            if message.id in self._queue:
                self.policy.touch(message.id, self.now)
                if message.id not in self._pinned:
                    pinned_tokens += self._queue[message.id].tokens
                self._pinned.add(message.id)
                hits.append(message.id)
                continue
            n = self.counter.count(message.render())
            # Pinned pages cannot be evicted, so never pin more than the queue holds.
            if pinned_tokens + n > cap:
                skipped.append(message.id)
                continue
            self._admit(message, paged_in=True)
            self._pinned.add(message.id)
            pinned_tokens += n
            admitted.append(message.id)
        self.stats.page_ins += len(admitted)
        self.stats.page_hits += len(hits)
        self.stats.page_skipped += len(skipped)
        tokens = sum(self._queue[k].tokens for k in admitted)
        self._enforce_capacity()
        return PageInResult(tuple(admitted), tuple(hits), tuple(skipped), tokens)

    def show_archival(self, hits: Sequence[ArchivalHit]) -> list[ArchivalHit]:
        """Place archival hits in the archival region, as many as fit, best first."""
        remaining = self.plan.archival - self.counter.count(ARCHIVAL_HEADER)
        shown: list[ArchivalHit] = []
        for hit in hits:
            n = self.counter.count(hit.passage.text)
            if n <= remaining:
                shown.append(hit)
                remaining -= n
        self._archival_view = shown
        return shown

    # -------------------------------------------------------------- eviction
    def _page_vector(self, message: Message) -> np.ndarray:
        """The stored embedding of a message, encoding and storing it on first use."""
        vector = self.recall.vector(message.id)
        if vector is None:
            vector = self._encoder.embed([message.render()])[0]
            self.recall.set_vector(message.id, vector)
        return vector

    def _admit(self, message: Message, *, paged_in: bool, vector: np.ndarray | None = None) -> None:
        tokens = self.counter.count(message.render())
        importance = 0.0
        if self.policy.uses_features:
            if vector is None:
                vector = self._page_vector(message)
            if self.importance is not None:
                importance = self.importance.score(message)
        self.policy.admit(message.id, PageInfo(tokens, importance, vector), self.now)
        self._queue[message.id] = QueueEntry(message, tokens, paged_in)
        self._queue_tokens += tokens

    def _enforce_capacity(self) -> None:
        cap = self.plan.queue_capacity
        if self._queue_tokens <= cap:
            return
        target = int(cap * self.flush_to)
        evicted: list[Message] = []
        while self._queue_tokens > target:
            key = self.policy.victim(self._pinned)
            if key is None:
                break
            self.policy.evict(key)
            entry = self._queue.pop(key)
            self._queue_tokens -= entry.tokens
            evicted.append(entry.message)
        self.stats.flushes += 1
        self.stats.evictions += len(evicted)
        fresh = [m for m in evicted if m.id not in self.summary.covers]
        if fresh:
            self.summary = self.summarizer.update(self.summary, fresh, self.plan.summary)
            self.db.set_kv(SUMMARY_KV, self.summary.to_json())
            self.stats.summary_updates += 1

    # -------------------------------------------------------------- assembly
    def _coverage(self, units: Iterable[tuple[str, frozenset[str]]]) -> tuple[set[str], set[str]]:
        """Split linked message ids into those whose full text is present and the rest."""
        full: set[str] = set()
        partial: set[str] = set()
        for text, sources in units:
            for mid in sources:
                message = self.recall.get(mid)
                if message is not None and message.text in text:
                    full.add(mid)
                else:
                    partial.add(mid)
        return full, partial - full

    def assemble(self) -> AssembledContext:
        items = [ContextItem(self.system_prompt, Priority.SYSTEM, pinned=True)]
        for i, block in enumerate(self.core.blocks.values()):
            full, partial = self._coverage((line.text, line.sources) for line in block.lines)
            items.append(
                ContextItem(
                    block.render(),
                    Priority.CORE,
                    position=i,
                    full_sources=frozenset(full),
                    partial_sources=frozenset(partial),
                )
            )
        if self.summary.units:
            full, partial = self._coverage((u.text, u.sources) for u in self.summary.units)
            items.append(
                ContextItem(
                    self.summary.render(),
                    Priority.SUMMARY,
                    full_sources=frozenset(full),
                    partial_sources=frozenset(partial),
                )
            )
        if self._archival_view:
            passages = [h.passage for h in self._archival_view]
            full, partial = self._coverage((p.text, p.sources) for p in passages)
            items.append(
                ContextItem(
                    "\n".join([ARCHIVAL_HEADER, *(p.text for p in passages)]),
                    Priority.ARCHIVAL,
                    full_sources=frozenset(full),
                    partial_sources=frozenset(partial),
                )
            )
        for entry in self._queue.values():
            message = entry.message
            items.append(
                ContextItem(
                    message.render(),
                    Priority.QUEUE,
                    position=message.seq,
                    keep_rank=message.seq,
                    pinned=message.id in self._pinned,
                    role="assistant" if message.role == "assistant" else "user",
                    full_sources=frozenset({message.id}),
                )
            )
        if self._input is not None and self._input.is_question:
            items.append(ContextItem(self._input.text, Priority.INPUT, pinned=True, role="user"))
        return self._assembler.assemble(items, self.plan.total)

    # ------------------------------------------------------------ checkpoint
    def checkpoint(self) -> _State:
        """Copy the in-context state so a step can be rewound.

        ``restore`` also writes the checkpointed summary and core blocks back to SQLite.
        Messages and archival passages stored after the checkpoint are not removed.
        """
        # Queue entries and summaries are immutable, so shallow copies are enough for them.
        return _State(
            dict(self._queue),
            self._queue_tokens,
            copy.deepcopy(self.policy),
            self.summary,
            self.core.snapshot(),
            self.now,
            copy.copy(self.stats),
        )

    def restore(self, state: _State) -> None:
        self._queue = dict(state.queue)
        self._queue_tokens = state.queue_tokens
        self.policy = copy.deepcopy(state.policy)
        if self.summary is not state.summary:
            # Evictions since the checkpoint saved a newer summary; reopening the
            # database must find the checkpointed one.
            self.summary = state.summary
            self.db.set_kv(SUMMARY_KV, self.summary.to_json())
        self.core.restore(state.core)
        self.now = state.now
        self.stats = copy.copy(state.stats)
        self._pinned.clear()
        self._archival_view = []
        self._input = None
