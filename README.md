# virtual-context-memory

A Python memory manager that treats an agent's context window like RAM. The prompt has
fixed regions (system text, core memory blocks, a recursive summary, archival search
results) and a message queue whose pages are evicted by a pluggable policy: FIFO, LRU,
ARC with ghost lists, or a Generative-Agents recency + importance + relevance score.
Evicted messages stay in SQLite-backed recall storage and come back on a search, the way a
page fault brings a page back from disk.

The repository also contains a replay benchmark that feeds the LoCoMo conversations
through the manager turn by turn and measures how much gold evidence each method keeps in
the prompt at 1K, 2K and 4K-token budgets.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repository> && cd virtual-context-memory
uv sync
uv run vcmem demo          # every policy on a short synthetic chat, no downloads
```

To use it as a library in another project: `uv add <path or git URL>`; the import name is
`vcmem`.

## Quick start

Everything below runs offline: `build_manager` defaults to an in-memory SQLite file, a
regex token counter and a hashing embedder. Pass `make_embedder("minilm")` and
`make_counter("wordpiece")` after `vcmem fetch` to use MiniLM and WordPiece counts.

```python
from vcmem import AssembledContext, MemoryManager, MemoryTools, Message, build_manager

manager: MemoryManager = build_manager(budget=400, policy="arc", core_blocks=("Ana", "Ben"))

turns = [
    ("Ana", "I adopted a greyhound called Pixel from the shelter."),
    ("Ben", "I started training for the Lisbon half marathon in May."),
    ("Ana", "My sister moved to Porto for a nursing job."),
] + [("Ben", f"Unrelated small talk number {i}, nothing to remember.") for i in range(30)]
for i, (speaker, text) in enumerate(turns):
    manager.ingest(Message(f"t{i}", text, speaker=speaker, timestamp="3 March 2024"))

print(manager.stats.evictions > 0, "t0" in manager.resident_ids())  # True False

# A question starts a new step; a search pages the old turn back into the queue.
manager.set_question("What is the name of Ana's dog?")
tools = MemoryTools(manager)
page = tools.conversation_search("greyhound dog")
print(page.paged_in)  # ['t0']

context: AssembledContext = manager.assemble()
assert context.tokens <= context.budget
print("t0" in context.full_sources())  # True
print(context.render())
```

Driving the same tools with an LLM goes through the provider interface, with any server
that speaks the OpenAI chat-completions protocol (OpenAI, Ollama, vLLM, LM Studio):

```python
from vcmem import HeartbeatAgent
from vcmem.providers import OpenAICompatibleProvider

llm = OpenAICompatibleProvider("http://localhost:11434/v1", model="<your model>")
agent = HeartbeatAgent(manager, llm)  # tool calls chain via request_heartbeat
trace = agent.ask("What is the name of Ana's dog?")
print(trace.reply, [c.name for c in trace.calls])
```

## Policy API

A policy orders victims; the manager owns the token accounting and decides how much to
evict. Sizes are in tokens.

| event | called when | 
|---|---|
| `admit(key, info, now)` | a page enters the queue: a new message, or a page fault from recall storage |
| `touch(key, now)` | a resident page is referenced again (a search hit already in context) |
| `victim(pinned) -> key or None` | the queue is over capacity; never return a pinned key |
| `evict(key)` | the manager removed the page (ARC keeps a ghost) |
| `resident() -> list[key]` | the resident set, for checkpoints and tests |
| `observe_query(embedding)` | once per step with the input's embedding, if `uses_features` is true |

`PageInfo` carries `size` (tokens), `importance` (0 to 1) and `embedding`; the last two are
only computed for policies that set `uses_features = True`.

Built-in policies:

| name | evicts | notes |
|---|---|---|
| `fifo` | the page admitted earliest | page faults re-admit at the tail; hits change nothing |
| `lru` | the page referenced least recently | |
| `arc` | LRU of T1 or T2, steered by the adaptive target `p` | token-weighted lists and ghost lists, see [ADR 0002](docs/adr/0002-token-weighted-arc.md) |
| `importance` | the lowest recency + importance + relevance score | Park et al. (2023) scoring, min-max normalised per candidate set |

A custom policy is a subclass plus one registration:

```python
from collections.abc import Container

from vcmem import EvictionPolicy, PageInfo, build_manager, register_policy


class LargestFirst(EvictionPolicy):
    """Evict the longest message first."""

    name = "largest"

    def __init__(self) -> None:
        self.sizes: dict[str, int] = {}

    def admit(self, key: str, info: PageInfo, now: int) -> None:
        self.sizes[key] = info.size

    def touch(self, key: str, now: int) -> None:
        return None

    def victim(self, pinned: Container[str]) -> str | None:
        candidates = [k for k in self.sizes if k not in pinned]
        return max(candidates, key=self.sizes.__getitem__, default=None)

    def evict(self, key: str) -> None:
        del self.sizes[key]

    def resident(self) -> list[str]:
        return list(self.sizes)


register_policy("largest", lambda capacity_tokens: LargestFirst())
manager = build_manager(budget=1024, policy="largest")
```

Budget regions come from `BudgetPlan.split(total, system_tokens)`: core memory, summary
and archival results get fixed shares of the total, a reserve is kept for the question,
and the rest is the queue capacity the policy works with. `MemoryManager(flush_to=...)`
sets the low watermark that batch eviction stops at.

Memory tools, available to the heartbeat loop (as OpenAI function schemas via
`tool_specs()`) and over MCP: `core_memory_append`, `core_memory_replace`,
`archival_insert`, `archival_search`, `conversation_search` (paginated), plus
`send_message` in the loop and `append_message` / `get_context` on the MCP server.

```bash
uv run vcmem serve                                  # MCP over stdio
uv run vcmem serve --transport http --port 8765     # streamable HTTP at /mcp, health at /healthz
```

## Policy versus budget on LoCoMo

The replay feeds all ten LoCoMo conversations (5,882 turns, 1,982 questions with
evidence) through every method at three budgets. A cell is the share of gold evidence
turns whose full text is in the assembled prompt, averaged over questions, in percent. In
brackets: also counting evidence that is only linked from a summary sentence, a
core-memory line or an archival passage. The baselines have none of those regions, so both
numbers agree for them.

Final probe, after the whole conversation (the standard LoCoMo setting):

| method | 1024 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| window | 3.2 (3.2) | 8.6 (8.6) | 17.2 (17.2) |
| bm25 | 66.5 (66.5) | 71.0 (71.0) | 75.5 (75.5) |
| dense | 59.0 (59.0) | 69.7 (69.7) | 79.8 (79.8) |
| hybrid | 71.2 (71.2) | 80.3 (80.3) | 86.8 (86.8) |
| memgpt-fifo | 61.6 (62.5) | 70.1 (71.8) | 74.8 (77.3) |
| memgpt-lru | 61.6 (62.5) | 70.0 (72.0) | 74.8 (77.5) |
| memgpt-arc | 61.6 (62.5) | 69.8 (71.6) | 73.8 (76.6) |
| memgpt-importance | 61.8 (62.6) | 70.5 (72.4) | 75.4 (78.1) |

Fresh probe, right after the session that holds the question's latest evidence:

| method | 1024 tokens | 2048 tokens | 4096 tokens |
|---|---:|---:|---:|
| window | 82.6 (82.6) | 91.6 (91.6) | 93.1 (93.1) |
| bm25 | 72.9 (72.9) | 77.6 (77.6) | 83.7 (83.7) |
| dense | 70.6 (70.6) | 80.3 (80.3) | 88.8 (88.8) |
| hybrid | 79.1 (79.1) | 86.8 (86.8) | 92.2 (92.2) |
| memgpt-fifo | 69.7 (75.4) | 83.5 (88.0) | 95.2 (96.2) |
| memgpt-lru | 69.8 (75.3) | 83.5 (87.7) | 95.1 (95.9) |
| memgpt-arc | 70.0 (75.8) | 79.7 (84.9) | 87.9 (91.2) |
| memgpt-importance | 71.1 (77.3) | 85.9 (89.5) | 95.0 (95.8) |

Final-probe evidence recall by question category at 2048 tokens, in percent:

| method | multi-hop | temporal | open-domain | single-hop | adversarial |
|---|---:|---:|---:|---:|---:|
| window | 4.5 | 7.7 | 7.2 | 9.3 | 11.0 |
| bm25 | 41.4 | 78.0 | 40.8 | 78.0 | 77.8 |
| dense | 62.1 | 80.7 | 52.8 | 78.8 | 53.0 |
| hybrid | 58.7 | 85.2 | 53.7 | 87.8 | 82.1 |
| memgpt-fifo | 43.3 | 77.6 | 39.8 | 76.6 | 75.3 |
| memgpt-lru | 43.3 | 77.6 | 39.5 | 76.6 | 75.3 |
| memgpt-arc | 43.2 | 77.5 | 38.9 | 76.3 | 75.1 |
| memgpt-importance | 44.3 | 78.7 | 40.8 | 76.8 | 75.3 |

Paging policies against FIFO paging: the paired difference in evidence recall
(percentage points) and a 95% interval from resampling whole conversations, since the
questions about one conversation are not independent. `vcmem report` prints the same
comparison against the window and hybrid baselines.

| probe | budget | method | compared with | difference | 95% interval |
|---|---:|---|---|---:|---|
| final | 1024 | memgpt-lru | memgpt-fifo | +0.0 | +0.0 to +0.0 |
| final | 1024 | memgpt-arc | memgpt-fifo | +0.1 | -0.1 to +0.2 |
| final | 1024 | memgpt-importance | memgpt-fifo | +0.2 | +0.1 to +0.3 |
| final | 2048 | memgpt-lru | memgpt-fifo | -0.0 | -0.0 to +0.0 |
| final | 2048 | memgpt-arc | memgpt-fifo | -0.3 | -0.7 to +0.1 |
| final | 2048 | memgpt-importance | memgpt-fifo | +0.4 | +0.1 to +0.7 |
| final | 4096 | memgpt-lru | memgpt-fifo | +0.0 | -0.2 to +0.2 |
| final | 4096 | memgpt-arc | memgpt-fifo | -1.0 | -2.0 to -0.1 |
| final | 4096 | memgpt-importance | memgpt-fifo | +0.7 | +0.2 to +1.1 |
| fresh | 1024 | memgpt-lru | memgpt-fifo | +0.1 | +0.0 to +0.3 |
| fresh | 1024 | memgpt-arc | memgpt-fifo | +0.3 | -0.0 to +0.6 |
| fresh | 1024 | memgpt-importance | memgpt-fifo | +1.5 | +1.0 to +2.1 |
| fresh | 2048 | memgpt-lru | memgpt-fifo | +0.0 | -0.5 to +0.6 |
| fresh | 2048 | memgpt-arc | memgpt-fifo | -3.7 | -5.3 to -2.1 |
| fresh | 2048 | memgpt-importance | memgpt-fifo | +2.5 | +1.6 to +3.3 |
| fresh | 4096 | memgpt-lru | memgpt-fifo | -0.1 | -0.4 to +0.2 |
| fresh | 4096 | memgpt-arc | memgpt-fifo | -7.3 | -9.3 to -5.4 |
| fresh | 4096 | memgpt-importance | memgpt-fifo | -0.2 | -0.8 to +0.2 |

Work a live deployment would do per step, counted during the replay. The replay serves
MiniLM vectors from a cache it fills before replaying, so the encoder is not inside any
latency below; these columns count the texts it would have to encode.

| method | budget | MiniLM texts per turn | MiniLM texts per final-probe question | SQLite KiB per conversation |
|---|---:|---:|---:|---:|
| memgpt-fifo | 1024 | 0.30 | 1.00 | 512.8 |
| memgpt-fifo | 2048 | 0.30 | 1.00 | 513.2 |
| memgpt-fifo | 4096 | 0.30 | 1.00 | 513.2 |
| memgpt-lru | 1024 | 0.30 | 1.00 | 512.8 |
| memgpt-lru | 2048 | 0.30 | 1.00 | 513.2 |
| memgpt-lru | 4096 | 0.30 | 1.00 | 513.2 |
| memgpt-arc | 1024 | 0.30 | 1.00 | 512.8 |
| memgpt-arc | 2048 | 0.30 | 1.00 | 513.2 |
| memgpt-arc | 4096 | 0.30 | 1.00 | 513.2 |
| memgpt-importance | 1024 | 1.00 | 1.00 | 1701.2 |
| memgpt-importance | 2048 | 1.00 | 1.00 | 1701.6 |
| memgpt-importance | 4096 | 1.00 | 1.00 | 1701.6 |

Latency per replayed turn (one `observe` call: ingest, eviction, summary update and the
scripted agent's tool calls) and per final-probe question (search, paging and assembly for
the paging methods; ranking and assembly over the prefix index for the baselines). These
figures come from the earlier run at `fc16b4e` and have not been re-measured since the
ARC, window, importance-scoring and encoder-reuse changes:

| method | budget | turn p50 ms | turn p95 ms | final-probe question p50 ms | final-probe question p95 ms |
|---|---:|---:|---:|---:|---:|
| window | 1024 | - | - | 1.90 | 4.86 |
| window | 2048 | - | - | 1.91 | 5.27 |
| window | 4096 | - | - | 1.95 | 5.02 |
| bm25 | 1024 | - | - | 1.90 | 4.82 |
| bm25 | 2048 | - | - | 1.88 | 4.92 |
| bm25 | 4096 | - | - | 1.92 | 5.32 |
| dense | 1024 | - | - | 2.27 | 6.08 |
| dense | 2048 | - | - | 2.27 | 5.63 |
| dense | 4096 | - | - | 2.26 | 5.43 |
| hybrid | 1024 | - | - | 3.16 | 7.71 |
| hybrid | 2048 | - | - | 3.25 | 7.82 |
| hybrid | 4096 | - | - | 3.19 | 7.71 |
| memgpt-fifo | 1024 | 1.06 | 3.76 | 3.29 | 6.63 |
| memgpt-fifo | 2048 | 1.31 | 5.07 | 6.41 | 11.85 |
| memgpt-fifo | 4096 | 1.23 | 6.22 | 8.14 | 13.29 |
| memgpt-lru | 1024 | 1.09 | 3.84 | 3.27 | 6.49 |
| memgpt-lru | 2048 | 1.31 | 4.51 | 6.30 | 12.38 |
| memgpt-lru | 4096 | 1.28 | 6.47 | 8.21 | 13.50 |
| memgpt-arc | 1024 | 1.11 | 3.84 | 3.18 | 5.86 |
| memgpt-arc | 2048 | 1.29 | 4.73 | 6.66 | 11.36 |
| memgpt-arc | 4096 | 1.20 | 6.23 | 8.03 | 13.83 |
| memgpt-importance | 1024 | 1.21 | 4.42 | 4.46 | 8.05 |
| memgpt-importance | 2048 | 1.38 | 5.51 | 8.89 | 15.18 |
| memgpt-importance | 4096 | 1.36 | 7.62 | 11.62 | 17.80 |

Where each table comes from:

| | recall, paired differences, work | latency |
|---|---|---|
| results file | [`bench/results/locomo-replay.json`](bench/results/locomo-replay.json) | [`bench/results/locomo-replay-fc16b4e.json`](bench/results/locomo-replay-fc16b4e.json) |
| code | commit `ba1b6df` | commit `fc16b4e` |
| Measured on | 2026-10-10, an Intel i5-8500 desktop: 6 cores, 12 GB RAM (11.9 GB visible to the OS), Windows 10, CPU only, Python 3.12 | 2026-10-10, the same desktop |
| run time | 686.1 s; other jobs shared the CPU, so its timings are not quoted | 655.4 s |

Both files also record package versions, the dataset and model sha256 and the exact
command. `vcmem report <file>` prints each table above from the file it is listed under.

### Reading the results

* **After the whole conversation, retrieval decides.** The four paging policies score
  61.6 to 61.8% at 1024 tokens and 73.8 to 75.4% at 4096. By then the evidence was evicted
  long ago and the question's own search brings it back the same way for every policy.
  Hybrid retrieval keeps more evidence in full at every budget (71.2, 80.3 and 86.8%).
  Counting summary, core and archival links moves the paging methods only to 62.5 to 62.6%
  at 1024 tokens, 71.6 to 72.4% at 2048 and 76.6 to 78.1% at 4096, still below hybrid's
  full-text figures at every budget; and a linked line is an extracted sentence or a note
  about the turn, not the turn.
* **Right after the evidence, recency decides at small budgets.** The plain window keeps
  82.6% at 1024 tokens, more than any paging method (69.7 to 71.1%). The manager gives part
  of the budget to core memory, the summary and archival results, flushes the queue down
  to a watermark, and the question's search pages older hits in over recent turns; its
  prompts are smaller too (852.1 tokens on average for `memgpt-fifo` at 1024, against
  993.9 for the window). At 4096 tokens FIFO, LRU and importance paging keep 95.0 to
  95.2%, above the window (93.1%) and hybrid retrieval (92.2%), and the paired intervals
  for those six comparisons all lie above zero.
* **Between policies, only importance and ARC separate from FIFO.** LRU stays within 0.1
  points of FIFO everywhere and every one of its intervals reaches zero. Importance scoring
  keeps more evidence than FIFO in the fresh probe at 1024 and 2048 tokens (71.1 and 85.9%
  against 69.7 and 83.5%), with intervals above zero; at 4096 the two are level. In the
  final probe its lead is a fraction of a point. ARC is level with FIFO at 1024 tokens and
  the weakest policy in the fresh probe at 2048 and 4096 (79.7 and 87.9%, against 83.5
  and 95.2% for FIFO). A plausible cause, not measured separately: new messages enter T1,
  a search hit on a resident page moves it to T2, and ARC evicts from T1 while T1 is above
  its target `p`, which only grows on a B1 ghost hit. Question-time faults are rolled back
  with the probe checkpoint, so only recall-cue searches during the replay produce ghost
  hits (239 on B1 and 101 on B2 over the ten conversations at 4096 tokens), and older
  pages that a search touched outlive the latest session's turns.
* **Cost.** In the `fc16b4e` run a replayed turn took 1.06 to 1.38 ms at the median and a
  paging question 3.18 to 11.62 ms (several search pages, then assembly), against 1.88 to
  3.25 ms for the baselines. Importance was the slowest policy at every budget; at that
  commit it rescored every resident page for each victim; the code now computes the raw
  scores once per flush and only re-normalises them per victim. None of this includes the
  encoder. Importance is the only policy that needs
  a vector for every turn (1.00 texts per turn, against 0.30 for the others, which encode
  only the turns they archive); at the 5.843 ms per text measured when the `fc16b4e` run
  primed its cache, that costs more than the median turn itself. Every policy encodes one
  text per question, for the archival search. The SQLite file holds every message, the
  archive and, for importance, one vector per message (1701.2 to 1701.6 KiB per
  conversation against 512.8 to 513.2 KiB); the BM25 and cosine indexes are rebuilt in
  memory from those rows when a database is opened and are not part of that size.

Not run, because they need an LLM endpoint and none was used:

| measurement | how to run it | status |
|---|---|---|
| answer accuracy (token F1 of an LLM answer from the assembled prompt; adversarial questions are scored against "not mentioned") | `vcmem bench --qa-base-url ... --qa-model ...` | not run |
| `memgpt-*` paging driven by an LLM agent instead of the scripted one | `vcmem bench --agent-base-url ... --agent-model ...` | not run |
| LLM importance rating (`ProviderImportance`) instead of `HeuristicImportance` | library only | not run |
| abstractive recursive summary (`ProviderSummarizer`) instead of TextRank | library only | not run |

Reproduce. `vcmem fetch` puts LoCoMo and the model in `~/.cache/virtual-context-memory`
unless `VCMEM_DATA_DIR` names another directory; the full replay takes about eleven
minutes on the desktop above.

```bash
uv run vcmem fetch                                       # LoCoMo + MiniLM, sha256-checked
uv run vcmem bench --out bench/results/locomo-replay.json
uv run vcmem report bench/results/locomo-replay.json
uv run vcmem inspect --sample conv-26 --question 0 --policy arc --budget 1024
```

### What is compared

* `window`: the newest messages that fit.
* `bm25`, `dense`, `hybrid`: the question's top-ranked messages (BM25, MiniLM cosine, or
  reciprocal rank fusion of both) until the budget is full, shown in chronological order.
* `memgpt-<policy>`: the full manager. Every turn is replayed through the heartbeat loop
  with a scripted deterministic agent that uses the same tools an LLM would:
  * on a turn with a recall cue ("remember", "last time", "how's the ..."), one page of
    `conversation_search`;
  * on a first-person statement, `core_memory_append` to the speaker's block, replacing
    the oldest line when the block is full;
  * on a turn whose heuristic importance is at least the archive threshold,
    `archival_insert`;
  * on a question, `conversation_search` pages until half the queue capacity is paged in,
    then `archival_search`.

Each question is asked twice from a checkpoint: after the last turn (final probe, the
LoCoMo setting) and right after the session holding its latest evidence (fresh probe),
see [ADR 0004](docs/adr/0004-measure-evidence-recall-at-two-probe-points.md). Budgets are
counted in WordPiece tokens of the MiniLM tokenizer.

## Concepts

| operating system | here |
|---|---|
| RAM | the prompt, a fixed token budget |
| kernel-resident pages | system text and core memory blocks, never evicted |
| page cache | the message queue |
| replacement policy | `EvictionPolicy` (FIFO, LRU, ARC, importance) |
| disk | recall storage (every message in SQLite, searched through an in-memory BM25 index) and archival storage (agent-written passages and their vectors in SQLite, searched by cosine) |
| page fault | a `conversation_search` hit that is not resident |
| swap-out | eviction, folded into the recursive summary |

```mermaid
flowchart LR
    IN([incoming message]) -->|ingest| REC[(recall storage<br/>SQLite + BM25)]
    IN -->|admit, pinned| Q
    subgraph CTX[main context, token budget]
        SYS[system] --- CORE[core blocks]
        CORE --- SUM[recursive summary]
        SUM --- AV[archival results]
        AV --- Q[message queue]
    end
    Q -->|over capacity: victims from EvictionPolicy| S[summarizer<br/>TextRank or provider]
    S --> SUM
    LOOP[heartbeat loop<br/>scripted agent or LLM provider] -->|tool calls| T[memory tools]
    MCP[MCP server<br/>stdio or HTTP] --> T
    T -->|conversation_search: page fault| REC
    REC -->|page_in| Q
    T -->|archival_insert / search| ARC[(archival storage<br/>SQLite + cosine)]
    ARC --> AV
    T -->|core_memory_append / replace| CORE
    CTX --> ASM[ContextAssembler<br/>priority classes, budget check]
    ASM -->|prompt| LOOP
```

## Limitations

* No LLM was run. Answer accuracy and an LLM-driven agent are wired up but not measured;
  evidence recall is an upper bound on what an answering model could use, not accuracy.
* The scripted agent's rules (recall-cue regex, first-person regex, importance threshold)
  shape every `memgpt-*` number. An LLM agent would search and archive differently.
* `HeuristicImportance` is a hand-written heuristic, not the LLM poignancy rating from
  Generative Agents (that path exists as `ProviderImportance`). The archive threshold was
  chosen so that just under a third of LoCoMo turns get archived (1,744 of 5,882 in the
  full replay), a selectivity target picked before any recall was measured, but still a
  choice made on this dataset.
* Budgets are WordPiece tokens, not the tokens of any particular LLM.
* Tool-call records are not kept in the queue, unlike MemGPT
  ([ADR 0001](docs/adr/0001-page-search-hits-as-message-pages.md)).
* LoCoMo has ten conversations; the open-domain category has 92 scored questions, so
  per-category differences there are noisy. Two evidence ids in the release point at
  turns that do not exist and four questions have no usable evidence; they are dropped
  and counted in the results file. Each configuration ran once: the replay is
  deterministic (the bm25, dense, hybrid, FIFO, LRU and importance full-text figures came
  out the same in both runs), and the latency percentiles come from one earlier run on one
  machine, before the latest code changes.
* The final probe mostly measures the question-time search, which is the same for every
  policy; policy differences show up only in the fresh probe. The paging prompts are not
  full by design (the queue is flushed to a watermark, and the core, summary and archival
  regions are often not used up), so they compete with baselines that fill the whole
  budget.
* The explanation given above for ARC's lower fresh-probe recall is an interpretation of
  the policy's rules and the ghost-hit counters, not a separate measurement.
* One agent per SQLite file, one process. On reopening a database the queue is refilled
  with the newest messages and the BM25 index is rebuilt from every stored message;
  policy state such as ARC ghost lists is not persisted. `restore` rewinds the summary and
  core blocks in SQLite as well, but messages and archival passages stored after the
  checkpoint stay.

## Data and model licences

| asset | source | licence | how it gets here |
|---|---|---|---|
| LoCoMo `locomo10.json` | [snap-research/locomo](https://github.com/snap-research/locomo), commit `cbfbc1d` | CC BY-NC 4.0 (non-commercial) | `vcmem fetch`; never committed |
| all-MiniLM-L6-v2, quantised ONNX + tokenizer | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), revision `751bff3` (export of sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | `vcmem fetch`; never committed |

`vcmem fetch` downloads into `$VCMEM_DATA_DIR` (default `~/.cache/virtual-context-memory`)
and checks each file against the sha256 recorded in `src/vcmem/assets.py`. LoCoMo's
non-commercial licence applies to anything you derive from the dataset; this code is MIT.

## Development

```bash
uv run pytest                      # default suite: no network, no downloads
uv run pytest -m integration       # needs `vcmem fetch`
uv run ruff check . && uv run ruff format --check .
docker build -t vcmem . && docker run --rm vcmem vcmem demo
```

Design records: [docs/adr](docs/adr/).

## License

MIT, see [LICENSE](LICENSE).
