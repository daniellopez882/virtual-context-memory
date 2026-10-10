# ADR 0003: Batch eviction to a low watermark, folded into an extractive recursive summary

Status: accepted

## Context

When the queue goes over capacity something has to leave, and what leaves should not
vanish from the prompt entirely. MemGPT flushes a large share of the queue at once and
asks the LLM for a new recursive summary of the previous summary plus the evicted
messages. Two questions follow for this project:

1. Evict one message per overflow, or a batch?
2. How to summarise without an LLM in the loop, in tests and in an offline benchmark,
   while keeping a link from the summary back to the messages it came from?

## Decision

* Eviction is batched: when the queue exceeds its capacity `c`, the manager asks the
  policy for victims until the queue is at or below `flush_to * c` (default 0.75).
* The evicted messages that the summary has not seen yet are folded into the summary:
  `summary_n = summarize(summary_{n-1}, evicted_n)`, limited to the summary region of the
  budget. A message that was paged back in and evicted again is not summarised twice.
* `TextRankSummarizer` is the default. Candidate units are the previous summary's units
  plus every sentence (with at least three content words) of the evicted messages,
  prefixed with speaker and date. PageRank over normalised word overlap ranks them, and
  the top units that fit are kept in chronological order. Each unit carries the ids of
  its source messages.
* `ProviderSummarizer` asks the provider for an abstractive summary and falls back to
  TextRank when the provider fails. Its single unit links to every message folded into
  the chain, because an abstractive sentence cannot be traced more precisely.

## Consequences

* Summaries run once per flush instead of once per message: the full LoCoMo replay at
  1024 tokens with FIFO made 1,468 flushes for 5,882 turns (`manager_totals` in
  `bench/results/locomo-replay.json`).
* The context is between 75% and 100% full in steady state, so some budget is idle by
  design. `flush_to` is a parameter and is recorded in every results file.
* Every summary unit is a verbatim sentence with source ids, so the benchmark can report
  evidence that survives only in summarised form (`evidence_recall_any`) separately from
  evidence present in full (`evidence_recall`). Hypothesis tests check that units always
  link to messages in the summary's coverage and that their text comes from those
  messages.
* Extractive summaries keep wording but drop connective information an abstractive
  summary would add; that difference is not measured here because the provider path is
  not run in the benchmark.

## Alternatives rejected

* **Evict one message per overflow.** Summarising after every message is expensive with
  a provider and churns the summary; a flush-to-watermark costs one call per batch.
* **Truncate without summarising.** Simplest, but the evicted content then exists only in
  recall storage and can only come back through a search.
* **Lead-k or first-sentence extraction.** Cheaper than TextRank but blind to which
  sentences the rest of the conversation keeps returning to.
