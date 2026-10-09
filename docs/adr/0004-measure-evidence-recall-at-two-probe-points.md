# ADR 0004: Measure evidence recall at two probe points, with WordPiece budgets

Status: accepted

## Context

The benchmark has to compare paging policies without an LLM (the agent and the QA step go
through the provider interface and are not run here). LoCoMo provides, for each question,
the dialog ids of the turns that contain the answer, so the measurable quantity is: is the
evidence in the assembled prompt?

A first trial asked every question after the whole conversation, as LoCoMo does. On one
conversation at 1024 tokens all four policies scored the same: by then the evidence is
far in the past, it is never resident, and only the question-time page faults (identical
for every policy) bring it back. That setting measures retrieval, not paging.

Two further choices had to be pinned down: what a "token" is, and how to embed.

## Decision

* Each question is probed twice from a checkpoint of the replay state:
  * **final**: after the last turn (the standard LoCoMo setting);
  * **fresh**: right after the session that contains the question's latest evidence.
  Probes restore the checkpoint first and the replay continues from the state before the
  probe, so questions never see each other's page faults.
* Baselines (window, BM25, dense, hybrid) are evaluated at the same two points over the
  conversation prefix seen so far, with indexes that grow session by session.
* `evidence_recall` counts an evidence turn only when its full text is in the prompt.
  `evidence_recall_any` also counts turns that are linked from a summary unit, a core
  memory line or an archival passage.
* Budgets are counted with the WordPiece tokenizer shipped with all-MiniLM-L6-v2. It is
  already downloaded for the embedder, and it is additive across whitespace, which keeps
  the assembler's accounting exact.
* MiniLM runs one text per inference call. The quantised export computes activation
  scales per input tensor, so batching made the same sentence's vector move by up to
  0.03 in cosine depending on its batch neighbours; single-text calls are as fast on this
  CPU (4.5 vs 4.3 ms per text measured on 400 LoCoMo turns) and reproducible.

## Consequences

* Policy differences become visible in the fresh probe, where the most recent session
  is still partly resident; the final probe still reports the standard setting.
* Probes double the number of question steps; the replay remains in the minutes range.
* WordPiece counts are not an LLM's BPE counts. A 1024-token budget here is a fixed,
  reproducible unit, not a promise about any specific model's window.
* Evidence recall is an upper bound on what an answering model could use; it says
  nothing about whether the answer would be right. QA F1 is wired up behind
  `--qa-base-url` and reported as not run.

## Alternatives rejected

* **Only the final probe.** Hides the property the project is about.
* **Probe after every session for every question.** About ten times more question steps
  for little extra information over the fresh probe.
* **A GPT-style BPE tokenizer for budgets.** One more download, and no single BPE
  vocabulary is "the" LLM tokenizer either.
* **Batch embeddings with padding.** Slightly faster, but results would depend on batch
  composition.
