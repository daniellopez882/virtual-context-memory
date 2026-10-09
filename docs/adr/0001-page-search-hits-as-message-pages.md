# ADR 0001: Search hits re-enter the queue as individual message pages

Status: accepted

## Context

In MemGPT, `conversation_search` returns its hits as one function-return message that is
appended to the FIFO queue. That message is a blob: whatever eviction does to it, it does
to all hits at once, and a later reference to one of the hits cannot be told apart from a
reference to the blob.

This project compares cache-replacement policies (FIFO, LRU, ARC, an importance score).
Those policies only behave differently when they see a stream of references to stable
keys: LRU needs re-references to reorder, ARC needs second references to promote pages to
T2 and needs faults on recently evicted keys to hit its ghost lists. With blob results
there would be no stable keys, every policy would degenerate to "evict the oldest blob",
and the comparison would measure nothing.

## Decision

The unit of paging is one message, keyed by its message id. `conversation_search`
returns a page of hits and the manager's `page_in` handles each hit separately:

* a hit that is already resident is a **reference** (`policy.touch`);
* a hit that is not resident is a **page fault**: the message is re-admitted
  (`policy.admit`), which for ARC may be a ghost hit in B1 or B2.

Pages brought in during a step are pinned for that step, and `page_in` refuses to pin
more tokens than the queue holds, so a large search cannot push the queue permanently over
capacity. Tool-call transcripts (the call and its JSON result) are shown to the provider
within the step and are not kept in the queue.

## Consequences

* Cache policies get a meaningful access stream, and ARC's ghost lists are exercised
  (the replay records `ghost_hits_b1` and `ghost_hits_b2`).
* Evidence coverage can be measured exactly: every resident page is a whole message with
  a known id.
* A paged-in message appears once in context even if several searches return it.
* Unlike MemGPT, the queue never holds function-call records, so an LLM agent does not
  see its own earlier tool calls from previous steps. This saves tokens but loses that
  history; the benchmark's scripted agent does not need it.

## Alternatives rejected

* **One blob per search result (MemGPT as published).** Faithful, but makes the policy
  comparison meaningless for the reasons above.
* **Separate "recalled" region outside the queue.** Simpler accounting, but then the
  eviction policy never sees recalled pages and paging is reduced to retrieval.
* **Keep tool-call records in the queue as well.** Closer to MemGPT; rejected for now
  because the records cost tokens that every method would have to pay for nothing in the
  evidence-recall benchmark. It is a reasonable extension for LLM-driven use.
