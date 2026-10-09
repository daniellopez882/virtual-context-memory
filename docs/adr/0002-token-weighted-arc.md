# ADR 0002: ARC measured in tokens, not pages

Status: accepted

## Context

ARC (Megiddo and Modha, 2003) manages a cache of `c` equally sized pages with four LRU
lists: T1 and T2 hold resident pages seen once and at least twice, B1 and B2 hold the keys
of recently evicted T1 and T2 pages. A target `p` for the size of T1 moves up on a B1 ghost
hit and down on a B2 ghost hit.

Here a page is a chat message. On LoCoMo a turn is 35 WordPiece tokens on average and up
to 116, and the constraint that matters is the token budget, not a message count. A
count-based ARC with `c` messages would either overflow the budget when long messages
cluster or waste it when short ones do.

## Decision

`ARCPolicy` keeps every list as an ordered map from key to size in tokens:

* `c` is the queue capacity in tokens from the budget plan;
* `p` is a token target for T1, clamped to `[0, c]`;
* on a B1 ghost hit, `p += max(1, |B2| / |B1|) * size`; on a B2 hit,
  `p -= max(1, |B1| / |B2|) * size`, with list sizes in tokens;
* the victim is the LRU page of T1 when `|T1| > p` (or `|T1| >= p` right after a B2
  fault), otherwise the LRU page of T2, skipping pinned pages;
* ghosts are trimmed after every admit and evict so that `|T1| + |B1| <= c` and
  `|T1| + |T2| + |B1| + |B2| <= 2c` hold whenever the resident set fits.

The manager, not the policy, decides how much to evict (it asks for victims until the
queue is under its low watermark). The policy only orders victims.

## Consequences

* With unit sizes and one eviction per miss the policy reduces to textbook ARC.
* The invariants are tested with hypothesis over random admit/touch sequences with
  random sizes and pinned keys (`tests/test_policies.py`).
* Ghost-list bounds are in tokens, so a few long evicted messages can push out many short
  ghosts. That is the intended trade: ghosts approximate "recently resident tokens".
* The same `EvictionPolicy` interface carries sizes for every policy, so size-aware
  variants of the other policies need no interface change.

## Alternatives rejected

* **Count-based ARC with `c` = average messages per budget.** Cannot guarantee the
  budget and makes `c` depend on the data.
* **Byte-level slab classes (as in memcached).** Overkill for a few hundred pages and it
  would hide the algorithm being compared.
* **Letting ARC trigger its own replacements.** Classic ARC evicts exactly one page per
  miss; batch eviction to a low watermark (ADR 0003) needs the manager to own the loop.
