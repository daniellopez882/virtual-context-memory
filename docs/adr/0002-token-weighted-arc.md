# ADR 0002: ARC measured in tokens, not pages

Status: accepted

## Context

ARC (Megiddo and Modha, 2003) manages a cache of `c` equally sized pages with four LRU
lists: T1 and T2 hold resident pages seen once and at least twice, B1 and B2 hold the keys
of recently evicted T1 and T2 pages. A target `p` for the size of T1 moves up on a B1 ghost
hit and down on a B2 ghost hit.

Here a page is a chat message. On LoCoMo a turn's text is about 33 WordPiece tokens on
average and up to 116 (about 41 and up to 124 once rendered with speaker and date, which
is what a page costs), and the constraint that matters is the token budget, not a message count. A
count-based ARC with `c` messages would either overflow the budget when long messages
cluster or waste it when short ones do.

## Decision

`ARCPolicy` keeps every list as an ordered map from key to size in tokens:

* `c` is the queue capacity in tokens from the budget plan;
* `p` is a token target for T1, clamped to `[0, c]`;
* on a B1 ghost hit, `p += max(1, |B2| / |B1|) * size`; on a B2 hit,
  `p -= max(1, |B1| / |B2|) * size`, with list sizes in tokens;
* the victim is the LRU page of T1 when `|T1| > p` (or `|T1| = p` right after a B2
  fault), otherwise the LRU page of T2, skipping pinned pages;
* `|T1|` in that test leaves out the pages admitted in the current step, and those pages
  are not candidates while anything else is. The paper runs REPLACE before the requested
  page enters T1; here the manager admits first and asks for victims afterwards, so the
  policy has to remember what just arrived;
* on a miss, ghosts are trimmed before the new page is inserted, following case IV of the
  paper: B1's LRU ghosts go while `|T1| + |B1| + size > c`, otherwise B2's go while the
  directory plus the new page would exceed `2c`. When B1 is empty and T1 alone fills
  its share, the paper drops T1's LRU page without a ghost; here that page is evicted
  into B1 and trimmed straight away, which ends in the same state;
* after every eviction ghosts are trimmed again, so `|T1| + |B1| <= c` and
  `|T1| + |T2| + |B1| + |B2| <= 2c` hold whenever the resident set fits.

The manager, not the policy, decides how much to evict (it asks for victims until the
queue is under its low watermark). The policy only orders victims.

## Consequences

* With unit sizes and one eviction per miss the four lists and `p` match the paper's
  Fig. 4 after every request. `tests/test_policies.py` checks this with hypothesis
  against a direct transcription of the paper's pseudocode.
* The first version counted the page being admitted in `|T1|` and trimmed ghosts after
  inserting it. That evicted from T1 where the paper evicts from T2 and dropped B2 ghosts
  the paper keeps (shortest case: `c = 2`, requests `a a b c b d`). Its invariant tests
  passed because they checked only list bounds; the differential test above now guards
  the decisions themselves. The recall benchmark was re-run after the fix (commit
  `ba1b6df`, and again with identical recall at `08b8f7d`, the run now in
  `bench/results/locomo-replay.json`): ARC's fresh-probe recall went
  from 69.7, 79.2 and 85.9% to 70.0, 79.7 and 87.9% at 1024, 2048 and 4096 tokens, and it
  is still the weakest policy at the two larger budgets.
* The size invariants are also tested with hypothesis over random admit/touch sequences
  with random sizes and pinned keys.
* Ghost-list bounds are in tokens, so a few long evicted messages can push out many short
  ghosts. That is the intended trade: ghosts approximate "recently resident tokens".
* The same `EvictionPolicy` interface carries sizes for every policy, so size-aware
  variants of the other policies need no interface change.
* The manager evicts in batches down to a watermark (ADR 0003), so in the benchmark
  several REPLACE steps follow one miss. Each one uses the same rule; the paper has no
  counterpart for this.

## Alternatives rejected

* **Count-based ARC with `c` = average messages per budget.** Cannot guarantee the
  budget and makes `c` depend on the data.
* **Byte-level slab classes (as in memcached).** Overkill for a few hundred pages and it
  would hide the algorithm being compared.
* **Letting ARC trigger its own replacements.** Classic ARC evicts exactly one page per
  miss; batch eviction to a low watermark (ADR 0003) needs the manager to own the loop.
