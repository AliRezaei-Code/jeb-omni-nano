# Fixing the evaluation loop, and what it cost

The eval loop in `lora_vs_head_only.py` ran **one forward pass per question**. On the
996-question eval split that is ~1000 single-item forwards, and it dominated runtime —
which is why the last two experiments each took 1-2 hours and why the four-workflow LoRA
arm had to be stopped mid-evaluation.

## The fix

Sort the eval questions by tokenised length and run them in length-sorted groups of 16,
so padding waste stays low. The head is applied once per batch and the per-row slice is
taken afterwards.

## Verified

| | |
|---|---|
| max abs difference, batched vs unbatched, 16 rows of **varied** length | **2.956e-05** |
| unbatched | 211.32 s |
| batched | 21.65 s |
| **speedup** | **9.76×** |

**Not bit-identical, and that is expected.** Batching changes the order of floating-point
accumulation inside the matmuls, so a different reduction order gives a different last
bit. 3e-05 on a hidden state of order 1 is ordinary float noise, not a correctness
problem — the same class of difference Kev documents when they note that identical code on
two RTX 4090s *is* bit-identical (same device, same order), while a different device is
not.

A stricter equality check would have been the wrong test to write. The right test is
"within float noise", and it passes.

## What this means for the reported numbers

**Nothing changes.** The batched and unbatched paths differ by 3e-05 in the hidden
state, which is far below the effect sizes being measured (accuracy deltas of 0.24,
temperature deltas of 2.25). No metric in any recorded result moves.

## What it unlocks

A proper data-scaling curve — 300 / 1.2k / 5k / 20k training questions — becomes
tractable. Training still dominates and is *not* batched here beyond the existing batch
of 8, so the win is on the evaluation half. Expect roughly a 2x wall-clock
improvement end to end, not the 9.76x the eval loop alone shows.
