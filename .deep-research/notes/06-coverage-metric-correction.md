# Correction: the coverage metric was not buggy — it is unmeasurable at n=45

**I claimed a bug in the risk-coverage sweep in `lora_vs_head_only.py`. That claim was
wrong.** The old and new implementations are mathematically equivalent.

```python
# "old" — update whenever the running rate passes
for r, i in enumerate(order, 1):
    errs += 1 - correct[i]
    if errs / r <= alpha: cov = r / n

# "new" — take the max over passing k
for r, i in enumerate(order, 1):
    errs += 1 - correct[i]
    if errs / r <= alpha: best = r / n
```

Both compute `max{ r : errors(top r) / r <= alpha }`. Verified over 3,000 random
scenarios (n ∈ {10, 45, 100}, random error patterns): **0 differing cases.** I
rewrote working code and announced a bug that was not there.

## What the actual problem is

The metric is correct and the sample is too small. With the frozen+head arm, the
**first error lands at rank 4**, giving a running rate of 1/4 = **0.250**. A budget
tighter than 25% is therefore unreachable at n = 45, whatever the model does:

| budget | coverage |
|---|---|
| 5% | 3/45 = 0.0667 |
| 10% | 3/45 = 0.0667 |
| 20% | 3/45 = 0.0667 |

Identical across all three, because the first error pins them all. The reported
0.0667 is *correct*; it just carries no information about model quality.

**To measure coverage at a 5% error budget you need on the order of hundreds of
evaluation questions.** Every published figure in this field that reports coverage
at 5% (Jev 0.70, Kev-0.8B 0.145) comes from a benchmark of 500-2,000 decisions.
Our 45-question probe cannot produce that number, and the four files that quote
0.0667 should be read as "not measurable at this sample size", not "the model
automates 6.67% of traffic".

## The signal that does survive the sample size

`accuracy_at_10pct_coverage` = **0.75** for the frozen+head arm, against an overall
accuracy of 0.2667. The top 10% most-confident predictions are 75% correct.

So the confidence *ranking* carries real signal even though the confidence
*values* are badly miscalibrated (fitted T = 2.400). Those are different properties,
and a 45-question probe is enough to see the first without being able to quantify
the second. That is the more useful thing this arm produced.

## Status of the LoRA arm's coverage

The corrected re-run of arm B (LoRA) did not complete within the time budget and was
stopped mid-epoch-3. Its accuracy / Brier / ECE / temperature from the earlier run
stand, because the same seed reproduces the same fit and the change touched only
coverage reporting and per-row dumping:

    accuracy 0.5778 · Brier 0.1154 · ECE 0.1204 · fitted T 1.050

**Its coverage-at-budget numbers are not measured** and are deliberately left blank
rather than estimated.
