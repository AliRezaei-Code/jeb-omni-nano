# The deployment policy this model actually supports

Derived from the saved per-question outcomes in `runs/cov-full/customer_service.json`
— no retraining. Reproduce with `python experiments/risk_coverage.py <path>`.

Intervals are exact Clopper–Pearson, verified against the rule of three
(0 failures in 15 → 95% lower bound on accuracy 0.819).

## LoRA arm (fitted T = 1.05)

| coverage | k | errors | accuracy | 95% CI |
|---:|---:|---:|---:|---:|
| 2% | 6 | 0 | 1.000 | [0.607, 1.000] |
| 5% | 15 | 0 | 1.000 | **[0.819, 1.000]** |
| 10% | 30 | 2 | 0.933 | [0.805, 0.988] |
| 20% | 60 | 11 | 0.817 | [0.715, 0.894] |
| 30% | 90 | 17 | 0.811 | [0.730, 0.876] |
| 50% | 150 | 45 | 0.700 | [0.632, 0.761] |
| 100% | 300 | 131 | 0.563 | [0.514, 0.611] |

| error budget | automatable coverage |
|---|---:|
| ≤ 2% | 8.0% |
| ≤ 5% | 9.7% |
| ≤ 10% | 11.0% |
| ≤ 20% | 33.7% |
| ≤ 30% | 50.0% |

## Frozen arm (fitted T = 3.3) — for contrast

| coverage | accuracy | 95% CI |
|---:|---:|---:|
| 2% | 0.667 | [0.271, 0.937] |
| 5% | 0.600 | [0.360, 0.809] |
| 10% | 0.500 | [0.339, 0.661] |
| 30% | 0.389 | [0.303, 0.481] |

**Coverage 0.0% at every budget from 2% to 30%.** The frozen model's confidence
ranking is *anti-correlated with correctness at the top*: its single
highest-confidence prediction was wrong. There is no threshold that makes it
usable.

## What to actually ship

Three bands, from the curve above:

| confidence band | expected accuracy | action |
|---|---|---|
| top ~8–10% | 0.93–1.00 (95% LB 0.81) | act automatically |
| next ~20–25% | 0.81–0.93 | act, log, sample for audit |
| bottom ~70% | ~0.50 | escalate to a person or a larger model |

**Do not promise "100% accurate".** The 5%-coverage point is 15 questions with
zero observed errors; its 95% lower bound is 0.819. The honest phrasing is "no
errors observed in 15", and any threshold set from it is optimistic.

**Do not trust the frozen arm's numbers at all.** Coverage 0.0% is not a weak
result, it is the absence of a usable operating point.

## Replicated on a second workflow

The same experiment on `invoice_processing` (n=200): LoRA 0.6450 accuracy / T 0.850
/ cov@5% 0.0550 / cov@20% 0.2750, against a frozen arm at 0.4700 / T 4.150 /
cov@5% 0.0300. Same direction, smaller magnitude. Full table and caveats in
[`09-cross-workflow-replication.md`](09-cross-workflow-replication.md).

## Scope

- One workflow (`customer_service`) of four.
- 360 training questions, 3 epochs, CPU. A feasibility probe.
- n = 300 for the curve. Adequate for the coverage gaps (0 vs 0.097); the
  intervals above are wide and should be quoted as ranges, not point estimates.
- The bands are **operating points, not a recommendation for your data.** Re-run
  `risk_coverage.py` against your own held-out set and set thresholds from that
  curve. A model retrained on 10k examples will move every row.
