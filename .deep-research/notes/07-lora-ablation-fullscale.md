# The LoRA ablation at full scale — the project's headline result

Supersedes the n=45 probe in `05-lora-ablation-results.md`. Same experiment, same
code, evaluated on a set large enough for coverage to mean something.

## Setup

- Backbone `LiquidAI/LFM2.5-350M` @ `9e6c6ccf`, read-out at the final layer (the
  layer the sweep measured as the winner).
- Data: `LocalLLaMA/typed-decisions` / `customer_service`, **all 400 cases / 1,200
  questions**, soft gold distributions, split by case.
- **360 train questions / 600 eval questions**, temperature fitted on the first half
  of eval and every metric reported on the second half. **n = 300** for metrics.
- The only difference between arms: whether the backbone's weights can move.

## Result

| arm | accuracy | Brier ↓ | ECE ↓ | fitted T | cov@5% | cov@10% | cov@20% | acc@10% cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — frozen backbone + head | 0.3233 | 0.1404 | 0.1467 | **3.300** | **0.0000** | **0.0000** | **0.0000** | 0.5000 |
| **B — LoRA r=16 all-linear + head** | **0.5633** | **0.1098** | **0.0646** | **1.050** | **0.0967** | **0.1100** | **0.3367** | **0.9333** |
| delta | **+0.2400** | −0.0306 | −0.0821 | −2.250 | **+0.0967** | **+0.1100** | **+0.3367** | **+0.4333** |

## What this actually says

**The frozen model can auto-decide nothing.** Coverage at a 5% error budget is
**0.0000** — not 0.07, not 0.1, exactly zero. At n=300 a 5% budget permits 15
errors, and the frozen model's confidence ranking is not good enough to spend
that budget safely. Its fitted temperature is **3.300**: the raw logits are more
than three times more confident than the model's accuracy justifies.

**LoRA turns it into a usable component.** Coverage goes 0.0000 → 0.0967 at a 5%
error budget, and 0.0000 → 0.3367 at 20%. The fitted temperature lands at
**1.050**, i.e. the tuned model is close to self-calibrating.

**The routing signal is strong at the top of the distribution.**
`accuracy_at_10pct_coverage` is **0.9333**: take the 10% most-confident decisions
and they are right 93% of the time, against 0.5633 overall. With the frozen arm
the same slice is 0.5000. This is the practical product number — a decision model
is worth building for the top of its confidence distribution, not its average.

**Three things move together**, which is why this is a real result rather than one
lucky metric: accuracy, calibration, and coverage all improve, and the temperature
moves to 1.05. A change that improved only accuracy, or only ECE, would be more
suspect.

## Scope, stated

- **One workflow** (`customer_service`) of four.
- **360 training questions, 3 epochs, CPU.** Published recipes use 10k–24k. This is a
  feasibility probe; a properly trained model should do better on all four columns.
- **n = 300** for metrics. Adequate for a coverage estimate, thin for a 1-point
  accuracy delta. The coverage gap (0 vs 0.0967) is far larger than the noise here.
- Task mixture is unbalanced (2-option `noul`, 5-option `choice`, 4-5 level `score`
  in one batch), so per-type accuracy is not reported and the aggregate is a blend.

## Relation to the n=45 probe

The earlier run reported cov@5% = 0.0667 for **both** arms. That number was
correct arithmetic and completely uninformative: with 45 questions the first
error lands at rank 4, so every budget below 25% collapses to the same 3/45. See
`06-coverage-metric-correction.md`. The full-scale run is the one to quote.

The two runs agree on direction and on temperature: arm A T=2.400 at n=45, T=3.300
at n=300; arm B T=1.050 at both.
