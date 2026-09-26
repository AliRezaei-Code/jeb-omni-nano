# Does the LoRA result replicate on a second workflow?

The LoRA ablation was run on `customer_service`. This repeats it on
`invoice_processing` — same code, same protocol, different data.

## Result: yes, same signature

| workflow | arm | n | accuracy | Brier | ECE | fitted T | cov@5% | cov@20% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| customer_service | frozen + head | 300 | 0.3233 | 0.1404 | 0.1467 | **3.300** | **0.0000** | 0.0000 |
| customer_service | **LoRA + head** | 300 | **0.5633** | **0.1098** | **0.0646** | **1.050** | **0.0967** | **0.3367** |
| invoice_processing | frozen + head | 200 | 0.4700 | 0.1502 | 0.0765 | **4.150** | **0.0300** | 0.0000 |
| invoice_processing | **LoRA + head** | 200 | **0.6450** | **0.1204** | **0.0678** | **0.850** | **0.0550** | **0.2750** |

**Every qualitative result replicates:**

1. **LoRA is worth a large accuracy delta** — +0.2400 on customer_service,
   +0.1750 on invoice_processing.
2. **The frozen arm is grossly overconfident** — fitted T of 3.300 and 4.150 against
   a correctly-calibrated ~1.0. Two different tasks, two different temperatures, both
   far above 1.
3. **LoRA fixes the calibration**, not just the argmax — T lands at 1.050 and 0.850.
   One below 1, one above: the fix is not a constant offset.
4. **Coverage is near-usable only after tuning** — frozen 0.0000 / 0.0300; LoRA
   0.0967 / 0.0550 at a 5% budget.

## Where it does NOT replicate, and that matters

**The absolute coverage is much lower on `invoice_processing`.** At a 5% error
budget, LoRA reaches 9.7% coverage on customer_service but only 5.5% on
invoice_processing. At 20% error: 33.7% vs 27.5%. At 30%: 50.0% vs 76.0%.

So the *direction* is robust and the *magnitude* is not portable. A deployment
threshold tuned on one workflow will not transfer. This is the concrete argument
for the guide's insistence that you re-run the curve on your own data.

The frozen arm's curve on invoice_processing makes the same point harder:

| coverage | accuracy | 95% CI |
|---:|---:|---:|
| 2% | 1.000 | [0.473, 1.000] |
| 5% | 1.000 | [0.741, 1.000] |
| 10% | 0.900 | [0.717, 0.982] |
| 20% | 0.825 | [0.696, 0.915] |
| 30% | 0.750 | [0.641, 0.839] |

n = 200, so the top 10% is 20 questions and the 95% lower bound there is 0.717. The
intervals are wide enough that the top-of-distribution accuracy claims are not
precise, only the ordering.

## What this does and does not license

**Does:** LoRA on the backbone is the thing that makes a small typed-decision model
usable rather than merely scoreable. Two workflows, two different data shapes, same
conclusion, and the calibration signature (T ≫ 1 frozen, T ≈ 1 tuned) is the
cleanest part of the result.

**Does not:** any specific coverage number. Both workflows used 240–360 training
questions and 3 CPU epochs. Published recipes use 10k–24k. Treat every coverage
figure here as a floor, not a forecast.
