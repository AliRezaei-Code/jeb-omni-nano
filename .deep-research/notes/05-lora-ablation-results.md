# Experiment: does LoRA actually help?

**The load-bearing assumption of this project, tested.** Every design decision assumes
the answer is yes — Kev, Jebadiah and Jev-Omni all fine-tune adapters, and we ported
that assumption without ever testing it. The layer sweep made the question sharper: a
frozen backbone with a head-only fit scored barely above the base-rate Prior.

## Result

Identical data, splits and evaluation. The **only** difference is whether the backbone's
weights can move. Read-out at the final layer (L15), the layer the sweep measured as
the winner.

| arm | accuracy | Brier ↓ | ECE ↓ | coverage @5% | fitted T | train loss |
|---|---:|---:|---:|---:|---:|---:|
| A — frozen backbone + head only | 0.2667 | 0.1371 | 0.1784 | 0.0667 | **2.400** | 1.2677 |
| **B — LoRA r=16 all-linear + head** | **0.5778** | **0.1154** | **0.1204** | 0.0667 | **1.050** | **1.1630** |
| delta | **+0.3111** | −0.0217 | −0.0580 | 0.0000 | −1.350 | −0.1047 |

**LoRA is worth 31.1 accuracy points on this task and this backbone.** For comparison,
`typed-decisions` reports a `Prior` (base-rate) baseline of 0.470 — arm B clears it, arm A
does not.

## The more interesting result: LoRA fixed calibration, not just the argmax

The fitted temperature moves from **2.400** to **1.050**. A large T means the model's raw
logits are much more confident than its accuracy justifies, and the fit has to soften
them heavily. A T of 1.05 means the LoRA-tuned model is *nearly* self-calibrating.

This is the concrete mechanism behind Jebadiah's reported finding that
"accuracy was flat while ECE fell 0.086 → 0.019: the model did not learn to be right
more often, it learned to be honest about when it isn't." Here both improve, and the
temperature is the diagnostic that shows which one happened.

It is also a practical warning: a frozen head-only model does not merely score worse,
it scores *dishonestly* worse. If you ship one, the probability you threshold on is
not meaning what you think it means.

## Caveats, stated

- **n = 45** evaluation questions. The accuracy delta is 14 questions.
- **One workflow** (`customer_service`) of four.
- **198 training questions, 3 epochs, CPU.** This is a feasibility probe, not a
  tuned result. Published work uses 10k–24k examples.
- **Coverage at 5% error is 0.0667 (3 questions) in both arms** and is too small to
  distinguish. It is reported because it is the number a routing policy cares about,
  not because it is informative at this sample size.
- Absolute accuracy is far from any published result; the claim is the **delta between
  arms**, not the endpoint.

## An inconsistency between my own two experiments, disclosed

The layer sweep read **pre-norm** activations: it hooked `backbone.layers[i]` and
captured that module's raw output. This experiment reads **post-norm**: it calls
`backbone(...)`, which applies the final RMSNorm before returning.

That is why the frozen+head arm scores **0.2667 here** versus **0.4889** in the layer
sweep, on the same data and the same final layer. They are different representations and
the post-norm read is the worse one for a frozen backbone.

**This does not affect the LoRA conclusion**, because both arms in *this* experiment
use the identical post-norm path and are compared against each other. But the two
experiments must not be compared to each other directly, and the layer sweep's absolute
numbers should be read as pre-norm figures. The layer *ranking* is unaffected, because
all five layers in that sweep were read the same (pre-norm) way.
