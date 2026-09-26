# Readout-layer sweep — measured result

Reproduce with:
`python experiments/layer_sweep.py --workflow=customer_service --max-cases=60 --layers=-16,-12,-8,-4,-1 --epochs=3`

**Question.** Nokia's AnyJev reports that a *middle* layer is a better feature space for
a linear head than the last one, because the top layers are busy turning the answer into
tokens. Does that hold for LFM2.5-350M on typed decisions?

**Answer: no. On this configuration the last layer wins, on four independent indicators.**

| layer (from end) | train loss | accuracy | Brier ↓ | ECE ↓ | fitted T |
|---|---|---|---|---|---|
| L0  (-16) | 1.2877 | 0.4000 | 0.1360 | 0.2258 | 0.350 |
| L4  (-12) | 1.2875 | 0.4000 | 0.1360 | 0.2256 | 0.350 |
| L8  (-8)  | 1.2760 | 0.4000 | 0.1358 | 0.2268 | 0.600 |
| L12 (-4)  | 1.2665 | 0.4000 | 0.1346 | 0.2203 | 0.750 |
| **L15 (-1)** | **1.2508** | **0.4889** | **0.1312** | **0.1360** | **0.900** |

**Protocol.** `LiquidAI/LFM2.5-350M` @ `9e6c6ccf`, **frozen backbone**, a fresh
`DecisionHead` trained per layer, so the read-out layer is the only variable. Data:
`LocalLLaMA/typed-decisions` / `customer_service` (Apache-2.0), trained against the
**soft gold distribution**, not the argmax. Split by case — 66/54/30 cases → 198/162/90
questions. Temperature fitted on one half of the held-out set, reported on the other.

## How much weight this carries

- **n = 45** evaluation questions. The accuracy gap is **4 questions**. On its own, that
  is not a result.
- What makes it more than one number: accuracy, Brier, ECE **and** final training loss
  all point the same way, and the fitted temperature rises monotonically with depth
  (0.35 → 0.90), meaning earlier layers are systematically more overconfident relative
  to how often they are right.
- **Absolute quality is low** — 0.4889 against a `Prior` base-rate baseline of ~0.470 on
  the full benchmark. This is a 198-example, head-only, frozen-backbone probe. The claim
  is about *which layer*, not about achievable accuracy.

## What it settles, and what it does not

**Settles:** for LFM2.5-350M on text-only typed decisions, reading the **last** layer is
the right default. `readout_layer=-1` stays, and AnyJev's finding is now marked
**not replicated at this scale**.

**Does not settle:** multimodal (the projector sits after the text stack); LoRA-tuned
models (the adapter reshapes the layers and could move the optimum); the other three
workflows; or whether the reversal appears at larger parameter counts — **AnyJev measured
Qwen2.5-7B**, and the mechanism it proposes ("the remaining blocks are busy turning the
answer into tokens") may simply be weaker at 350M.
