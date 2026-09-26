# Experiment: the readout-layer sweep on LFM2.5-350M

**Status:** run in this repo, on real data, with real weights. This is the experiment
the report had flagged five times as "wired but unvalidated".

## The claim under test

Nokia + Tencent Hunyuan, `AnyJev`:

> "Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy slightly *higher* and
> calibration better, and was faster: **a middle block is a better feature space for a
> linear head than the last one**, where the remaining blocks are busy turning the
> answer into tokens."

Every implementation in this research reads the **last** layer — Jev-Omni, Kev,
Jebadiah, Laya, this-that-model. We ported Jev-Omni, so we read the last layer too.
If a middle layer wins, that is free accuracy *and* it licenses truncating the layers
above it.

## Design

| Choice | Value | Why |
|---|---|---|
| Backbone | `LiquidAI/LFM2.5-350M` @ `9e6c6ccf` | The project's chosen backbone, pinned |
| Backbone weights | **frozen** | Makes the read-out layer the *only* variable. Adding LoRA would confound the comparison with adapter quality. |
| Head | fresh `DecisionHead(1024, max_options=32)` per layer | Each layer gets its own head, so no layer is advantaged by a shared fit |
| Data | `LocalLLaMA/typed-decisions`, `customer_service` | Apache-2.0, ships **gold probability distributions**, not just labels |
| Target | the **soft gold distribution** | Argmax alone hides distribution shape — the benchmark's own Prior row has the best ECE while knowing nothing |
| Split | **by case**, then by question | Every case contributes ~3 questions about the same state; splitting per question would leak the state across train and eval |
| Temperature | fitted on one half of the held-out set, reported on the other | Fitting and reporting on the same rows is exactly the mistake the MLX Jev-Omni card documents |

## RESULT: the finding does NOT replicate

| layer (from end) | train loss | accuracy | Brier | ECE | fitted T |
|---|---|---|---|---|---|
| L0  (-16) | 1.2877 | 0.4000 | 0.1360 | 0.2258 | 0.350 |
| L4  (-12) | 1.2875 | 0.4000 | 0.1360 | 0.2256 | 0.350 |
| L8  (-8)  | 1.2760 | 0.4000 | 0.1358 | 0.2268 | 0.600 |
| L12 (-4)  | 1.2665 | 0.4000 | 0.1346 | 0.2203 | 0.750 |
| **L15 (-1)** | **1.2508** | **0.4889** | **0.1312** | **0.1360** | **0.900** |

**The last layer wins on accuracy, Brier, ECE and training loss, and the fitted
temperature rises monotonically with depth.** AnyJev's finding does not replicate here.

Weight: n=45, so the accuracy gap is 4 questions. What carries it is that four
independent indicators agree and the temperature trend is monotonic.

AnyJev measured a **7B** model. We measured 350M, text-only, one workflow. The
mechanism AnyJev proposes -- "the remaining blocks are busy turning the answer into
tokens" -- may simply be weaker at 350M scale.

**Decision: `readout_layer=-1` stays the default**, and the guide now says the sweep
found nothing better rather than leaving an open question.

Full write-up: [`04-layer-sweep-results.md`](04-layer-sweep-results.md).

## Bugs found by running it

Four, all of which would have produced a plausible-looking wrong answer:

1. **Case dedup collapsed everything.** Keying cases on `prompt[:160]` is identical for
   every sibling question of a case, so all 450 questions landed in one bucket and the
   split returned 0/0/0. Fixed by keying on the **state** (the text before the `---`
   separator), not the prompt.
2. **Wrong module passed to `resolve_backbone`.** It walks `.model`/`.backbone`/… from a
   raw HF module; I passed the `JebNanoModel` wrapper, which has no `.layers`.
3. **Gold schema was assumed, not read.** I guessed `options` on the question and
   `{true: p}` on the gold. The real schema is `criteria` (a **dict** for choice/noul, a
   **list** for score) and gold `probabilities: {option: p}`. Verified by fetching and
   printing the Hub rows before writing the parser.
4. **Ragged targets.** Option counts are 2 for `noul` and 5 for `category`, so a
   `torch.tensor([...])` over a batch raised. Same class of bug already fixed once in
   `train.collate`.

Bug 3 is the one worth generalising: **I would have written a plausible loader against an
imagined schema had I not printed one row first.** Every field name in this file came
from an actual dump.

## What it can and cannot tell us

**Can:** which layer of *this* backbone gives the most linearly separable representation
for typed decisions, with everything else held fixed.

**Cannot:**
- Speak to multimodal, where the projector sits after the text stack and changes the
  picture.
- Speak to a LoRA-tuned model, where the adapter reshapes the layers and the optimum
  could move.
- Generalise past one workflow. `customer_service` is one of four; the others are
  `invoice_processing`, `security_incidents` and `agent_trace_observability`.
- Establish prevalence. `n=90` evaluation questions is enough to separate a large effect
  and not enough to separate a small one.
