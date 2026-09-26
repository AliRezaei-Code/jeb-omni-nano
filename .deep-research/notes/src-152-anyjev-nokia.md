# Source: AnyJev — turn any LLM into a Jev-style decision model

- **URL:** https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/README.md
- **Publisher/Author:** Nokia (Sunnyvale, CA) + Tencent Hunyuan
- **Published:** 2026-09 (session)
- **Tier:** A
- **Maps to report section:** SQ2 (read-out geometry), SQ4 (calibration), SQ8 (failure modes)
- **Ledger row:** 152

## Key claims

- **A middle layer beats the last layer as a linear read-out feature space** → this
  changed our code; `readout_layer` is now a constructor argument.
- **Depth reduction is a gain, not a trade** → directly contradicts the intuition that
  the final layer is best.
- **A closed-form head needs only 100–300 labels and no gradients** → changes the
  recommended order of operations: fit closed-form *before* LoRA.
- **Order-flip has a zero-label fix** → maps to SQ8, corroborates the option-name paper
  by a different route.
- **fp8 is a bad trade for a linear head** → maps to SQ6, third independent
  quantisation caveat.

## Data points / quotes

Qwen3-8B, BANKING77 20-way, 300 test items:

| | raw logits | L0 (zero labels) | L1 (+temperature) |
|---|---:|---:|---:|
| Labels required | none | **none** | 100–500 |
| Answer flips when options are reversed | 0.230 | **0.073** | 0.077 |
| Accuracy | 0.747 | 0.803 | 0.807 |
| Calibration error (ECE) | 0.240 | 0.184 | **0.095** |
| **Auto-decidable at ≤5% error** | **7.7%** | 46.3% | **52.0%** |

> "The last row is the point: accuracy moves six points, but the share of traffic you can
> safely automate goes **7.7% → 52.0%**. With raw logits a '0.9' is not trustworthy
> enough to act on, so everything goes to a human. Once the probability means what it
> says, you can set a threshold."

The truncation finding, verbatim:

> "Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy slightly *higher* and calibration
> better, and was faster: **a middle block is a better feature space for a linear head
> than the last one, where the remaining blocks are busy turning the answer into
> tokens.**"

> "**Depth is usually a gain, not a trade.**"

> L2: "A closed-form head per question, solved on 100–300 labels in seconds — no
> gradients, the model's weights untouched"

> "`--quantization fp8` is available and not recommended — **it buys single-question
> latency and costs accuracy**."

Deployment parity: "A head fit through `transformers` and served by vLLM answers the same
as one fit and served on either alone — **99.0% identical answers, mean |dp| 0.0011** on
BANKING77-20."

## Contradictions with other sources

- **vs. Jev-Omni and Kev, which read the final layer.** AnyJev says an intermediate
  layer is better. This is a direct, unresolved disagreement between implementations.
  We default to `-1` to preserve Jev-Omni parity and **expose the knob rather than
  choosing**, because nobody has swept it on LFM2.5.
- **vs. the assumption that quantisation is free.** Corroborated by the Q4_K_M Jev-Omni
  card (Δp 0.210) and by the per-format temperature refit. AnyJev adds the stronger
  claim that even **fp8** costs accuracy for a head.
- **vs. `this-that-model`'s note that multi-step arithmetic fails** (0.560 vs 0.98–1.00).
  Not a conflict, but together they describe the same ceiling: a single forward pass
  cannot carry intermediate results.

## Credibility notes

- **Corporate research lab with named authors** (Nokia Sunnyvale + Tencent Hunyuan), not
  an individual blog. That raises the baseline substantially.
- Apache-2.0, on PyPI, with a reproducible pipeline command
  (`python -m anyjev.pipeline <model> --labels-from banking20`) that truncates, serves,
  fits, measures, shuts down, and re-runs at full depth for comparison.
- **The authors tell you to verify on your own hardware**: "Measure it on your own box
  instead of trusting ours."
- Single backbone (Qwen2.5/Qwen3-8B) and a single benchmark (BANKING77-20). The
  *truncation* finding is a 7B model and has not been shown at 350M scale, which is
  where we intend to use it.
- Their timing caveat is unusually good: "on a shared machine a single pass can report
  the same configuration as both faster and slower than the baseline."
