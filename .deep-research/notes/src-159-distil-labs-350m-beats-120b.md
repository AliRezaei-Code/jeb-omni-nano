# Source: Fine-Tuning Liquid's LFM2.5 — Accurate Tool Calling at 350M Parameters

- **URL:** https://www.distillabs.ai/blog/fine-tuning-liquids-lfm25-accurate-tool-calling-at-350m-parameters/
- **Publisher/Author:** distil labs (Liquid AI's named fine-tuning partner)
- **Published:** 2026-03-30
- **Tier:** B
- **Maps to report section:** SQ6 (distillation), SQ3 (backbone justification)
- **Ledger row:** 159

## Key claims

- **A 350M student can match or exceed a 120B teacher on structured output.** This is the
  strongest published support for the project premise.
- The pipeline's middle step — **validate and filter** the teacher's output — is the one
  people skip.
- **Epoch 1 captures nearly all of the gain**, independently corroborating Kev and
  Jebadiah's "one or two epochs" finding on a different codebase.
- The architecture, not just the fine-tune, is doing work: LFM2.5's **base** scores 2–6×
  higher than FunctionGemma-270M before any training.

## Data points / quotes

| Task | Teacher (GPT-oss-120B) | LFM2.5-350M base | **LFM2.5-350M tuned** |
|---|---:|---:|---:|
| Shell command execution (Gorilla) | 97.03% | 61.4% | **98.0%** |
| Smart home control | 92.11% | 63.2% | **96.7%** |
| Banking voice assistant | 96.95% | 34.5% | **95.9%** |

**The student exceeds the teacher on two of three.**

Epoch curve on tool-call equivalence:

| Dataset | Epoch 0 (base) | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|---:|
| Gorilla | 61.4% | **98.0%** | 97.0% | 98.0% | 98.0% |
| Smart Home | 63.2% | 96.1% | 96.1% | **96.7%** | 96.7% |
| Voice Assistant | 34.5% | 86.8% | 92.4% | **95.9%** | 95.4% |

The pipeline, quoted:

> "define the task with a prompt and examples, generate synthetic training data using a
> large teacher model, **validate and filter that data**, then fine-tune the student."

Multi-turn compounding, the number worth internalising:

| Task | Base | 2-turn | 5-turn |
|---|---:|---:|---:|
| Shell (Gorilla) | 61.4% | ~37.7% | ~8.7% |
| Smart Home | 63.2% | ~39.9% | ~10.1% |
| Banking Voice | 34.5% | ~11.9% | ~0.5% |

Mechanism note, relevant to a hybrid backbone:

> "standard transformers hit a memory wall on edge devices because their KV cache grows
> with every token, while LFM2's architecture **cuts that cache by up to 90%** by
> replacing most attention layers with zero-cache convolution blocks."

vs. Google's FunctionGemma (270M): LFM2.5 base scores **2–6× higher** on all three
(Gorilla 61.4% vs 9.9%), and the tuned models are comparable or better.

## Contradictions with other sources

- **vs. Jebadiah's finding that a large synthetic pool made the 9B *worse*.** No real
  conflict: Jebadiah's pool was 14,714 questions labelled end-to-end by teacher models
  with no filter step reported; this one is 5,000 examples with an explicit
  validate-and-filter stage. The difference is plausibly the filtering, and the contrast
  is the most useful thing in this note.
- **vs. "bigger is better" expectations.** A 350M model at 98.0% on Gorilla beating a
  120B at 97.03% is the counterexample, and it holds only for *structured* output.
- Corroborates Liquid's own use-case guide: "After task-specific fine-tuning, it can
  match or beat that larger model on your task at lower cost and latency."

## Credibility notes

- **Vendor partner blog with a commercial interest.** distil labs sells a fine-tuning
  platform and Liquid AI is a named partner. The numbers are specific and internally
  consistent, but the framing is promotional. Treated as Tier B for that reason.
- Model cards for the fine-tuned models are published
  (`distil-labs/distil-lfm25-shellper`, `-home-assistant`, `-voice-assistant`), so the
  artefacts are inspectable even if the blog's framing is not neutral.
- **Metric caveat the authors do not flag:** "tool call equivalence" is a binary
  exact-match score, and the three tasks are constrained function catalogues. This is
  not evidence about open-ended judgement.
- The tasks are *generative* tool calling (emit a call), not our *readout* setting. The
  transfer to a decision head is an inference, not a measurement.
