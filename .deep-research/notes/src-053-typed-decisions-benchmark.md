# Source: LocalLLaMA/typed-decisions

- **URL:** https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/main/README.md
- **Publisher/Author:** LocalLLaMA
- **Published:** 2026-09 (session)
- **Tier:** A
- **Maps to report section:** SQ2 (specialist vs generalist), SQ4 (why ECE alone is a trap)
- **Ledger row:** 53

## Key claims

- **The Prior row is the single most important number in this ledger for calibration
  design**: a model that ignores the input and answers each question's base rate has the
  **best ECE on the table** while being useless.
- **Specialist and generalist scores are not comparable**, and the benchmark says so in a
  table column.
- Soft targets cut KL by a third and score MAE by 15% "while barely moving accuracy."
- The scale of the reference points: 0.52 floor, ~0.70 strong, ~0.75 saturation.

## Data points / quotes

Baselines, all on the 2,000-decision test split:

| Model | Kind | Acc | KL↓ | Brier↓ | ECE | ms/case |
|---|---|---|---|---|---|---|
| Uniform | reference | 0.308 | 0.444 | 0.238 | 0.169 | 0 |
| **Prior** (ignores input) | reference | 0.470 | 0.347 | 0.189 | **0.088** | 0 |
| MiniLM-L6 (22M) | specialist | 0.587 | 0.262 | 0.143 | 0.108 | 22 |
| ModernBERT-base (149M) | specialist | 0.646 | 0.223 | 0.119 | 0.179 | 349 |
| Perfect scenario understanding | ceiling | 0.704 | — | — | — | — |
| Teacher self-agreement | ceiling | 0.735 | — | — | — | — |
| TypeSafe Jev 1.13.0 | generalist | 0.727 | **1.442** | 0.148 | 0.144 | 710 |
| meraGPT Decider 1 | generalist | **0.768** | 0.096 | 0.052 | 0.180 | 526 |

> "Prior also has the best ECE on the table, at 0.088, while knowing nothing. Guessing
> the base rate is perfectly calibrated by construction. That is the clearest argument
> for reading KL and Brier here instead of ECE."

> "Read 0.52 as the floor. Around 0.70 is strong. Around 0.75 is saturation… **A score
> much above 0.75 means a model has learned the teacher's quirks rather than the task.**"

> "Jev at 0.727 against the specialist's 0.646 has not beaten it by eight points… **Read
> the gap as the price of generality, not as a quality ranking.**"

Soft-target result:

> "That single change cut **KL by a third** and **score MAE by 15%**, while barely moving
> accuracy. The argmax was already right. What improved was the shape of the predicted
> distribution, which is what this benchmark is for."

**Jev's KL of 1.442 is 4× the Prior's and 15× Decider 1's** — the outlier in that
column, invisible to the accuracy column.

Data: 1,600 cases × 5 questions over 4 workflows (agent_trace_observability,
customer_service, invoice_processing, security_incidents). Gold = **mean of 3 teacher
samples at temperature 0.7**, so the labels are soft distributions by construction.

## Contradictions with other sources

- **vs. JevBench's board**, where Jev 1.13 scores at or near the top. Here its **KL is
  the worst in the table by 4×** while its accuracy is mid-pack. Two leaderboards, same
  model, opposite impressions — because one scores argmax and the other scores the whole
  distribution. Neither is wrong; they answer different questions.
- **vs. every claim in this ledger that a decision model's probabilities are "the
  product."** This benchmark's own ECE column shows a useless model winning it.
- Corroborates the DataBuilder claim elsewhere: soft targets help *when the distribution
  is good*. Here the gold is a mean of three samples; Jebadiah's synthetic pool made
  things worse.

## Credibility notes

- Apache-2.0, published dataset with per-question gold distributions, splits verified
  disjoint by id **and** by state, with the packaging refusing to build on overlap.
- The four workflows are **the same four** used on `evals.typesafe.ai` (TypeSafe's own
  workflow-eval site), which is how this dataset connects to the upstream taxonomy.
- **Self-declared ceiling of 0.735** for teacher self-agreement, stated up front. Few
  benchmarks publish their own saturation point; this one does, and it is why "beats
  Jev" claims are treated sceptically throughout this research.
- "A better model can score worse here" — the authors state that anything right where the
  teacher is wrong counts against you. Honest framing of an agreement-with-teacher metric.
- **Not a correctness benchmark.** It measures agreement with a ~4B teacher. The authors
  say so: "A score measures agreement with that teacher. It does not measure correctness."
