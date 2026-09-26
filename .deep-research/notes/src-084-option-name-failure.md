# Source: Type-Safe Is Not Error-Free — a Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It

- **URL:** https://arxiv.org/abs/2609.26758
- **Publisher/Author:** Yu Sun, Junhao Xu, Jiajia Shi, Zijin Yang
- **Published:** 2026-09-22
- **Tier:** A
- **Maps to report section:** SQ8 (failure modes), SQ2 (read-out geometry)
- **Ledger row:** 84

## Key claims

- Changing only the **name↔rubric binding** — holding the question, state, rubric
  wording and the *set* of option names fixed — causes a **systematic ranking reversal**,
  not uncertainty.
- The failure is **stronger as the option count increases**.
- **Read-out geometry moderates it:** a mean-pooling family flips 4.1× less often.
- **Random character-string option names eliminate it entirely at no accuracy cost.**
- **Type-error rate stays 0% throughout.**

## Data points / quotes

> "On 1200 workflow decisions with task-specific rubrics, renaming binary options 0/1 to
> no/yes changes **70.4 more answers per hundred (95% CI: [67.6, 73.1])** and shifts
> **AUC from .94 to .23**, revealing a systematic reversal in the decision ranking
> rather than simple uncertainty."

| Condition | Result |
|---|---|
| 1,200 decisions, 4 predicates | **+70.4 answers per hundred** (CI [67.6, 73.1]) |
| AUC | **.94 → .23** |
| Effect vs neutral-name control | **≥ 7.4×**, across all 4 predicates |
| Scaling with option count | effect **grows** |
| Read-out geometry | mean-pooling family flips **4.1× less often** |
| Hosted Jev | AUC **.8146 → .5806**, **24×** its test-retest flip floor |
| Random character-string names | all families **return to neutral, no accuracy loss** |
| **Type-error rate** | **0% in every condition** |

## Contradictions with other sources

- **vs. every model card in the ledger reporting a calibration figure.** A 0% type-error
  rate alongside a ranking reversal is the sharpest available demonstration that
  **schema conformance is not a correctness signal**. Several projects (Laya, OpenDecision)
  advertise "it cannot produce a malformed answer" as a headline property; this paper
  shows that property is orthogonal to being right.
- **vs. pointer scoring (Kev).** Kev's pointer head weights each option's own
  representation. If the mean-pooling family here is a pointer-style reader, the
  4.1× result is an argument *against* Kev's geometry and *for* slot logits. It is an
  inference from one sentence, not a stated finding — flagged as interpretation.
- **Corroborated independently** by `r-ms/mini-jev`, which found that reading a **symbol**
  beats reading the option's **name** by +10.0 pp. Different experiment, same underlying
  worry: the readout is sensitive to what the option *is called*.
- **Corroborated again** by Nokia's AnyJev, whose zero-label L0 readout cuts the
  order-flip rate from 0.230 to **0.073** with no training at all.

## Credibility notes

- Peer-style arXiv preprint, four named authors, published 2026-09-22 — **four days
  before this research session**. Brand new, and not yet replicated by anyone else in
  this ledger.
- The design is a clean intervention: one variable changed, the name *set* held constant,
  and a neutral-name control plus a random-string control. The random-string arm is what
  makes the conclusion specific — it rules out "renaming" as the cause and isolates
  **semantic polarity** as the cause.
- Sample is 1,200 decisions with confidence intervals given. Adequate for the effect
  size claimed.
- **Scope limit stated by the authors:** the effect is demonstrated on the hosted Jev and
  two open models. It has **not** been measured on a slot-logit model at 350M scale,
  which is our configuration. We therefore present the neutral-identifier mitigation as
  a *diagnostic to run*, not as a known-required fix.
