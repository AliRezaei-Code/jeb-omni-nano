# Source: mini-Jev — read the letter

- **URL:** https://raw.githubusercontent.com/r-ms/mini-jev/main/README.md
- **Publisher/Author:** r-ms
- **Published:** 2026-09 (session)
- **Tier:** A
- **Maps to report section:** SQ2 (read-out geometry), SQ6 (cost of constrained decoding)
- **Ledger row:** 179

## Key claims

- Reading an option **letter** at the answer position is **accuracy-neutral** against
  generating JSON under a grammar → maps to **SQ2**.
- A letter beats writing the option's **name** by a large margin → maps to **SQ2**,
  supports slot logits over label-token scoring.
- Model-written probabilities are a **disaster** → maps to **SQ2/SQ4**, reinforces the
  whole thesis.
- Raw option shares are **not** calibrated probabilities → maps to **SQ4**, the caveat
  our temperature fit exists to close.
- Shared-prefix caching is worth **1.4–2.4×** on long inputs; naive per-field re-read
  is *slower* than one JSON → maps to **SQ6**.

## Data points / quotes

> "Does reading the letter lose accuracy vs generating JSON under a grammar? **No.**
> Intent field, 6750 paired observations on 450 texts: JSON 0.909, letters 0.907,
> Δ −0.22 pp, 95 % CI [−1.44, +1.04]. **Every k from 2 to 16 has a CI covering zero.**"

> "Is a letter better than writing the option's name? Yes: **+10.0 pp [+8.3, +11.7]** for
> the letter on intent, **+13.2 pp** on domain, +1.2 pp (CI covers 0) on a boolean whose
> names are already single tokens."

> "Should the model write probabilities instead (the TypeSafe adapter shape)? **No.** With
> a bounded enum grid and a 640-token cap the form finishes but scores **0.346** on intent
> vs **0.896** for letters on the same units; **62 % of its 'choices' are the first
> option.**"

> "the shares are *normalized candidate scores*, not calibrated probabilities; use the gap
> for abstention, **do not read the percentage as P(correct)**"

Speed, on the same runs:

| setting | speedup vs JSON |
|---|---|
| 32-token text | **4×** (0.24× the time) |
| 2048-token text, 1 field, shared cache | 1.4× (0.71×) |
| 2048-token text, 2 fields, shared cache | 1.8× (0.56×) |
| 2048-token text, 3 fields, shared cache | 2.4× (0.41×) |
| 2048-token text, **naive re-read per field** | **1.10–1.16× SLOWER** |

Dependent-field result: a boolean that follows from an earlier field **gains 5 pp** when
written after it in one JSON rather than read separately.

Out-of-scope behaviour: on 50 out-of-scope texts, JSON picked "none of the above" on
47/50 and letters on 41/50; both were 1/50 on in-scope texts.

## Contradictions with other sources

- **vs. `decider-2b-vision` / Visual Jev (arXiv 2609.25845):** that paper reports "A
  matched typed-head control offers **no consistent accuracy advantage** over the
  language-model-head readout." mini-jev finds reading a *symbol* is worth +10 pp over
  reading a *name*, which is compatible — both say a trained/typed readout helps, and
  Visual Jev's control was matched, not weaker. Not a true conflict, but the two
  findings are in tension about *how much* the head matters.
- **vs. Jebadiah's `hard` vs `train` temperature fits:** mini-jev explicitly declines to
  fit a temperature at all. It therefore reports ranking quality, not calibrated
  probabilities, and its accuracy numbers are not comparable to any ECE column in this
  report.

## Credibility notes

- **Preregistered** (`PREREG.md`, amendments v1.1–v1.3). This is the single strongest
  methodological property in the entire 200-source ledger: the design was fixed before
  the runs, which forecloses the most common failure mode in self-reported LLM results.
- Every number is **recomputed from stored run records** by the authors' own scripts.
- **Adversarially relevant limitation the authors state themselves:** "extraction-as-choice
  … is a recommendation from the design, **not measured in this study**", and Score was
  not measured. We cite only the Choice and boolean results.
- Frozen model (Qwen3-4B-Instruct-2507, greedy), CLINC150 task. Single backbone, single
  task — generalisation to LFM2.5 is an inference, not a measurement.
- Reported determinism: "The same code on two different RTX 4090s is bit-identical
  (607 / 607)."
