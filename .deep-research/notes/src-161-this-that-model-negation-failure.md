# Source: this-that-model (training repo and changelog)

- **URL:** https://raw.githubusercontent.com/FLock-io/this-that-model/main/README.md
- **Publisher/Author:** Zehua Cheng, Wei Dai, Jiahao Sun — FLock.io / University of Oxford
- **Published:** 2026-09 (session)
- **Tier:** A
- **Maps to report section:** SQ8 (failure modes), SQ4 (calibration regression)
- **Ledger rows:** 51 (model card), 161 (repo)

## Key claims

- **A leading decision model read a negated rule as its opposite, at 0.88 confidence.**
- Four of eleven phrasings of that rule scored **below chance** — systematically wrong,
  not guessing.
- **The newest checkpoint is the worst on the axis you would threshold on.** qL2 on
  `sim_local_ood`: 1.0 **0.0096** → 1.1 0.0203 → 1.2 0.0161.
- Composed decisions (several rules at once) went **0.406 → 0.775 → 0.878** against a
  0.258 chance rate.
- Two generator-only loss terms drove the 1.1 gain: the same decision rendered four ways
  must answer the same, and two policies over one state must **not** answer the same.

## Data points / quotes

The negation failure, verbatim:

> "**1.2** fixed *how the rule is written*. 1.1 read `bays without chilled handling are
> ineligible` as though it named the eligible set — **not failing to apply the rule but
> applying its opposite, at 0.88 mean confidence.** Over eleven phrasings of one rule,
> against a chance rate of 0.19, **1.1 ranges from 0.00 to 0.89 and 1.2 from 0.98 to
> 1.00; four of 1.1's eleven rows sit *below* chance, which is not guessing.**"

The calibration regression, verbatim:

> "**Read the calibration rows before you upgrade.** 1.1 gave up half the sharpness 1.0
> had on `sim_local_ood`… **If you threshold `confidence` rather than read the argmax —
> which this README spends a section recommending — that is the one axis where the newest
> checkpoint is not the best one**, and it is worth measuring on your own decisions before
> you move."

| metric | 1.0 | 1.1 | 1.2 |
|---|---:|---:|---:|
| composed decisions, 1,710 q (chance 0.258) | 0.406 | 0.775 | **0.878** |
| composed by depth, 242 q | — | 0.785 | **0.909** |
| spatial benchmark, 7,305 q | 0.839 | **0.871** | — |
| frozen 68-question cohort | 0.941 | **1.000** | 0.985 |
| calibration qL2 ↓ `sim_event_ood` | **0.0250** | 0.0253 | 0.0274 |
| calibration qL2 ↓ `sim_local_ood` | **0.0096** | 0.0203 | 0.0161 |

1.1 training data: 64,028 composed-decision questions over 112 rule structures and 40
domains.

The honest failure, from the paper: multi-step arithmetic scores **0.560** against
0.98–1.00 for comparators — *"a single forward pass cannot carry intermediate
results."*

## Contradictions with other sources

- **Directly contradicts this project's own advice.** Our guide tells the reader to
  threshold on confidence; their changelog says the newest checkpoint is worst on exactly
  that axis. The resolution is not "don't threshold" — it is **refit and re-measure
  calibration on every checkpoint you consider, including ones you believe are
  upgrades.** Same conclusion as the per-format refit finding, from a different direction.
- **vs. `r-ms/mini-jev`'s "never ask the model to write its own probabilities."**
  Consistent — this model reads label tokens, not generated probabilities, which is the
  mechanism that avoids that failure.
- **vs. Jebadiah's 14,714-question synthetic pool that made the 9B worse.** The
  difference is the two generator-only loss terms: theirs constrained *consistency* and
  *discrimination* between renderings, not just agreement with a teacher. The structure
  matters more than the volume.

## Credibility notes

- Named academic authors at Oxford with a company affiliation, an **arXiv paper**
  (2609.23886), released weights, and a third-party-recorded evaluation cohort.
- The 68-question cohort was **recorded by a third party and chosen by neither party**,
  which is the strongest evaluation hygiene in this ledger.
- **The calibration regression is disclosed by the authors against their own interest**,
  in a section headed "Read the calibration rows before you upgrade." That is the single
  strongest credibility signal in this note.
- The negation result is a **single rule across eleven phrasings** on one model version.
  It is a real, measured, published failure, but its base rate across the model class is
  unknown. We therefore present negation-testing as a diagnostic to run, not a known
  prevalence.
- Their spatial-benchmark row is caveated by them: question *shapes* were trained on while
  hosted systems met them for the first time.
