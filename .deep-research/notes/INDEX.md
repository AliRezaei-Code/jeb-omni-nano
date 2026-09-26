# Deep Notes — Index

Raw per-source notes for the deep-read sources, following the deep-research template's
per-source skeleton (Key claims → Data points/quotes → Contradictions → Credibility
notes). Roughly 60 of the 200 ledger sources were read at depth; these are the ones that
changed the design, plus the ones most likely to be argued with.

Every numbered `src-NNN` file maps to ledger row NNN in `../sources-ledger.md`.

## Per-source notes

| File | Source | Why it is here |
|---|---|---|
| `src-084-option-name-failure.md` | arXiv 2609.26758 — Type-Safe Is Not Error-Free | The newest and most consequential failure result. Renaming option names moves AUC .94 → .23 with a 0% type-error rate. |
| `src-053-typed-decisions-benchmark.md` | LocalLLaMA/typed-decisions | The benchmark whose Prior row has the best ECE while knowing nothing — the clearest argument for reporting KL and Brier. |
| `src-152-anyjev-nokia.md` | Nokia + Tencent Hunyuan, AnyJev | Changed the code: a middle layer beats the last for a linear head; a closed-form head needs 100–300 labels. |
| `src-159-distil-labs-350m-beats-120b.md` | distil labs × Liquid AI | The strongest published support for the project premise: a 350M student beats its 120B teacher on structured output. |
| `src-161-this-that-model-negation-failure.md` | FLock.io / Oxford, this-that-model | A leading model read a negated rule as its opposite at 0.88 confidence, and the newest checkpoint is worst on the axis we advise thresholding. |
| `src-179-mini-jev-preregistered.md` | r-ms, mini-Jev | The only **preregistered** study in the ledger. Measured directly on our core design decision. |
| `01-jev-omni-deep-read.md` | The ten load-bearing sources | Long form: Jev-Omni's source files, RCLD, Liquid configs, the calibrators, the option-name paper, Jebadiah's negative results. |
| `02-design-decisions.md` | 12 design decisions | Each with evidence, cost, and what would change our mind — plus what is explicitly **not** decided. |
| `00-scope.md` | Frozen scope | Topic, 7 sub-questions, cutoff, audience, and the method constraint. |

## The five findings that most changed the build

1. **A middle layer beats the last** for a linear read-out (AnyJev). `readout_layer` is now
   a constructor argument. Untested on LFM2.5 — it is the first experiment to run.
2. **Temperature varies by option count** (per-cardinality calibrator): a 3.3× spread, so
   one global T is 3.3× wrong at the ends. Implemented as buckets with geometric
   shrinkage, verified against a published calibrator to <1e-4.
3. **A 350M student can beat a 120B teacher** on structured output (distil labs). The
   strongest evidence the project premise is sound.
4. **Option names carry semantic weight** (arXiv 2609.26758) and **a symbol beats a name
   by +10 pp** (mini-jev, preregistered). Two independent lines on the same worry.
5. **Negation inverts silently** at high confidence (this-that-model), and the newest
   checkpoint can be the worst on calibration. Both argue for testing before shipping.

## Sources deliberately not given their own note

Roughly 140 ledger rows were read at the depth needed to extract their claims — a config
file's fields, a model card's benchmark table, a dataset card's schema. Their extracted
claims are in `../sources-ledger.md` with a Relevance score. The ones that *were* deep-read
are the nine files above; no source was deep-read and left undocumented.
