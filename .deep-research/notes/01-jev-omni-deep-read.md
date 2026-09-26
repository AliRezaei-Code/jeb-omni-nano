# Deep Notes — The Ten Sources That Decided the Design

Per-source notes for the sources that actually changed what we built. Sources read
but not decisive are summarised in `sources-ledger.md` with a Relevance score; this
file is the long form for the load-bearing ten.

---

## 1. `akhilaaa3/Jev-Omni` — `jev_omni.py`

**The single most important file in the entire project.** It is the whole runtime in
~150 lines, and it hands you the architecture for free.

Read this, verbatim, from the repo:

```python
class _Head256(torch.nn.Module):
    """Decision head: normalise the last hidden state, one logit per option slot."""
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, hidden))
        self.register_buffer("sd", torch.ones(1, hidden))
        self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

    def forward(self, features, counts):
        z = self.linear((features.float() - self.mu) / self.sd)
        return z.masked_fill(torch.arange(256, device=z.device)[None] >= counts[:, None], -1e30)
```

Four load-bearing details, each of which we had to understand to port it:

1. **The 256 outputs are positional slots, not vocabulary.** Slot *i* is "option
   *i+1*". This is why one checkpoint handles any question with 2–256 options and
   unseen option *words*. It is the reason the design scales to a 350M backbone at all:
   head cost is `hidden_size × 256`, nothing to do with vocabulary or task count.
2. **`mu`/`sd` are `register_buffer`, not `Parameter`.** Frozen standardisation
   statistics. They keep the head's input well-scaled regardless of what the backbone
   is doing, and they are saved and shipped.
3. **`dtype=torch.float32` on the head**, while the backbone runs bf16 autocast. A
   deliberate choice: bf16 has ~3 significant decimal digits, which is not enough for
   a probability you are going to threshold on.
4. **`masked_fill(..., -1e30)`** then the caller slices `[:len(options)]` and
   softmaxes. We kept the `-1e30` sentinel rather than switching to `-inf` because that
   is what the reference does and bit-exact parity with it is now a test.

The rest of the file, also load-bearing:

- the prompt template (numbered options, "Reply with only the number")
- `predict(state=, question=, options=, media=, modality=)` with 2–256 validation
- video: `sorted({int(round((total-1)*(k+.5)/count)) for k in range(count)})`, 16
  frames, each passed as a **separate image item**
- audio: `ffmpeg -v error -i IN -t 30 -ac 1 -ar 16000 out.wav`
- `_find_backbone` probing four attribute paths, because the multimodal wrapper nests
  the text stack differently from a text-only model
- `load_jev_omni` raises unless CUDA

**Contradiction to flag:** the model card says "30,000-question fine-tuning run";
`decision_config.json` says `size: 24000`. Both are the author's. The config is the
more specific artefact, so we use 24,000 and note the discrepancy rather than picking
whichever is convenient.

---

## 2. `akhilaaa3/Jev-Omni` — `decision_config.json`

The training recipe, complete. This is the second file you must read.

| Field | Value | Why it matters |
|---|---|---|
| `size` | 24000 | The fine-tune is *small*. This is the good news. |
| `rank` / `alpha` | 512 / 512 | **17.5% of a 12B network is trainable.** Far above the rank-16 Kev and Jebadiah use. |
| `lr` / `head_lr` | 1e-05 / 1e-05 | Low. The backbone is barely touched. |
| `microbatch` / `effective_batch` | 8 / 32 | |
| `epochs` | **1** | Matches Jebadiah's "two epochs were worse than one" and the distil-labs epoch curve. |
| `warmup_ratio` / `steps` | 0.1 / 750 | 24,000 / 32 = 750 exactly. The arithmetic checks out. |
| `schedule` | "linear warmup then linear decay to 10% of peak" | We copied this schedule. |
| `seed` | 3407 | |
| `adapters_merged` | `["trained v1 rank128", "lora-v1merged-…"]` | **Two stacked LoRA stages.** Kev-0.8B round 13 shows stacking can erode the first stage. |
| `initialization` | "FP32 merged trained v1 + trained head + fresh LoRA" | |

The takeaway that reframed the project: **the expensive part is the backbone, not the
fine-tune.** 24K examples, one epoch, lr 1e-5. Our 350M equivalent needs
6,004,744 trainable params (1.7%), not 2,099,183,872 (17.5%).

---

## 3. `akhilaaa3/Jev-Omni` — `verification.json`

Four cases with reference distributions, `worst_abs_diff 0.01936584711074829`.

The two that matter are the **unanswerable** ones:

    die roll:  {1: 0.234, 2: 0.093, 3: 0.044, 4: 0.089, 5: 0.081, 6: 0.459}
    marble urn:{Blue: 0.413, Yellow: 0.126, Red: 0.079, Green: 0.382}

These are near-uniform, and correctly so — the state does not determine the answer.
**A calibrated model on an unanswerable question produces a flat distribution.** That
single fact is the entire product, and it is visible in the author's own shipped
reference data. It is also the thing to test first on any model you build: feed it a
question whose answer is unknowable and check that it does not sound confident.

---

## 4. `notnotsamuel/LFM2.5-350M-RLCD` — this is "RCLD"

The repo the user was asking about. Its README says, in its own words:

> "**Inference only: no training or fine-tuning, and no reproduction of TypeSafe.ai's
> proprietary Jev training method.**"

So it shares the *idea* of constrained decoding with Jev and shares nothing else.
Its contribution is the cache-branching technique: prefill the state once, deep-copy
the hybrid cache, branch, score every JSON-Schema candidate in one batch, argmax per
field, assemble JSON in Python.

**The cache subtlety, from `rlcd/engine.py`:**

```python
def fork_cache(cache, count):
    """Copy all state and reorder batch rows, including convolution history.
    Generic batch_repeat_interleave is not implemented for LFM2 convolution
    layers in the pinned Transformers release. reorder_cache handles both.
    index_select allocates independent storage; never broadcast mutable views.
    """
```

LFM2 has **two** caches: attention KV *and* `LinearAttentionLayer.conv_states`
(`conv_L_cache: 3`). Copying only the first produces a model that is subtly, silently
wrong. Anyone branching LFM2 must copy both.

**What it does not do, in its own words:**

> "Valid JSON does not mean correct answers: neither method produced a fully correct
> 28-field object."

> "On the 12-case diagnostic suite the constrained method was *faster* but *less
> accurate*: 77.8% vs 80.6% field accuracy."

> "With 255 candidate options on the Mac, constrained inference was 3.23× SLOWER."

> "They are not calibrated confidence. No calibration evaluation was performed and no
> normalized probability API is exposed."

**The lesson:** constraining the decoder buys you *structure*, not accuracy and not
calibration, and it does nothing to shrink the model. We want the head approach *and* a
smaller backbone. The 28-field speedups (8.46× to 62.91×) are real and impressive, and
they are also orthogonal to what we set out to do.

---

## 5. `LiquidAI/LFM2.5-350M` — `config.json` and card

The backbone. Measured from the shipped file, not quoted from prose:

    hidden_size 1024 · 16 layers (10 conv + 6 full_attention) · conv_L_cache 3
    vocab 65536 · max_position_embeddings 128000 · rope_theta 1e6 · bf16
    actual parameter count: 354,483,968

Three details that only appear in the file:

- **`conv_L_cache: 3`** — the convolution state you must copy when branching.
- **10 of 16 layers are convolutions.** This is why a LoRA targeting only
  `q_proj/k_proj/v_proj/o_proj` leaves most of the network frozen. Use
  `target_modules="all-linear"`.
- **`max_position_embeddings: 128000` contradicts the card's "Context length: 32,768".**
  Both are Liquid's. Plan for 32,768 and flag the discrepancy.

And the recommendation that made the choice obvious:

> "We recommend using it for **data extraction, structured outputs, and tool use**. It
> is not recommended for knowledge-intensive tasks and programming."

Compare Kev-0.8B's measured gaps: MMLU 0.42 vs Jev 0.90, PAWS 0.55 vs 0.79 — knowledge
and paraphrase. Those are precisely the two categories Liquid steers you away from. The
vendor's stated strength and the demonstrated small-model weakness line up, and the
overlap is the task we want.

---

## 6. `distil labs` × Liquid — the 350M-beats-120B result

The single strongest published support for the project premise, and the reason the
backbone choice is defensible.

| Task | Teacher GPT-oss-120B | LFM2.5-350M base | **tuned** |
|---|---:|---:|---:|
| Shell (Gorilla) | 97.03% | 61.4% | **98.0%** |
| Smart home | 92.11% | 63.2% | **96.7%** |
| Banking voice | 96.95% | 34.5% | **95.9%** |

**The student exceeds the teacher on two of three.** And the base was far below, so this
is not a strong base coasting — 34.5% on the hardest task.

The pipeline, and the step people skip:

    seeds (20-100) → teacher generates → **validate and filter** → fine-tune student

Epoch curve: `61.4 → 98.0 → 97.0 → 98.0 → 98.0`. **Epoch 1 is almost the whole
gain** — independent confirmation of the one-to-two-epoch finding from Kev and
Jebadiah, on a completely different codebase.

Two transferable numbers: 5,000 synthetic examples for one task; and at 63% per-call
accuracy a five-turn conversation succeeds at `0.63^5 ≈ 10%`, which is why multi-turn
tasks look hopeless on a base model and are not.

---

## 7. `nokia-applied-research/AnyJev` — two findings that changed the code

Nokia (Sunnyvale) with Tencent Hunyuan. Apache-2.0. Turns *any* LLM into a Jev-style
decision model with no fine-tuning.

**(a) Read a middle layer, not the last one.**

> "Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy slightly *higher* and
> calibration better, and was faster: a middle block is a better feature space for a
> linear head than the last one, where the remaining blocks are busy turning the answer
> into tokens."

> "Depth is usually a gain, not a trade."

We were reading the last layer because that is what Jev-Omni does. Now `readout_layer=`
is a constructor argument (default `-1`, preserving parity) and the integration test
sweeps `-1 / -8 / -12` on real weights. **The finding that a middle layer wins is
**since been measured, and it did not replicate** — the last layer won on accuracy,
   Brier, ECE and training loss. See [`04-layer-sweep-results.md`](04-layer-sweep-results.md).**

**(b) A closed-form head beats LoRA when you have 100–300 labels.**

| | raw logits | L0 (zero labels) | L1 (+temperature) |
|---|---|---|---|
| Labels needed | none | **none** | 100–500 |
| Order-flip rate | 0.230 | **0.073** | 0.077 |
| Accuracy | 0.747 | 0.803 | 0.807 |
| ECE | 0.240 | 0.184 | **0.095** |
| **Auto-decidable @ ≤5% error** | **7.7%** | 46.3% | **52.0%** |

Accuracy moves six points. **The share of traffic you can safely automate moves 7.7% →
52.0%.** That is the argument for this entire class of model in one table: with raw
logits a "0.9" is not trustworthy enough to act on, so everything goes to a human; once
the probability means something, you can set a threshold.

Practical consequence: **do a closed-form fit before reaching for LoRA.** It takes
minutes, needs almost no data, and it tells you whether the backbone's features are
usable at all. If a linear head on frozen features cannot separate your classes, LoRA
will not rescue it.

Also: **order-flip has a zero-label fix** (0.230 → 0.073, readout change only). Third
independent confirmation that read-out geometry matters.

---

## 8. The published calibrators — what they actually do

Three checked-in calibration artefacts, each of which changed our implementation.

**`Jev-Style-0.8B-Decision-v3` (`readout_config.json` + `jev_style_decision.py`)**

- 20 group temperatures + a global `0.8800546821789332`; clamp `[0.3, 5.0]`;
  `shrinkage_k 100.0`; buckets `["2","3-5","6-10","11-20","21+"]`; `n_rows 15655`
- fit quality: `nll 0.37752 → 0.36671`, **`ece 0.032893 → 0.011376`** (65% reduction)
- **verified arithmetically: `weight = n/(n+100)` exactly for all 20 groups**
- **verified: the blend is geometric, not arithmetic.** Two groups reproduce to ~5
  decimal places under `exp(w·ln T_raw + (1-w)·ln T_global)`; on the n=15 group the
  arithmetic blend is off by 8.5% (1.09567 vs the published 1.01016)
- **the readout uses zero new parameters** — `h·(w_yes − w_no)` from tied embeddings
- **T can be below 1.0.** This model was *under*confident, so calibration sharpened it.
  Do not assume `T >= 1`.

**`heman10x/rlcd-modernbert-151m` (`calibrator.json`)** — the per-cardinality result

    k=2 → 5.0069   k=9  → 1.6668
    k=3 → 5.0069   k=11 → 3.3919
    k=4 → 4.0314   k=17 → 1.7200
    k=5 → 3.0560   k=25 → 1.5144
    k=6 → 2.3898   global → 2.8039
    k=7 → 2.3898

**A 3.3× spread driven by option count alone.** A single global T is 3.3× too wrong at
the ends of the range. (k=11 is non-monotone — per-k fits are themselves noisy at these
sample sizes, which is exactly why the shrinkage exists.)

**`Jev-Style-Qwen3.5-2B-Decision-v2-GGUF`** — the procedure, stated

- `"objective": "sample_mean_soft_cross_entropy"` — against the SOFT target, not one-hot
- `bounds [0.05, 20]`, `calibration_n 3100`
- **a different temperature per quantisation format**: BF16 `1.0409`, Q4_K_M `1.0123`
- argmax agreement `0.996` (BF16) vs **`0.914`** (Q4_K_M), with **relaxed gates** for
  the lossy format
- `"temperature_folded": true, "folded_tensor": "output_norm.weight"` — the score is
  `h·(w_yes−w_no)` and `output_norm` is an RMSNorm gain on `h`, so dividing the gain by
  T divides the score by T. Zero inference cost, one scalar edit, runtime T = 1.0

**And the biggest single win in that project's history was a bug fix.** From
`Verdict-open-jev`:

> "The engine previously failed to load `calibrator.json` during standalone
> instantiation, **running at uncalibrated temperature 1.0**. A scope check also limited
> calibration exclusively to 5-candidate queries, leaving other cardinalities unscaled."

Fixing it moved hard-tier ECE **0.298 → 0.118 (−60.4%)**. Check that yours loads.

---

## 9. arXiv 2609.26758 — the option-name failure

The newest and most consequential result, and it is about *this exact architecture*.

The experiment holds the question, the state, the rubric wording and the *set* of option
names fixed, and changes only which **name** is bound to which **rubric**. Renaming two
options `0`/`1` → `no`/`yes`:

    70.4 more answers changed per hundred   (95% CI [67.6, 73.1])
    AUC  .94 → .23
    effect ≥ 7.4x the neutral-name control, across all 4 predicates
    grows with option count
    a mean-pooling family flips 4.1x LESS often
    hosted Jev: AUC .8146 → .5806, 24x its test-retest floor
    random character-string names → all families return to neutral, no accuracy loss
    TYPE-ERROR RATE 0% THROUGHOUT

AUC below 0.5 is a systematic ranking reversal, not uncertainty. And the type-error
rate staying at 0% is the part that should worry you most: **a decision model can be
perfectly schema-conformant and completely wrong.**

Three actionable consequences, all in the guide:

1. Run the test. It takes an hour and it is the highest-value hour in the project.
2. If affected, use neutral or random-ish option identifiers with the human-readable
   name in the rubric. A prompt change, not a retrain.
3. It is a second argument for slot logits over pointer scoring, and a reason to keep
   option counts at or below 20.

---

## 10. `getainode/jebadiah` — the negative-results log

The most valuable single document in the field, because it publishes what did **not**
work.

**Failed:** a 14,714-question synthetic pool from their own teachers made the 9B worse;
re-targeting it with a better teacher did nothing; public human yes/no data did not
lift their human yes/no eval sets; two epochs were worse than one; stacking a second
LoRA delta on top of a first eroded it.

**Worked:** human-rated score data + an ordinal target took HelpSteer2 Decision Score
from **−21.4 to +11.9** and ECE from **0.39 to 0.045**, with rubric accuracy unchanged.
Their summary is the sentence I quote most in the guide:

> "The model became honest about not knowing helpfulness rather than better at judging
> it."

And the base-checkpoint finding:

> the chat checkpoint with thinking off was worth **2.15 headline points** on the 4B —
> *"more than any data change we made."*

That is why our default is `LiquidAI/LFM2.5-350M` (instruction-tuned), not `-350M-Base`.

**The two temperature fits**, which changed our calibration code:

| Fit | Objective | choice | noul | score | Effect |
|---|---|---|---|---|---|
| `hard` | NLL vs **argmax** | 0.68 | 0.96 | 0.83 | sharpens |
| `train` | NLL vs the **soft/ordinal target** | 1.12 | 1.33 | 1.20 | softens |

They ship `train`, because *"the ordinal target is what makes rubric probabilities
honest, and the `hard` fit is exactly the sharpening that made v0's HelpSteer2
calibration worse."* Our `fit_temperature` takes targets, not argmaxes, for this reason.
