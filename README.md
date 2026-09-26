# Jeb-Omni-Nano

**A small, cheap, calibrated typed-decision model built on Liquid AI LFM2.5.**

Give it a situation, a question and a list of options. It returns a calibrated
probability for every option. It never writes a sentence — there is no decoding
loop, no parser, and no refusal path.

```
state:      "I was charged twice for order 4411. Please refund one charge."
question:   "Which team should handle this?"
options:    1. billing   2. shipping   3. technical support
answer:     billing   {billing: 0.91, shipping: 0.06, technical: 0.03}
```

---

## Why this exists

`akhilaaa3/Jev-Omni` is a 12B-parameter decision classifier on Gemma 4 12B.
It is good — 87.57% on DecisionBench Medium, ECE 0.0400. It also needs a CUDA GPU
and about 24 GB of weights, and its own card reports 83 ms per text decision on
an H200.

This is the same *interface* on a 350M-parameter backbone that runs on a
Raspberry Pi.

|                      | Jev-Omni          | Jeb-Omni-Nano        |
|----------------------|-------------------|----------------------|
| Backbone             | Gemma 4 12B       | LiquidAI/LFM2.5-350M |
| Parameters           | 11,959,730,224    | 354,483,968          |
| Decision head        | 983,456 (fp32)    | 262,400 (fp32)       |
| Head as % of network | 0.008%            | **0.0740%**          |
| Trainable params     | 2,099,183,872     | 6,004,744            |
| Hardware             | CUDA GPU required | CPU, Apple Silicon, Pi |
| Runs on              | a datacenter      | a laptop, a phone, a Pi |

The head is the whole trick. Everything expensive about a decision model is the
pretrained backbone and the calibration you fit afterwards.

---

## Start here

**[`GUIDE.txt`](GUIDE.txt)** — 13 parts, written for someone who has never
trained a model. It reads the Jev-Omni source code, explains every term, and
gives the build, the evaluation protocol, and the failure modes.

The single most important section is **Part 11.1**, on a published failure where
renaming two options from `0`/`1` to `no`/`yes` moved AUC from .94 to .23. Run
that test before you ship a threshold policy.

---

## The design

Five decisions, each with a reason:

1. **Backbone** — `LiquidAI/LFM2.5-350M` (instruction-tuned, not `-Base`).
   16 layers: 10 short-convolution + 6 grouped-query attention. 28T training
   tokens. Liquid recommends it for structured extraction and tool use.
2. **Readout** — the last position of the final layer's hidden state.
3. **Head** — `Linear(hidden_size, 256)` scoring **positional option slots**,
   masked past the real option count, softmaxed. 256 not vocabulary logits, so
   one checkpoint answers any question with 2–256 options.
4. **Objective** — cross-entropy at that one position. No target tokens, nothing
   generated. This is a *strictly proper scoring rule*, which is the
   theoretical reason the probabilities are usable as thresholds.
5. **Calibration** — one fitted temperature per question type, on a held-out
   split, fitted against the soft target you trained on rather than the argmax.

Ported verbatim from Jev-Omni's published `_Head256`; our `tests/test_smoke.py`
asserts bit-exact equality with that reference implementation.

---

## Usage

```python
from jeb_nano import Question
from jeb_nano.model import JebNanoModel

model = JebNanoModel(
    "LiquidAI/LFM2.5-350M",
    revision="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",   # pin this
    device="cpu",
)

answer = model.decide(
    {"subject": "Charged twice", "body": "Two charges on my card."},
    Question(key="team", instructions="Which team should handle this?",
             options=("billing", "shipping", "technical")),
)

answer.prediction      # "billing"
answer.probabilities   # {"billing": 0.91, ...}  <- the calibrated output
answer.confidence      # (p_max - 1/K) / (1 - 1/K), TypeSafe's formula
```

Route on `probabilities`, not on `confidence`:

```python
p = answer.probabilities[answer.prediction]
if p >= 0.90:   act_automatically()
elif p >= 0.60: queue_for_review()      # sample these; that is where errors live
else:           escalate()
```

Thresholds are yours to pick from a measured accuracy-at-coverage curve on your
own data. Report coverage at your error budget.

---

## Training

```bash
python -m jeb_nano.train \
    --backbone LiquidAI/LFM2.5-350M \
    --revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f \
    --data train.jsonl --calibration calib.jsonl \
    --out runs/jeb-nano \
    --epochs 2 --batch-size 8 --lr 1e-4 --lora-r 16 --lora-alpha 32
```

Data is one JSON object per line, with a `label` index (or a soft
`{"index": 2, "soft": [...]}`) per question. See `GUIDE.txt` Part 8.

Start with `LocalLLaMA/typed-decisions` (Apache-2.0, ships gold probability
distributions) and hold out `akhilaaa3/decision-bench` for evaluation.

`score` questions train toward an **ordinal kernel** by default — 20% of the mass
on each adjacent level. Jebadiah measured this: rubric Decision Score went from
-21.4 to +11.9 and calibration error from 0.39 to 0.045, with accuracy unchanged.
The model stopped being confidently wrong about what it cannot judge.

---

## Tests

```bash
python tests/test_smoke.py         # 57 checks, no weights, no network
python tests/test_integration.py   # 16 checks, real pinned LFM2.5-350M weights
```

All 73 pass. Verified, not asserted:

- head is exactly `hidden*256 + 256` = 262,400 params = 0.0740% of the
  354,483,968-param backbone
- head output is **bit-identical** to the Jev-Omni reference implementation
- `fit_temperature` recovers a planted `T=2.5` as `2.5000`
- confidence formulas reproduce Kev's published worked examples
  (`{0.47,0.28,0.25}` → 0.21; `{0.00,0.56,0.44}` → score 1.44, confidence 0.34)
- inference is deterministic across repeated calls
- a real training run closes the loop: 80 examples, 40 steps, 127 s on CPU,
  loss 1.80 → 0.31, 85.0% accuracy, Brier 0.0775, ECE 0.0758

> The 85% figure is a **plumbing demonstration** on 12 hand-written sentences,
> not an accuracy claim. It proves the loop closes. Real numbers come from
> Part 10 of the guide.

---

## Repository layout

```
jeb_nano/head.py     DecisionHead, TemperatureFit, fit_temperature,
                     probabilities, expected_calibration_error, brier_score
jeb_nano/prompt.py   Question, DecisionRequest, render_request,
                     render_single_question, prompt_sha256, permute
jeb_nano/model.py    JebNanoModel: backbone loading, hidden-state hook, decide()
jeb_nano/train.py    LoRA + head trainer, ordinal/soft targets, temperature fitting
tests/               smoke (no weights) and integration (real weights)
GUIDE.txt            the full guide — read this first
LICENSE              MIT for the code; weights inherit the LFM Open License v1.0
```

---

## The read-out layer sweep — a negative result

The strongest external claim behind the design (Nokia's AnyJev: a *middle* layer is a
better feature space for a linear head than the last) is from a single unreplicated
source. We tested it on real data with real weights:

| layer (from end) | train loss | accuracy | Brier | ECE | fitted T |
|---|---|---|---|---|---|
| L0 (-16) | 1.2877 | 0.4000 | 0.1360 | 0.2258 | 0.350 |
| L4 (-12) | 1.2875 | 0.4000 | 0.1360 | 0.2256 | 0.350 |
| L8 (-8) | 1.2760 | 0.4000 | 0.1358 | 0.2268 | 0.600 |
| L12 (-4) | 1.2665 | 0.4000 | 0.1346 | 0.2203 | 0.750 |
| **L15 (-1)** | **1.2508** | **0.4889** | **0.1312** | **0.1360** | **0.900** |

**The last layer wins on all four indicators.** Frozen backbone, fresh head per layer,
soft gold distributions, temperature fitted and reported on different halves.

`readout_layer=-1` stays the default. n=45, so read the accuracy delta with care; the
weight comes from four indicators agreeing plus a monotonic temperature trend.
Full write-up: `.deep-research/notes/04-layer-sweep-results.md`.
Reproduce: `python experiments/layer_sweep.py`

## What is verified, and what is not

**Verified here:** head shape and dtype, bit-exact parity with the Jev-Omni
reference head, temperature fitting, ECE and Brier, the confidence formulas, the
LFM2.5-350M config, the parameter counts, determinism, and that the training loop
runs and learns.

**Not claimed:** any accuracy number for a trained Jeb-Omni-Nano. No trained
checkpoint is published yet. The numbers above are from a tiny smoke training
run and are labelled as such.

**Not attempted:** multimodal. The `model.py` media path and the video/audio
preprocessing conventions are implemented and match Jev-Omni's, but no
multimodal checkpoint has been trained or evaluated. `GUIDE.txt` Part 9 is
explicit about why that is a later project, not a first one.

---

## Licence

**Code: MIT.** See [`LICENSE`](LICENSE).

**Weights: not MIT.** Any model derived from `LiquidAI/LFM2.5-*` remains under
the **LFM Open License v1.0** — based on Apache 2.0, but **not** OSI-approved
and **not** MIT. It restricts free commercial use to legal entities with under
**USD 10,000,000** annual revenue, and exempts qualified non-profits for
non-commercial or research use. There is no copyleft, so you may keep fine-tuned
weights proprietary, but you must ship the licence, retain attribution, and mark
modified files.

This is not a formality: the LFM licence terminates automatically on breach, and
a public repo that says "MIT" above a set of LFM-derived weights is simply wrong.

---

## Independence

Jeb-Omni-Nano is an independent project. It is **not** affiliated with, endorsed
by, sponsored by, or derived from TypeSafe AI or its Jev model. **No Jev output
was used in training.** The design lineage is the *pattern* — a pretrained causal
backbone with a small trained readout that returns probabilities instead of
generating text — which is described in the open literature and implemented in
many independent public projects.
