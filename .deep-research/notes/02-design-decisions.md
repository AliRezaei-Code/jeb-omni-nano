# Design Decision Record

Every choice we made, the evidence for it, and what would change our mind. Ordered by
how much the evidence constrained us.

---

## D1 — Attach a small trained head; do not constrain the decoder

**Decision.** A trained readout on the last-position hidden state, softmaxed over
option slots.

**Evidence.**
- Jev-Omni ships its head in `jev_omni.py`; it is one `Linear(hidden, 256)`.
- RCLD, which *does* constrain the decoder, reports in its own words that "valid JSON
  does not mean correct answers", that the constrained method was *less accurate* on
  its diagnostic suite (77.8% vs 80.6%), that 255 options made it 3.23× *slower* on a
  Mac, and that it performs "no calibration evaluation".
- Six independent teams (Jev-Omni, Kev, Jebadiah, Laya, this-that, jeff) all ship a
  trained head.

**Convergence is the strongest evidence in the field.** Implementations that cannot see
each other agreeing on the same structure is worth more than any single paper.

**What would change our mind.** Evidence that a constrained decoder with calibration
matches a trained head on accuracy *and* latency at equal option counts. Nobody has
published that.

---

## D2 — Slot logits, not pointer scoring and not label-token logits

**Decision.** `Linear(hidden, 256)`; option *i* → slot *i*.

**Evidence for over label-token logits.** Laya scores **exactly 0.425** on Banking77
(77 labels) on *both* checkpoints, against Jev's 0.870, and diagnoses it as
architectural: options share a fixed `head_max_len` budget (192/256 tokens), so 77
labels get ~3–4 tokens each and become indistinguishable. "Keep choice questions under
~20 options." Slot logits have no such ceiling — Jev-Omni's head accepts 256.

**Evidence for over pointer scoring.** The option-name paper (arXiv 2609.26758) found
the failure is **4.1× weaker** in "a second model family that mean-pools over the full
option span", i.e. a pointer-style reader. Our choice happens to be the more robust
one on the newest available evidence.

**A preregistered study supports the symbolic-slot half of this.** `r-ms/mini-jev`
runs a preregistered experiment on a frozen Qwen3-4B and measures that giving the model
a **one-token identifier** to answer with beats writing the option's name by **+10.0 pp
[+8.3, +11.7]** on intent and +13.2 pp on domain. It also shows the choice between
letter-readout and grammar-constrained JSON is accuracy-neutral (Δ −0.22 pp, CI
[−1.44, +1.04], with every k from 2 to 16 covering zero) and 4x faster.

**Cost of the choice.** Slot logits need a prompt that numbers the options, and they
cannot score 500+ options without growing the head. Pointer scoring is richer; we accept
the trade for a 350M model where simplicity and robustness win.

**The caveat we must not skip.** That same study warns that its raw option shares are
"normalized candidate scores, **not calibrated probabilities**" and that you must not
read the percentage as P(correct). A trained head plus a fitted temperature is what
converts scores into probabilities -- which is exactly what D8 exists for.

**What would change our mind.** A published permutation-robustness result showing slot
logits are *equally* affected by the option-name failure at small scale. Then
neutral-identifier prompting becomes mandatory rather than optional.

---

## D3 — LFM2.5-350M, instruction-tuned

**Decision.** `LiquidAI/LFM2.5-350M`, not `-Base`; text-only for v1.

**Evidence.**
- Jebadiah measured the Base→chat-with-thinking-off swap at **+2.15 headline points**
  on the 4B, *"more than any data change we made"* across many experiments.
- Liquid's own card recommends the family for "data extraction, structured outputs, and
  tool use" and steers away from "knowledge-intensive tasks and programming" — which is
  exactly where Kev-0.8B measures its worst gaps (MMLU 0.42 vs 0.90, PAWS 0.55 vs 0.79).
- The LFM2 architecture was found by **hardware-in-the-loop search under edge latency
  and memory constraints**, i.e. optimised for the one-forward-pass-no-decode operation
  a decision model performs.
- It runs on a Raspberry Pi 5 at 30 tok/s decode in 300 MB.

**Size arithmetic.** hidden 1024 → head = `1024×256 + 256` = **262,400** params =
**0.0740%** of 354,483,968. Trainable: LoRA 5,996,544 + head 8,200 = **6,004,744** (1.7%),
against Jev-Omni's 2,099,183,872 (17.5%).

**What would change our mind.** If the readout-layer sweep (D6) shows most of the
backbone is unnecessary, a truncated LFM2.5-350M — or the 230M encoder — becomes
better. If the closed-form head (D7) plateaus, the 2.6B is the fallback.

---

## D4 — Cross-entropy at one position; no target tokens

**Decision.** `loss = - Σ_k target_k · log_softmax(logits)_k`, computed at the answer
position only. No decoding, no target masking.

**Why.** This is a **strictly proper scoring rule** applied to a distribution whose
support is exactly the declared option list. Laya calls training against strictly proper
scoring rules "RLCD" and that is the formal reason the probabilities are usable as
thresholds. We get the main benefit without a reinforcement-learning loop.

**Why not TRL's `SFTTrainer`.** It computes a causal-LM loss over target tokens. This
task has no target tokens. Writing the loop out keeps the loss, the masking and the
temperature handling visible, which is where this class of model actually goes wrong.

---

## D5 — Ordinal kernel for `score`, soft targets where available

**Decision.** `score` questions train toward an ordinal kernel: 20% of the mass on each
adjacent level. `_ordinal_target(2,5) = [0, 0.2, 0.6, 0.2, 0]`. Opt-in soft targets
where a real distribution exists.

**Evidence.** Jebadiah: human-rated score data + an ordinal target took HelpSteer2
Decision Score from **−21.4 to +11.9** and ECE from **0.39 to 0.045**, with rubric
accuracy unchanged. `typed-decisions`: carrying the soft gold into a learner cut **KL by
a third** and **score MAE by 15%** with accuracy barely moving.

**The caveat, which we take seriously.** Jebadiah's 14,714-question *synthetic* soft-
label pool made their 9B **worse**. Soft targets help when the distribution is good and
hurt when it is not. Prefer human or public labels; `tasksource/procedural-typed-decisions`
is the best answer here because its posteriors are *computed exactly*, not voted on.

---

## D6 — Sweep the readout layer (open, and the first experiment to run)

**Decision.** `readout_layer=-1` by default, for Jev-Omni parity — but the guide tells
you to sweep it, and the integration test proves the option works.

**Evidence.** AnyJev: *"Cutting Qwen2.5-7B from 28 blocks to 18 left accuracy slightly
higher and calibration better... a middle block is a better feature space for a linear
head than the last one, where the remaining blocks are busy turning the answer into
tokens."* "Depth is usually a gain, not a trade."

The intuition generalises: the last layers specialise for emitting the next token; a
linear readout wants a layer that still represents meaning. On a 16-layer LFM2.5-350M
the middle is around 10–12.

**Status.** Wired and tested. **The claim that a middle layer wins is untested on our
data.** Two bugs were found while wiring it, both only visible when actually run:
`Lfm2DecoderLayer` returns a plain `Tensor` (so the old `out[0]` indexed the *batch*
axis), and one of our own edits had dropped the `register_forward_hook` call.

---

## D7 — Do a closed-form head fit before LoRA

**Decision.** Recommended in the guide, not yet implemented in this repo.

**Evidence.** AnyJev L0/L1/L2: a closed-form head on 100–300 labels, no gradients,
weights untouched, moves auto-decidable-at-≤5%-error from **7.7% to 52.0%**. Kev and
Jebadiah both ship `--init_from` so you start from a released adapter rather than the
base. this-that-model's whole adaptation is a single scalar `θ(λ) = θ₀ + λΔ` with
`θ(0)` bit-exact the prior checkpoint.

**Why it matters here.** It is minutes of work on almost no data, and it answers the
question that matters before you spend GPU-hours: *are the backbone's frozen features
separable at all?* If a linear head on frozen features fails, LoRA will not rescue it.

**Gap.** Not implemented. The guide says so; the repo does not pretend otherwise.

---

## D8 — Per-bucket temperatures with geometric shrinkage

**Decision.** `T` looked up by option-count bucket first, then question type, then a
global value. Small groups shrink toward global in **log space** with `k = 100`.

**Evidence, and it is verified rather than quoted.**
- A published 151M calibrator shows a **3.3× spread by option count alone**
  (k=2 → 5.01 … k=25 → 1.51). One global T is 3.3× too wrong at the ends.
- A published 0.8B calibrator stores `weight = n/(n+100)` for all 20 groups; we
  verified that identity arithmetically.
- The blend is **geometric**: two of its groups reproduce to ~5 decimal places under
  `exp(w·ln T_raw + (1-w)·ln T_global)`, and on the n=15 group arithmetic is wrong by
  8.5% (1.09567 vs published 1.01016). Our `tests/test_smoke.py` checks both
  reproductions *and* the arithmetic rejection.
- Fit against the **soft target**, not the argmax (Jebadiah's `train` fit).
- Grid-search rather than L-BFGS: NLL in T is smooth but nearly flat near the optimum,
  and a 96-point grid is dependency-free and reproducible.

**One honest limitation.** In the smoke test, the `T[noul]` fit came out at the grid
floor (0.250) on 8 calibration rows. That is exactly the overfit the shrinkage exists to
catch — with 8 rows a group should be pulled hard to global, and on a real run the
bucket groups will be far larger. But it is a reminder that **a calibration split
smaller than a few hundred rows per bucket is not measuring the model, it is measuring
the fit.**

---

## D9 — Do not quantise the head; refit temperature per format

**Decision.** Head stays fp32. After any quantisation, re-measure ECE/Brier and refit.

**Evidence.** Three independent results:
- Jev-Omni's Q4_K_M card: **max absolute probability difference 0.210** vs fp32, and
  "Do not assume source-equivalent probabilities or calibration."
- A published calibrator ships **different temperatures per format** (BF16 1.0409,
  Q4_K_M 1.0123) with argmax agreement 0.996 vs **0.914** and *relaxed* gates.
- A model trained against a 4-bit NF4 base and served bf16 saw **ECE roughly double,
  0.036 → 0.069**.

Plus Nokia: for a linear head, **fp8 "buys single-question latency and costs
accuracy"** — they recommend against it.

The head is 262,400 fp32 params ≈ 1 MB. Quantising it buys nothing and risks the only
thing the product is for.

---

## D10 — Isolate questions: one forward pass each

**Decision.** `decide_many` runs one pass per question. No shared-prefix cache reuse
implemented.

**Why.** "Does the customer sound angry" silently changing "which queue gets the
ticket" is a nasty production bug, and TypeSafe documents that questions are evaluated
"in parallel **and in isolation** against the same state in one go."

**The cost, stated honestly.** A five-question request does five prefills. RCLD shows
how to avoid this with a shared KV cache, and we point at its `fork_cache` — including
the subtlety that you must copy LFM2's *convolution* state as well as the attention KV.

**On hybrid bases there is no choice anyway.** Kev documents that Qwen3.5/3.8 Gated
DeltaNet layers "are recurrent and ignore attention masks", so on those backbones each
question must be its own row. LFM2.5 is also hybrid.

**Status.** Implemented and correct; the optimisation is not. Flagged in the guide.

---

## D11 — MIT for code, LFM Open License v1.0 for weights

**Decision.** `LICENSE` says MIT and states, in the file itself, that weights derived
from LFM2.5 are under the LFM Open License v1.0 and **not** MIT.

**Evidence.** The licence text, read in full. §1 defines "Threshold" as annual revenue
of **$10,000,000 or more**; §5(a–c) caps free commercial use at that threshold and
exempts qualified non-profits for non-commercial/research use; §11 terminates
automatically on breach and requires deleting all copies.

**A correction we made to ourselves.** We initially wrote that the licence "resembles
Llama's MAU scheme". It does not. We checked: **there is no MAU threshold, no use-based
gating, and no field-of-use restriction.** The revenue cap is the whole of it. The
report now says so explicitly.

**Also checked.** `google/gemma-4-12b-it` is tagged `license: apache-2.0` on the Hub with
a `license_link` to Google's terms and **ships no LICENSE file**. Jev-Omni ships
`apache-2.0` "following Gemma 4". The SPDX tag and the linked terms are not obviously
the same instrument; anyone redistributing Gemma-4-derived weights should read the
linked terms.

---

## D12 — Ship the research report despite the skill's gitignore default

**Decision.** `.deep-research/research-report.md` and `sources-ledger.md` are committed;
only `notes/*` is ignored.

**Why.** `GUIDE.txt` and `README.md` both cite the report. The deep-research skill's
default is to gitignore the whole directory, but the user asked for a public repo, and
a public repo with a dangling reference to its own evidence base is broken. The commit
message records the one-line revert.

---

## What is explicitly NOT decided here

- **No trained checkpoint exists.** Every accuracy figure for the proposed model is
  hypothetical. The only measured numbers are the smoke-test plumbing (loss 1.80 → 0.31,
  85.0% on 12 hand-written sentences) and the integration test.
- **Multimodal is designed, not trained.** The media code path and preprocessing
  conventions match Jev-Omni's, but nothing has been run on real media. The
  LFM2.5-VL-450M recommendation is a prediction from published benchmarks.
- **The readout-layer finding is untested on our data.** Wired, not validated.
- **The source count is 160, not 200.** `websearch` failed for the entire session, which
  removed the whole discovery surface for forums, blogs, HN and news. We logged the
  shortfall rather than padding the count, and no row is duplicated to close the gap.
