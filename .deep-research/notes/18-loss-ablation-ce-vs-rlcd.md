# Note 18 — CE vs RLCD: the loss-objective ablation

**Launched:** 2026-09-26
**Status:** running at time of writing; results appended below when they land.

## Why this experiment exists

kyr0, who built the strongest open system on the largest cross-model benchmark
(76.46% against Jev's 88.08% over 22,001 decisions on `typed-decision-bench`), wrote:

> **"btw. just one bit on that. RLCD is overrated guys! Use CE as the primary loss with
> an eye on Brier+NLL; it's cheaper and more effective!"**

That is the only direct challenge found anywhere in the corpus to this project's
conceptual prior, and Key Takeaway 10 states the prior in the same terms ("cross-entropy
is a strictly proper scoring rule — you already have RLCD's main benefit"). It was
flagged in Contradiction 8 as "the one free experiment in the corpus."

Five turns later it is the obvious thing to run, so it is being run.

## The complication, stated up front

**This project was already training with cross-entropy.** `loss = -(g * logp).sum(-1)`
on a soft target distribution `g` is soft-target cross-entropy, and it *is* a strictly
proper scoring rule on the declared option set. So the naive reading of kyr0's claim —
"use CE instead of RLCD" — is already satisfied and would test nothing.

What is actually on the table is a sharper three-way question, and the arms are chosen
to separate the two things that are genuinely different:

| arm | objective | what it is |
|---|---|---|
| `soft_ce` | `-(g * logp).sum(-1)` | **the control.** soft-target CE, strictly proper on the option distribution. This is what the project has always done. |
| `hard_ce` | `-logp[argmax(g)]` | kyr0's *literal* reading: plain CE on the argmax label. Proper but **not strictly proper** — the model is indifferent to the non-argmax mass. |
| `ce_brier` | `-(g * logp).sum(-1) + w * ((p - g)**2).sum(-1)` | "CE with an eye on Brier" taken literally: CE plus the Brier score as a second term. |

If `hard_ce` matches or beats `soft_ce` on coverage and ECE, the strictly-proper
argument is worth less than the report claims. If `ce_brier` beats both, kyr0's
specific suggestion is right and the extra term is free. If both match `soft_ce`,
the project's recipe is confirmed and kyr0's claim does not replicate here.

**The strongest prediction, stated before seeing results:** `soft_ce` and `ce_brier`
should tie or near-tie on accuracy (both see the full target) and `ce_brier` should be
*worse or equal* on ECE, because adding a squared-error term to a well-specified
cross-entropy mostly adds gradient noise near the optimum. `hard_ce` should lose
calibration specifically, because discarding the non-argmax mass is precisely the
information a calibration term needs. If `hard_ce` ties on ECE, the strictly-proper
framing in Part 5.1 of the guide is overstated and should be rewritten.

## Controls held fixed

Identical across all three arms, so nothing but the objective varies:

- `LiquidAI/LFM2.5-350M` @ `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`
- `customer_service`, `readout_layer=-1`, LoRA r=16 all-linear, 3 epochs
- **seed 17**, batch 8, `--eval-frac 0.5`, lr 1e-4, head-lr 1e-3
- temperature fitted on one half of eval, reported on the other

**`soft_ce` is being re-run with the byte-identical command rather than reusing the
historical baseline.** The saved `runs/cov-full/customer_service.json` records
`n_eval: 300` but not the seed or the split fraction, and a loss ablation whose arms
might not share a split is not an ablation. The historical arm-B figures (acc 0.5633,
Brier 0.1098, ECE 0.0646, T 1.050, cov@5% 0.0967) are quoted in the report and the
guide; if the re-run reproduces them to four decimals, the two are the same run and
that is worth stating as an extra reproducibility check.

## Execution notes

The host has been at load 85–92 for many hours. Diagnosis: four `nvidia-smi` processes
at 83% CPU and one `nvidia-settings` at 92%, all in a spin loop against a **wedged GPU** —
`nvidia-smi` itself could not complete in 300 s and had to be interrupted. Those D-state
tasks inflate the load average without competing for CPU, which is why the machine is
usable at all.

Mitigation for this run, so the three arms do not contend with each other or with the
spin loops:

- `taskset` pinning to **disjoint** core sets: `soft_ce` on 0–5, `ce_brier` on 20–31,
  `hard_ce` on 8–19
- `OMP_NUM_THREADS`/`MKL_NUM_THREADS` capped per arm
- `nice -n 10` so the spin loops win any contention

This is why the experiment is running at all when it was declared blocked for many
turns. **The load average was never the actual constraint — D-state tasks do not
consume CPU.** That is a correction to this project's own earlier assessment, which
repeated "blocked by host load" without diagnosing which state the load was in.

## Limitations, stated before results

- **Single seed (17).** A loss ablation that moves ECE by less than a few points is not
  distinguishable from noise on one seed, and the historical arm-B ECE is already
  single-seed. If the three arms land within ~0.01 ECE of each other, the honest
  statement is "no detectable difference at one seed", not a winner.
- **A LoRA arm only.** kyr0's claim is about small decision models generally; this tests
  it at 350M with LoRA r=16. It does not test the frozen arm, and it does not test any
  RLCD-trained model, because no public RLCD training method exists to compare against.
- **`brier_w = 1.0` is arbitrary.** The three terms are not on comparable scales — CE
  is a mean negative log-probability, the Brier term here is a summed squared error
  over up to 32 options. A weight of 1.0 may effectively ignore the Brier term. If
  `ce_brier` ties `soft_ce` exactly, that is the first thing to suspect, and the weight
  should be swept before concluding anything.
- **This tests kyr0's *suggestion*, not his *system*.** He trains a 27B quantised
  backbone with a post-hoc calibration file; this trains a 350M backbone with an
  in-loop objective. Agreeing or disagreeing is evidence about the objective, not about
  his model.
