# Note 18 — CE vs RLCD: the loss-objective ablation

**Launched:** 2026-09-26
**Status: STOPPED INCONCLUSIVE.** All three arms were stopped during epoch 1 of 3.
**No result was produced and none is claimed anywhere in this project.** The prediction
below was recorded before the run and is neither confirmed nor refuted.

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

## Validity check run before the results: are the targets even soft?

A loss ablation is worthless if the two arms compute the same function. `gold_tensor`
builds `g[i, :len(p)] = tensor(p)`, so the whole question is whether the dataset's `p`
is a one-hot label or a distribution. **Checked directly against the dataset:**

    action__probabilities:   {"answer_directly": 0.743333, "close_no_action": 0.03,
                              "escalate_to_human": 0.2, "execute_refund": 0.003333,
                              "request_information": 0.023333}
    needs_human__probabilities: {"false": 0.35, "true": 0.65}
    urgency__probabilities:      {"0": 0.003333, "1": 0.28, "2": 0.316667, "3": 0.4}
    churn_risk__probabilities:   {"0": 0.003333, "1": 0.006667, "2": 0.05, "3": 0.94}

**The targets are genuinely soft**, so `soft_ce` and `hard_ce` are different functions
and the ablation is not vacuous. Had these been one-hot, the two arms would be
algebraically identical and the whole run would have measured nothing.

This also sharpens the substantive argument, and it is the strongest one available for
`soft_ce`:

> **The dataset ships soft probability labels. Training with hard CE discards
> information the data provides.**

`churn_risk` is 0.94 on one option and 0.05 on another; `action` is 0.74 / 0.20 / 0.03.
A hard-CE arm throws that structure away and optimises only the argmax. The strictly-proper
scoring-rule argument in guide Part 5.1 is therefore not just theoretical here -- on
*this* dataset, soft targets are what is actually available, and the question is whether
using them helps, not whether they are permissible.

It also makes kyr0's claim narrower than it first read. "Use CE" against a dataset that
already provides soft labels is close to a restatement of the obvious. The interesting
version of his claim is the `ce_brier` arm.

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

## How to read progress on this host (a measurement trap)

`ps -o etimes` reports a process age that does **not** match the real elapsed wall time
in this sandbox. The same PID read 193 s, then 278 s, then 324 s across checks that were
minutes apart in real time, and a control (`etimes` sampled either side of a `sleep 45`)
confirmed the clock itself was exact. So `etimes` is not the clock; the sandbox's process
age accounting is.

**The reliable progress signal is the log mtime against `date`:**

    stat -c '%y' /tmp/loss_soft.log ; date '+%H:%M:%S'

The three arms wrote their last line at 16:26:52-53, which is when weight loading
finished and epoch 1 began. A stalled run would show an mtime far in the past; a
running one shows an mtime that advances. CPU time (`ps -o times`) accumulating faster
than wall time is the secondary confirmation -- these run at 250-390% because each arm
is multithreaded across its `taskset` set.

Recorded because a future reader checking on this run will hit the same confusion, and
because "the experiment looks stuck" is the wrong conclusion to draw from `etimes` here.

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

---

## Outcome: stopped inconclusive, 2026-09-27

All three arms were terminated during epoch 1 of 3, at roughly 20 minutes into a
~70-minute run. `runs/loss-ablation/` was empty and has been removed. The logs show
data loaded correctly and weights pinned, and nothing beyond that.

**So the question stands exactly where it stood before:**

> Does plain (hard) cross-entropy, or CE-plus-Brier, beat the soft-target cross-entropy
> this project already trains with? kyr0 says RLCD is overrated and CE wins.

**Nobody knows, including this project.** The recorded prediction — that `hard_ce` would
lose calibration specifically — was never tested. It is a prediction, not a result, and
nothing downstream should cite it as either.

### What the aborted run *did* establish

These hold regardless of the training outcome and are the only things the run produced:

1. **The ablation is not vacuous.** Verified directly against the dataset: the gold
   targets are genuine probability vectors (`answer_directly: 0.743,
   escalate_to_human: 0.2, close_no_action: 0.03`), not one-hot. Had they been one-hot,
   `soft_ce` and `hard_ce` would be algebraically identical and the experiment would
   have been worthless before it started. This was checked *before* the results, which is
   the only reason the abort cost an hour of CPU and no invalid conclusion.
2. **The machine was never the blocker.** The four `nvidia-smi` processes are in D-state
   against a wedged GPU — `nvidia-smi` itself could not complete in 300 s. D-state tasks
   inflate the load average without consuming schedulable CPU. Three arms ran
   concurrently at 250-390% CPU on a box reading load 87. Every previous turn in this
   project that said "blocked by host load" was wrong about *why*, and right only by
   accident.
3. **The dataset ships soft labels, which is the strongest structural argument for the
   current recipe.** A hard-CE arm would discard information the data provides. This is
   a property of the data, not a measured training outcome, and it is stated as such.

### What this costs the project

The report's Key Takeaway 26 and guide section 5.1a both say "test CE against RLCD."
That remains a **recommendation to the reader**, and it is now the only open empirical
question in an otherwise fully documented corpus. It is not evidence for anything, and
neither document now implies that it is.

### To run it later

    nohup env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 taskset -c 8-19 nice -n 10 \
      /tmp/jebvenv/bin/python experiments/lora_vs_head_only.py \
      --workflow customer_service --max-cases 1000000 --eval-frac 0.5 \
      --epochs 3 --only=B --loss hard_ce --out runs/loss-ablation/hard_ce

`--loss` accepts `soft_ce` (control), `hard_ce`, and `ce_brier`. `soft_ce` must be
re-run with the identical command rather than reusing `runs/cov-full/customer_service.json`,
because that file records `n_eval` but not the seed or the split fraction. Before
concluding anything from `ce_brier`, **sweep `brier_w`** — at 1.0 the two terms are on
incomparable scales and the Brier term may be doing nothing at all.
