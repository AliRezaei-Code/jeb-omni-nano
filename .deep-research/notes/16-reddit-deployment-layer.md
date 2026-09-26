# Note 16 — The Reddit and practitioner-deployment layer

**Date written:** 2026-09-26
**Why written:** two previous passes recorded Reddit as uncovered. `reddit.com/search.json`
returns 403 and `old.reddit.com` redirects to login. The fix was a public **Redlib**
instance — `safereddit.com` — a privacy front-end that serves Reddit search and full
comment threads with no auth. Fifteen ledger rows (212–226) exist because of it.

This is the layer that contains **outcomes** rather than artefacts. Everything before it
was model cards, config files and papers. This is what happened when people actually ran
the things.
## 1. The number the project's own docs are missing

The `typed-decisions` benchmark — 2,000 decisions across four workflows — is the
benchmark this project trains and evaluates on. **[`laya-typed-decisions` scores 0.766 on
it](https://huggingface.co/convaiinnovations/laya/raw/main/README.md), against 0.362 for
the base English checkpoint.**

| model | accuracy on `typed-decisions` | weights | licence |
|---|---:|---|---|
| base `laya` (English, zero-shot) | 0.362 | open | Apache-2.0 |
| `laya-typed-decisions` (fine-tuned) | **0.766** | open | Apache-2.0 |
| **this project, frozen backbone + head only** | **0.3233** | — | — |
| **this project, LoRA + head (360 training questions)** | **0.5633** | — | — |

**Correction to what I first wrote here.** I opened this section claiming the 0.766
figure "changes the project's target" and that I had been implying 0.5633 was a
reasonable place to be. Checking the files, **the report already carries the 0.766 row**
in its "Reported accuracy" comparison table, alongside the 0.343 chance rate for base
Laya. That claim was wrong and I am striking it.

What is actually true, and is still worth acting on: **`README.md` and `GUIDE.txt` do
not mention Laya or 0.766 anywhere.** They report the LoRA arm's 0.5633 with no
reference point, so a reader of the project's own documentation sees 0.5633 as the
score and has no way to know an Apache-2.0 checkpoint at the same size class already
scores 0.766 on the same 2,000 decisions. The report is right and the user-facing
artifacts are incomplete. Both now carry the comparison.

The gap is still ~20 points, and still Apache-2.0 open weights trained by a different
person on **a single RTX 6000 Pro (96 GB)**. The frozen-vs-LoRA gap, the
coverage-at-5% finding, and the threshold-does-not-transfer finding are all about
*method* and none of them depend on being SOTA — so this is a correction to what the
project's docs imply about its result, not a refutation of the experiments.

The card also carries an operational warning worth stealing: Laya **ships with a
1,024-token default that silently truncates long documents**, and says so in bold. Our
context handling deserves the same treatment.

## 2. The best-controlled cross-model table in the corpus

[`typed-decision-bench`](https://kyr0.github.io/typed-decision-bench/) (row 212), last
updated 2026-09-23: 7 models, 275 capabilities, 22,001 decisions, 5,499 calibrations,
**all on one H200 NVL** so the latencies are actually comparable.

| Model | Macro Soft Acc. | Δ vs Jev | p50 (ms) | p95 (ms) | VRAM |
|---|---:|---:|---:|---:|---|
| `jev-1.13.0` | **88.08%** | — | 716.4 | 778.8 | proprietary |
| `bonsai-2-27b-calibrated` | 76.46% | −11.6 pp | **170.9** | 448.0 | 9.0 GB |
| `openjev-qwen3.5-4b` | 74.13% | −13.9 pp | 1,066.1 | 1,478.9 | 12.6 GB |
| `spark-X2.5` | 70.97% | −17.1 pp | 1,056.6 | 1,783.5 | 9.6 GB |
| `von-1.1` | 48.57% | −39.5 pp | 38.5 | 46.5 | 3.8 GB |
| `laya` | **46.70%** | **−41.4 pp** | **36.7** | 44.4 | **1.4 GB** |

Three things fall out, and the first is a correction to what I wrote yesterday.

**(a) The fastest model is the worst model.** Laya is 36.7 ms and 46.70%; Jev is 716 ms
and 88.08%. That is a **19.5x latency inversion against a 41.4-point accuracy gap.** I
added Contradiction 7 yesterday on the strength of an HN comment saying Laya "does only
slightly better than a small classifier" on real decisions. On this benchmark it does
considerably *worse*. The discipline I recorded — *never quote a latency win without the
accuracy number* — is right, and this is the cleanest possible demonstration of it.

**(b) The Ollaya table I recorded yesterday is superseded for accuracy purposes.** Ollaya
leads with Laya at 8.1 ms; here Laya is 36.7 ms and last on accuracy. Both are
single-party measurements on different hardware, and they do not agree. **Neither should
be quoted alone.** The Ollaya table remains useful for the API-compatibility surface; the
kyr0 table is the one to cite for accuracy.

**(c) Laya's calibration is the real problem, not its size.** kyr0's write-up says Laya
"only performs well on a non-diverse benchmark, and **calibration is really bad**," and
gives its own ECE-15 as **13% against Jev's 8.4%** — i.e. the 76% model is *worse
calibrated* than the 88% one. This is the same failure this project measured directly
(frozen arm: fitted T of 3.300, **0.0000 coverage at a 5% error budget**), arriving from
a completely independent direction. A small decision model that is badly calibrated has
**no usable operating point**, which is a much more serious defect than being 20 points
behind.

## 3. A direct challenge to this project's premise

kyr0, in the Bonsai-Llama-Jev post:

> **"btw. just one bit on that. RLCD is overrated guys! Use CE as the primary loss with an
> eye on Brier+NLL; it's cheaper and more effective!"**

This project is built on RLCD as its conceptual prior — the whole framing is "you already
have RLCD's main benefit, because cross-entropy is a strictly proper scoring rule"
(Key Takeaway 10). The claim that RLCD is *overrated* is a **single source, from someone
with a competing system and a commercial interest in it**, so it does not overturn
anything. But it is the first direct challenge to the design basis I have found, it is
specific, it is testable, and it costs nothing to test: train the same LoRA arm with CE
as the primary loss and Brier/NLL as reported metrics, on the same seed and split.

Note the subtlety: kyr0 is not saying *calibration doesn't matter* — he is saying CE gets
you there more cheaply. That is consistent with this project's own finding that the
calibration gap is a **fitting** problem, not a training-objective problem.

## 4. The most valuable negative result in the corpus

[u/Obside_AI, r/ai_trading](https://www.reddit.com/r/ai_trading/comments/1wkq4lt/) (row 215):
**731 trades over 24 hours, −3.15% (−$3,150), of which ~$1,650 was fees. 21% of trades
profitable after fees. 779 ms average per decision.**

The setup was unusually fair to Jev. The author supplied candle history, twelve
indicators, and — critically — **explicit trading costs** (commission, exchange fees,
slippage, round-trip ticks), and asked directly whether a move was likely to clear the
cost. Entry required >50% probability. Position management had no threshold. A hard
$500 stop was imposed outside the model.

Even with the costs handed to it, it churned. The author's summary: *"So far, it's very
efficient at generating commissions."*

The open question the author could not answer in 24 hours is the one that matters: **"whether higher probabilities actually lead to better trades after costs."** That is precisely this project's coverage question, asked in production instead of on a held-out split.

**And the paired anecdote cuts the other way.** [u/artguerilla, r/hermesagent](https://www.reddit.com/r/accelerate/comments/1wn1gck/) (row 226), 278 upvotes, 82 comments, same week, same model class: *"So far it actually looks promising. Which probably means I've misunderstood something and will discover it in 6 hours."*

**Two retail trading experiments reached opposite conclusions in the same week. Neither is
evidence.** Recording both, and refusing to average them into a finding, is the only
defensible handling — which is also why the project's own single-seed caveat is the
right standard.

## 5. What people actually build, and the one commercial number

A source-reviewed directory of **287 projects** (row 216) names the pattern its author
sees everywhere:

> **big model → Jev → code → Jev → tool → Jev → big model**

Jev handles the small decisions between the large model's steps. Not writing code, not
generating, not reasoning deeply — choosing, filtering, routing, gating, judging
completion.

The single commercial datapoint: **Vercel Labs** tried Jev inside `json-render`. Their
train-ticket demo took **3.21 s on the default JSONL path and 0.88 s on the Jev path — a
3.6x reduction**, by having Jev pick from predefined components while ordinary code
assembles the interface. This is the cleanest real-world validation in the corpus of the
latency claim, and it comes from a company with every incentive not to oversell.

## 6. A same-accuracy local substitute

[choosekit](https://www.reddit.com/r/LLMDevs/comments/1wkc9hp/) (row 219): Qwen3.8 27B
Q4 XL via llama.cpp against Jev on **SemIf's 144-task benchmark — both 96.53%**, median
**239 ms local vs 368 ms Jev hosted**. On a narrow, well-matched task set, a quantised
27B general model matches a specialised decision model and does it locally. It is 60x
larger than this project's backbone and it still wins on wall-clock.

## 7. Prior art, and why I am not claiming it

The 3.3k-upvote, 98%-upvoted, 317-comment thread (row 214) is a prior-art claim: that
the Jev architecture was built and open-sourced in **March 2025** (arXiv 2503.23303) and
again in September 2025 (arXiv 2510.01237), with PPO over sequence embeddings, and that
TypeSafe released the same idea as a "literal breakthrough without technical papers."

**I could not verify either arXiv ID.** The arXiv API returned an empty feed for
2503.23303. So this is recorded as an unverified claim, not as prior art, and it is not
used to support anything.

The same author shipped [Laya](https://huggingface.co/convaiinnovations/laya) — the model
in §1. Whatever one thinks of the priority dispute, that is unambiguously a real
contribution, and it is the strongest argument for the prior-art point being partly
self-serving.

The thread's most substantive comment is not about priority at all. nullc (471 points):
publishing it means *"they will not be able to obtain a valid patent on the general idea
and lock it away from everyone."* The follow-on discussion is careful and correct about
what prior art does and does not do — the examiner may narrow the application, invalid
claims help future defendants, and a patent office "usually grants the patent without
really bothering to check if the work is novel."

## 8. The conceptual challenge worth taking seriously

r/learnmachinelearning (row 220): *"When the state space is explicit and the answer is
verifiable, why use a model to make the decision at all?"* The author's hybrid framing:

- **ML/LLM** — perception, ambiguity, language, hypothesis generation
- **Symbolic layer** — explicit state, constraints, invariants, verification
- **Runtime** — deterministic execution when the answer is knowable

This is not an argument against decision models and the author says so. It is an
argument about **where the boundary belongs**, and it lines up with the strongest
practical finding in this corpus: every successful project uses a decision model for the
*small* choices and keeps deterministic code for the *large* ones. `json-render` does
not let Jev build the UI. The trading bot does not let Jev place orders unguarded. The
drone project keeps classical control for flight stability and uses Jev one level up.

**Decision models earn their place in the gaps between deterministic steps, not across
them.** That is a sharper version of this project's own conclusion than I had written.

## 9. Two name collisions, flagged to prevent a wrong merge

- **"Blink"** appears both as [sqliteai/blink](https://github.com/sqliteai/blink) (row 204,
  a C99 typed-decision runtime) and as a Jev-powered semantic codebase navigator in the
  287-project list. **Different projects.**
- **"Reflex"** is both [kaustav1996/reflex](https://github.com/kaustav1996/reflex) (row 89,
  a cascade system) and [lateos-ai/reflex](https://github.com/lateos-ai/reflex) (row 207, a
  Jev-like runtime). **Different projects.**

## 10. What I did not get

- **No arXiv verification** of the prior-art claim. Empty feed, not a negative result.
- **Only one Redlib instance was used** (`safereddit.com`, SFW-only). A pass through a
  non-SFW instance may surface more, and the 287-project directory was not itself crawled.
- **The 287 projects were not individually read.** The list is a directory, not evidence.
- **No new experiment.** Read-only network calls again. Every number above is somebody
  else's measurement.
