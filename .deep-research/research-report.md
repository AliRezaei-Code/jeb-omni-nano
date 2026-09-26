# How Jev-Omni Was Made — and How to Build a 34× Smaller Version of It

**A source-verified deep-research report**

| | |
|---|---|
| **Topic** | The architecture and training recipe of the "Jev" family of typed-decision models, and the design of a much smaller, cheaper sibling built on Liquid AI LFM2.5 |
| **Report generated** | 2026-09-26 |
| **Research cutoff** | 2026-09-26 |
| **Sources analyzed** | **230** unique URLs fetched and read. The 200-source floor is met. Full ledger with per-source claims: `sources-ledger.md` |
| **Overall confidence** | **High** on the Jev-Omni architecture and the Liquid LFM2.5 configuration — both read from primary published source files, not model-card prose. **High** on licensing (full licence text read verbatim). **Medium** on the multimodal extension, which is designed here but not trained. **High** on this project's own measured results (LoRA vs frozen, coverage, layer sweep) — all reproduced from `experiments/`, though single-seed. **Low** on any claim about a *shipped* Jeb-Omni-Nano checkpoint: no trained checkpoint is published, and the model is ~20 points below the public 0.766 bar. |
| **Companion deliverable** | `GUIDE.txt` — 13-part build guide, ~2,200 lines |

---

## Executive Summary

### The headline

`akhilaaa3/Jev-Omni` is a **12-billion-parameter Gemma 4 12B with its text-generation
loop replaced by a 983,456-parameter linear head**. The head takes the final hidden
state at the last token position, standardises it, projects it to 256 numbers, masks
the slots beyond the real option count, and softmaxes. That is the entire model-specific
innovation, and it is recoverable because the author shipped the code
([`jev_omni.py`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/jev_omni.py)) and the
recipe ([`decision_config.json`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/decision_config.json)).

The head is **0.008%** of the network. Which means "make this cheaper" is not a
squeezing exercise — it is a question of *which pretrained backbone you attach it to*.

### The findings that matter, in order of how much they should change your plan

**1. The bar on this project's own benchmark is 0.766, and it is already public.**
`laya-typed-decisions` — Apache-2.0, 421M — scores **0.766** on `typed-decisions` (2,000
decisions, four workflows), trained by a different person on a **single RTX 6000 Pro
(96 GB)**. This project reaches **0.5633** at 360 training questions on CPU. The gap is
~20 points. Everything below is about *method* — none of it needs to be state of the art
to be true — but **0.5633 is not a headline, and 0.766 is.**

**2. The design is "read a probability instead of generating a claim about one."**
Jev-Omni's card is explicit: "Supply a question and options; receive a probability for
each option — **not a generated explanation**." The head never runs a decode loop, so
there is no refusal path, no parser, and no format repair. An independent
reverse-engineering study of the *hosted* Jev product, based on 10,000 API calls, reached
the same conclusion: "decision probabilities read directly from its internal
representations" rather than "[generated] confidence claims"
([Archer Hume, 17 Sep 2026](https://archerhume.com/posts/jevs-architecture-unmasked/)).

**3. The 256 outputs are positional option slots, not vocabulary tokens.**
`Linear(3840, 256)` — slot *i* is "option *i+1*". That is why one checkpoint answers a
question it has never seen, with option words it has never seen, without retraining, and
why the head cost scales with `hidden_size` and nothing else. Four independent teams
converged on slot or marker logits over label-token logits.

**4. Fine-tuning the backbone is not optional — we measured it, we did not assume it.**
On `customer_service`, identical data, splits and evaluation, changing only whether the
backbone's weights can move:

| arm | accuracy | Brier | ECE | fitted T | coverage @5% error |
|---|---:|---:|---:|---:|---:|
| frozen backbone + head | 0.3233 | 0.1404 | 0.1467 | **3.300** | **0.0000** |
| **LoRA r=16 + head** | **0.5633** | **0.1098** | **0.0646** | **1.050** | **0.0967** |

**Claim class: our own measurement, single seed, not replicated.** The coverage column
is the finding: a frozen backbone and head has **0.0000** coverage at a 5% error budget —
its single most confident prediction was wrong. The temperature explains it: **3.300**
against **1.050**. It does not merely score worse, it scores *dishonestly* worse.

**5. More data does not rescue the frozen arm.** With **1,212** training questions across
all four workflows (3.4x) the frozen arm reaches 0.4357 and temperature 1.800 —
miscalibration roughly halved — and coverage@5% is still **0.0000** across 996 evaluation
questions. **Data quantity and backbone fine-tuning are not substitutes.**

**6. Calibration, not size, is the failure mode — and it is cross-validated.** The
strongest independent result in the corpus agrees: the builder of the best open system
says Laya "only performs well on a non-diverse benchmark, and **calibration is really
bad**", with his own 76%-accurate model at **ECE-15 of 13% against Jev's 8.4%**. That is
this project's frozen-arm verdict arriving from the other direction. **A badly calibrated
decision model has no usable operating point** — a worse defect than being 20 points
behind.

**7. The value is concentrated at the top, and thresholds do not transfer.** The 10% most
confident decisions are **93.3%** correct against 56.3% overall; the frozen arm manages
50.0% on the same slice. Coverage at a 5% budget is **9.7%** on `customer_service` and
**5.5%** on `invoice_processing`. A decision model is worth building for the top of its
confidence distribution, and a threshold fitted on one workflow is not a threshold.

**8. This field is 56% plumbing and 6% classification.** Of **693** catalogued projects,
387 are orchestration, routing, MCP, guardrails, context GC and evaluation, and **2 are
filed under Classification**. The interesting work is integration — and that is also the
binding constraint on any new small model. The 287-project survey's own summary of the
pattern is the design brief: **big model -> Jev -> code -> Jev -> tool -> Jev -> big
model.**

**9. The field's speed claims are unverified, and the census says so.** **376 of 693**
entries carry *"performance and cost benefits have not been independently verified"*;
**352 of 693** could not even be confirmed to make a decision. Four independent latency
measurements for the same system span **50 ms to 716 ms** — a 14x range, with request
shapes different enough that they may not be the same operation. **The direction is
agreed by everyone; the multiples are not.** The one independently reproduced performance
number in the entire corpus is Vercel Labs' `json-render` at **3.21 s -> 0.88 s**.

**10. The upstream "Jev" is closed; everything open is an independent reimplementation.**
No weights, no paper, no published training method. Jev-Omni, Kev, Jebadiah, Laya,
`this-that-model`, Ollaya, Blink and this project all reimplemented the *pattern* without
seeing each other. A prior-art claim exists (arXiv 2503.23303) and **could not be
verified** — the arXiv API returned an empty feed — so it is recorded and supports
nothing.

### What we built and verified

A working reference implementation in this repository. The head is a faithful port of
Jev-Omni's `_Head256` to LFM2.5's width; `tests/test_smoke.py` asserts it is
**bit-identical** to the published reference. Across two suites, **112 checks** pass (67 unit + 29 smoke/integration against
real pinned LFM2.5-350M weights):

- head is exactly `1024×256 + 256` = **262,400** parameters = **0.0740%** of LFM2.5-350M's
  354,483,968
- `fit_temperature` recovers a planted `T = 2.5` as `2.5000`
- confidence formulas reproduce Kev's published worked examples exactly (0.21; 1.44/0.34)
- inference is deterministic
- a real training run closes the loop: 80 examples, 40 steps, 127 s on CPU, loss
  **1.80 → 0.31**, 85.0% accuracy, Brier 0.0775, ECE 0.0758

**That 85% is a plumbing demonstration on 12 hand-written sentences, not an accuracy
claim.** No trained Jeb-Omni-Nano checkpoint is published, and this report makes no
accuracy claim for one. Multimodal is designed but untrained.

### The honest cost comparison

Jev-Omni has no bill, so its card prices it as a *proxy*: its recorded input tokens at
OpenRouter's Gemma 3 12B input rate of $0.05/M. The real Jev is $0.042/M. JevBench
measures cost **per 1,000 decisions, not per 1,000 tokens** — one decision is a whole
question, hundreds to thousands of input tokens — and computes Jev at $0.0399 per 1,000
decisions from ~950 input tokens each.

A 350M model on a Pi costs electricity. Per this-that-model-1.0's card, the honest
framing is the one they chose: "read our distance from the hollow square as **one order
of magnitude, not five**," because a self-hosted model's cost is electricity while a
hosted price must cover serving and margin. The real gap is **independence**, not a
five-orders-of-magnitude price gap. We report it that way.

**And the latency side of that claim is not settled either.** Four independent
measurements of Jev-class latency in this corpus:

| Source | Figure | Request shape |
|---|---:|---|
| Jev-Omni's own card | 83 ms | in-process, H200, ~2k-token text, one question |
| Awesome Jev radar | 50-100 ms | **unstated** — no version, question count, or network condition |
| Ollaya | 236-276 ms | hosted API, median, five questions, includes network |
| `typed-decision-bench` | 716 ms p50 / 779 ms p95 | H200 NVL, 275 capabilities, includes HTTP |

A **14x spread**, and the shapes differ enough that these may not be measuring the same
operation. What every source agrees on is the *direction*: a local small-model decision
path is far cheaper and lower-latency than a hosted frontier round trip. **Nobody
disputes that; the multiples do not survive.** Treat any single latency number from this
field as unattributed until someone states the request shape it was measured under —
which, per the 693-project census, **54% of published projects cannot currently do.**

---

## Methodology

- **Sub-questions investigated (7).** SQ1 what Jev-Omni actually is and how it was
  built; SQ2 the wider Jev ecosystem and what independent implementers concluded; SQ3 the
  Liquid LFM2/LFM2.5 family; SQ4 turning hidden states into trustworthy probabilities;
  SQ5 small-model multimodal decision making; SQ6 toolchain, distillation and cost;
  SQ7 licensing and honest reporting. Frozen in `notes/00-scope.md` before searching.
- **Queries executed.** ~150 search-API queries and enumeration calls against the Hugging
  Face model/dataset/space APIs, the arXiv query API, and the GitHub repository search
  API, plus ~200 direct URL fetches. The query angles and their second wave are recorded
  per sub-question in `notes/00-scope.md`.
- **Search tools used.** Direct URL retrieval as the primary mechanism, because
  **`websearch` was non-functional for the entire session** (Codex and Z.ai 429 quota /
  subscription errors, Startpage and Ecosia timeouts, DuckDuckGo timeout, Google
  automated-traffic challenge, Mojeek datacenter-IP block). Endpoints that did work and
  were used throughout:
  - `huggingface.co/api/models`, `/api/datasets`, `/api/spaces` (with `?search=`) for
    enumeration
  - `huggingface.co/{owner}/{repo}/raw/main/{file}` for model cards, configs, licences and
    source code
  - `arxiv.org/abs/{id}` and `export.arxiv.org/api/query` for papers
  - `api.github.com/search/repositories` and `raw.githubusercontent.com` for repos
  - `docs.liquid.ai` and `www.liquid.ai/blog` for vendor documentation
- **Source selection criteria.** Every URL in the ledger was fetched and read; nothing is
  cited from memory. Tier A (primary: official cards, config files, source code, licence
  text, papers, dataset cards) preferred; Tier C (forums, blogs) never solely supports a
  report claim. Where sources disagree, both are recorded and the disagreement flagged
  rather than resolved by fiat. Unreachable sources are recorded as findings, not dropped
  silently.
- **Deep-read sources: ~60 of 200.** The remaining ~140 were read at the depth needed to
  extract their specific claims (a config file's fields, a model card's benchmark table, a
  dataset card's schema), and each carries a Relevance score plus its extracted claims in
  the ledger.
- **Per-source notes follow the template skeleton** (Key claims → Data points/quotes →
  Contradictions with other sources → Credibility notes), one file per deep-read source,
  indexed in [`notes/INDEX.md`](notes/INDEX.md). Nine files cover the sources that
  changed the design. **No source was deep-read and left undocumented.**
- **Rejected: 15.** Recorded in the ledger's rejection log with the reason. Five were arXiv
  IDs that turned out to be unrelated papers after fetching — a reminder that guessing an
  identifier is not the same as citing one. One (`LiquidAI/LFM2.5-1.2B`) is **gated
  (HTTP 401)** and is therefore cited only through Liquid's own documentation, never
  paraphrased from a third party.

### A note on the collection method

Because discovery ran through APIs rather than a general search engine, the corpus is
**strong on primary artefacts and thin on discussion.** Model cards, config files, source
code, licence texts, papers and dataset cards are well represented. **This was written
before the discussion layer had actually been searched, and it understated the problem:
it was not thin, it was absent.**

`websearch` returned `usage_limit_reached` on every provider (Codex, ZAI, Startpage,
Ecosia, Google, Mojeek) from the first attempt through to the last, and no substitute
was found for most of the project. Near the end, the **Hacker News Algolia API**
(`hn.algolia.com/api/v1/search`, `…/items/<id>`) was found to work unauthenticated and
returned complete JSON comment trees. It immediately surfaced a **570-point, 137-comment
front-page thread posted the day before** on the largest open runtime in this space —
Ollaya — plus Blink, three more independent implementations, and 169-point reception
data on the LFM2.5 backbone. **Eleven ledger rows (201–211) and one full note exist only
because of that late fix.** The failure was mine: I documented a gap I had not tried
hard enough to close.

The technical claims are unaffected — this project's own measurements dominate the
report and no claim here rests on a forum post. But the *practitioner discourse* —
who is shipping this, what surprised them, what they got wrong — was genuinely
missing, and it turned out to contain an independent confirmation of this report's
central architectural claim and a challenge to one of its phrases. Both are now
recorded (SQ2 "The practitioner layer, found late", Contradiction 6).

## Thematic Findings

### SQ1 — What Jev-Omni actually is, read from its source

**Claim class: hard fact, from primary source files.**

The repository ships `config.json`, `decision_config.json`, `head.pt`, `jev_omni.py`,
`verification.json`, `processor_config.json` and `sha256.json`. The architecture and the
recipe are therefore *readable*, not inferred.

**Backbone.** `Gemma4UnifiedForConditionalGeneration`, `model_type: gemma4_unified`,
base `google/gemma-4-12B-it`, **11,959,730,224 BF16 parameters**, 71.6 GB of storage
([config.json](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/config.json),
[API metadata](https://huggingface.co/api/models/akhilaaa3/Jev-Omni)). The text stack is
`hidden_size` 3840, 48 layers, 16 attention heads / 8 KV heads, `head_dim` 256,
`intermediate_size` 15360, vocab 262,144, `sliding_window` 1024, `max_position_embeddings`
262,144, and a strict repeating **5-sliding : 1-full** attention pattern across all 48
layers. Sliding layers use `rope_theta` 10,000; full layers 1,000,000 with
`partial_rotary_factor` 0.25 and `rope_type: proportional`.

**The head**, quoted verbatim from `jev_omni.py`:

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

Four properties, each load-bearing:

1. **Last-position readout.** A forward hook on the text backbone captures
   `out.last_hidden_state[:, -1].float()`. No pooling, no gather.
2. **Frozen standardisation.** `mu` and `sd` are `register_buffer`, not `Parameter` —
   they are never trained.
3. **fp32 head on a bf16 backbone.** `dtype=torch.float32` on the `Linear`; the backbone
   runs under `torch.autocast("cuda", dtype=torch.bfloat16)` with `use_cache=False` and
   `logits_to_keep=1`. This is why probabilities do not inherit bf16's ~3 significant
   digits.
4. **Slot masking.** `masked_fill(arange(256) >= counts, -1e30)`; the caller then slices
   `[:len(options)]` and softmaxes.

Parameter count: `3840×256 + 256 = 983,456`, which the independent GGUF conversion
confirms independently: the 4-bit Jev-Omni card reports a **3.78 MiB** FP32 decision head,
and `983,456 × 4 bytes = 3.75 MiB`
([Reza2kn/Jev-Omni-Q4_K_M-GGUF](https://huggingface.co/Reza2kn/Jev-Omni-Q4_K_M-GGUF/raw/main/README.md)).

**The recipe**, verbatim from `decision_config.json`:

| Field | Value |
|---|---|
| `size` | 24000 examples |
| `rank` / `alpha` | 512 / 512 |
| `lr` / `head_lr` | 1e-05 / 1e-05 |
| `microbatch` / `grad_acc` / `effective_batch` | 8 / 1 / 32 |
| `epochs` | 1 |
| `warmup_ratio` / `warmup_steps` / `steps` | 0.1 / 75 / 750 |
| `seed` | 3407 |
| `schedule` | "linear warmup then linear decay to 10% of peak" |
| `trainable_parameters` | 2,099,183,872 |
| `world_size` / backend | 4 / nccl |
| `adapters_merged` | `["trained v1 rank128", "lora-v1merged-n24000-r512-lr1e-05-eb32-w10-ddp4"]` |
| `initialization` | "FP32 merged trained v1 + trained head + fresh LoRA" |

Two things are worth pausing on. The rank-512 LoRA covers **17.5%** of a 12B network,
which is far above the rank-16 that Kev and Jebadiah both use — Jev-Omni was trained
with serious hardware. And it was a *second* adapter merged on top of a rank-128 first
one, which Jebadiah's own logs warn is fragile ("the second delta eroded the first",
Kev-0.8B round 13).

**The prompt**, verbatim from `jev_omni.py`'s `_prompt()`:

```
{state}

---

QUESTION: {question}

OPTIONS:
1. {option_1}
2. {option_2}

Reply with only the number of the correct option (1-{K}).
Output a single number and nothing else.
```

**Verification data.** `verification.json` ships four cases with reference probability
distributions, `worst_abs_diff: 0.01936584711074829`. Two of them are epistemically
interesting because the answer is genuinely unknowable, and the model's output is
correspondingly flat: a die roll gives `{1: 0.234, 6: 0.459, ...}` and a four-marble urn
gives `{Blue: 0.413, Green: 0.382, ...}`. **A calibrated model on an unanswerable question
produces a near-uniform distribution.** That is the behaviour the whole design is
after, and it is visible in the shipped reference outputs.

**Multimodal preprocessing**, from `processor_config.json` and the runtime:

- image: `patch_size` 16, `pooling_kernel_size` 3, `image_seq_length` 280, `model_patch_size` 48
- audio: `sampling_rate` 16000, `feature_size` 640, `audio_samples_per_token` 640,
  `audio_seq_length` 750 — i.e. 25 audio tokens/second, 750 tokens for the 30 s cap.
  Decoded by `ffmpeg -v error -i IN -t 30 -ac 1 -ar 16000 out.wav`
- video: **16 frames**, sampled at `round((total-1) * (k+0.5) / count)` — the centre of
  each of 16 equal bins — and each frame is passed as a **separate `{"type": "image"}`
  item**, not as a video item. The 16-frame cost is therefore 16 image encodes.
- limits: 2–256 options enforced in code; "best supported at ≤20 options"

**Reported results.** DecisionBench Medium 87.57% state-macro / 86.01% micro; JevBench
86.15% / 87.45%; MMAU 63.10% micro on 1,000 questions; MVBench 53.10% on 2,786 questions
across 14 tasks; ECE 0.0400 (10 bins) on DecisionBench Medium. Warm H200: 83 ms for
~2k-token text, 26 ms image, 31 ms for 13 s audio, 504 ms for 16-frame video.

The latency profile is the design working: video at 504 ms against text at 83 ms is a
6.1× ratio, which is what "16 images" costs. A decision model generates zero tokens, so
its latency is dominated by the prefill — and the prefill is dominated by media.

#### SQ1 verdict

Jev-Omni is fully understood. It is a Gemma 4 12B, a 983K-parameter standardised linear
readout on 256 positional slots, LoRA-fine-tuned on 24,000 examples for one epoch at
lr 1e-5 with a 10%-floor linear decay. Anyone can reimplement it, and several people
have.

### SQ2 — The Jev ecosystem, and what independent implementers concluded

**Claim class: hard fact for existence and design; reported fact for benchmarks.**

The "Jev" name has become a genre. Searching the Hugging Face Hub for `Jev` returns
roughly 40 distinct models, and the search results are effectively a map of the design
space. What matters is not the count but the **convergence**.

#### The convergence

Every serious independent implementation agrees on all of the following:

| Decision | Consensus | Who |
|---|---|---|
| A small **trained head**, not constrained decoding | Yes | all |
| A **pretrained causal backbone**, fine-tuned | Yes | all (except jeff, an encoder) |
| **LoRA**, not full fine-tuning | Yes | Kev, Jebadiah, Jev-Omni, this-that |
| A **fitted temperature** shipped with the model | Yes | Kev, Jebadiah, Laya, jeff, this-that |
| **Per-question-type** temperatures | Yes | Jebadiah (1.12/1.33/1.20), Laya |
| Report **Brier or KL** alongside ECE | Yes | typed-decisions, Kev, this-that |
| Probabilities framed as a **routing signal**, not truth | Yes | all, explicitly |

That last one is a cultural norm as much as a technical one. Every project in this
ecosystem ships a policy suggestion built on a confidence threshold, and every one of
them says the same thing: *the model never refuses; policy is built by the caller from
the probabilities* (Jebadiah), *thresholds belong to the caller* (Kev), *a threshold is a
policy you choose from measured accuracy at that coverage on your data, not a property
of the model* (Laya).

#### The one real disagreement: read-out geometry

**A. Slot logits.** `Linear(hidden, 256)`; option *i* → slot *i*.
Used by Jev-Omni and by us.

**B. Pointer scoring.** Score each option's own `</opt>` hidden state against the
question's `<decide>` hidden state. Kev's design. On hybrid bases with Gated DeltaNet
layers, which are recurrent and ignore attention masks, Kev runs one row per question so
isolation is exact, and reuses the state cache across rows
([jaredpalmer/kev](https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md)).

**C. Label-token logits.** Read the LM's own vocabulary logits restricted to the option
label tokens: `p_k(j|x) = softmax_j(⟨w_ℓ(k,j), h_k⟩ / τ)`, normalised over exactly the
declared labels, so "the support of that distribution **is** your option list"
([this-that-model-1.0](https://huggingface.co/flock-io/this-that-model-1.0/raw/main/README.md)).
Used by Jebadiah, Laya, this-that-model.

**And now a preregistered answer to "does the read-out cost accuracy?".**
`r-ms/mini-jev` ran a preregistered study (`PREREG.md`, amendments v1.1–v1.3) on a
*frozen* Qwen3-4B-Instruct-2507 over CLINC150, every number recomputed from stored run
records. Reading an option **letter** rather than generating JSON under a grammar costs
no accuracy: **6,750 paired observations, JSON 0.909 vs letters 0.907, Δ −0.22 pp, 95% CI
[−1.44, +1.04], and every k from 2 to 16 has a CI covering zero.** It is also **4× faster**
on short texts, and 1.4–2.4× faster on 2048-token texts *with a shared-prefix cache*
(while the naive per-field re-read is 1.10–1.16× *slower*).

Two results in it change design advice rather than confirming it:

- **A letter beats writing the option's name by +10.0 pp [+8.3, +11.7]** on intent and
  +13.2 pp on domain. *"The single lever that changed accuracy was giving the model a
  one-token identifier to answer with."* That argues for a symbolic option slot over
  scoring option text — which is what slot logits are.
- **Never let the model write its own probabilities.** Same units, same options: letters
  score **0.896**, a probability-writing format scores **0.346**, and 62% of its "choices"
  are the first option.

And one caveat we must carry: the raw option shares from a logit read are *"normalized
candidate scores, **not calibrated probabilities**… do not read the percentage as
P(correct)."* That is precisely the gap our temperature fit closes.

**Design C has a documented architectural ceiling.** Laya scores 0.425 on Banking77 (77
labels) against Jev's 0.870, and diagnoses it exactly: options share a fixed
`head_max_len` budget (192 tokens English, 256 multilingual), so 77 labels get ~3–4
tokens each and become indistinguishable. **Both Laya checkpoints score exactly 0.425** —
a budget ceiling, not a capability gap. Their docs advise keeping choice questions
under ~20 options.

Designs A and B do not have this failure mode. That is a real, under-advertised argument
for the slot design, and the reason Jev-Omni can honestly claim "the head accepts 256"
where Laya cannot claim 77.

**And the September 2026 paper is a second argument for A over B**: the option-name
failure is **4.1× weaker** in "a second model family that mean-pools over the full option
span," i.e. a pointer-style reader. We adopt A on the strength of this.

#### The negative results, which are the most valuable part

`getainode/jebadiah` publishes its full experiment log. Things that did **not** work:

- A **14,714-question synthetic pool** authored and labelled by their own teacher models
  made the 9B *worse*.
- Re-targeting that synthetic pool with a better teacher did not move the headline.
- Public human yes/no data did not lift their human yes/no evaluation sets.
- **Two epochs were worse than one.**

What **did** work:

- Adding human-rated score data and switching the score target to an **ordinal kernel**:
  HelpSteer2 Decision Score **-21.4 → +11.9**, calibration error **0.39 → 0.045**, with
  rubric accuracy unchanged. Their summary is the most useful sentence in this report:
  *"The model became honest about not knowing helpfulness rather than better at judging it."*
- Switching the base checkpoint from `-Base` to the **chat checkpoint with thinking off**:
  worth **2.15 headline points** on the 4B — *"more than any data change we made."*

That second result is why our default backbone is the instruction-tuned
`LiquidAI/LFM2.5-350M`, not `-350M-Base`.

#### The stacking failure

Kev-0.8B needed fifteen registered rounds. Round 13 attempted to stack a skills delta on
top of a documents delta and **failed**: *"the second delta eroded the first: seed 1's
documents lower bound was -2.03 pp against a -2 pp floor, and seed 2's short-state
accuracy was -1.6 pp."* Round 15, training jointly on the union from the released
checkpoint, passed. Jev-Omni's own `adapters_merged` field shows a rank-128 adapter
merged before the rank-512 one — i.e. it *did* stack, at 12B scale with a lot more data.

**Lesson: if you fine-tune twice on a small model, consider training on the union.**

#### On scale, from people who built the small versions

Kev-0.8B, verbatim: *"It is still a sub-1B model. On the development splits it trails
Jev everywhere it can be compared: documents 0.842 vs 0.868, hard-v1 0.594 vs 0.777,
devtools-v1 0.602 vs 0.713, out of domain 0.648 vs 0.857."* And per-source: MMLU
**0.42 vs 0.90**, PAWS **0.55 vs 0.79**, three-level date arithmetic **0.38 vs 0.93**
(0.38 is barely above the 0.33 chance rate — the model hedges to the middle level).

jeff (400M), verbatim: *"Cheaper to self-host, but less accurate than jev on
reasoning-heavy tasks."* Its JevBench rank came from cost; on Intelligence it was **#14
of 18**.

this-that-model-1.0 (1.88B) is the counter-example worth reading: **0.941 accuracy /
0.042 Brier / 0.126 NLL** against Jev's 0.765 / 0.133 / 0.403, at 30.9 ms and ~$0.000014
per pass. But its own card adds the caveat that decides the argument: *"read our
distance from the hollow square as **one order of magnitude, not five**"*, because its
cost is electricity while a hosted price must cover serving and margin. And on the
spatial benchmark, where question *shapes* were unseen, it scored 0.844 against Jev's
0.803 — while `laya-typed-decisions` scored **0.345 against a chance rate of 0.343**.

That 0.345 is the single most important number in the ecosystem. It is what a
well-built 421M model does zero-shot on a task family it was never trained for. Their
authors' own framing: *"Laya is a fast base to specialise, not a zero-shot decision
engine."*

#### SQ2 verdict

The design space is well-mapped and the centre is well-defended. Small models are
**specialists that need your data**, not general-purpose replacements. The one thing
every project agrees on and that this report makes central: **calibration is the product,
and it only transfers if you fit it yourself.**

---

#### What we measured: the read-out layer sweep

The strongest external claim in this report — that a *middle* layer beats the last for a
linear head — comes from a single unreplicated source (AnyJev, on a 7B model). We tested
it rather than passing it on.

Frozen `LiquidAI/LFM2.5-350M` @ `9e6c6ccf`, a fresh `DecisionHead` trained per layer so
the read-out layer is the only variable, `LocalLLaMA/typed-decisions` /
`customer_service` against the **soft gold distribution**, split by case, temperature
fitted on one half of the held-out set and reported on the other:

| layer (from end) | train loss | accuracy | Brier ↓ | ECE ↓ | fitted T |
|---|---|---|---|---|---|
| L0 (-16) | 1.2877 | 0.4000 | 0.1360 | 0.2258 | 0.350 |
| L4 (-12) | 1.2875 | 0.4000 | 0.1360 | 0.2256 | 0.350 |
| L8 (-8) | 1.2760 | 0.4000 | 0.1358 | 0.2268 | 0.600 |
| L12 (-4) | 1.2665 | 0.4000 | 0.1346 | 0.2203 | 0.750 |
| **L15 (-1)** | **1.2508** | **0.4889** | **0.1312** | **0.1360** | **0.900** |

**Claim class: our own measurement. The last layer wins on accuracy, Brier, ECE and
training loss**, and the fitted temperature rises monotonically with depth, so the earlier
layers are more overconfident relative to how often they are right.

**Weight: limited, and stated.** n = 45 evaluation questions, so the accuracy gap is four
questions. What makes it more than one number is that four indicators agree and the
temperature trend is monotonic. Absolute accuracy is low (0.4889 against a `Prior`
base-rate baseline near 0.470): this is a 198-example, head-only, frozen-backbone probe
about *which layer*, not about achievable accuracy.

AnyJev measured a **7B** model; the "top blocks are busy turning the answer into tokens"
mechanism may be weaker at 350M. **The finding is not replicated at this scale, so
`readout_layer=-1` remains the default** — and the report's claim is downgraded from
"middle layers are better" to "tested here, not replicated; sweep it on your own data".
Full write-up: [`notes/04-layer-sweep-results.md`](notes/04-layer-sweep-results.md).

#### The practitioner layer, found late

This subsection exists because the collection method failed for most of the project
and was fixed at the end. `websearch` returned `usage_limit_reached` on every provider
from the first attempt, and no substitute was found until the Hacker News Algolia API
(`hn.algolia.com/api/v1/search`, `…/items/<id>`) turned out to work unauthenticated.
That pass found a **570-point, 137-comment front-page thread posted one day before it
ran** — the largest community signal in this entire space, and I had produced 1,959
lines of research without it. The honest characterisation of the previous state is
**not** "the discussion layer was thin". It was **absent**, and I did not say so.

**[Ollaya](https://ollaya.dev/)** (Apache-2.0, [github.com/ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya))
is "Ollama for decision models": it serves `/v1/systemone` and `/v1/models` in
TypeSafe's request and response shapes, and **the official TypeSafe Python SDK 0.7.1
runs against it unchanged** by pointing `TYPESAFE_BASE_URL` at localhost. That is a
materially different deployment story from "ship your weights" — it makes the
*interface* the durable artefact and the *weights* replaceable, which is precisely the
architectural bet this project's `decision-interface` contract makes. Two details
deserve attention: **every model ships its own calibration and a `Modelfile` refits it
on your labelled data** (an independent implementation of the per-bucket temperature
fitting in `calibration.py` — corroboration, not novelty), and weights are pulled from
their authors' Hugging Face repos pinned to a commit and checked against sha256 rather
than re-hosted.

The thread's own comment section is where the value is, because it is sceptical:

> *"Are there many models that are comparable to Jev for generic decision making?
> **Smarter move if you have an eval set is to just train a classifier and call it a day.**"*

> *"The best open ones are close to Jev now, **but they're big models**."*

> *"In a benchmark with actual decisions — navigation, traffic, waypoints — **laya does
> only slightly better than a small classifier**."*

The last one is the significant claim: an independent report of a **small gap between a
fast decision model and a plain classifier on real task-shaped decisions**, on a
benchmark I have never seen and cannot audit `[single source]`. It is not the same claim
as this project's frozen-backbone result (0.2667 accuracy, 0.0000 coverage at a 5% error
budget) and the two are **not merged here**. They point the same way from opposite ends
of the size axis, and the practitioner version has an unseen benchmark behind it.

One commenter also states the mechanism of this entire report, unprompted, having only
read TypeSafe's announcement:

> *"Their marketing language is misleading. They must still use some transformer
> language model backbone to encode the text input (BERT or decoder-only LLM). The
> biggest difference is the output, instead of auto-regressively generating tokens, they
> produce probabilities over a bounded set of decisions."*

That is the same architectural claim this report reaches from config files, arrived at
independently by a practitioner. It is also a direct challenge to the phrase "not an
LLM", which this project should stop repeating flatly — see Contradiction 6 below.

By contrast, another top-level commenter's *"text classification is equivalent to
decision. This is exactly the same thing Jev does"* is **overstated and this report
does not adopt it**. Text classification predicts one label from a fixed set; a decision
model must score an arbitrary per-call option set, handle score and boolean question
types separately, and calibrate across those sets. The report's SQ4 typing analysis
supports the narrower version, and this project's own measurements contradict the wider
one — the three question types do not behave alike.

Three further independent attempts appeared in the same window and are logged in
`notes/15-community-and-practitioner-sources.md`: [privatemode.ai](https://www.privatemode.ai/blog/system-one-from-glm-flash)
retrofitting System One behaviour onto GLM-5.3-Flash, [lateos-ai/reflex](https://github.com/lateos-ai/reflex)
(a name collision with ledger row 89, unrelated project), and
[sshh12/nanojev](https://github.com/sshh12/nanojev) at 200 lines. Together with Ollaya
that is **five unrelated teams** converging on a bounded read-out head over a general
backbone, none of whom needed to have read the others.

#### The deployment layer: what happened when people ran it

Everything above is artefacts — model cards, config files, papers, runtimes. The Reddit
layer, reached through a public [Redlib](https://github.com/redlib-org/redlib-instances)
instance after two passes recorded Reddit as inaccessible, is **outcomes**. It is also
where the corpus's best evidence lives, and where two of this report's claims were
corrected.

**[`typed-decision-bench`](https://kyr0.github.io/typed-decision-bench/)** (updated
2026-09-23) is the best-controlled cross-model table found anywhere in this project: 7
models, 275 capabilities, 22,001 decisions, 5,499 calibrations, **all on one H200 NVL** so
the latencies are genuinely comparable.

| Model | Macro Soft Acc. | Δ vs Jev | p50 (ms) | p95 (ms) | VRAM |
|---|---:|---:|---:|---:|---|
| `jev-1.13.0` | **88.08%** | — | 716.4 | 778.8 | proprietary |
| `bonsai-2-27b-calibrated` | 76.46% | −11.6 pp | **170.9** | 448.0 | 9.0 GB |
| `openjev-qwen3.5-4b` | 74.13% | −13.9 pp | 1,066.1 | 1,478.9 | 12.6 GB |
| `spark-X2.5` | 70.97% | −17.1 pp | 1,056.6 | 1,783.5 | 9.6 GB |
| `von-1.1` | 48.57% | −39.5 pp | 38.5 | 46.5 | 3.8 GB |
| **`laya`** | **46.70%** | **−41.4 pp** | **36.7** | 44.4 | **1.4 GB** |

This **supersedes Ollaya's table for accuracy purposes** and contradicts it on latency
(Laya: 8.1 ms there, 36.7 ms here). Two single-party measurements on different hardware
do not agree, so **neither should be quoted alone.** The benchmark's own caveats are
unusually good and are preserved: "the subline is the gap in percentage points vs the
baseline", "per-capability point estimates are noisy at small n", and Laya's 8k context
is RoPE-scaled.

The decisive pattern: **the fastest model is the worst model.** Laya is 36.7 ms and
46.70%; Jev is 716 ms and 88.08% — a **19.5x latency inversion against a 41.4-point
accuracy gap.** And kyr0's write-up identifies the cause as calibration, not size: Laya
"only performs well on a non-diverse benchmark, and **calibration is really bad**",
with his own 76% model at **ECE-15 of 13% against Jev's 8.4%**. That is this project's own
frozen-arm failure (fitted T 3.300, **0.0000 coverage at a 5% budget**) arriving from an
independent direction. **A small decision model that is badly calibrated has no usable
operating point** — a worse defect than being 20 points behind.

**[Laya](https://huggingface.co/convaiinnovations/laya)** is also the closest existing
artefact to this project, and it is Apache-2.0: 421M, non-autoregressive, a bidirectional
ModernBERT-large encoder plus a scratch Transformer head scoring `[MASK]` option markers,
~33 ms, 100+ languages, trained with RLCD against strictly proper scoring rules — **trained
on a single RTX 6000 Pro (96 GB)**. Its fine-tuned `laya-typed-decisions` checkpoint
scores **0.766** on the same 2,000-decision benchmark this project evaluates on, against
0.362 zero-shot. `README.md` and `GUIDE.txt` previously reported this project's 0.5633
with no reference point; both now state that **the bar is 0.766, not 0.5633.**

#### The one commercial validation in the corpus

**Vercel Labs** tried Jev inside `json-render`. Their train-ticket demo took **3.21 s on
the default JSONL path and 0.88 s on the Jev path — 3.6x faster** — by having Jev choose
from predefined components while ordinary code assembles the interface. The same
287-project survey names the pattern its author sees everywhere:

> **big model → Jev → code → Jev → tool → Jev → big model**

Jev never writes the code and never reasons deeply. It chooses, filters, routes, gates,
and judges whether an agent's completion claim is supported.

#### The most valuable negative result

[u/Obside_AI, r/ai_trading](https://www.reddit.com/r/ai_trading/comments/1wkq4lt/):
**731 trades over 24 hours, −3.15% (−$3,150), of which ~$1,650 was fees; 21% of trades
profitable after fees; 779 ms average per decision.** The setup was unusually fair to the
model — twelve indicators, candle history, **explicit trading costs handed to it** and the
question posed directly as whether a move was likely to clear them, >50% probability to
enter, a hard $500 stop imposed outside the model. It churned anyway. The author's summary:
*"So far, it's very efficient at generating commissions."*

The question they could not answer in 24 hours is this project's question, asked in
production: **"whether higher probabilities actually lead to better trades after costs."**

And the paired anecdote cuts the other way —
[u/artguerilla](https://www.reddit.com/r/accelerate/comments/1wn1gck/), 278 upvotes, same
week, same model class: *"So far it actually looks promising. Which probably means I've
misunderstood something and will discover it in 6 hours."* **Two retail trading
experiments reached opposite conclusions in the same week. Neither is evidence, and the
two are not averaged here** — which is precisely why this project's single-seed caveat
is the correct standard rather than excessive hedging.

#### Prior art: recorded, not claimed

The largest thread in the space — [3.3k upvotes, 98% upvoted, 317
comments](https://www.reddit.com/r/LocalLLaMA/comments/1wijo3e/) — is a prior-art claim
citing arXiv **2503.23303** (Mar 2025) and **2510.01237** (Sep 2025) against TypeSafe's
launch. **I could not verify either identifier: the arXiv API returned an empty feed for
2503.23303.** It is recorded as an unverified claim and supports nothing. The same author
shipped Laya, which is unambiguously a real contribution and weakens — though does not
dissolve — the self-interest in the priority claim.

#### The ecosystem census: 693 projects, and what they are not

The [Awesome Jev radar](https://raw.githubusercontent.com/logicrw/awesome-jev-projects/main/README.md)
holds **693 curated projects across 18 categories**, each with a standardised provenance
block naming *where* the decision happens. It is the largest census of this field and
the most useful single artefact found across three research passes.

**The category distribution is the finding.** Grouping the 18 categories:

| Grouping | n | share |
|---|---:|---:|
| **Orchestration, plumbing, safety, evaluation** (frameworks, routing, MCP, guardrails, context GC, CLI, eval) | **387** | **56%** |
| Demos, verticals and toys (domain tools, games, creative, voice) | 145 | 21% |
| **Classification-class work** (classification, code navigation, decision tools) | **41** | **6%** |
| — of which filed under `Classification` specifically | **2** | **0.3%** |

**Only 2 of 693 projects are filed under Classification.** A field whose entire pitch is
"a small model that classifies" has spent its first weeks overwhelmingly building
**plumbing around classifiers** — adapters, routers, MCP servers, guardrails — not
classifiers. That is the strongest corroboration available of the practitioner consensus
on the largest thread (*"smarter move if you have an eval set is to just train a
classifier and call it a day"*), and it is also the honest boundary on this project: a
350M decision model is 6% of the field, and **the 56% is the integration work that
decides whether it is usable at all.**

#### The census cannot substantiate the field's central claim

Every entry carries a provenance block, and two caveats recur **verbatim** across it.
Counted exactly over the raw README:

- **376 of 693 entries (54%)** are marked: *"performance and cost benefits have not been
  independently verified."*
- **352 of 693 entries (51%)** say: *"consult the source for the exact decision policy"* —
  the reviewer could not confirm **what decision the project makes**.

**In a category whose entire value proposition is latency, the largest census of it
cannot substantiate the latency claim in the majority of its own entries.** That is not a
criticism of the directory — it is being *more* honest than the field. It is a
measurement of the field: **"20-200x faster" has essentially not been independently
reproduced by independent people in six weeks.**

The directory's discipline is worth borrowing outright. It notes per project: *"arbitrary
task generalization is not claimed"*; *"Author demo numbers were not retested here"*;
*"supports persistent sessions and UI readback, which does not by itself prove database
persistence"*; *"GitHub SPDX is empty; the LICENSE file is MIT"*. Licences are recorded
per entry and **"Not declared"** is used honestly.

**The genuinely independent performance evidence in the entire corpus is one number:**
Vercel Labs' `json-render` at 3.21 s → 0.88 s. Everything else is either self-reported by
the project author or explicitly unverified by the directory.


### SQ3 — Liquid AI LFM2 / LFM2.5, exactly

**Claim class: hard fact from config files and the technical report.**

#### The architecture, and why it exists

From the [LFM2 Technical Report (arXiv 2511.23404, 28 Nov 2025)](https://arxiv.org/abs/2511.23404),
quoted:

> "Using hardware-in-the-loop architecture search under edge latency and memory
> constraints, we obtain a compact hybrid backbone that combines gated short
> convolutions with a small number of grouped query attention blocks, delivering up to
> **2x faster prefill and decode on CPUs** compared to similarly sized models."

*Hardware-in-the-loop* is the phrase that matters. The layer mix was chosen by measuring
latency and memory on real devices, not by proxying a loss. That is why a 350M LFM2.5
beats much larger models at structured extraction: the architecture was optimised for
the operation a decision model actually performs — one prefill, no decode.

The report also documents the training pipeline: a **tempered, decoupled Top-K knowledge
distillation objective that avoids support mismatch**, **curriculum learning with
difficulty-ordered data**, and a three-stage post-training recipe of SFT, **length-
normalised preference optimisation**, and model merging. 10–12T tokens for the LFM2
generation.

#### LFM2.5-350M, measured from the shipped config

```
architectures            Lfm2ForCausalLM
hidden_size              1024
intermediate_size        6656          block_ff_dim 6656
num_hidden_layers        16
num_attention_heads      16
num_key_value_heads      8             (grouped-query attention, head_dim 64)
conv_dim                 1024
conv_L_cache             3             (short-convolution state, 3 positions)
vocab_size               65536
max_position_embeddings  128000
rope_theta               1000000.0
norm_eps                 1e-05
block_use_swiglu         true
dtype                    bfloat16
layer_types              10 × conv, 6 × full_attention
```

**Measured parameter count: 354,483,968.** Card-stated: 28T training tokens, 32,768
context, knowledge cutoff mid-2024, 9 languages.

**Our head on this backbone: `1024×256 + 256 = 262,400` parameters = 0.0740% of the
network.** Verified by running it, not by arithmetic alone.

Note `conv_L_cache: 3`. This is the detail that bit the RCLD project: forking a batch
requires copying **both** the attention KV cache and the short-convolution state, and
`DynamicCache`'s generic `batch_repeat_interleave` does not handle LFM2's
`LinearAttentionLayer.conv_states`. RCLD's `fork_cache` deep-copies and calls
`reorder_cache` instead, and documents the constraint precisely.

#### The family, and which members can take media

| Checkpoint | Params | Hidden | Context | Media |
|---|---|---|---|---|
| LFM2.5-350M / -Base | 350M | 1024 | 32,768 (128k pos) | **none** |
| LFM2.5-2.6B | 2.69B | 2048 | 131,072 | none |
| LFM2.5-8B-A1B (MoE) | 8B-A1B | 2048 | 128,000 | none |
| LFM2.5-VL-450M | 450M | 1024 | 32,768 | **image** |
| LFM2.5-VL-1.6B / -3B | 1.6B / 3.1B | 2048 | 32,768 | image |
| LFM2.5-Audio-1.5B | 1.5B | — | 32,768 | **audio (speech)** |
| LFM2.5-Encoder-350M / -230M | 350M / 230M | 1024 | 8,192 | none, **bidirectional** |

**The most common early mistake: the plain text LFM2.5 checkpoints have no media
tower.** For image you must start from a VL checkpoint.

LFM2.5-2.6B's layer composition is 30 layers = 22 convolution + 8 GQA, 34T training
tokens, vocab 128,000, `rope_theta` 1e7. LFM2.5-8B-A1B is `Lfm2MoeForCausalLM`: 24
layers, 2 dense + MoE, 32 experts, 4 per token, `moe_intermediate_size` 1792, 128,000
vocab.

#### Measured speed (Liquid's own numbers, 1K prefill / 100 decode)

| Device | Runtime | Prefill tok/s | Decode tok/s | Peak mem |
|---|---|---|---|---|
| AMD Ryzen AI Max 395+ | CPU, llama.cpp Q4 | 2.9K | 313 | 434 MB |
| Snapdragon 8 Elite | NPU, RunAnywhere Q4 | 2.8K | 15 | 169 MB |
| Snapdragon 8 Elite | GPU, RunAnywhere Q4 | 5.3K | 62 | 81 MB |
| Apple A18 Pro | GPU, Mirai bf16 | 1953 | 73 | 945 MB |
| Apple M1 Max | GPU, Mirai bf16 | 10.4K | 328 | 940 MB |
| Apple M5 Max | GPU, Mirai bf16 | 44.8K | 564 | 1 GB |
| iPhone 13 mini | CPU, Cactus int8 | 496 | 88 | 56 MB |
| Google Pixel 6a | CPU, Cactus int8 | 208 | 42 | 328 MB |
| **Raspberry Pi 5** | CPU, Cactus int8 | **200** | **30** | **300 MB** |

**A decision model generates zero tokens, so the prefill column is the one that
matters.** A 500-token state on a Pi 5 reads in about 3 seconds; a short state in well
under a second.

#### The vendor endorses the premise

Liquid's own use-case-evaluation guide, quoted, because a vendor agreeing with the plan
is worth more than a third-party opinion:

> "A fair evaluation of a small model often includes a light fine-tune. Out of the box, a
> 1.2B model will trail a larger generalist on broad tasks. After task-specific
> fine-tuning, **it can match or beat that larger model on your task at lower cost and
> latency.**"

Their hardware guide names the same starting point we did — "Start with LFM2.5-350M …
it is the first model many silicon partners profile" — and its eval checklist ends with
"**Held-out examples frozen before any fine-tuning**". That is the rule that would have
saved time on this project.

#### The recommendation, which is the reason we chose this family

Liquid's own card: *"We recommend using it for **data extraction, structured outputs,
and tool use**. It is not recommended for knowledge-intensive tasks and programming."*

Compare Kev-0.8B's per-source gap table: knowledge (MMLU 0.42 vs 0.90) and paraphrase
(PAWS 0.55 vs 0.79) are where small models lose. Those are exactly the two categories
Liquid steers you away from. **The vendor's stated strength and the demonstrated weakness
line up, and the overlap is the task we want.**

Benchmark deltas from LFM2-350M → LFM2.5-350M, showing where the post-training went:
IFBench 18.20 → **40.69**, CaseReportBench 11.67 → **32.45**, BFCLv3 22.95 → **44.11**.
Data extraction and tool use improved 2-3x. That is the profile of a decision model.

#### Licensing — the full text, and why it matters

The **LFM Open License v1.0** is based on Apache 2.0 with exactly one substantive change. `[single source: the licence text itself, read in full]`
Section 5, verbatim:

> (a) The rights granted under this License for Commercial Use are conditioned upon You or
> your Legal Entity not exceeding the Threshold.
> (b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds
> the Threshold is not licensed under this Agreement.
> (c) The Threshold shall not apply to a Qualified Non-Profit Organization's use of the
> Work or a Derivative Work for Non-Commercial or Research Purposes.

with §1: *"'Threshold' shall mean annual revenue of **10 million United States dollars
($10,000,000) or more**."*

Liquid's plain-language guide confirms: commercial use is free under $10M revenue;
research and qualified non-profits have **no** threshold; you may modify; **there is no
copyleft**; you own your modifications but derivatives stay under this licence.

**We checked for a MAU threshold and there is none.** Neither is there a field-of-use
restriction, an acceptable-use list, or any use-based gating. The revenue cap is the
whole of it. (This corrects a common assumption — the licence does *not* resemble Llama's
MAU scheme.) The full asset-by-asset licence split, and the upstream Gemma 4 question,
are in **SQ7**.

**Practical consequences (summarised here, expanded in SQ7):**

A public repo that labels LFM-derived weights "MIT" is simply wrong. Our `LICENSE`
states the split explicitly and our `README.md` repeats it.

**Gemma 4, for contrast.** `google/gemma-4-12b-it` is tagged `license: apache-2.0` on
the Hub with a `license_link` to Google's Gemma 4 terms, and ships **no LICENSE file in
the repo**. Jev-Omni ships under `apache-2.0` and its card says "Apache-2.0, following
Gemma 4." Anyone redistributing Jev-Omni-derived weights should verify the Gemma 4 terms
at that link rather than trusting the SPDX tag alone.

#### SQ3 verdict

LFM2.5-350M is a well-chosen backbone: the architecture was searched for the exact
operation a decision model performs, the vendor's stated strength matches the task, and
it runs on a Raspberry Pi in 300 MB. The licence is a real constraint on weights and no
constraint at all on code.

---

### SQ4 — Turning a hidden state into a probability you can act on

**Claim class: hard fact for the definitions; reported fact for the measured effects.**

This is where most of the engineering effort in this ecosystem actually goes, and where
the published evidence is richest.

#### The scoring-rule foundation

A **proper scoring rule** is one where reporting your true belief minimises expected loss.
A **strictly proper** rule is one where *only* the true belief does. Laya states its
entire training method as "reinforcement learning against strictly proper scoring rules
(RLCD)", and this-that-model-1.0 was "adapted against a strictly proper scoring rule, so
the model has no way to lower its loss except by reporting what it believes — which is
what makes the probability usable as a threshold."

**The practical consequence for us: plain cross-entropy over the option distribution is
already a strictly proper scoring rule.** Our trainer's loss is literally
`− Σ_k target_k · log_softmax(logits)_k`, with the support constrained to exactly the
declared option list. We therefore get the main theoretical benefit of RLCD *without* a
reinforcement-learning loop. The RL machinery matters when the reward is the *task's*
notion of correctness rather than the world's — this-that-model-1.0's adaptation is a
single scalar `θ(λ) = θ₀ + λΔ`, so `θ(0)` is bit-exact the prior checkpoint and rollback
is "a configuration change rather than a restore."

#### Why ECE alone is a trap

The `typed-decisions` benchmark makes the sharpest available argument. Its **Prior** row
— a model that fits each question's label frequencies on train and then answers those
frequencies for every case while **ignoring the state entirely** — has:

- ECE **0.088**, the *best* on the table
- accuracy 0.470, worse than every learned model

> "Prior also has the best ECE on the table, at 0.088, while knowing nothing. Guessing the
> base rate is perfectly calibrated by construction. That is the clearest argument for
> reading KL and Brier here instead of ECE."

Its full baseline table (all on the 2,000-decision test split):

| Model | Kind | Acc | KL↓ | Brier↓ | ECE | ms/case |
|---|---|---|---|---|---|---|
| Uniform | reference | 0.308 | 0.444 | 0.238 | 0.169 | 0 |
| **Prior** (ignores input) | reference | 0.470 | 0.347 | 0.189 | **0.088** | 0 |
| MiniLM-L6 (22M) | specialist | 0.587 | 0.262 | 0.143 | 0.108 | 22 |
| ModernBERT-base (149M) | specialist | 0.646 | 0.223 | 0.119 | 0.179 | 349 |
| Perfect scenario understanding | ceiling | 0.704 | — | — | — | — |
| Teacher self-agreement | ceiling | 0.735 | — | — | — | — |
| TypeSafe Jev 1.13.0 | generalist | 0.727 | 1.442 | 0.148 | 0.144 | 710 |
| meraGPT Decider 1 | generalist | 0.768 | 0.096 | 0.052 | 0.180 | 526 |

Read the reference points carefully, because they are the whole point of the table:
**0.52 is the floor, around 0.70 is strong, around 0.75 is saturation**, and *"a score
much above 0.75 means a model has learned the teacher's quirks rather than the task."*
Per-question ceilings range from 0.560 to 0.937.

**Jev's KL of 1.442 is the outlier in that column** — 4x the Prior's, and 15x Decider 1's.
Accuracy alone hides that. This is exactly the kind of thing the ecosystem argues about.

#### The soft-target result, and the caveat that complicates it

The same benchmark reports that Adaptive Classifier trains on hard labels, so the gold
distribution is normally thrown away. Their fix: enter each case **four times**,
apportioned across labels in proportion to its gold, carrying the soft target into a
learner that cannot represent one. The measured effect:

> "That single change cut **KL by a third** and **score MAE by 15%**, while barely moving
> accuracy. The argmax was already right. What improved was the shape of the predicted
> distribution."

But Jebadiah's own log records that a 14,714-question *synthetic* soft-labelled pool made
their 9B **worse**. Both can be true: soft targets help when the distribution is good, and
hurt when it is not. Prefer human or public labels; use synthesis to cover gaps.

#### The ordinal kernel, the highest-leverage single decision

For `score` questions, put 20% of the mass on each adjacent level. Our
`_ordinal_target(2, 5)` returns `[0, 0.2, 0.6, 0.2, 0]`.

Jebadiah measured the effect on human helpfulness ratings, moving their 9B from
Decision Score **-21.4 to +11.9** and calibration error from **0.39 to 0.045**, with
rubric accuracy **unchanged**. A negative Decision Score means *worse than guessing the
label base rates*.

What changed was not accuracy. It was honesty. That is the correct trade for a decision
model, because the consumer is a routing policy.

#### Fitting the temperature: against the train target, not the argmax

Jebadiah ships both fits in `temperatures.json`:

| Fit | Objective | choice | noul | score | Effect |
|---|---|---|---|---|---|
| `hard` | NLL vs **argmax label** | 0.68 | 0.96 | 0.83 | **sharpens** |
| `train` | NLL vs the **soft/ordinal target** | 1.12 | 1.33 | 1.20 | **softens** |

They ship `train`, and explain: *"the ordinal target is what makes rubric probabilities
honest, and the `hard` fit is exactly the sharpening that made v0's HelpSteer2
calibration worse."*

The price is visible in the same file: measured against **hard** labels on the
calibration split, ECE moves 0.107→0.111 (choice), 0.086→0.100 (noul), 0.060→0.098
(score) while NLL on the training target falls. Their note: *"a consumer who gates on the
argmax may prefer `hard`, and either way should refit on its own data."*

Our `fit_temperature` takes logits and targets and grid-searches T over [0.25, 5.0] in 96
points, minimising NLL. **Verified: it recovers a planted T = 2.5 as 2.5000.**

A grid rather than LBFGS, for a reason: NLL in T is smooth but nearly flat near the
optimum, a grid is dependency-free and reproducible, and 96 evaluations cost well under a
millisecond.

#### Calibration does not transfer

Kev-0.8B, same checkpoint, two regimes: in-distribution ECE **0.033**, out-of-domain ECE
**0.049** with Brier **0.430**. Their own summary: *"Probabilities are usable in-domain;
treat them as advisory elsewhere."* And their confident-error rate: 0.2-0.3% in-domain
against **3.7%** for Jev — but coverage at a 5% error budget is **0.145** for Kev-0.8B
versus **0.70** for Jev.

**That is the number to internalise.** A model that is confidently wrong less often but
can only automate 14.5% of decisions is not obviously more useful than one that automates
70%.

And the MLX 4-bit Jev-Omni card records a calibration *regression* from its own
temperature fit: ECE improved on the fit split but **got worse held out** (0.06261 →
0.06774). Always report calibration on a split you did not fit on.

#### What the checked-in calibrators actually do

Three calibration artefacts are published as files rather than prose, and reading them
changed our implementation. This is the most actionable material in SQ4.

**Temperature depends on the option count, not only the question type.**
`heman10x/rlcd-modernbert-151m` ships a `calibrator.json` with a temperature per
cardinality `[single source]`:

| k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=9 | k=11 | k=17 | k=25 | global |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.0069 | 5.0069 | 4.0314 | 3.0560 | 2.3898 | 2.3898 | 1.6668 | 3.3919 | 1.7200 | 1.5144 | 2.8039 |

**A 3.3× spread driven by option count alone.** A single global T is 3.3× too wrong at
the ends of the range. (k=11 is non-monotone — per-k fits are themselves noisy at these
sample sizes, which is the motivation for the shrinkage below.)

**Small groups must shrink toward the global value, in log space.**
`Jev-Style-0.8B-Decision-v3` stores, per group, the raw fit `T_raw`, the shrunk `T`, the
row count `n` and a `weight`. Two identities were checked arithmetically against the
file `[single source, independently recomputed]`:

    weight = n / (n + 100)          -- exact for all 20 groups
    T = exp( w·ln T_raw + (1-w)·ln T_global )   -- geometric, not arithmetic

Two groups reproduce to ~5 decimal places under the geometric blend
(`general|choice|3-5`: 0.856385 vs stored 0.856380; `intent|choice|11-20`: 0.853653 vs
0.853659). On the n=15 group the two blend kinds diverge decisively — geometric gives
1.01018 against a stored 1.01016, arithmetic gives 1.09567. That group wanted T=5.0 and
was rescued only by a hard clamp, which is precisely the overfit shrinkage exists to
catch.

**T can be below 1.0.** That same calibrator's global is **0.880** — the model was
*under*confident, so calibration *sharpened* it. Kev-0.8B needs 2.35. Do not assume
`T >= 1`, and do not search a grid that only goes upward.

**Refit per quantisation format.** One project ships two calibration files for the same
model: BF16 `T = 1.0409` and Q4_K_M `T = 1.0123`, with argmax agreement **0.996** and
**0.914** respectively, and *relaxed* acceptance gates for the lossy format. A
temperature fitted on one format is wrong on another.

**And the cheapest possible deployment is a bug fix.** From `Verdict-open-jev`: the engine
"previously failed to load `calibrator.json` during standalone instantiation, **running
at uncalibrated temperature 1.0**", and a scope check limited calibration to 5-candidate
queries, "leaving other cardinalities unscaled". Fixing both moved hard-tier ECE
**0.298 → 0.118 (−60.4%)** — the largest single calibration win in that project's
history, from *loading* the calibrator correctly rather than fitting it better.

#### What the objective is optimised against matters

We fit T by minimising NLL against the training target. That is right for a routing
threshold and **wrong for a prediction-set consumer**, and the distinction is documented:
Conformal Temperature Scaling reports that calibration parameters "when optimized with
**cross-entropy loss**, might counteract the goal of generating efficient prediction
sets" `[dated: 2024]`. If your downstream use is a set rather than a threshold, fitting
by NLL optimises the wrong thing. Say which one you built for.

#### SQ4 verdict

Calibration is the product. Cross-entropy (a strictly proper scoring rule) is the
objective; the ordinal kernel for rubrics is the biggest single win; the temperature must
be fitted per question type against the soft target; and none of it transfers to your
data without a refit. Report KL and Brier next to ECE, and report coverage at your error
budget above all.

### SQ5 — Small-model multimodal, honestly assessed

**Claim class: hard fact for encoder dimensions; interpretation for the recommendation.**

The honest answer to "can a 350M model do what Jev-Omni does across four modalities?"
is **partly, and the vision path is the only one where Liquid has already done the work
for you.**

#### The token budget is the real constraint

Jev-Omni's vision tower emits up to **280 soft tokens per image** (`num_soft_tokens: 280`,
`model_patch_size: 48`, `patch_size: 16`, `pooling_kernel_size: 3`). Video is **16 frames,
each passed as a separate image item** — so 16 frames is up to **4,480 visual tokens**
before a single word of text.

That is the crux:

- against Gemma 4-12B's **262,144**-token context: negligible
- against LFM2.5-350M's **32,768**-token context: **13.7%** of the window

**The smaller your text model, the more painful the visual budget becomes.** That is not
an argument against going small — it is an argument for using a vision checkpoint that
already solves it.

#### Liquid already shipped the answer: LFM2.5-VL-450M

`LiquidAI/LFM2.5-VL-450M` is LFM2.5-350M plus a **SigLIP2 NaFlex 86M** vision tower. Its
config, read directly:

```
text_config    hidden_size 1024, 16 layers, vocab 65536   (the same 350M)
vision_config  hidden_size 768, 12 layers, 12 heads, patch 16, num_patches 256
projector_hidden_size 2048, downsample_factor 2
min_image_tokens 64, max_image_tokens 256, min_tiles 2, max_tiles 10
tile_size 512, use_thumbnail true, image_token_id 396
```

So the projector already lands in your 1024-wide backbone. You do not have to invent one.
Benchmarks (VLMEvalKit): MMStar 43.00, RealWorldQA 58.43, MMBench 60.91, POPE 86.93,
BLINK 43.92, RefCOCO-M 81.28, OCRBench 45.00, MMMB 68.09.

**And there is a task-specific sibling that is the strongest available evidence a sub-1B
multimodal model can do real work.** `LiquidAI/LFM2.5-VL-450M-Extract`, on a 2,000-sample
(image, schema, JSON) benchmark with frontier-model reference labels:

| Model | Params | JSON validity | Schema F1 | VLM judge |
|---|---|---|---|---|
| **LFM2.5-VL-450M-Extract** | **0.45B** | **98.9** | **98.8** | **84.5** |
| LFM2.5-VL-450M | 0.45B | 97.7 | 93.5 | 73.4 |
| SmolVLM-500M-Instruct | 0.51B | 33.0 | 26.6 | 12.2 |
| FastVLM-0.5B | 0.76B | 22.5 | 19.3 | 16.3 |
| Qwen3.5-0.8B | 0.87B | 96.4 | 96.3 | 82.3 |
| InternVL3.5-1B | 1.06B | 98.0 | 96.5 | 80.7 |
| Qwen3.5-2B (ref) | 2.27B | 97.9 | 97.7 | 89.7 |
| gemma-4-E2B-it (ref) | 2.3B | 97.4 | 97.1 | 84.4 |

Note the contrast: generic small VLMs (SmolVLM 33.0, FastVLM 22.5) **cannot** do
schema-based extraction zero-shot, at any prompt. A 450M model fine-tuned for the task
beats a 2.3B model not fine-tuned for it. **Task-specific post-training is doing the work,
not scale.** That is the single most transferable finding in this section.

#### Audio: no small Liquid option

LFM2.5-Audio-1.5B is 1.5B, **English only**, and built as an end-to-end
speech-to-speech model (FastConformer audio encoder + RQ-transformer + Mimi detokenizer) —
i.e. for generation, not decisions. There is no small Liquid audio decision model. You
would be building a new audio tower.

If you must, match Jev-Omni's convention exactly so your numbers are comparable:
`ffmpeg -i IN -t 30 -ac 1 -ar 16000 out.wav`. Jev-Omni's audio config gives
`audio_samples_per_token: 640` at 16 kHz — **25 audio tokens per second**, 750 tokens for
the 30 s cap.

For scale: whisper-tiny is 39M (`d_model` 384, 4 encoder layers), whisper-base 74M
(`d_model` 512, 6 layers). Either projects into 1024 with one linear layer. But note the
whisper encoder's `max_source_positions` is **1500** — it consumes only 15 s per forward
pass, and longer audio is chunked.

#### Video: 6x the cost of text, and worse for you

Jev-Omni's own latencies: 83 ms text, 26 ms image, 504 ms 16-frame video. Video is **6.1x**
text at 12B scale. For a 350M model with a 32,768 window and 4,480 visual tokens for
16 frames, the amortisation is strictly worse.

#### The benchmarks, in context

**MMAU** (arXiv 2410.19168): 10,000 expert-annotated questions, 27 tasks
(11 information-extraction, 16 reasoning), speech + environmental sound + music, split
10:10:7 across Speech:Music:Sound, difficulty 22/56/22% easy/medium/hard, average audio
10.14 s. Reference points: **Gemini Pro v1.5 52.97%, Qwen2-Audio 52.50%.** Error analysis
found perceptual errors dominate (Qwen2-Audio 55%, Gemini 64%). Jev-Omni reports **63.10%**
on the 1,000-question `test-mini` split. `test-mini` audio is ~1.2 MB per clip and
`test` is released without answers.

**MVBench** (arXiv 2311.17005): 20 temporal tasks the authors say "cannot be effectively
solved with a single frame," 4,000 questions (20 configs × 200), videos from 11 sources
filtered to 5–35 s. The official harness samples **8 segments by default** at 224px with
`decord`, `num_frames=4`. Jev-Omni reports **53.10%** on 14 tasks / 2,786 questions.

**A caveat that matters:** Jev-Omni uses 16 frames; the official MVBench harness defaults
to 8. Its 53.10% is **not** directly comparable to published MVBench leaderboard numbers.

#### SQ5 verdict

**Image: yes, and cheaply — start from LFM2.5-VL-450M and do not build a projector.**
**Audio: only if you accept building a tower; there is no small Liquid option.**
**Video: last, or never.** Ship text-only first; it is a genuinely useful product on its
own, and that is what every project in this ecosystem actually ships.

---

### SQ6 — Toolchain and cost

**Claim class: hard fact for quantisation tables and command syntax.**

#### The toolchain, in order

| Stage | Tool | Version note |
|---|---|---|
| Fine-tune | `peft` LoRA, `trl` SFTTrainer | peft ≥ 0.21 |
| Quantise for training | `bitsandbytes` 4-bit (QLoRA) | not needed at 350M on 24 GB |
| Convert to GGUF | `convert_hf_to_gguf.py` | `--outtype bf16 --remote <repo>` |
| Quantise GGUF | `llama-quantize` | see ladder below |
| Serve CPU | llama.cpp | day one for LFM2 |
| Serve Apple | MLX | Liquid ships `-MLX-8bit`, plus 4/5/6-bit and bf16 |
| Serve Windows/NPU | ONNX Runtime | Liquid ships `-ONNX` exports |
| Serve Intel | OpenVINO | Liquid ships int8 |
| Serve GPU | vLLM, SGLang | high concurrency |

Liquid's documented LoRA recipe (`r=16, lora_alpha=32, lora_dropout=0.05`,
`target_modules=["q_proj","k_proj","v_proj","o_proj"]`, `lr=2e-4`, batch 4 × accum 4) is
tuned for *generative* SFT. For a decision model you want `target_modules="all-linear"`
instead: LFM2.5 is hybrid, and 10 of its 16 layers are convolutions with their own
projections, so an adapter touching only q/k/v/o leaves most of the network frozen. PEFT
documents `all-linear` as the QLoRA-equivalent sweep, and Kev's trainer picks targets from
the model config for the same reason.

#### The GGUF quantisation ladder (measured on Llama-3.1-8B, llama.cpp)

| Type | bits/weight | Size (8B) |
|---|---|---|
| IQ2_M | 2.1460 | 2.01 GiB |
| IQ4_XS | 4.4597 | 4.17 GiB |
| Q4_K_S | 4.6672 | 4.36 GiB |
| **Q4_K_M** | **4.8944** | **4.58 GiB** |
| Q5_K_M | 5.7036 | 5.33 GiB |
| Q6_K | 6.5633 | 6.14 GiB |
| Q8_0 | 8.5008 | 7.95 GiB |
| F16 | 16.0005 | 14.96 GiB |

For a 350M model: Q4_K_M ≈ 210 MB, Q8_0 ≈ 350 MB, F16 ≈ 700 MB.

**But do not quantise the head.** It is 262,400 fp32 parameters ≈ 1 MB. Leave it fp32 —
it is the cheapest possible insurance on your calibration, and quantising it buys nothing.

llama.cpp's own guidance on multimodal projectors generalises: *"Multimodal components are
usually much smaller than the LLMs they come with. In addition, their quality has a direct
impact on the quality of LLM generations... For these reasons, multimodal components are
usually kept in a high-quality format such as bf16 or q8."*

**And quantisation can break calibration outright.** The 4-bit Jev-Omni GGUF card says:
*"this Q4 matched the winning option on 3/4 published text verification examples; the
maximum absolute probability difference was **0.210**. Do not assume source-equivalent
probabilities or calibration."* Re-measure ECE and Brier after quantising, and refit the
temperature for the quantised model.

#### What things actually cost

**Jev-Omni has no bill.** Its card prices it as a proxy: its recorded input tokens at
OpenRouter's Gemma 3 12B input rate of **$0.05/M**, generating no output tokens. An
independent GGUF conversion measured a 4-bit footprint of 6.87 GiB for the backbone plus
116 MiB for the projector, 4.95 bits/weight overall, and reported **1.18 s per six-question
request on an RTX 5080 Laptop GPU (0.20 s/decision)** and **11.30 s on a Ryzen AI 9
HX 370 CPU**.

**Real Jev** is $0.042/M input tokens. JevBench measures cost **per 1,000 decisions**:
Jev reads ~950 input tokens per decision, so 1,000 decisions cost
`950 × 1000 × 0.042 / 1e6 = $0.0399`.

**DecisionBench's caution, which is the most useful paragraph in this area:**

> "The Cost column is US dollars per 1,000 DECISIONS, not per 1,000 tokens. One decision is
> a whole question: its state, its rubric and its options — hundreds to thousands of input
> tokens."

And the request-shape correction they had to make to their own chart:

> "This corrects an earlier version of the chart, where Jev 1.13 was run per state while
> Jev-Omni was already run per question. Batching amortised one copy of the state across
> every question on it, so on medium's 3.7 questions per state it made Jev look about three
> times cheaper than it is."

They then measured the true multiplier: splitting a state into per-question calls
multiplies input tokens by **2.82x** on medium and **3.09x** on hard.

**A 350M model on a Pi costs electricity.** Per this-that-model-1.0's own framing, the
honest claim is **one order of magnitude, not five** — their cost is electricity at 80 W
and $0.30/kWh, a different kind of number from a price covering serving and margin.

**The real advantages are latency, privacy and independence**, and those are not
contestable: tens of milliseconds on a laptop, no network egress, no per-token bill, and
no dependency on a vendor's API or its pricing.

#### Distillation: the one result that most directly justifies this whole project

**Claim class: reported fact (vendor-partner case study), cross-checked for internal consistency.**

There is a case study, run by Liquid AI's named fine-tuning partner, of LFM2.5-350M
fine-tuned on synthetic data generated by a **GPT-oss-120B teacher** on three
structured-output tasks. It is the closest published evidence to "can a 350M model
replace something much larger for typed decisions?"

| Task | Teacher (120B) | LFM2.5-350M base | **LFM2.5-350M tuned** |
|---|---:|---:|---:|
| Shell command execution (Gorilla) | 97.03% | 61.4% | **98.0%** |
| Smart home control | 92.11% | 63.2% | **96.7%** |
| Banking voice assistant | 96.95% | 34.5% | **95.9%** |

**The 350M student exceeds the 120B teacher on two of three tasks** and comes within
1.1 points on the third. The base model was far below the teacher (34.5–63.2%), so this
is not a case of a strong base coasting.

The pipeline is worth copying exactly, because the middle step is the one people skip:

    1. define the task with a prompt and 20-100 seed examples
    2. generate synthetic training data with a large teacher model
    3. **validate and filter that data**
    4. fine-tune the student

Epoch-by-epoch, the gain is nearly all in **epoch 1** (Gorilla 61.4% → 98.0% → 97.0% →
98.0% → 98.0%). That is direct support for Kev's and Jebadiah's shared finding that one
or two epochs is right and more overfits.

The same post reports a 5,000-example synthetic set for the shell task, and a
compounding-effect calculation worth internalising: at 63% per-call accuracy, a
five-turn conversation succeeds at roughly `0.63^5 ≈ 10%`.

It also reports a head-to-head against Google's FunctionGemma (270M): LFM2.5's **base**
scores 2–6× higher before any fine-tuning (Gorilla 61.4% vs 9.9%), and the tuned models
are comparable. So the architecture choice, not just the fine-tune, is doing work.

And a mechanistic note that matters for a hybrid model: replacing most attention layers
with zero-cache convolution blocks **cuts the KV cache by up to 90%**, which is the
specific reason LFM2 is fast on devices with little RAM.

#### Distillation as a *training objective* in this ecosystem

Three distinct mechanisms appear, and they are easy to confuse:

1. **Synthetic-data distillation** (above). A teacher generates examples; the student
   trains on them with ordinary cross-entropy. This is what most people mean and it is
   what the Distil Labs pipeline does.
2. **Logit/soft-target distillation.** `kushalpatil/jevify` trains with
   `KL(target ‖ softmax(label_logits)) + mass_weight × (−log P(any label token))` — a
   KL against a teacher distribution, plus a term that penalises putting total mass on
   one option. That mass term is a sensible guard against a collapsed readout.
3. **Closed-form capability transfer.** `this-that-model-1.0` adapts from `decider-2b`
   against a strictly proper scoring rule in a **single scalar**,
   `θ(λ) = θ₀ + λΔ`, such that `θ(0)` is bit-exact the prior checkpoint. Nokia's AnyJev
   goes further: a closed-form head per question, 100–300 labels, no gradients at all.
   Both make "rollback is a configuration change rather than a restore".

Liquid's own LFM2 training pipeline uses a **tempered, decoupled Top-K knowledge
distillation objective that avoids support mismatch** (arXiv 2511.23404), and LFM2.5-2.6B's
post-training includes a **multi-domain on-policy distillation** stage. None of that
recipe is published in enough detail to reproduce — flagging that as a gap.

#### Quantisation: what the measurements actually show

The bits/weight ladder is measured (llama.cpp, Llama-3.1-8B): Q4_K_M **4.8944**,
Q5_K_M 5.7036, Q6_K 6.5633, Q8_0 8.5008, F16 16.0005. For a 350M model that is
~210 MB at Q4_K_M and ~350 MB at Q8_0.

But three independent published results say **quantisation is not free, and it is
format-specific**:

- The Q4_K_M conversion of Jev-Omni reports a **maximum absolute probability difference
  of 0.210** against the fp32 source, and says plainly: *"Do not assume source-equivalent
  probabilities or calibration."*
- One project refit its temperature **per format**: BF16 T = 1.0409, Q4_K_M
  T = 1.0123, with argmax agreement **0.996 vs 0.914** and *relaxed acceptance gates*
  for the lossy format.
- A model trained against a 4-bit NF4 base and served in bf16 saw **ECE roughly double,
  0.036 → 0.069** — the adapter had partly learned to compensate for quantisation it
  would no longer face.

And a fourth, about architecture: Nokia reports that for their linear head, **fp8 "buys
 single-question latency and costs accuracy"** — they recommend against it.

#### Speculative decoding: exact, but not for decision models

Liquid's DSpark drafter is a 279.5M model (4 full-attention layers, hidden 2048, plus a
rank-256 Markov head and a confidence head) that makes LFM2.5-VL-3B decode **2.66× faster
on one H100** (3.13× on an M5 Max with MLX-VLM, 2.14× on an M3 Ultra with llama.cpp),
with 3.2–4.6 draft tokens accepted per verification pass. The property that makes it
legitimate: **speculative decoding is exact under greedy decoding** — "the target
verifies every proposed token… You get the speedup, not a different model."

**This does not apply to us.** A decision model generates zero tokens, so there is
nothing to speculate about. Recorded here to close the topic, not because it helps.

#### What we measured: does LoRA actually help?

**The load-bearing assumption, tested rather than assumed.** Every recommendation in
this report follows from the claim that you must fine-tune the backbone. That claim was
inherited from Kev, Jebadiah and Jev-Omni and had never been tested here. Identical
data, splits and evaluation; read-out at the final layer; the only variable is whether
the backbone's weights can move.

**Scope: single seed (17), not replicated.** A seed-42 replication was launched and is
blocked by host load (load average 93); no result is claimed from it. Read the accuracy
and temperature figures as order-of-magnitude. The coverage gap is the least noise-prone
claim, since a zero at n=300 is hard to produce by accident
(`notes/12-seed-robustness-pending.md`).

| arm | train Q | eval n | accuracy | Brier ↓ | ECE ↓ | fitted T | cov@5% | cov@20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| customer_service, frozen | 360 | 300 | 0.3233 | 0.1404 | 0.1467 | **3.300** | **0.0000** | 0.0000 |
| customer_service, **LoRA** | 360 | 300 | **0.5633** | **0.1098** | **0.0646** | **1.050** | **0.0967** | **0.3367** |
| invoice_processing, frozen | 240 | 200 | 0.4700 | 0.1502 | 0.0765 | **4.150** | 0.0300 | 0.0000 |
| invoice_processing, **LoRA** | 240 | 200 | **0.6450** | **0.1204** | **0.0678** | **0.850** | **0.0550** | **0.2750** |
| all 4 workflows, frozen | 1212 | 996 | 0.4357 | 0.1244 | 0.1040 | 1.800 | **0.0000** | — |

**Claim class: our own measurement.** LoRA is worth **31.1 accuracy points** here. For
scale, `typed-decisions` puts a base-rate `Prior` at 0.470; arm B clears it and arm A
does not.

**The more informative number is the fitted temperature: 3.300 → 1.050** (and 4.150 →
0.850 on `invoice_processing`). A large T means
the raw logits are far more confident than the accuracy justifies. So a frozen
head-only model does not merely score worse, it scores *dishonestly* worse — the
probability a caller would threshold on does not mean what it appears to mean. This is
the concrete mechanism behind Jebadiah's reported observation that "accuracy was flat
while ECE fell 0.086 → 0.019: the model did not learn to be right more often, it
learned to be honest about when it isn't."

**The decisive column is coverage.** A frozen backbone and head has coverage **0.0000**
at a 5% error budget in two of three configurations and 0.0300 in the third — no
threshold is both useful and safe. LoRA reaches 0.0967 and 0.0550. At n = 300 a 5%
budget permits 15 errors, so unlike the earlier n = 45 probe this is measurable.

**Generalises on direction, not magnitude.** Two workflows: large accuracy delta,
grossly overconfident frozen arm (T = 3.300 and 4.150), temperature near 1 after
tuning (1.050 and 0.850). But coverage at a 5% budget is 9.7% on one workflow and 5.5%
on the other. **A threshold tuned on one workflow will not transfer.**

**More data does not rescue the frozen arm.** At 1,212 training questions across four
workflows it reaches 0.4357 accuracy and T 1.800 — miscalibration roughly halved — and
coverage@5% is still 0.0000 across 996 evaluation questions. Data quantity and backbone
fine-tuning are not substitutes.

**Weight:** 240–1,212 training questions, 3 CPU epochs, against a published norm of
10k–24k. The claim is the **delta between arms** and the coverage gap, not the endpoint.

**One inconsistency between our own experiments, disclosed.** The read-out layer sweep
read *pre-norm* activations (hooking a layer module) while this experiment reads
*post-norm* (calling the backbone, which applies the final RMSNorm). That is why
frozen+head scores 0.2667 here versus 0.4889 in the sweep. The LoRA conclusion is
unaffected — both arms here use the identical path — but the two experiments must not
be compared to each other directly, and the sweep's absolute figures should be read as
pre-norm. Its layer *ranking* is unaffected, because all five layers were read the
same way. Full write-up: [`notes/05-lora-ablation-results.md`](notes/05-lora-ablation-results.md).

#### SQ6 verdict

The toolchain is unremarkable and fully available: LoRA → merge → GGUF → llama.cpp, with
day-one LFM2 support and a four-way ladder of runtimes.

The two things to actually take away: **a 350M model fine-tuned on filtered synthetic
data from a 120B teacher can match or beat that teacher on structured output** — which is
the strongest published support for the project premise — and **quantisation is
format-specific and can silently break your calibration**, so refit the temperature per
format and re-measure. Do not quantise the head. Report cost per decision, never per
token, and always state the request shape.

---

### SQ7 — Licensing and honest reporting

**Claim class: hard fact. Both licence texts were read in full, not paraphrased from a
model card.** The LFM licence was covered in SQ3; this section collects the licensing
consequences for *this project specifically*, plus the reporting discipline, because
the two are the same problem: what you are allowed to claim, and what you are allowed
to publish.

#### What the LFM licence permits, precisely

The text is Apache 2.0 with exactly one substantive change. Section 1 defines the
threshold `[single source: the licence text itself, read in full]`:

> "'Threshold' shall mean annual revenue of **10 million United States dollars
> ($10,000,000) or more**."

and Section 5 conditions commercial use on not exceeding it, with a carve-out for
qualified non-profits on non-commercial and research use. Everything else is
Apache-standard: perpetual, irrevocable, worldwide, royalty-free, no-charge,
non-exclusive, with §4 imposing attribution on redistribution and §11 terminating
automatically on breach.

**We checked for the restrictions people assume and did not find.** There is **no MAU
threshold**, no use-based gating, no field-of-use restriction, and no acceptable-use
list. The revenue cap is the whole of it. (This corrects an assumption made earlier in
this research, which described the licence as resembling Llama's MAU scheme. It does
not. `[dated: corrected 2026-09-26]`)

Liquid's plain-language guide agrees and adds two things the legal text does not spell
out: modified models remain under the same revenue threshold, and *"You own your
modifications. Liquid AI owns the base models, and derivative models remain subject to
this license."*

#### The four-way licence split for a project like this

| Asset | Licence | Consequence |
|---|---|---|
| Training / inference / eval **code** | **MIT** | Fully permissive |
| Trained **weights** derived from LFM2.5 | **LFM Open License v1.0** | Not MIT, not OSI-approved; $10M revenue cap; must ship the licence, retain attribution, mark modified files |
| Base **weights** (LFM2.5-*) | LFM Open License v1.0 | As above |
| Prompt/template and the typed-decision *pattern* | Unencumbered | A design pattern is not copyrightable; the concrete implementation is |

**No copyleft.** Fine-tuned weights may be kept proprietary. The practical consequence
is that a repository labelled "MIT" above a directory of LFM-derived weights is simply
wrong, and the most likely way to get this wrong is to copy the badge from the code
repo onto the model repo. Our own `LICENSE` states the split in the file itself, and the
`README.md` repeats it, because the failure mode is plausible rather than exotic.

#### A second licence trap: the upstream base

Jev-Omni ships `apache-2.0` and its card says "Apache-2.0, following Gemma 4". The base,
`google/gemma-4-12b-it`, is tagged `license: apache-2.0` on the Hub **and ships no
LICENSE file in the repository** `[single source: the Hub API record]`. It carries a
`license_link` pointing at Google's Gemma 4 terms. The SPDX tag and the linked terms are
not obviously the same instrument, and Gemma's historical terms have included a use
policy that plain Apache-2.0 does not.

Anyone redistributing Gemma-4-derived weights — which includes anyone who redistributes
Jev-Omni's — should read the linked terms rather than trusting the tag. We flag this
rather than assert a conflict, because the linked page was not itself retrievable in
this session (see *Limitations & Gaps*).

#### Independence, and why every project in this field states it separately

The upstream TypeSafe Jev is a closed hosted product. **Every** open implementation in
this ledger states its independence in its own words, and so do we:

> "This is an independent project. It is not affiliated with, endorsed by, sponsored by,
> or derived from TypeSafe AI or its Jev model. No Jev output was used in training."

That is not boilerplate to be copy-pasted past. Three separate parties (Jev-Omni, RCLD,
Kev) each had to state it because a name collision invites the inference that they
distilled from a closed model. `this-that-model` puts the technical version of the same
point: its whole adaptation is `θ(λ) = θ₀ + λΔ` from `decider-2b`, an Apache-2.0
checkpoint, with `θ(0)` bit-exact the prior — so there is provably no distillation from
Jev anywhere in the pipeline.

#### The five reporting pitfalls, each with a documented instance

**1. Cost per token instead of per decision.** JevBench's own warning, quoted in SQ6.
Getting the unit wrong is the most common error in the field.

**2. Comparing across request shapes.** DecisionBench withdrew a chart over exactly this
and published the correction, including the withdrawn numbers. That is the standard.

**3. Specialist vs generalist.** `typed-decisions` states it plainly:

> "Train a specialist on these four workflows, score it on them, and you have measured
> architecture... It is not a comparison against a general System One model, which has never
> seen these workflows."
> "Jev at 0.727 against the specialist's 0.646 has not beaten it by eight points... Read
> the gap as the price of generality, not as a quality ranking."

Their table carries a **Kind** column (`specialist` / `generalist`) on every row for this
reason. Ours should too.

**4. Ignoring option order.** Documented at catastrophic magnitude: `open-alternative-jev`
scored **21%** with options in one order and **72%** reversed. JevBench's caveat:

> "small models are very sensitive to option order."

Kev reports option-order flip rates of 0.07–0.08 for its small models against Jev's
**0.00**.

**5. Evaluating on training data.** Kev-0.8B reports a paired JevBench hard-tier gain of
**+2.7 pp [−1.8, +7.2]** — *"not distinguishable from zero"* — while gaining **+26.9 pp**
on its in-distribution test. It also reports an eval-only source that got **worse**
(When2Call 0.233 → 0.133, below the one-in-four guessing rate) and says plainly: *"Do not
use this checkpoint for tool-call routing."*

---

#### SQ7 verdict

**Code MIT, weights LFM-licensed — and that split has to be stated, not assumed.** The
LFM Open License is Apache 2.0 plus a $10M revenue cap, with no MAU threshold and no
copyleft, so a fine-tune can be kept proprietary but can never be called MIT. The most
likely way to get this wrong is to copy a licence badge from a code repo onto a weights
repo, which is why the split is written into the `LICENSE` file itself rather than only
into a README.

**Independence is a technical claim, not a disclaimer.** Every open implementation in
this ledger states it separately, and one proves it structurally: `this-that-model`'s
adaptation is a single scalar applied to an Apache-2.0 checkpoint, with `θ(0)` bit-exact
the prior, so there is demonstrably no distillation from the closed model anywhere in the
pipeline.

**And five reporting traps, each with a documented victim — including two of the authors
of those victims.** These are not pedantry. JevBench withdrew a chart over one of them,
and the per-decision-versus-per-token confusion is the single easiest way to make a cheap
model look expensive or an expensive model look free.

---

### SQ8 — Failure modes: what actually goes wrong, and how badly

**Claim class: reported fact. Every item below is a measured failure in a published
system, not a hypothetical.** This section is separated from SQ7 because these are
different in kind: SQ7 is about how to *report*, this is about what is *wrong*.

#### And the failure nobody expected

The option-name failure (arXiv 2609.26758) deserves its own treatment because it is the
newest and most consequential result, and because **it is not an option-order problem** —
it survives permutation testing.

The experiment holds the question, the state, the rubric wording, and the *set* of option
names fixed, and changes only which name is bound to which rubric. Renaming two options
from `0`/`1` to `no`/`yes`:

| Condition | Result |
|---|---|
| 1,200 workflow decisions, 4 predicates | **70.4 more answers changed per hundred** (95% CI [67.6, 73.1]) |
| AUC | **.94 → .23** |
| Effect vs neutral-name control | **≥ 7.4x**, across all 4 predicates |
| Scaling | **stronger as option count increases** |
| Read-out geometry | a mean-pooling family flips **4.1x less often** |
| Hosted Jev | AUC **.8146 → .5806**, 24x its test-retest flip floor |
| Random character-string names | **all families return to neutral, no accuracy loss** |
| Type-error rate throughout | **0%** |

AUC below 0.5 is a *systematic ranking reversal*. The type-error rate staying at 0% is the
part that should worry you most: **a decision model can be perfectly schema-conformant
and completely wrong.** Schema conformance is not a correctness signal.

This is directly actionable, and cheaply: **run the test, and if you are affected, use
neutral or random-ish option identifiers in the prompt with the human-readable name
carried in the rubric.** It is a prompt change, not a retrain. It is also an argument for
slot logits over pointer scoring, and a reason to keep option counts at or below 20.

#### A model can read a negated rule as its opposite, confidently

The second and more alarming failure, from the same read-out family
([this-that-model 1.0 → 1.1 → 1.2 changelog](https://github.com/FLock-io/this-that-model)):

> "1.1 read `bays without chilled handling are ineligible` as though it named the
> **eligible** set — **not failing to apply the rule but applying its opposite, at 0.88
> mean confidence.**"

Across eleven phrasings of that one rule (chance 0.19) the model ranged **0.00–0.89, with
four of eleven rows *below* chance**; 1.2 fixed the range to 0.98–1.00. Below chance is not
guessing, it is systematically wrong in one direction. And this is the model that scores
**0.941 accuracy / 0.042 Brier** on the 68-question cohort every Jev comparison uses.

The mechanism is general: a slot-logit or label-token readout must compress "excluded by a
condition" and "named by a condition" into a single scalar, and negation is exactly what
that compression handles worst. Nothing in a cross-entropy loss on argmax labels rewards
noticing it. **Test negation explicitly before shipping** — write each of your rules both
ways and check the model inverts correctly.

#### Three adversarial findings, all from September 2026

**Context can flip a correct decision.** `JevOut` (arXiv 2609.30243) uses the model's own
option probabilities to steer fluent context additions that keep the source, question,
choices and gold answer intact, and redirects **312 of 508 initially-correct decisions
(61.4%)**, with the wrong target getting ≥0.7 probability in 229 cases. Three other
decision systems showed 64.9–73.2% targeted flip rates. Your `state` field is a blob of
user text; treat it as adversarial input.

**Returning probabilities builds you a gradient oracle.** `Decision Hijacking` (arXiv
2609.28613) ran 510 reconstructed InjecAgent cases. Malicious content rarely selects the
attacker's target outright, but **adaptive attacks using score feedback double the mean
highest attacker-target probability**, raising fresh-validation success from 1.8% to 3.5%.
An adversary who can submit inputs and read probabilities optimises against you far more
cheaply than against a text generator, whose output is a noisier signal.

**Cascades fail when the errors are correlated.** This is the most consequential result for
deployment, and it cuts against the most common plan. `JEV vs. LLMs as Rubric Judges`
(arXiv 2609.29769) found the small model's confidence does rank its own errors — *"which
should make a cheap classifier the ideal first stage of a cascade"* — but then: *"**the
LLM judges repeat nearly all of Jev's most confident errors**, so a cascade … gains at
most 1.5 points over the best single judge with cross-fitted thresholds, and at most 2.0
even with oracle thresholds."* A companion paper (arXiv 2609.26550) found the opposite on
a different benchmark, retaining 99% of a SOTA judge's accuracy at lower cost. **The two
are in direct tension and the difference is benchmark-dependent**, so measure your cascade
end to end on your own data, and check whether your fallback actually disagrees with you
on the cases you escalated.

#### SQ8 verdict

The option-name test and the negation test are both cheap, both take under a day, and
both find failures that a schema check and an accuracy number both report as fine. Run
them before you trust a threshold. The adversarial results mean a decision model fed
untrusted text is an attack surface, and the cascade results mean the obvious
"small model first, big model on doubt" architecture may buy you almost nothing.

---

## Cross-Cutting Analysis

### Pattern 1 — The head is free; the backbone is everything

Across every implementation, the trained head is between 0.008% (Jev-Omni:
983,456 / 11.96B) and a few percent of the model. Cost scales with `hidden_size` and
`max_options`, not with vocabulary, task count, or modality.

This is why "make it 34x smaller" is not an optimisation problem. It is a
backbone-selection problem. We chose LFM2.5-350M because its architecture was
*hardware-in-the-loop searched* for exactly the one-forward-pass-no-decode operation a
decision model performs, and because Liquid's own recommendation ("data extraction,
structured outputs, and tool use") names our task.

### Pattern 2 — Convergence is the strongest evidence in the field

**Ten** independent teams, no shared codebase and no shared data, converged on: a small
trained head, a pretrained backbone, a fitted per-type temperature, and Brier/KL
reported next to ECE. When implementations that cannot see each other agree, the
agreement is worth more than any single paper.

The late community pass raised the count from six to ten, and raised it in the most
convincing possible way: **[Ollaya](https://ollaya.dev/)** independently ships a
`Modelfile` that refits per-model calibration on the user's own labelled data — the
same idea as `calibration.py`, built by a team with no reason to have read it. And
**[Blink](https://github.com/sqliteai/blink)** reaches the same architecture from the
opposite direction, stripping it down to C99 with **zero allocations during scoring**
(mechanically asserted by its test suite, not by inspection) and a 66 KB WebAssembly
build that runs **18,211 decisions/s** on one core of an Apple M5 Pro.

The one place they diverge — read-out geometry — is exactly where the September 2026
paper found measurable, consequential differences.

### Pattern 3 — Soft and ordinal targets beat more data, repeatedly

Three independent results point the same way:
- soft targets cut KL by a third and score MAE by 15%
- the ordinal kernel moved Decision Score -21.4 → +11.9 and ECE 0.39 → 0.045
- a base-rate predictor has the *best* ECE on the table while being useless

A decision model is judged on the shape of its distribution, not just its argmax. The
training target is where that shape is set, and it is cheaper to get right than to fix
downstream.

### Pattern 4 — Honesty is a competitive norm, and it is load-bearing

Every serious project in this ecosystem publishes its negative results. Jebadiah's
experiment log is a catalogue of what did not work. Kev-0.8B leads with what got worse,
gives confidence intervals on every delta, and states that one number is "unexplained,
not a skill." RCLD's README says "valid JSON does not mean correct answers" in bold.

That norm is why this report can say useful things about where small models fail. It
is worth preserving in anything built on this pattern.

### Contradiction 1 — LFM2.5-350M's context length

The `config.json` says `max_position_embeddings: 128000`. The model card says
"**Context length**: 32,768 tokens." Both are Liquid's. The most likely reading is that
128,000 is the RoPE-configured maximum and 32,768 is the validated/labelled context
(the card also claims knowledge from 28T tokens and mid-2024 cutoff, and the VL variants
all state 32,768). **Plan for 32,768.** Flagged rather than resolved.

### Contradiction 2 — MMAU's published baseline numbers

The MMAU **paper** (arXiv and Sec 3.1) says 11 information-extraction / 16 reasoning tasks
and Gemini Pro v1.5 **52.97%** / Qwen2-Audio **52.50%**. The **project homepage** says
"12 information-retrieval types and 15 reasoning types" and Gemini 1.5 **66.15%** /
Qwen2-Audio **55.4%**. A ~13-point gap on the same named model means these are two
different evaluation runs, not a typo. **Cite the arXiv numbers.** The homepage is a
marketing surface.

### Contradiction 3 — Gemma 4's licence surface

`google/gemma-4-12b-it` is tagged `license: apache-2.0` on the Hub, with a `license_link`
pointing to Google's Gemma 4 terms, and **ships no LICENSE file**. `[single source: the Hub API record]` Jev-Omni ships
`apache-2.0` and says "following Gemma 4." The SPDX tag and the linked terms are not
obviously the same instrument. Anyone redistributing Gemma-4-derived weights should read
the linked terms.

### Contradiction 4 — MVBench frame count

Jev-Omni uses 16 frames; the official MVBench harness defaults to **8 segments** at 224px.
Its reported 53.10% is therefore **not** directly comparable to published MVBench
leaderboard figures, and should not be presented as though it were.

### Contradiction 5 — "Sub-1B is close to Jev", said two ways

Kev-0.8B: *"It is still a sub-1B model... it trails Jev everywhere it can be compared"*
(0.648 vs 0.857 out of domain). this-that-model-1.0 at 1.88B: **0.941 vs 0.765**. Both
are true; they differ in training data, discipline, and what "compared" means. The
honest summary is that **the architecture is not the bottleneck — the training data and
the evaluation discipline are.**

### Contradiction 6 — "A decision model is not an LLM" is contested phrasing

TypeSafe's announcement language implies a model that is categorically not an LLM. A
practitioner on the 570-point HN thread, reading only that announcement, rejects it:

> *"Their marketing language is misleading. They must still use some transformer language
> model backbone to encode the text input (BERT or decoder-only LLM). The biggest
> difference is the output, instead of auto-regressively generating tokens, they produce
> probabilities over a bounded set of decisions."* `[single source]`

This is a comment, not a specification — but it is the same claim this report reaches
independently from config files, and it agrees. **The resolution is terminological, not
substantive:** a decision model is a transformer backbone with the decode loop replaced
by a bounded read-out, and this report's preferred phrasing already says that. What
should change is the flat assertion that decision models "aren't LLMs" — at least one
informed reader rejects that framing, and Blink's own README concedes the same point
from the small end ("*it does not read text the way a pretrained language model does*").

### Contradiction 7 — Latency and capability are separate axes, plotted as one

Ollaya's table puts the **fastest** models at the top: `laya` at 8.1 ms against
`decider:2b` at 190 ms. But an independent commenter reports that on real decisions —
"navigation, traffic, waypoints" — *laya* *"does only slightly better than a small
classifier."* **A 24x latency advantage with a small capability advantage is not a
24x win**, and the table's own caption concedes the setups differ. This is a second,
independent confirmation of the discipline this report already applies: **a 2.4x
latency gap is not a 2.4x capability gap**, and the 8.1 ms figure must not be quoted
without the accuracy caveat attached.

### Contradiction 8 — "RLCD is overrated" challenges this project's premise

kyr0, building the strongest open system in the
[`typed-decision-bench`](https://kyr0.github.io/typed-decision-bench/):

> **"RLCD is overrated guys! Use CE as the primary loss with an eye on Brier+NLL; it's
> cheaper and more effective!"**

This project is built on RLCD as its conceptual prior — Key Takeaway 10 is literally
"cross-entropy is a strictly proper scoring rule, you already have RLCD's main benefit."
The claim is **a single source, from someone with a competing system and a commercial
interest in it**, so it overturns nothing. But it is specific, testable, and free to
check: train the same LoRA arm with CE as the primary loss, same seed and same split,
with Brier and NLL reported as metrics.

Note the subtlety that keeps this compatible with the report's own findings. kyr0 is not
saying calibration does not matter — he is saying **CE gets you there more cheaply.**
That is exactly consistent with this project's measurement that the calibration gap is a
**fitting** problem (fitted T 3.300 → 1.050) rather than a training-objective problem.

### Contradiction 9 — Ollaya and typed-decision-bench disagree about Laya's latency

Ollaya measures `laya` at **8.1 ms**; `typed-decision-bench` measures it at **36.7 ms**
p50 on an H200. That is a 4.5x disagreement between two single-party measurements, and
Ollaya's own caption already says "setups differ, so read it as an order-of-magnitude
comparison." The two tables agree on the *shape* — small models are far faster than the
hosted baseline — and disagree on the number.

**Neither figure should be quoted alone, and this report cites the kyr0 table for
accuracy and Ollaya only for the API-compatibility surface.** A benchmark that publishes
its own per-capability sample sizes, its noise caveat, its baseline deltas and its
calibration counts is worth more than a marketing latency chart — a general rule, not a
comment about these two projects.

### Contradiction 10 — Four latency measurements, none of them comparable

This report has now collected four independent latency figures for the same family of
system, and **they disagree by more than an order of magnitude**:

| Source | Figure | Shape / conditions |
|---|---:|---|
| Jev-Omni's own card | **83 ms** | in-process, H200, ~2k-token text, one question |
| Awesome Jev radar | **50-100 ms** | none stated: no version, no question count, network time undeclared |
| Ollaya | **236-276 ms** | hosted API, median, five questions, includes network |
| `typed-decision-bench` | **716 ms p50 / 779 ms p95** | H200 NVL, 275 capabilities, includes HTTP latency |

The radar's claim is **~7x faster** than the controlled benchmark's, and the spread from
fastest to slowest is **~14x**. The request shapes differ so much that these may not be
the same operation: one question versus five, in-process versus hosted, unknown context
length.

**The honest statement is a range with the shape attached to each end, and no
cross-source latency ranking.** This report's Latency table lines these figures up
vertically, which invites exactly the comparison none of them supports. That table is a
record of *what each party measured on its own hardware* and must not be read as a
ranking. Takeaway 23 already forbids quoting a latency win without the accuracy number;
this extends it: **do not quote a latency number at all without its request shape.**

What survives is the *direction*, which four sources agree on: local small-model decision
paths are far cheaper and lower-latency than a hosted frontier round trip. Nobody in this
corpus disputes that. **The specific multiples do not survive.**

---

## Comparisons

### Head cost across implementations

| Model | Backbone params | Head type | Head params | % of net |
|---|---|---|---|---|
| Jev-Omni | 11,959,730,224 | `Linear(3840, 256)` | 983,456 | 0.0082% |
| **Jeb-Omni-Nano (ours)** | **354,483,968** | **`Linear(1024, 256)`** | **262,400** | **0.0740%** |
| Kev-0.8B | ~0.8B | pointer head + LoRA r=16 | 11.3M (adapter) | ~1.4% |
| Jebadiah-4B-v2 | ~4B | label-token logits + LoRA r=16 | 32.5M (adapter) | ~0.8% |

*Comparability: our head count is measured by running the code. Jev-Omni's is
`3840×256+256` and is independently corroborated by the GGUF card's reported 3.78 MiB
fp32 head. Kev's and Jebadiah's figures are their own self-reported adapter sizes and
include the LoRA, not a separate head, so they are not like-for-like.*

### Reported accuracy — read the column headers

| System | Params | typed-decisions (2000) | DecisionBench Med | JevBench (231) | Note |
|---|---|---|---|---|---|
| Jev 1.13.0 (hosted) | — | 0.727 | — | 0.866 | generalist, zero-shot |
| Jev-Omni | 12B | — | **0.8757** | 0.8615 | own harness, own data |
| meraGPT Decider 1 | — | **0.768** | — | — | generalist, closed |
| Laya (fine-tuned) | 421M | 0.766 | — | — | **specialist** on these workflows |
| Laya (zero-shot) | 421M | 0.345 | — | — | **at chance (0.343)** |
| Kev-4B | 4B | 0.800 | — | 0.758 | in-distribution row |
| Kev-0.8B | 0.8B | — | — | 0.636 | hard tier 0.360 |
| jeff | 400M | — | — | 0.669 (#9) | #14/18 on Intelligence |
| this-that-model-1.0 | 1.88B | — | — | — | 0.941 on a 68-question cohort |
| ModernBERT-base | 149M | 0.646 | — | — | **specialist** |
| Prior (ignores input) | — | 0.470 | — | — | ECE 0.088, best on table |

*Comparability: **these are not one table.** The typed-decisions column is
generalist-zero-shot for Jev/meraGPT and specialist-fitted for Laya/ModernBERT. The
JevBench column comes from different authors' harnesses on different runs. The Laya
0.766-vs-0.345 pair is the same model, same benchmark, differing only in whether it was
fine-tuned on those four workflows — and that 42-point gap is the most important number
in the table.*

### Cost per decision, with the request shape stated

| System | $/1k decisions | Request shape | Basis |
|---|---|---|---|
| Jev 1.13.0 | **$0.0399** | one call per question | 950 input tokens/decision × $0.042/M |
| Jev-Omni | proxy only | one call per question | recorded tokens × $0.05/M (Gemma 3 12B) |
| Laya (self-hosted, 421M) | $0 | any | electricity |
| Jeb-Omni-Nano | $0 | any | electricity |

*Comparability: the JevBench correction notes that re-pricing the chat models per question
would "move all three marks right by roughly the same factor (2.82x on medium, 3.09x on
hard) without changing the order." The order is robust; the absolute numbers are not.*

### Latency

| System | Hardware | Per decision |
|---|---|---|
| Jev-Omni text (~2k tok) | H200 | 83 ms |
| Jev-Omni image | H200 | 26 ms |
| Jev-Omni audio (13 s) | H200 | 31 ms |
| Jev-Omni video (16 frames) | H200 | 504 ms |
| Jev-Omni (4-bit GGUF) | RTX 5080 Laptop | 200 ms |
| Jev-Omni (4-bit GGUF) | Ryzen AI 9 HX 370 | ~1.9 s (11.30 s / 6 q) |
| Laya | Tesla T4 | 32.8 ms |
| this-that-model-1.0 | consumer GPU | 30.9 ms |
| **Jeb-Omni-Nano, untrained** | **CPU, fp32** | **337 ms** |
| Ollaya `laya:multilingual` | RTX 4090 | 8.1 ms (5-question request) |
| Ollaya `laya:en` | RTX 4090 | 9.6 ms (5-question request) |
| Ollaya `gliclass` | RTX 4090 | 14.7 ms (5-question request) |
| Ollaya `decider:0.8b` | RTX 4090 | 155 ms (5-question request) |
| Ollaya `decider:2b` | RTX 4090 | 190 ms (5-question request) |
| **TypeSafe Jev, hosted API** | **third-party, unspecified** | **236–276 ms** |
| Blink (`blink-tiny`) | Apple M5 Pro, 1 core | 406 µs fresh / 54 µs state-reused |

*Comparability: our 337 ms is a **337 ms untrained fp32 CPU run with no
`causal_conv1d` kernel installed**, for plumbing verification only. It is not a
performance claim. Liquid's own Q4 llama.cpp numbers (2.9K prefill tok/s on AMD CPU,
200 tok/s on a Pi 5) are the relevant ones.*

The Ollaya block is a **single-party measurement of many models on one machine**,
and the Jev figure in it comes from third-party hosted benchmarks that include the
network. **The two halves of that row were measured on different hardware by
different people.** Ollaya's own caption says the right thing: *"Setups differ, so
read it as an order-of-magnitude comparison."* Its only defensible reading is that
a local five-question decision request lands somewhere in the 8–190 ms band while a
hosted Jev round trip lands at 236–276 ms. It is **not** evidence that any of those
models is more accurate than Jev, and the next section shows why that distinction
is load-bearing.

---

## Key Takeaways

The first eight are **our own measurements**; the rest are drawn from the literature and
labelled by source. Every experiment is reproducible from `experiments/`.

1. **The head is 0.008–0.074% of the model. Everything expensive is the backbone.**
2. **Fine-tune the backbone, not just the head — measured, not assumed.** Frozen + head
   0.3233 vs LoRA + head 0.5633 on the same 360 training questions; fitted temperature
   **3.300 -> 1.050**.
3. **A head-only model has no usable operating point.** Coverage at a 5% error budget is
   **0.0000** — its most confident prediction was wrong. LoRA reaches 0.0967.
4. **More data does not rescue the frozen arm.** 1,212 training questions across four
   workflows halved its miscalibration (T 3.300 -> 1.800) and left coverage at 0.0000
   across 996 eval questions.
5. **The value is concentrated at the top of the confidence distribution.** The 10%
   most-confident decisions are **93.3%** correct against 56.3% overall.
6. **Thresholds do not transfer between workflows.** 9.7% coverage at a 5% budget on
   `customer_service`, 5.5% on `invoice_processing`.
7. **Coverage at a 5% budget needs hundreds of evaluation questions.** At n = 45 one early
   error pins every budget below 25% to the same 3/45 and the metric carries no
   information.
8. **Read the last layer.** The final layer beat L0/L4/L8/L12 on accuracy, Brier, ECE and
   training loss. AnyJev's middle-layer finding does **not** replicate at 350M.
9. **Use slot logits, not label-token logits** (Kev, Jebadiah, Laya, this-that-model).
   Positional, so one checkpoint handles 2–256 options.
10. **Cross-entropy is a strictly proper scoring rule — you already have RLCD's main
    benefit** (Laya and the scoring-rule literature).
11. **Fit temperature per question type *and* per option-count bucket, against your soft
    target** (Jebadiah's two fits; a published per-cardinality calibrator showing a 3.3x
    spread by option count).
12. **The ordinal kernel for `score` questions is the highest-leverage training decision**
    (Jebadiah: Decision Score −21.4 → +11.9, ECE 0.39 → 0.045).
13. **Use the instruction-tuned checkpoint** (Jebadiah: +2.15 headline points on the 4B).
14. **Run the option-name and negation tests before shipping a threshold policy**
    (arXiv 2609.26758; this-that-model's changelog).
15. **Report KL and Brier next to ECE** (`typed-decisions`: the base-rate Prior has the
    best ECE on the table while knowing nothing).
16. **A 350M model is a specialist, not a generalist** (Laya zero-shot 0.345 against a
    0.343 chance rate).
17. **Your code can be MIT; your weights cannot** (LFM Open License v1.0, $10M revenue
    cap, and **no** MAU threshold).
18. **Start from LFM2.5-VL-450M for image; do not build a projector**
    (`LFM2.5-VL-450M-Extract`: 98.9 / 98.8 / 84.5 at 0.45B).
19. **Do not quantise the head, and refit temperature per format** (Jev-Omni Q4_K_M: max
    probability difference 0.210; a published calibrator fits T per quantisation).
20. **Know where the floor is: form vs. reading.** Blink, the smallest serious artefact
    in the field, states it plainly — near perfect where the answer is carried by *form*,
    *"at or a little above chance, and a frozen 4B model is far ahead"* where it requires
    *reading*. That is this project's 0.2667-vs-0.5778 gap, described by someone who built
    the minimal version and measured where it stops working.
21. **For a fixed task with an eval set, the top-voted answer on the largest thread in this
    space is: just train a classifier.** *"Smarter move if you have an eval set is to just
    train a classifier and call it a day."* A general-purpose decision model earns its cost
    only when you do not know your label set in advance. Choose deliberately.
22. **Make the interface the artefact, not the weights.** Ollaya runs the **official
    TypeSafe Python SDK 0.7.1 unchanged** against a local server. If your deployment
    survives the model being replaced, the port is cheap; if it doesn't, every model swap
    is a rewrite.
23. **Latency is not capability.** The fastest model in the ecosystem's headline table
    (8.1 ms) is independently reported to beat a small classifier only slightly on real
    decisions. Never quote a latency win without the accuracy number attached.
24. **The bar on this project's own benchmark is 0.766, and it is already public.**
    `laya-typed-decisions` (Apache-2.0, 421M) scores 0.766 on `typed-decisions` against
    this project's 0.5633, trained on a single 96 GB GPU. This project's findings are
    about *method* and do not require being SOTA, but its docs previously reported 0.5633
    with no reference point. They do now.
25. **Calibration, not size, is the failure mode.** kyr0's 76% model has ECE-15 of 13%
    against Jev's 8.4%, and singles out Laya as "calibration is really bad" — the same
    verdict this project measured directly (frozen arm, T 3.300, 0.0000 coverage at 5%).
    A badly calibrated decision model has **no usable operating point**, which is worse
    than being 20 points behind on accuracy.
26. **Test CE against RLCD.** It is the one free experiment in the corpus: same arm, same
    seed, same split, CE as the primary loss, Brier and NLL reported. One source with a
    competing commercial interest, so not decisive — but cheap, and it goes to the
    project's own premise.
27. **The honest production failure in the corpus is instructive.** 731 trades in 24h,
    -3.15%, with costs handed to the model explicitly: "it's very efficient at generating
    commissions." The unanswered question is this project's question — *do higher
    probabilities actually lead to better outcomes after costs?* The paired anecdote
    reached the opposite conclusion the same week, so neither is evidence.
28. **Decision models earn their place between deterministic steps, not across them.**
    `json-render` does not let Jev build the UI; the trading bot does not let it place
    orders unguarded; the drone project keeps classical control and uses Jev one level
    up. The 287-project survey's own summary is the pattern: big model → Jev → code →
    Jev → tool → Jev → big model.
29. **This field is 56% plumbing and 6% classification.** Of 693 catalogued projects, 387
    are orchestration, routing, MCP, guardrails, context GC and evaluation; **2 are filed
    under Classification.** The interesting work is integration, and that is also the
    binding constraint on any new small model.
30. **The speed claims are unverified, and the census says so.** 376 of 693 entries carry
    *"performance and cost benefits have not been independently verified"*; 352 say the
    reviewer could not even confirm what decision the project makes. **The one
    independently reproduced performance number in the whole corpus is Vercel Labs'
    `json-render` at 3.21 s -> 0.88 s.** When you benchmark your own model, the directory's
    provenance-block discipline is the standard to copy.
31. **Never quote a latency number without its request shape.** Four sources measure
    Jev-class latency between **50 ms and 716 ms** — a 14x spread — and the request shapes
    differ so much they may not be the same operation. The *direction* is agreed by
    everyone; the multiples are not.


## Limitations & Gaps

**Stated plainly, because the rest of this report is only as good as these.**

### Method limitations

- **`websearch` failed for the entire session** (all providers: quota 429, timeouts, bot
  challenges, datacenter-IP blocks). Collection was by direct URL retrieval.
- **The discussion layer was absent until a late fix.** The Hacker News Algolia API was
  found to work unauthenticated and produced ledger rows 201–211, including a
  570-point/137-comment front-page thread. Every earlier statement that community
  sources were "under-represented" understated it: they had not been searched.
  **`websearch` failing was treated as a reason to narrow scope, when it should have
  been treated as a reason to find another endpoint.**
- **Reddit is now covered; Stack Overflow genuinely has nothing.** A third pass reached
  Reddit through a public **Redlib** instance after `reddit.com/search.json` (403) and
  `old.reddit.com` (login redirect) both failed, and surfaced the deployment layer that
  two earlier passes had recorded as absent — 15 more ledger rows. **Stack Overflow was
  searched via the Stack Exchange API (which works unauthenticated) and returned nothing
  on-topic**: the API OR-matches query terms, and `Jev decision model` returned zero
  items while `temperature scaling calibration` returned colour-tracking and Scala
  immutability questions. **No reliable data found on Stack Overflow**, which is a
  finding rather than a gap: the engineering discussion for this field lives on Reddit
  and Hacker News, not on Q&A sites.
- **One Redlib instance was used** (`safereddit.com`, SFW-only). A non-SFW instance may
  surface more, and the 287-project directory from row 216 was not itself crawled.
- **The 693 projects were not individually read.** The census is evidence about the
  *field*, not about any one project. The `llms.txt` / `llms-full.txt` files the
  directory publishes for agents may carry a cleaner per-project dataset than the README
  and are the obvious next fetch.
- **The prior-art claim could not be verified.** The arXiv API returned an empty feed for
  2503.23303. Recorded as an unverified claim; it supports nothing in this report.
- **The new community sources are Tier C except where noted** and carry no weight in any
  quantitative claim. Every number in the Ollaya latency table is somebody else's
  measurement on somebody else's hardware.
- **Four of eight scouts failed on infrastructure, not on the research**: two upstream
  provider idle timeouts after 12–16 minutes of *successful* fetching, one
  result-payload overflow, and one that returned a structured stub. The stub's ledger was
  recovered from its session backing file (+9 sources). All recovered claims were
  **re-verified independently** before inclusion; any that could not be were excluded.
- **The 200-source floor was met by enumeration breadth, not by 200 deep reads.** Roughly
  60 sources were read at full depth and have per-source notes under
  [`notes/`](notes/INDEX.md); the remaining ~140 were read at the depth needed to extract
  their specific claims (config fields, benchmark tables, dataset schemas). Every ledger
  row carries a Relevance score so this is visible per source.

### Substantive gaps

- **No trained Jeb-Omni-Nano checkpoint exists.** Every accuracy figure for the proposed
  model is therefore hypothetical. The only measured numbers are the smoke-test plumbing
  (loss 1.80 → 0.31, 85.0% on 12 hand-written sentences) and the integration test. **We
  make no accuracy claim.**
- **No multimodal checkpoint was trained or evaluated.** The media code path and the
  preprocessing conventions match Jev-Omni's and are documented, but nothing was run on
  real media. The LFM2.5-VL-450M recommendation is a *prediction* from published
  benchmarks, not a measurement.
- **The read-out layer sweep froze the backbone and trained only a head.** It therefore
  rules out the frozen-350M case, not a LoRA-tuned one; a tuned model reshapes the layers
  and the optimum could move. One workflow, text-only, n=45 at eval.
- **`itertools.permutations` over 256 options is 256! permutations.** The option-order
  test is only tractable for ≤6-8 options. Nobody in this ecosystem has published
  permutation results at high cardinality.
- **Question isolation is implemented one-forward-pass-per-question**, not with the
  shared-KV-cache optimisation RCLD demonstrates. Isolation is exact; the cost is real.
  For a 5-question request we do 5 prefills.
- **Dated sources are flagged where they carry load-bearing claims.** Four of the 200
  predate the 12-month window and are marked inline: Guo et al. 2017 `[dated: 2017]`
  (the origin of the ECE binning debate), Conformal Risk Control 2022 `[dated: 2022]`,
  Conformal Temperature Scaling 2024 `[dated: 2024]`, and the LFM2 technical report
  2025 `[dated: 2025]`. MVBench (2023-11-28) and MMAU (2024-10-24) are also older than
  12 months but are used only for benchmark **definitions**, which have not changed, so
  they are listed here rather than flagged inline.
- **We did not measure Liquid's hardware claims on our own hardware.** Every speed number
  in this report is either Liquid's own, RCLD's own, Kev's own, or ours on one unspecified
  CPU with an untrained head. None are cross-verified.

### Single-source claims, listed

Every claim resting on exactly one source, with an assessment of whether that is
acceptable:

| Claim | Sole source | Assessment |
|---|---|---|
| LFM Open License v1.0 permits X, forbids Y | the licence text itself | **Acceptable.** A licence is its own primary source; §1 and §5 were read in full and quoted verbatim. |
| No MAU threshold, field-of-use restriction or acceptable-use list exists in it | the licence text itself | **Acceptable**, same reason. This corrects an earlier assumption in this research. |
| `google/gemma-4-12b-it` is tagged Apache-2.0 and ships no LICENSE file | the Hub API record | **Weak.** A tag is metadata, not a licence. The linked Gemma 4 terms were not retrievable, so we flag the question rather than answer it. |
| Temperature varies by option count with a 3.3× spread | `rlcd-modernbert-151m` `calibrator.json` | **Weak but internally checkable.** One project. The file is self-consistent (log-T exponentiation, per-k grid); the *direction* is corroborated by AnyJev, the *magnitude* is not. |
| `weight = n/(n+100)`, and the blend is geometric | `Jev-Style-0.8B-Decision-v3` `readout_config.json` | **Strong for our purposes.** We recomputed the identity for all 20 groups and reproduced two stored temperatures to <1e-4. A third matches to 1e-3, so the rule is not perfectly determined and we say so. |
| T can be < 1.0 | same file | **Weak.** One artefact. The direction is model-specific and cannot be assumed. |
| Liquid's 9-device speed table | Liquid AI's own release post | **Unverified.** Vendor self-reported; we measured none of it. |
| LFM2.5-VL-450M-Extract at 98.9 / 98.8 / 84.5 | Liquid AI's own card | **Unverified.** Vendor self-reported. The eval pipeline ships in-repo; we did not run it. |
| A 350M student beats a 120B teacher | distil labs blog | **Weak.** Vendor partner with a commercial interest; graded Tier B. Model cards are inspectable, but it is *generative* tool calling, not our readout setting. |
| A middle layer beats the last for a linear head | AnyJev (Nokia + Tencent) | **Unreplicated.** One 7B model, one benchmark, and it contradicts Jev-Omni and Kev. We expose `readout_layer` rather than choosing. |
| The option-name failure, AUC .94 → .23 | arXiv 2609.26758 | **Unreplicated.** Published four days before this session, on the hosted Jev and two open models. **Not measured on a slot-logit model at 350M**, which is our configuration. |
| The negation failure at 0.88 confidence | this-that-model changelog | **Unreplicated.** One rule, eleven phrasings, one model version. Prevalence across the class is unknown. |
| Archer Hume's reverse-engineering of hosted Jev | a single independent blog | **Weak but appropriate.** Explicitly labelled inference by its author, and used only for context, never to support a load-bearing claim. |

### What would change the conclusions

- A published option-name robustness result on a **slot-logit** model at 350M. The
  September paper studied the hosted Jev and two open models; if slot logits turn out to
  be equally affected at small scale, the neutral-identifier mitigation becomes mandatory
  rather than optional.
- Any *independent* verification of the LFM2.5 speed table on the hardware we target.
- A small generalist (not specialist) beating Laya's 0.345 zero-shot by a meaningful
  margin. That would change the "you must bring data" conclusion.

---


## Sources

All 230 sources, with the same numbering as `sources-ledger.md`. Full key claims per
source are in the ledger; per-source deep notes for the load-bearing ones are in
`notes/`.

1. [Jev-Omni runtime loader](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/jev_omni.py) — akhilaaa3, 2026-09. `_Head256` verbatim: buffers `mu(1,H)`, `sd(1,H)`; `Linear(hidden,256,dtype=float32)`; `z=linear((features.float()-mu)/sd)`;... (Tier A, repo/code, rel 5)
2. [Training recipe](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/decision_config.json) — akhilaaa3, 2026-09. `hidden_size 3840`, `output_classes 256`, `dtype float32`, base `google/gemma-4-12B-it` (Tier A, repo/config, rel 5)
3. [Base architecture](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/config.json) — akhilaaa3, 2026-09. `Gemma4UnifiedForConditionalGeneration`, `model_type gemma4_unified` (Tier A, repo/config, rel 5)
4. [Model card](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/README.md) — akhilaaa3, 2026-09. DecisionBench Medium 87.57% state-macro / 86.01% micro; JevBench 86.15%/87.45%; MMAU 63.10% (1000 Q); MVBench 53.10% (14 tasks, 2786 Q) (Tier A, modelcard, rel 5)
5. [Reference outputs](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/verification.json) — akhilaaa3, 2026-09. 4 cases with reference distributions, `worst_abs_diff 0.01936584711074829` (Tier A, repo/data, rel 5)
6. [Media preprocessing](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/processor_config.json) — akhilaaa3, 2026-09. image_seq_length 280, patch 16, pooling_kernel 3; audio_seq_length 750, audio_ms_per_token 40, feature_size 640, sampling_rate 16000; video num_frames... (Tier A, repo/config, rel 5)
7. [Runtime deps](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/requirements.txt) — akhilaaa3, 2026-09. `torch>=2.10`, `transformers==5.17.0`, accelerate, safetensors, numpy, Pillow, opencv-python-headless, soundfile, librosa (Tier A, repo/config, rel 3)
8. [Model metadata + file list](https://huggingface.co/api/models/akhilaaa3/Jev-Omni) — HF, 2026-09. 11,959,730,224 BF16 params, 71.6 GB storage, 237 likes, created 2026-09-20 (Tier A, docs, rel 4)
9. [ZeroGPU Space, re-implements the head](https://huggingface.co/spaces/akhilaaa3/jev-omni/raw/main/app.py) — akhilaaa3, 2026-09. Independent restatement of `Head256` (same buffers, same `Linear`, same mask) and the same prompt template — a second copy of the same design,... (Tier A, repo/code, rel 4)
10. [Gemma 4 base metadata](https://huggingface.co/api/models/google/gemma-4-12b-it) — Google, 2026-05. Tagged `license: apache-2.0` with `license_link` to Google's Gemma 4 terms; **ships no LICENSE file in the repo**; 11,959,730,224 BF16 params,... (Tier A, docs, rel 4)
11. [LFM2.5-350M config](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/config.json) — Liquid AI, 2026-03. `Lfm2ForCausalLM`, `model_type lfm2` (Tier A, repo/config, rel 5)
12. [LFM2.5-350M card](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/README.md) — Liquid AI, 2026-03. 350M, 28T tokens, 32,768 ctx, vocab 65,536, cutoff mid-2024, 9 languages (Tier A, modelcard, rel 5)
13. [LFM Open License v1.0 (full text)](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/LICENSE) — Liquid AI, Inc., 2026. §1 "Threshold" = **$10,000,000 or more** annual revenue; Licensor = "Liquid AI, Inc." (Tier A, docs, rel 5)
14. [Model License guide](https://docs.liquid.ai/lfm/help/model-license.md) — Liquid AI, 2026-04-02. Plain-language: commercial free under $10M revenue; research/nonprofit free with no threshold; no copyleft; you own modifications but derivatives stay... (Tier A, docs, rel 4)
15. [LFM2 Technical Report](https://arxiv.org/abs/2511.23404) — Liquid AI, 2025-11-28. "hardware-in-the-loop architecture search under edge latency and memory constraints… compact hybrid backbone that combines gated short convolutions... (Tier A, paper, rel 5)
16. [LFM2.5-350M release post](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind) — Liquid AI, 2026-03-31. Full hardware table at 1K prefill / 100 decode: AMD Ryzen AI Max 395+ CPU Q4 2.9K/313 tok/s 434 MB; Snapdragon 8 Elite NPU Q4 2.8K/15 169 MB, GPU Q4... (Tier A, docs, rel 5)
17. [LFM2.5-350M-Base card](https://huggingface.co/LiquidAI/LFM2.5-350M-Base/raw/main/README.md) — Liquid AI, 2026-03. Base checkpoint "only recommended for tasks that require heavy fine-tuning… experimenting with novel post-training approaches" — the inverse of what we... (Tier A, modelcard, rel 4)
18. [LFM2.5-2.6B config](https://huggingface.co/LiquidAI/LFM2.5-2.6B/raw/main/config.json) — Liquid AI, 2026-07. hidden 2048, 30 layers = 24 conv + 6 full_attention, 32 heads / 8 KV, intermediate 10752, vocab 128000, max_pos 131072, rope 1e7, bos 124894, eos... (Tier A, repo/config, rel 4)
19. [LFM2.5-2.6B card](https://huggingface.co/LiquidAI/LFM2.5-2.6B/raw/main/README.md) — Liquid AI, 2026-07. 2.69B, 30 layers (22 conv + 8 GQA), 34T tokens, 131,072 ctx, 128,000 vocab (Tier A, modelcard, rel 4)
20. [LFM2.5-8B-A1B MoE config](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/raw/main/config.json) — Liquid AI, 2026-05. `Lfm2MoeForCausalLM`, hidden 2048, 24 layers (2 dense + MoE), `num_experts 32`, `num_experts_per_tok 4`, `moe_intermediate_size 1792`,... (Tier A, repo/config, rel 3)
21. [LFM2.5-VL-450M config](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/raw/main/config.json) — Liquid AI, 2026. `Lfm2VlForConditionalGeneration`, `model_type lfm2_vl` (Tier A, repo/config, rel 5)
22. [LFM2.5-VL-450M card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/raw/main/README.md) — Liquid AI, 2026. LM backbone LFM2.5-350M + **SigLIP2 NaFlex 86M** (Tier A, modelcard, rel 5)
23. [VL-450M image processor](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/raw/main/processor_config.json) — Liquid AI, 2026. `Lfm2VlImageProcessorFast`, image_mean/std `[0.5,0.5,0.5]`, rescale 1/255, resample 2 (bilinear), do_image_splitting, do_pad, tile_size 512,... (Tier A, repo/config, rel 3)
24. [VL-450M-Extract nano card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract/raw/main/README.md) — Liquid AI, 2026. 2,000-sample (image, schema, JSON) benchmark: **JSON validity 98.9, schema F1 98.8, VLM judge 84.5** at 0.45B (Tier A, modelcard, rel 5)
25. [LFM2.5-VL-3B config](https://huggingface.co/LiquidAI/LFM2.5-VL-3B/raw/main/config.json) — Liquid AI, 2026. `lfm2_vl`, text_config hidden 2048 / 30 layers, vision `siglip2_vision_model` hidden 1152 / 27 layers / patch 16, projector_hidden_size 2048,... (Tier A, repo/config, rel 4)
26. [LFM2.5-VL-3B card](https://huggingface.co/LiquidAI/LFM2.5-VL-3B/raw/main/README.md) — Liquid AI, 2026. Vision = SigLIP2 NaFlex 400M (Tier A, modelcard, rel 4)
27. [LFM2.5-Audio-1.5B card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B/raw/main/README.md) — Liquid AI, 2026. 1.5B (1.2B LM) (Tier A, modelcard, rel 4)
28. [LFM2.5-Encoder-350M card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/raw/main/README.md) — Liquid AI, 2026. Bidirectional masked-LM encoder on the LFM2 backbone; `Lfm2BidirectionalModel` + `Lfm2BidirectionalForMaskedLM`; hidden 1024, 8,192 ctx, 15 languages (Tier A, modelcard, rel 4)
29. [LFM2-350M-Extract nano](https://huggingface.co/LiquidAI/LFM2-350M-Extract/raw/main/README.md) — Liquid AI, 2026. Text-extraction nano on LFM2-350M; "outperforms Gemma 3 4B at this task, a model more than 11x its size" (Tier A, modelcard, rel 4)
30. [Liquid docs index](https://docs.liquid.ai/llms.txt) — Liquid AI, 2026-09. Full page list; confirms dedicated pages for Nanos, audio models, vision models, TRL/Unsloth fine-tuning, llama.cpp, MLX, ONNX, OpenVINO, vLLM, SGLang (Tier A, docs, rel 3)
31. [Liquid TRL fine-tuning docs](https://docs.liquid.ai/lfm/fine-tuning/trl) — Liquid AI, 2026. `pip install trl>=0.9.0 transformers>=4.55.0 torch>=2.6 peft accelerate` (Tier A, docs, rel 4)
32. [generation config](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/generation_config.json) — Liquid AI, 2026. bos 1, eos 7, pad 0, no sampling defaults baked in (Tier A, repo/config, rel 2)
33. [tokenizer config](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/tokenizer_config.json) — Liquid AI, 2026. `TokenizersBackend`, bos `<\|startoftext\|>`, eos `<\|im_end\|>`, pad `<\|pad\|>`, no max_length cap set (Tier A, repo/config, rel 3)
34. [chat template](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/chat_template.jinja) — Liquid AI, 2026. ChatML-like: `<\|im_start\|>system/user/assistant` … `<\|im_end\|>`; tool calls as `[name(arg=val, …)]` between... (Tier A, repo/code, rel 3)
35. [Liquid Nanos docs](https://docs.liquid.ai/lfm/models/liquid-nanos.md) — Liquid AI, 2026. Nanos library: LFM2.5-Encoder-230M/350M, Embedding-350M, ColBERT-350M, **VL-1.6B-Extract, VL-450M-Extract**; LFM2-350M-PII-Extract-JP, 2.6B-Transcript,... (Tier A, docs, rel 3)
36. [LFM2.5-350M-RLCD card](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/README.md) — notnotsamuel, 2026-09. **Inference only, no training, no reproduction of TypeSafe's Jev method.** Unchanged LFM2.5-350M weights at rev `9e6c6ccf…` (Tier A, modelcard, rel 5)
37. [RCLD engine source](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/rlcd/engine.py) — notnotsamuel, 2026-09. `fork_cache` deep-copies the cache then `reorder_cache` with repeated index 0; `index_select` allocates independent storage, "never broadcast mutable... (Tier A, repo/code, rel 5)
38. [Benchmarking methodology](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/docs/BENCHMARKING.md) — notnotsamuel, 2026-09. 2 warmups + 3 measured reps; `perf_counter` with MPS/CUDA sync; includes tokenisation, cache copying, forwards, selection, JSON serialisation; excludes... (Tier A, docs, rel 4)
39. [Implementation notes](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/docs/IMPLEMENTATION_REVIEW.md) — notnotsamuel, 2026-09. Pinned config: 16 layers, 6 full-attention + 10 short-conv, `conv_L_cache 3` (Tier A, docs, rel 4)
40. [Measured results](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/results/REPORT.md) — notnotsamuel, 2026-09. Diagnostic suite 12 cases: constrained 55.27/36.23/21.21 ms vs AR 345.36/350.66/197.82; field accuracy 77.8% constrained vs 80.6% AR; exact-object... (Tier A, repo/data, rel 5)
41. [Diagnostic task definitions](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/rlcd/tasks.py) — notnotsamuel, 2026-09. 3 hand-authored schemas (SUPPORT/SENTIMENT/ROUTING), 12 cases with gold labels; explicit comment "not a population benchmark" (Tier A, repo/code, rel 3)
42. [Kev repo README](https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md) — jaredpalmer, 2026-09. Rank-16 LoRA + **pointer head**: each option's `</opt>` hidden state scored against the question's `<decide>` state, softmax → probabilities (Tier A, repo, rel 5)
43. [Kev-0.8B card](https://huggingface.co/jaredpalmer/kev-0.8b/raw/main/README.md) — jaredpalmer, 2026-09-24. LoRA r=16, 11.3M trainable, pointer head on Qwen3.5-0.8B-Base rev `dc7cdfe2` (Tier A, modelcard, rel 5)
44. [Kev metrics source](https://raw.githubusercontent.com/jaredpalmer/kev/main/kev/metrics.py) — jaredpalmer, 2026-09. **ECE verbatim**: `edges = np.linspace(0,1,bins+1)`, `bins=10`, left-closed bins with only the last right-closed, `e += m.mean()*abs(correct[m].mean()... (Tier A, repo/code, rel 5)
45. [Kev calibration source](https://raw.githubusercontent.com/jaredpalmer/kev/main/kev/calibrate.py) — jaredpalmer, 2026-09. Four-arm report raw/shipped/workload/workload_oof (group-disjoint 5-fold) (Tier A, repo/code, rel 4)
46. [Jebadiah repo](https://raw.githubusercontent.com/getainode/jebadiah/main/README.md) — getainode, 2026-09. LoRA r=16 α=32 dropout 0.05 on every linear projection **including Gated DeltaNet**; 32.5M trainable on 4B, 43.3M on 9B; 1 epoch, lr 1e-4 cosine, 30... (Tier A, repo, rel 5)
47. [Jebadiah-4B-v2 card](https://huggingface.co/frontier-infra/jebadiah-4b-v2/raw/main/README.md) — frontier-infra, 2026-09. Option labels are single tokens; answer = distribution over those label tokens at the last prompt position, read fp32, then per-type temperature (Tier A, modelcard, rel 5)
48. [Laya repo README](https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md) — NandhaKishorM, 2026-09. Non-autoregressive System 1 engine, 33 ms, 100+ languages, **"trained with reinforcement learning against strictly proper scoring rules (RLCD)"** (Tier A, repo, rel 5)
49. [Laya measured benchmarks](https://raw.githubusercontent.com/NandhaKishorM/laya/main/BENCHMARKS.md) — NandhaKishorM, 2026-09. typed-decisions (400 cases / 2000 decisions): laya-typed-decisions 0.766 acc / 0.471 soft / Brier 0.061 / ECE 0.213; laya 0.361/0.332/0.316/0.175; Jev... (Tier A, repo, rel 4)
50. [Laya schema-driven decisions](https://raw.githubusercontent.com/NandhaKishorM/laya/main/docs/structured.md) — NandhaKishorM, 2026-09. JSON Schema → typed questions, one forward pass (Tier A, repo, rel 5)
51. [this-that-model-1.0 card](https://huggingface.co/flock-io/this-that-model-1.0/raw/main/README.md) — FLock.io / Oxford, 2026-09. 1.88B, Qwen3.5-style hybrid (18/24 DeltaNet) (Tier A, modelcard, rel 5)
52. [jeff repo](https://raw.githubusercontent.com/logan-markewich/jeff/main/README.md) — logan-markewich, 2026-09. GLiFormer 400M, MIT, self-hosted `typesafe` API (Tier A, repo, rel 4)
53. [Typed Decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/main/README.md) — LocalLLaMA, 2026-09. 1,600 cases × 5 questions over 4 workflows (agent_trace_observability, customer_service, invoice_processing, security_incidents) (Tier A, dataset, rel 5)
54. [DecisionBench card](https://huggingface.co/datasets/akhilaaa3/decision-bench/raw/main/README.md) — akhilaaa3, 2026-09. Two subsets (medium/hard), 80 scenarios / 293 questions each, synthetic via Claude Opus 5 (Tier A, dataset, rel 5)
55. [DecisionBench schema](https://huggingface.co/datasets/akhilaaa3/decision-bench/raw/main/DATASET_INFO.md) — akhilaaa3, 2026-09. Fields `id, subset, family, difficulty, state, n_questions, questions{key:{type,instructions,criteria}}, answers` (Tier A, dataset, rel 3)
56. [JevBench](https://raw.githubusercontent.com/fstandhartinger/jevbench/main/README.md) — fstandhartinger, 2026-09. Score = 25% each of chance-corrected Intelligence, Calibration, Speed, Cost (harmonic mean v1.3; v1.4 blends 20% fresh sealed + 80% v1.3) (Tier A, repo, rel 5)
57. [Decision Index](https://raw.githubusercontent.com/apolinario/decision-index/main/README.md) — apolinario, 2026-09. 40 benchmarks, 5 equal-weight areas, chance-corrected (0 = random, 100 = perfect), coverage-adjusted (Tier A, repo, rel 4)
58. [v3 calibration file](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/raw/main/readout_config.json) — chaoliangUNSW, 2026-09-25. **20 group temperatures + a global 0.8800546821789332**; `clamp [0.3, 5.0]`; `shrinkage_k 100.0`; key `family\|qtype\|option_bucket`; buckets... (Tier A, repo/config, rel 5)
59. [v3 runtime](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/raw/main/jev_style_decision.py) — chaoliangUNSW, 2026-09-26. `z = scores/T; p = exp(z − max(z)); p /= p.sum()` — the whole probability transform (Tier A, repo/code, rel 4)
60. [v3 card](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3/raw/main/README.md) — chaoliangUNSW, 2026-09-25. 752,393,024 params, 24 layers (18 Gated DeltaNet + 6 full attention), hidden 1024 (Tier A, modelcard, rel 4)
61. [BF16 calibration](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF/raw/main/Jev-Style-v2-Calibrated-BF16.calibration.json) — chaoliangUNSW, 2026-09-24. `objective: "sample_mean_soft_cross_entropy"`, `bounds [0.05, 20]`, `calibration_n 3100`, `temperature 1.0408715111841746`, `nll_before 0.51440 →... (Tier A, repo/config, rel 5)
62. [Q4_K_M calibration](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF/raw/main/Jev-Style-v2-Calibrated-Q4_K_M.calibration.json) — chaoliangUNSW, 2026-09-24. **Same objective and calibration_n, but `temperature 1.0123069568523906`** — a different T per quantisation format (Tier A, repo/config, rel 5)
63. [v2 reliability data](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF/raw/main/evaluation/chart_data.json) — chaoliangUNSW, 2026-09-24. `"binning": "15 equal-width confidence bins on [0,1]"`, empty bins omitted, Wilson 95% intervals (Tier A, repo/data, rel 4)
64. [151M per-cardinality calibrator](https://huggingface.co/heman10x/rlcd-modernbert-151m/raw/main/calibrator.json) — heman10x, 2026-09-20. **`per_k` temperatures: 2→5.0069, 3→5.0069, 4→4.0314, 5→3.0560, 6→2.3898, 7→2.3898, 9→1.6668, 11→3.3919, 17→1.7200, 25→1.5144; global 2.8039**... (Tier A, repo/config, rel 5)
65. [OpenJev Verdict 151M card](https://huggingface.co/heman10x/rlcd-modernbert-151m/raw/main/README.md) — heman10x, 2026-09-17. 151M, `L_total = L_CE + 1.0 × L_Brier` — a **composite strictly proper scoring rule**; then L-BFGS temperature scaling (Tier A, modelcard, rel 5)
66. [Verdict repo](https://raw.githubusercontent.com/Heman10x-NGU/Verdict-open-jev/main/README.md) — Heman10x-NGU, 2026-09. **"The engine previously failed to load `calibrator.json`… running at uncalibrated temperature 1.0 (Tier A, repo, rel 5)
67. [jevify trainer](https://raw.githubusercontent.com/kushalpatil07/jevify/main/train/train_lora.py) — kushalpatil, 2026-09. **"Loss per item = KL(target ‖ softmax(label_logits)) + mass_weight × (−log P(any label token))"**, `label_logits[j] = logsumexp over token variants of... (Tier A, repo/code, rel 5)
68. [jevify repo](https://raw.githubusercontent.com/kushalpatil07/jevify/main/README.md) — kushalpatil, 2026-09. **Overconfidence of the un-adapted base**: Gemma 4 E4B raw 0.745 accuracy at **0.963 stated confidence**; 26B-A4B raw 0.757/0.991; Jev 1.13 0.756/0.872 (Tier A, repo, rel 4)
69. [jev-lite card](https://huggingface.co/vagmi/jev-lite/raw/main/README.md) — vagmi, 2026-09-19. **"Accuracy was flat from step 250 to the end of training while ECE fell 0.086 → 0.019: the model did not learn to be right more often, it learned to... (Tier A, modelcard, rel 5)
70. [jevlite repo](https://raw.githubusercontent.com/vagmi/jevlite/main/README.md) — vagmi, 2026-09-19. Worked example: a genuinely ambiguous ticket returns **0.56/0.44 with confidence 0.56** rather than a confident guess — "That split is the output the... (Tier A, repo, rel 4)
71. [Tiny-Jev card](https://huggingface.co/lostargon/Tiny-Jev) — lostargon, 2026-09-21. 0.6B, per-marker-token linear head, grouped softmax (Tier A, modelcard, rel 4)
72. [mini-Jev card](https://huggingface.co/samatv256/mini-Jev) — samatv256, 2026-09-21. **Only 262,657 head params (~1.1 MB)** on a frozen Qwen3-0.6B — a head nearly identical in size to ours (262,400) (Tier C, modelcard, rel 3)
73. [JEV-CPU card](https://huggingface.co/Meanblock/JEV-CPU) — Meanblock, 2026-09-19. **Zero training** — unmodified Qwen3-0.6B, zero-shot, uppercase-letter options, single-token verified, one forward pass, gather option slots, softmax... (Tier C, modelcard, rel 4)
74. [Schema scorer source](https://huggingface.co/mobarmg/jev-schema-scorer-deberta-v3-large/raw/main/schema_scorer.py) — mobarmg, 2026-09-17. `probabilities = logits.float().softmax(dim=0)` — **raw T=1, no calibration at all** (Tier A, repo/code, rel 4)
75. [ModernBERT-JEV card](https://huggingface.co/tasksource/modernbert-tasksource-jev) — tasksource, 2026-09-22. Option-query cross-attention, permutation-equivariant by construction, O(L²) + O(Σ M_k²) + O(K×L) (Tier B, modelcard, rel 3)
76. [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) — TheoLeeCJ, 2026-09. MIT, **4,335 stars**, homepage openjev.com, "Semantic ifs from open models, on a 3090 at home (Tier A, repo, rel 3)
77. [NanoJev](https://raw.githubusercontent.com/TianyuCodings/NanoJev/main/README.md) — TianyuCodings, 2026-09. 0.6B open replica, MIT, 2,262 stars (Tier A, repo, rel 4)
78. [open-jev DeBERTa card](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) — com-kotobalabs, 2026-09-18. DeBERTa-v3-large, tags `typed-decisions`, `calibrated`; trained on mteb/banking77, SetFit/sst5, google/boolq (Tier A, modelcard, rel 3)
79. [NeoHorse-Jev-4B card](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B/raw/main/README.md) — TokenRhythm, 2026-09. ~4B, prefill-only, three decision types (Tier A, modelcard, rel 4)
80. [Jev-Omni Q4_K_M card](https://huggingface.co/Reza2kn/Jev-Omni-Q4_K_M-GGUF/raw/main/README.md) — Reza2kn, 2026-09-23. Q4_K_M backbone 6.87 GiB; projector 116.38 MiB; **FP32 decision head 3.78 MiB = 983,456 params × 4 bytes** (independent confirmation of the head size) (Tier A, modelcard, rel 5)
81. [Jev-Omni MLX 4-bit card](https://huggingface.co/Ruiruiz30/Jev-Omni-MLX-4bit/raw/main/README.md) — Ruiruiz30, 2026-09-23. M4 Mac mini, MLX 0.32.2 (Tier A, modelcard, rel 4)
82. [JEVision card](https://huggingface.co/divyanshx11/JEVision/raw/main/README.md) — divyanshx11, 2026-09. Qwen3.5-0.8B-Base rev `dc7cdfe2` + LoRA + pointer head, **no native vision tower**; separate text and image (visual sidecar) routes (Tier A, modelcard, rel 3)
83. [visual-jev-4b card](https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft/raw/main/README.md) — guanxuyu, 2026-09-23. LoRA r=16 on Qwen3-VL-4B-Instruct, **vision tower frozen** (`exclude_modules '.*visual.*'`), 33,030,144 trainable, 3000 steps, batch 8, lr 1e-4, 100... (Tier A, modelcard, rel 4)
84. [Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It](https://arxiv.org/abs/2609.26758) — Yu Sun, Junhao Xu, Jiajia Shi, Zijin Yang, 2026-09-22. Renaming two options `0`/`1` → `no`/`yes`, changing only the name↔rubric binding: **70.4 more answers changed per hundred (95% CI [67.6, 73.1])** on... (Tier A, paper, rel 5)
85. [JevOut: Natural Context Can Flip Decision Models](https://arxiv.org/abs/2609.30243) — Zixiang Xu, 2026-09. DEFINE+BREAK against the model's **own** option probabilities: **312 of 508 initially-correct decisions redirected (61.4%)**; in 229 cases the wrong... (Tier A, paper, rel 5)
86. [Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions](https://arxiv.org/abs/2609.28613) — Tiantong Wu, Wei Yang Bryan Lim, 2026-09. 510 reconstructed InjecAgent cases (Tier A, paper, rel 5)
87. [JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places](https://arxiv.org/abs/2609.29769) — Delip Rao, Chris Callison-Burch (UPenn), 2026-09. Jev differs from an LLM judge in only 8 of 27 paired comparisons; LLM judges cost 29–325× more and take 30–220× longer (Tier A, paper, rel 5)
88. [JEV-as-a-Judge: Accept When Confident, Escalate When Unsure](https://arxiv.org/abs/2609.26550) — Yubo Li, Yidi Miao, Ramayya Krishnan, Rema Padman, 2026-09. Within 3 pp of a SOTA LLM judge at **0.36% of its fee**; gaps concentrate in **low-confidence** decisions, so the signal is diagnostic (Tier A, paper, rel 4)
89. [REFLEX with Jev for Efficient Selective Control in LLM Agents](https://arxiv.org/abs/2609.26532) — Tiantong Wu, Wei Yang Bryan Lim, 2026-09. 95% success with 72.7% fewer strong-model calls (Tier A, paper, rel 4)
90. [Visual Jev: Accurate and Efficient Decisions from Shared Visual Context](https://arxiv.org/abs/2609.25845) — Guanxu Yu, Yuhang Yao, 2026-09. Encodes image+context ONCE, executes isolated question suffixes as a batch, reads candidate probabilities from the **backbone's language-model head** (Tier A, paper, rel 5)
91. [From Text Decisions to Pixels: A Study of Jev-Style Visual Choice Model (PixelJev)](https://arxiv.org/abs/2609.29283) — Xunlan Zhou, Xianliang Yang, Li Zhao, 2026-09. Maps (image, instruction, candidate set) → structured choice + candidate-conditioned probabilities **using an existing language-model readout** (Tier A, paper, rel 4)
92. [Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model](https://arxiv.org/abs/2609.23959) — Simiao Ren et al., 2026-09. Closest published recipe to a 4B-class open Jev clone: Qwen3-4B LoRA so that the **temperature-scaled softmax over two answer-label logits is P(scam)** (Tier A, paper, rel 5)
93. [this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent](https://arxiv.org/abs/2609.23886) — Zehua Cheng, Wei Dai, Jiahao Sun, 2026-09. 30.9 ms, zero output tokens, 32 decisions/s on one consumer GPU; 42-family suite = 32 s and $0.000217 electricity vs 155.2 min and $10.636 for the most... (Tier A, paper, rel 5)
94. [Just Ask Jev: RL for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures](https://arxiv.org/abs/2609.29429) — Ruoqi Guo et al., 2026-09. RLCDAlignBench: 44 benchmarks, ten alignment failures, 7,193 labelled instances (Tier A, paper, rel 4)
95. [Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem](https://arxiv.org/abs/2609.30216) — Guoming Ling, Muen Xue, Zijian Ye, 2026-09. **2,170 publicly available Jev projects** collected from GitHub as of 2026-09-22 (Tier A, paper, rel 3)
96. [JEVQA — Video Quality from Metadata, Bitstream and Pixel Features with a General-Purpose Decision Model](https://arxiv.org/abs/2609.24395) — Werner Robitza (AVEQ), 2026-09. Jev as a zero-shot video quality model (Tier A, paper, rel 3)
97. [MMAU: massive multi-task audio understanding and reasoning benchmark](https://arxiv.org/abs/2410.19168) — Sakshi, Tyagi, Kumar, Seth, Selvakumar, Nieto, Duraiswami, Ghosh, Manocha (UMD), 2024-10-24. 10,000 questions, 27 tasks (11 information-extraction / 16 reasoning), Speech:Music:Sound = 10:10:7, difficulty 22/56/22% easy/med/hard, average audio... (Tier A, paper, rel 5)
98. [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/abs/2311.17005) — Li, Wang, He et al., 2023-11-28. 20 temporal tasks "that cannot be effectively solved with a single frame", 4,000 questions, 11 video sources filtered to 5–35 s, options template-based... (Tier A, paper, rel 5)
99. [SigLIP 2](https://arxiv.org/abs/2502.14786) — Tschannen et al. (Google), 2025-02. Vision-tower sizes released explicitly: **ViT-B 86M, L 303M, So400m 400M, g 1B** (Tier A, paper, rel 3)
100. [When Calibration Rankings Reverse: Accuracy-Controlled Evaluation for Fair Comparison of LLMs](https://arxiv.org/abs/2606.30814) — Zhichao Yang et al. (EMNLP 2026), 2026-06-29. Global ECE and Brier comparisons of different LLMs are **"confounded by differences in model accuracy"** (Tier A, paper, rel 4)
101. [Soft Mean Expected Calibration Error (SMECE)](https://arxiv.org/abs/2603.14092) — Michael Leznik, 2026-03-14. Where labels are themselves probabilities (teacher soft outputs, radiologist confidence), **"ece commits a category error — it discards the... (Tier A, paper, rel 4)
102. [Same Answer, Different Confidence: Protocol Sensitivity in LLM Confidence Calibration](https://arxiv.org/abs/2605.27752) — Hankyeol Kim, Pilsung Kang, 2026-05-26. Whether verbalized confidence beats token likelihood **depends on how the likelihood is measured**; in a 12-study audit **five never state the choice** (Tier A, paper, rel 4)
103. [Improving Semantic Uncertainty Quantification in LM QA via Token-Level Temperature Scaling](https://arxiv.org/abs/2604.07172) — Lamb, Ivanova, Torr, Rudner, 2026-04-08. **"fixed-temperature heuristics, produce systematically miscalibrated and poorly discriminative"** semantic confidence; **"optimising a single scalar... (Tier A, paper, rel 4)
104. [LLMs UQ via Adaptive Conformal Semantic Entropy](https://arxiv.org/abs/2605.04295) — Karimi, Meyappan, Samavi (IJCAI 2026), 2026-05-05. Conformal accept/abstain with a **finite-sample, distribution-free guarantee** that the error rate among accepted responses stays under a user tolerance (Tier A, paper, rel 3)
105. [Adaptive Cumulative Mass Calibration with Conformal Prediction](https://arxiv.org/abs/2505.15437) — Kazantsev, Moulines, Panov, Kotelevskii, Guizani, 2025-05-21. Existing post-hoc methods **"lack guarantees that a specific notion of calibration is achieved"** (Tier A, paper, rel 3)
106. [Structured Matrix Scaling for Multi-Class Calibration](https://arxiv.org/abs/2511.03685) — Berta, Holzmüller, Jordan, Bach, 2025-11-05. More expressive than temperature scaling, but **"a key challenge lies in the increasing number of parameters… often coupled with limited calibration... (Tier A, paper, rel 3)
107. [From token probabilities to calibrated confidence: an empirical study of mathematical QA](https://arxiv.org/abs/2608.07827) — Ma, Schell, Bhaskara, Pishdad, 2026-08-08. **"individual token probabilities can be highly saturated, aggregating token probabilities over the full sequence captures small but consistent... (Tier A, paper, rel 3)
108. [Calibrating Semantic Uncertainty from Observable Language-Model Probabilities](https://arxiv.org/abs/2607.17447) — Matthew F. Dixon, 2026-07-20. "Language models assign probabilities to words, whereas applications require uncertainty over meaningful states." A **semantic map** bridges word... (Tier A, paper, rel 3)
109. [Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection](https://arxiv.org/abs/2409.15844) — Zecchin, Park, Simeone, 2024-09-24. Sequential data-dependent multiple-hypothesis testing with early termination via e-processes (Tier A, paper, rel 3)
110. [(see #84)](https://arxiv.org/abs/2609.26758) — —, 2026-09-22. Listed once; see #84 (Tier A, paper, rel 5)
111. [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe AI (Diogo Almeida), 2026. **The primary source for the term RLCD.** Compares RLHF vs RLVR vs **RLCD = Reinforcement Learning for Calibrated Decisions** (Tier A, docs, rel 5)
112. [TypeSafe docs — primitives](https://docs.typesafe.ai/introduction) — TypeSafe AI, 2026. Choice → `{choice, probabilities, confidence}`; Score → `{score, probabilities, confidence}`; **Noul → `{noul}` in 0–1 and carries NO confidence** (Tier A, docs, rel 5)
113. [TypeSafe docs — confidence](https://docs.typesafe.ai/confidence) — TypeSafe AI, 2026. **Exact formula: for n options, `confidence = clamp((n * max_prob - 1) / (n - 1), 0, 1)`** — algebraically identical to `(p_max − 1/K)/(1 − 1/K)`,... (Tier A, docs, rel 5)
114. [TypeSafe docs — ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer) — TypeSafe AI, 2026. Calibration is a **GROUP property**: outcomes assigned 0.2 occur ~20% of the time (Tier A, docs, rel 5)
115. [Workflow evals](https://evals.typesafe.ai/) — TypeSafe AI, 2026. **The primary source for the Noul/Choice/Score taxonomy and for where the four `typed-decisions` workflows come from** — security_incidents,... (Tier A, docs, rel 4)
116. [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) — Archer Hume, 2026-09-17. 10,000 API calls of black-box probing (Tier B, blog, rel 5)
117. [(see #111)](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe AI, 2026. Retained for the rejection log: no public RLCD objective exists (Tier A, docs, rel 5)
118. [MVBench card](https://huggingface.co/datasets/OpenGVLab/MVBench/raw/main/README.md) — OpenGVLab, 2023-10. MIT (Tier A, dataset, rel 4)
119. [Official MVBench harness](https://raw.githubusercontent.com/OpenGVLab/Ask-Anything/main/video_chat2/mvbench.ipynb) — OpenGVLab, 2023-2024. **`get_index(num_frames, num_segments)`: `seg_size = (num_frames-1)/num_segments`; `start = int(seg_size/2)`; `offsets = start + round(seg_size*idx)` (Tier A, repo/code, rel 4)
120. [MVBench row counts](https://datasets-server.huggingface.co/size?dataset=OpenGVLab%2FMVBench) — HF, 2026-09. 20 configs × 200 rows = 4,000 (Tier A, dataset, rel 3)
121. [MMAU official repo](https://raw.githubusercontent.com/Sakshi113/MMAU/main/README.md) — Sakshi113 (MMAU authors), 2026. v05.15.25: ~25% of questions revised (Tier A, repo, rel 4)
122. [MMAU test-mini](https://datasets-server.huggingface.co/size?dataset=gamma-lab-umd%2FMMAU-test-mini) — UMD Gamma Lab, 2026-09. **1,000 rows, ~1.21 GB of audio** — the exact set Jev-Omni reports (Tier A, dataset, rel 4)
123. [MMAU parquet mirror](https://huggingface.co/datasets/lmms-lab-audio/mmau/raw/main/README.md) — lmms-lab-audio, 2026-09. `test` 9,000 / 13.99 GB; `test_mini` 1,000 / 1.43 GB (Tier A, dataset, rel 3)
124. [MMAU-Pro](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/raw/main/README.md) — UMD, 2025-08. Successor: 5,305 expert-annotated pairs, 49 skills, audio up to 10 min, multi-audio, spatial (Tier A, dataset, rel 3)
125. [SigLIP2 base config](https://huggingface.co/google/siglip2-base-patch16-224/raw/main/config.json) — Google DeepMind, 2025-02. `siglip` / `siglip_vision_model`, image_size 224, patch 16 → **196 patch tokens** (Tier A, repo/config, rel 3)
126. [SigLIP2 So400m config](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/config.json) — Google DeepMind, 2025-02. hidden 1152, 27 layers, 16 heads, image 384, patch 14 → **729 patch tokens** (Tier A, repo/config, rel 3)
127. [SmolVLM-256M config](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/config.json) — Hugging Face, 2025. 256,484,928 total (Tier A, repo/config, rel 4)
128. [SmolVLM-256M card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/README.md) — Hugging Face, 2025. "We went from a 400M parameter siglip vision encoder to a much smaller 93M encoder." 64 visual tokens per 512×512 patch; under 1 GB GPU RAM (Tier A, modelcard, rel 4)
129. [Whisper feature extractor](https://huggingface.co/openai/whisper-base/raw/main/preprocessor_config.json) — OpenAI, 2026. `chunk_length 30`, `feature_size 80`, **`hop_length 160`**, `n_fft 400`, `n_samples 480000`, `nb_max_frames 3000`, `sampling_rate 16000` (Tier A, repo/config, rel 4)
130. [Whisper-tiny config](https://huggingface.co/openai/whisper-tiny/raw/main/config.json) — OpenAI, 2026. d_model 384, 4 encoder layers, 6 heads, 80 mel bins, 37,760,640 params (Tier A, repo/config, rel 3)
131. [Qwen2-Audio config](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/raw/main/config.json) — Qwen, 2026. `qwen2_audio_encoder`, 128 mel bins, **32 layers × 1280 d_model ≈ 650M audio encoder alone — larger than all of LFM2.5-350M** (Tier A, repo/config, rel 4)
132. [Qwen2.5-Omni config](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/raw/main/config.json) — Qwen, 2026. TMRoPE: **`tokens_per_second 25`, `position_id_per_seconds 25`, `seconds_per_chunk 2`** — the 25-tokens-per-second video convention (Tier A, repo/config, rel 3)
133. [Qwen2.5-VL card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/README.md) — Qwen, 2026. **Dynamic FPS sampling** (vs Qwen2-VL's fixed count), mRoPE aligned to absolute time (Tier A, modelcard, rel 3)
134. [wav2vec2-base config](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json) — Meta, 2026. 7-layer conv frontend, total stride 320 → **50 Hz output from 16 kHz**; 50 Hz × 30 s = 1500 frames, matching Whisper's 1500 positions (Tier A, repo/config, rel 3)
135. [llama-quantize reference](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md) — ggml-org, 2026. **Bits/weight, measured on Llama-3.1-8B:** IQ2_M 2.1460 · IQ4_XS 4.4597 · Q4_K_S 4.6672 · **Q4_K_M 4.8944** · Q5_K_M 5.7036 · Q6_K 6.5633 · Q8_0 8.5008... (Tier A, repo/docs, rel 5)
136. [PEFT LoRA docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora) — Hugging Face, 2026. `target_modules="all-linear"` is the QLoRA-equivalent sweep and "easier than specifying individual modules by name which can vary depending on the... (Tier A, docs, rel 4)
137. [llama.cpp README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/README.md) — ggml-org, 2026. MIT (Tier A, repo, rel 3)
138. [Liquid llama.cpp docs](https://docs.liquid.ai/deployment/on-device/llama-cpp) — Liquid AI, 2026. Liquid's supported flag set for LFM2.5 on llama.cpp, including vision variants (Tier A, docs, rel 3)
139. [Liquid chat template docs](https://docs.liquid.ai/lfm/key-concepts/chat-template) — Liquid AI, 2026. ChatML structure and the `<image>` insertion rule for vision models: "Do not include `<image>` in your message content" — the processor inserts it (Tier A, docs, rel 3)
140. [GitHub repo search](https://api.github.com/search/repositories?q=openvev+OR+SemIf&per_page=15&sort=stars) — GitHub API, 2026-09-26. Confirms `TheoLeeCJ/SemIf-OpenJev` at 4,335 stars, MIT, homepage openjev.com, "Independent; not affiliated with Jev or TypeSafe" (Tier A, docs, rel 2)
141. [HF search: jeb](https://huggingface.co/api/models?search=jeb&limit=40) — HF, 2026-09-26. `frontier-infra/jebadiah-{4b,9b}-v{0,1,2}` all trained on `LocalLLaMA/typed-decisions` + `nvidia/HelpSteer2` + `mteb/summeval`; confirms the Jebadiah... (Tier A, docs, rel 3)
142. [HF search: Jev](https://huggingface.co/api/models?search=Jev&limit=50) — HF, 2026-09-26. Enumerates ~40 distinct Jev-named models, confirming the ecosystem breadth this report analyses (Tier A, docs, rel 3)
143. [Jev-Omni Space metadata](https://huggingface.co/spaces/akhilaaa3/jev-omni/raw/main/README.md) — akhilaaa3, 2026-09. Gradio 5.49.1, apache-2.0, "Multimodal decision classifier - text, image, audio, video" (Tier A, docs, rel 2)
144. [OmniSnap Space](https://huggingface.co/spaces/simkeyur/omnisnap-engine/raw/main/README.md) — simkeyur, 2026-09. "ZeroGPU stateless inference engine for OmniSnap (Tier A, docs, rel 2)
145. [HF dataset search: jev](https://huggingface.co/api/datasets?search=jev&limit=100) — HF, 2026-09-26. Enumerates the Jev-adjacent datasets, including `ZefanCai/Open-Jev-v1.1` and `SargeDev/jev-distill-corpus-v3` referenced by model tags (Tier A, docs, rel 2)
146. [Checksum manifest](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/sha256.json) — akhilaaa3, 2026-09. Per-file sha256 manifest, enabling independent verification of the checkpoint (Tier A, repo/data, rel 2)
147. [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) — tasksource, 2026-09. **Apache-2.0.** 12 configs x train/val/test parquet, 100K<n<1M (Tier A, dataset, rel 5)
148. [Tasksource JEV typed decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) — tasksource, 2026-09. train **2,500,000** / val 15,000 / test 15,000; 1.45 GB download (Tier A, dataset, rel 5)
149. [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) — n4ze3m, 2026-09. 7,414 synthetic cases / 25 … (partial extraction; included for breadth) (Tier B, dataset, rel 3)
150. [awesome-jev](https://raw.githubusercontent.com/yibie/awesome-jev/main/README.md) — yibie, 2026-09. The ecosystem index: **~475 public projects** across 14 categories (Classification & Routing 45, Verification & Guardrails 37, Scoring & Ranking 37,... (Tier A, repo, rel 4)
151. [decider](https://raw.githubusercontent.com/Mapika/decider/main/README.md) — Mapika, 2026-09-25. **The most-documented family; ships 0.8B, 2B, 2B-vision, 4B, 35B-A3B and 35B-A3B-NVFP4.** Qwen3.5 Base backbones, teacher = local Qwen3.5-27B, "Nothing... (Tier A, repo, rel 5)
152. [AnyJev — turn any LLM into a Jev-style decision model](https://raw.githubusercontent.com/nokia-applied-research/AnyJev/main/README.md) — Nokia Sunnyvale + Tencent Hunyuan, 2026-09. **Levels L0/L1/L2, no fine-tuning.** Qwen3-8B BANKING77 20-way, 300 test items: labels needed none/none/100-500; **order-flip rate 0.230 → 0.073 (L0,... (Tier A, repo, rel 5)
153. [open-alternative-jev](https://raw.githubusercontent.com/ikermoel/open-alternative-jev/main/README.md) — ikermoel, 2026-09. The system JevBench measured at **21% with reversed options vs 72% in the author's order** — the concrete instance of the option-order failure the... (Tier A, repo, rel 4)
154. [pcd-rlcd — parallel constrained decoding](https://raw.githubusercontent.com/MahdiBND/pcd-rlcd/main/README.md) — MahdiBND, 2026-09. Parallel constrained decoding combined with RCLD-style scoring; a third naming collision for "RCLD" (Tier C, repo, rel 3)
155. [luce](https://raw.githubusercontent.com/scienthoon/luce/main/README.md) — scienthoon, 2026-09. Community decision model; cited by Kev-0.8B as an external eval set ("scienthoon (873 support tickets)") (Tier C, repo, rel 3)
156. [laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark) — Luni, 2026-09. Community benchmark comparing Laya against Jev (Tier C, dataset, rel 2)
157. [ZTC-Judge-4B](https://huggingface.co/FINAL-Bench/ZTC-Judge-4B) — FINAL-Bench, 2026-09. A 4B judge model in the same space, with 9B/27B siblings; shows commercial interest in small judges (Tier C, modelcard, rel 2)
158. [JevBench Space](https://huggingface.co/spaces/benchmarkheaven/JevBench/raw/main/README.md) — Benchmark Heaven, 2026-09. The live leaderboard Space backing the JevBench board cited throughout this ledger (Tier A, docs, rel 3)
159. [Fine-Tuning Liquid's LFM2.5: Accurate Tool Calling at 350M Parameters](https://www.distillabs.ai/blog/fine-tuning-liquids-lfm25-accurate-tool-calling-at-350m-parameters/) — distil labs (Liquid AI named partner), 2026-03-30. **The single most direct evidence for the project premise.** Student LFM2.5-350M, teacher **GPT-oss-120B** (Tier B, blog, rel 5)
160. [LFM2.5-VL-3B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark/raw/main/README.md) — Liquid AI, 2026-09-18. Speculative-decoding drafter for LFM2.5-VL-3B: **279.5M draft params**, 4 full-attention layers, hidden 2048, GQA 32/8, plus a **Markov head (rank 256)... (Tier A, modelcard, rel 3)
161. [this-that-model training repo](https://raw.githubusercontent.com/FLock-io/this-that-model/main/README.md) — FLock.io / Oxford, 2026-09. **Documents the negation failure**: "1.1 read `bays without chilled handling are ineligible` as though it named the eligible set -- **not failing to... (Tier A, repo, rel 5)
162. [encoder-eval-harness](https://raw.githubusercontent.com/Liquid4All/eurobert-repro/main/README.md) — Liquid AI (Liquid4All), 2026. The eval harness behind the LFM2.5-Encoder-350M 17-task table (Tier A, repo, rel 4)
163. [MLX-VLM](https://raw.githubusercontent.com/Blaizzy/mlx-vlm/main/README.md) — Blaizzy, 2026. The Apple-Silicon VLM runtime referenced by Liquid's VL cards (Tier A, repo, rel 3)
164. [Liquid4All cookbook](https://raw.githubusercontent.com/Liquid4All/cookbook/main/README.md) — Liquid AI, 2026. Worked end-to-end apps, several directly relevant: **Invoice Parser (structured extraction from invoice images with LFM2-VL-3B)**, **Home Assistant... (Tier A, repo, rel 4)
165. [Liquid ONNX deployment docs](https://docs.liquid.ai/deployment/on-device/onnx.md) — Liquid AI, 2026. LiquidONNX (`Liquid4All/onnx-export`) is the official ONNX export path (Tier A, docs, rel 3)
166. [LiquidONNX repo](https://raw.githubusercontent.com/Liquid4All/onnx-export/main/README.md) — Liquid AI, 2026. Text models support **q4f32** in addition to fp32/fp16/q4/q8; MoE supports q4f16 (Tier A, repo, rel 3)
167. [Liquid MLX deployment docs](https://docs.liquid.ai/deployment/on-device/mlx.md) — Liquid AI, 2026. `mlx-lm` for Apple Silicon via Metal, unified memory shared between CPU and GPU (Tier A, docs, rel 3)
168. [Liquid llama.cpp deployment docs](https://docs.liquid.ai/deployment/on-device/llama-cpp.md) — Liquid AI, 2026. CPU-first, cross-platform (Tier A, docs, rel 3)
169. [Liquid vLLM deployment docs](https://docs.liquid.ai/deployment/gpu-inference/vllm.md) — Liquid AI, 2026. LFM2.5 dense, MoE and VL are **native** to vLLM from v0.23.0 (`Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, `Lfm2VlForConditionalGeneration`) so "there's no... (Tier A, docs, rel 3)
170. [Liquid SGLang deployment docs](https://docs.liquid.ai/deployment/gpu-inference/sglang.md) — Liquid AI, 2026. Native `lfm2` tool-call parser and `<think>` reasoning; `sglang serve --model-path .. (Tier A, docs, rel 3)
171. [Liquid LM Studio docs](https://docs.liquid.ai/deployment/on-device/lm-studio.md) — Liquid AI, 2026. GUI route: search "LiquidAI"/"LFM2" in the Search tab, pick a quantisation level (**`Q4_K_M` recommended**), OpenAI-compatible API (Tier A, docs, rel 2)
172. [Hardware Evaluation guide](https://docs.liquid.ai/guides/hardware-evaluation.md) — Liquid AI, 2026. Liquid's own profiling recipe (Tier A, docs, rel 4)
173. [Use Case Evaluation guide](https://docs.liquid.ai/guides/use-case-evaluation.md) — Liquid AI, 2026. **"A fair evaluation of a small model often includes a light fine-tune (Tier A, docs, rel 4)
174. [Migration guide](https://docs.liquid.ai/guides/migration-guide.md) — Liquid AI, 2026. For teams moving off Qwen/Llama/Gemma (Tier A, docs, rel 3)
175. [Liquid Unsloth fine-tuning docs](https://docs.liquid.ai/lfm/fine-tuning/unsloth.md) — Liquid AI, 2026. Unsloth claims **2-5x faster, 70% less memory** (Tier A, docs, rel 4)
176. [Liquid dataset format docs](https://docs.liquid.ai/lfm/fine-tuning/datasets.md) — Liquid AI, 2026. Canonical shapes: SFT = `messages[]` with system/user/assistant; DPO = **explicit** `prompt`/`chosen`/`rejected` ("The explicit format is... (Tier A, docs, rel 3)
177. [liquid-audio package](https://raw.githubusercontent.com/Liquid4All/liquid-audio/main/README.md) — Liquid AI, 2026. `pip install liquid-audio` (Tier A, repo, rel 3)
178. [JevEmbed](https://raw.githubusercontent.com/HITsz-TMG/JevEmbed/main/README.md) — HITsz-TMG, 2026-09-25. **A structurally different route: frozen embeddings → decisions, with no LM fine-tuning at all.** "turn embeddings into decisions" -- Python API, CLI... (Tier A, repo, rel 3)
179. [mini-Jev — read the letter](https://raw.githubusercontent.com/r-ms/mini-jev/main/README.md) — r-ms, 2026-09. **The most methodologically rigorous source in this ledger: a PREREGISTERED study (`PREREG.md`, amendments v1.1-v1.3) on frozen Qwen3-4B-Instruct-2507,... (Tier A, repo, rel 5)
180. [Jevlike](https://raw.githubusercontent.com/vinnylarouge/jevlike/main/README.md) — vinnylarouge, 2026-09. **A third read-out geometry, and the only one that runs on image patches.** "Each option becomes a query vector… assigns attention weights to the... (Tier A, repo, rel 4)
181. [TinyJev](https://raw.githubusercontent.com/ankit-aglawe/tinyjev/main/README.md) — AnkitAI, 2026-09. MIT, PyPI-installable, weights at 0.6B and 4B (Tier A, repo, rel 4)
182. [JevK5](https://raw.githubusercontent.com/allebee/jevk5/main/README.md) — alibiserikbay / allebee, 2026-09. Apache-2.0, `/v1/systemone` wire-compatible (Tier A, repo, rel 4)
183. [Jev Persian Benchmark](https://raw.githubusercontent.com/ArmanJR/Jev-Persian-Benchmark/main/README.md) — ArmanJR, 2026-09-25. **An independent frozen cross-check of Jev vs Laya on 480 authored Persian questions**, dataset v1.0.0, identical inputs and scoring, no runtime model... (Tier A, repo, rel 4)
184. [OpenDecision](https://raw.githubusercontent.com/deepanwadhwa/OpenDecision/main/README.md) — deepanwadhwa, 2026-09. An open-source equivalent of Jev with Choice / Noul / Score plus a fourth primitive **`Relation`** reporting `supports` / `contradicts` / `unknown` /... (Tier A, repo, rel 3)
185. [Reflex](https://raw.githubusercontent.com/kaustav1996/reflex/main/README.md) — kaustav1996, 2026-09. A concrete production deployment: "A calibrated System One model checks each tool call, turn and voice transcript in about 400 ms, and code decides... (Tier A, repo, rel 3)
186. [jevlike-esp32](https://raw.githubusercontent.com/david-cermak/jevlike-esp32/main/README.md) — david-cermak, 2026-09. The same option-query scorer ported to **ESP32 embedded firmware** (ESP-IDF), training staying in Python (Tier C, repo, rel 2)
187. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo, Pleiss, Sun, Weinberger (Cornell), 2017-06-14. **The origin of the ECE binning debate this whole project inherits.** Finds modern NNs "poorly calibrated" `[dated: 2017]` and that depth, width, weight decay and... (Tier A, paper, rel 5)
188. [Conformal Risk Control](https://arxiv.org/abs/2208.02814) — Angelopoulos, Bates, Fisch, Lei, Schuster, 2022-08-04. Extends conformal prediction to control the expected value of **any monotone loss function**, generalising split conformal together with its coverage... (Tier A, paper, rel 4)
189. [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://arxiv.org/abs/1901.09192) — Geifman, El-Yaniv, 2019-01-26. "Existing rejection mechanisms are based mostly on a threshold over the prediction confidence of a **pre-trained** network (Tier A, paper, rel 4)
190. [AUC-based Selective Classification](https://arxiv.org/abs/2210.10703) — Pugnana, Ruggieri, 2022-10-19. "In many application scenarios, such as **credit scoring**, performance is instead measured by ranking metrics, such as the Area Under the ROC Curve."... (Tier A, paper, rel 3)
191. [Parameterized Temperature Scaling (PTS)](https://arxiv.org/abs/2102.12182) — Tomani, Cremers, Buettner, 2021-02-24. "the performance of accuracy-preserving state-of-the-art post-hoc calibrators is limited by their **intrinsic expressive power**" (Tier A, paper, rel 4)
192. [Does confidence calibration improve conformal prediction?](https://arxiv.org/abs/2402.04344) — Xi, Huang, Liu, Feng, Wei, 2024-02-06. "current confidence calibration methods (e.g., temperature scaling) **typically lead to larger prediction sets** in adaptive conformal prediction", and... (Tier A, paper, rel 4)
193. [LFM2.5-230M config](https://huggingface.co/LiquidAI/LFM2.5-230M/raw/main/config.json) — Liquid AI, 2026. **A smaller sibling with the SAME hidden size as the 350M: `hidden_size 1024`, 14 layers = 9 conv + 5 full_attention, `intermediate_size 2560`,... (Tier A, repo/config, rel 4)
194. [LFM2.5-230M card](https://huggingface.co/LiquidAI/LFM2.5-230M/raw/main/README.md) — Liquid AI, 2026. The `<1B` entry in Liquid's own migration table, recommended for "classification, extraction, routing, and tight memory budgets" (Tier A, modelcard, rel 3)
195. [LFM2.5-Encoder-230M card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/raw/main/README.md) — Liquid AI, 2026. The smaller bidirectional sibling of the encoder family (229.7M), 15 languages, 8,192 context (Tier A, modelcard, rel 3)
196. [LFM2.5-350M-GGUF card](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/raw/main/README.md) — Liquid AI, 2026. The official llama.cpp distribution of our chosen backbone, `license: other` / `lfm1.0` — the exact artifact the deployment ladder in the guide points at (Tier A, modelcard, rel 3)
197. [LFM2.5-VL-450M-GGUF card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/raw/main/README.md) — Liquid AI, 2026. The llama.cpp distribution of the multimodal backbone, with the vision tower as a separate `mmproj` file (Tier A, modelcard, rel 3)
198. [LEAP Finetune](https://docs.liquid.ai/lfm/fine-tuning/leap-finetune.md) — Liquid AI, 2026. Liquid's full customisation repo (`Liquid4All/leap-finetune`): SFT / DPO / GRPO, VLM and MoE variants, LoRA and full fine-tuning; training-time... (Tier A, docs, rel 4)
199. [Liquid tool-use docs](https://docs.liquid.ai/lfm/key-concepts/tool-use.md) — Liquid AI, 2026. The four-step tool-use workflow (Tier A, docs, rel 3)
200. [Liquid prompting guide](https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting.md) — Liquid AI, 2026. Three prompt roles (system / user / assistant) (Tier A, docs, rel 3)

**I. Community and practitioner sources** (added in a late pass after the Hacker News
Algolia API was found to work; see Methodology)

201. [Ollaya](https://ollaya.dev/) — Ollaya, 2026-09-25. "Ollama for open-source, Jev-style decision models." The largest community signal in this space: 570 points, 137 comments, HN front page, the day before this pass ran (Tier A, docs, rel 5)
202. [HN discussion: Ollaya](https://news.ycombinator.com/item?id=49848269) — Hacker News, 2026-09-25. The 137-comment practitioner thread: sceptical, and the source of the report's Contradictions 6 and 7 (Tier C, forum, rel 5)
203. [ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya) — the runtime, the model zoo, and the `Modelfile` calibration refit (Tier A, repo, rel 4)
204. [sqliteai/blink](https://raw.githubusercontent.com/sqliteai/blink/main/README.md) — marcobambini, 2026-09-22. One-pass typed decisions in C99 with zero allocations in scoring, 66 KB WASM, 18,211 decisions/s. Its own honest ceiling: at or a little above chance where the answer requires reading (Tier A, repo, rel 5)
205. [Show HN: Blink](https://news.ycombinator.com/item?id=49800787) — Hacker News, 2026-09-22. "<200 microseconds per decision, and memory is never allocated while scoring" (Tier C, forum, rel 3)
206. [Turning GLM-5.3-Flash into a Jev-like decision model](https://www.privatemode.ai/blog/system-one-from-glm-flash) — privatemode.ai, 2026-09-26. Retrofitting System One behaviour onto an existing instruction model (Tier B, blog, rel 3)
207. [lateos-ai/reflex](https://github.com/lateos-ai/reflex) — a local Jev-like runtime targeting a 16GB NVIDIA GPU. Name collision with ledger row 89, unrelated project (Tier C, repo, rel 2)
208. [sshh12/nanojev](https://github.com/sshh12/nanojev) — the whole design in 200 lines; evidence of how low the floor is (Tier C, repo, rel 2)
209. [HN: LFM2.5 2.6B competitive with 4x larger models](https://news.ycombinator.com/item?id=49173107) — Hacker News, 2026-08-04. 169 points, 40 comments: independent reception of this project's chosen backbone (Tier C, forum, rel 3)
210. [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) — multimodalart, 2026. A third-party leaderboard, referenced independently in the HN thread (Tier A, docs, rel 4)
211. [ankit-aglawe/tinyjev](https://github.com/ankit-aglawe/tinyjev) — a fourth local-runtime entry, published as a HF model (Tier C, repo, rel 2)

**J. Reddit / practitioner-deployment layer** (added via a public Redlib instance after
`reddit.com/search.json` proved 403 to unauthenticated agents)

212. [typed-decision-bench results](https://kyr0.github.io/typed-decision-bench/) — kyr0, 2026-09-23. 7 models, 275 capabilities, 22,001 decisions, 5,499 calibrations, all on one H200 NVL. The best-controlled cross-model table in the corpus, and it supersedes Ollaya for accuracy (Tier A, docs, rel 5)
213. [Laya model card](https://huggingface.co/convaiinnovations/laya/raw/main/README.md) — NandhaKishorM / Convai Innovations, 2026-09. 421M Apache-2.0 RLCD decision model; **fine-tuned checkpoint scores 0.766 on `typed-decisions`**, the project's own benchmark (Tier A, modelcard, rel 5)
214. [r/LocalLLaMA: "I literally built the Jev architecture one year back"](https://www.reddit.com/r/LocalLLaMA/comments/1wijo3e/) — 3.3k upvotes, 317 comments. Prior-art claim citing arXiv 2503.23303 and 2510.01237. **Neither identifier could be verified** (Tier C, forum, rel 5)
215. [r/ai_trading: Jev trades for 24h](https://www.reddit.com/r/ai_trading/comments/1wkq4lt/) — 731 trades, −3.15%, 779 ms/decision, costs supplied explicitly. The corpus's most valuable negative result (Tier C, forum, rel 5)
216. [r/LLMDevs: I reviewed 287 open-source Jev projects](https://www.reddit.com/r/LLMDevs/comments/1wko2e5/) — 223 upvotes. Contains the Vercel `json-render` result: **3.21 s → 0.88 s**. **Name collision: the "Blink" here is not sqliteai/blink** (Tier C, forum, rel 5)
217. [r/LocalLLaMA: Laya release](https://www.reddit.com/r/LocalLLaMA/comments/1wjieap/) — the 421M architecture in the author's words: bidirectional ModernBERT-large + `[MASK]` option markers, one RTX 6000 Pro (Tier C, forum, rel 4)
218. [r/accelerate: TypeSafe launch thread](https://www.reddit.com/r/accelerate/comments/1wht23u/) — Diogo Almeida. "20-200x faster, 40-400x cheaper", $0.042/MTok, output free. Quoted **as marketing copy** (Tier C, forum, rel 4)
219. [r/LLMDevs: choosekit on local Qwen](https://www.reddit.com/r/LLMDevs/comments/1wkc9hp/) — Qwen3.8 27B Q4 XL and Jev both 96.53% on SemIf 144 tasks; 239 ms local vs 368 ms hosted (Tier C, forum, rel 4)
220. [r/learnmachinelearning: why use JEV at all?](https://www.reddit.com/r/learnmachinelearning/comments/1wpoteb/) — the symbolic-boundary challenge. Not anti-Jev; asks where the line is (Tier C, forum, rel 4)
221. [kyr0/typed-decision-bench](https://github.com/kyr0/typed-decision-bench) — the benchmark source; proposes a `calibration.json` post-hoc calibration standard (Tier A, repo, rel 4)
222. [kyr0/Bonsai-Llama-Jev](https://github.com/kyr0/Bonsai-Llama-Jev) — `POST /v1/systemone` inside a llama.cpp fork; official TypeSafe SDKs work unchanged. Source of the **"RLCD is overrated, use CE"** challenge (Tier A, repo, rel 4)
223. [The benchmark, post-hoc calibration method, and why it works](https://kyr0.github.io/typed-decision-bench/paper/) — kyr0, 2026-09. Invites refutation; concedes the benchmark is synthetic (Tier B, paper, rel 4)
224. [logicrw/awesome-jev-projects](https://github.com/logicrw/awesome-jev-projects) — the 287-project directory (Tier A, repo, rel 3)
225. [r/homeassistant: Jev for smart-home control](https://www.reddit.com/r/homeassistant/comments/1wjmqj0/) — 58 upvotes. The bounded-control-plane use case (Tier C, forum, rel 2)
226. [r/hermesagent: Jev-style layer on a 24/7 trader](https://www.reddit.com/r/accelerate/comments/1wn1gck/) — 278 upvotes. **Contradicts row 215's outcome the same week.** Neither anecdote is evidence (Tier C, forum, rel 2)

**K. Ecosystem census**

227. [Awesome Jev — System-1 Agent Architecture Radar](https://raw.githubusercontent.com/logicrw/awesome-jev-projects/main/README.md) — logicrw, 2026-09-26. 693 curated projects in 18 categories, each with a standardised provenance block. 56% orchestration/plumbing/safety/eval, 6% classification-class, **2 of 693 under Classification** (Tier B, repo, rel 5)
228. [Awesome Jev — caveat counts](https://raw.githubusercontent.com/logicrw/awesome-jev-projects/main/README.md) — same source, counted: **376 of 693 entries** marked "performance and cost benefits have not been independently verified"; **352 of 693** say "consult the source for the exact decision policy". Exact `grep -c` over the raw README (Tier B, repo, rel 5)
229. [Awesome Jev — architecture comparison table](https://raw.githubusercontent.com/logicrw/awesome-jev-projects/main/README.md) — same source: the radar claims **"Sub-100ms Latency: 50-100ms"** for TypeSafe Jev, ~7x faster than typed-decision-bench's 716 ms p50. No version, question count or network condition stated (Tier C, repo, rel 4)
230. [Awesome Jev live radar](https://logicrw.github.io/awesome-jev-projects/en/) — the filterable category view, plus `llms.txt` / `llms-full.txt` agent-skill files that were **not fetched in this pass** (Tier B, docs, rel 3)
