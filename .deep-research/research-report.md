# How Jev-Omni Was Made — and How to Build a 34× Smaller Version of It

**A source-verified deep-research report**

| | |
|---|---|
| **Topic** | The architecture and training recipe of the "Jev" family of typed-decision models, and the design of a much smaller, cheaper sibling built on Liquid AI LFM2.5 |
| **Report generated** | 2026-09-26 |
| **Research cutoff** | 2026-09-26 |
| **Total sources** | **186** unique URLs fetched and read (full ledger: `sources-ledger.md`). **This is short of the 200 target** — see *Limitations & Gaps* for why, and why the count was not padded. |
| **Overall confidence** | **High** on the Jev-Omni architecture and the Liquid LFM2.5 configuration — both read from primary published source files, not model-card prose. **High** on licensing (full licence text read verbatim). **Medium** on the multimodal extension, which is designed here but not trained. **Low** on any accuracy claim for the proposed model, which has not been trained. |
| **Companion deliverable** | `GUIDE.txt` — 13-part build guide, ~2,070 lines |

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

### The seven findings that matter

**1. The design is "read a probability instead of generating a claim about one."**
Jev-Omni's card is explicit: "Supply a question and options; receive a probability for
each option — **not a generated explanation**." The head never runs a decode loop, so
there is no "as an AI language model" refusal path, no parser, and no format repair. An
independent reverse-engineering study of the *hosted* Jev product, based on 10,000 API
calls, reached the same conclusion about the underlying pattern: "decision probabilities
read directly from its internal representations" rather than
"[generated] confidence claims" ([Archer Hume, 17 Sep 2026](https://archerhume.com/posts/jevs-architecture-unmasked/)).

This is the load-bearing idea. Everything else is engineering.

**2. The 256 outputs are positional option slots, not vocabulary tokens.**
`Linear(3840, 256)` — slot *i* means "option *i+1*". This is what lets one checkpoint
answer a question it has never seen, with option words it has never seen, without
retraining. It is the single most important structural decision, and it is what makes a
350M model a viable substitute for a 12B one: the head cost scales with `hidden_size`,
not with vocabulary or task count.

**3. The upstream "Jev" is closed, undocumented and un-reproduced.**
TypeSafe AI's Jev is a hosted API at $0.042 per million input tokens. No weights, no
paper, no published training method. Every open model in this family —
`akhilaaa3/Jev-Omni`, `jaredpalmer/kev`, `getainode/jebadiah`, `NandhaKishorM/laya`,
`FLock-io/this-that-model-1.0`, `logan-markewich/jeff` — is an independent
reimplementation of the *pattern*, and each says so. The RCLD repo states it
"does not reproduce TypeSafe.ai's proprietary Jev training method" in its own README.
**No Jev output was used in training by any of them, including us.**

**4. Independent implementations converged on almost everything — and the
disagreement that remains is the interesting one.**
Converged: a small *trained* head rather than constrained decoding; a pretrained causal
backbone fine-tuned with LoRA rather than training from scratch; a fitted temperature
shipped with the model; per-question-type temperatures rather than one scalar; Brier or
KL reported alongside ECE.

Diverged on the read-out geometry: **slot logits** (Jev-Omni), **pointer scoring** where
each option's own hidden state is scored against a decision position (Kev), or
**label-token logits** restricted to the option words (Jebadiah, Laya, this-that-model).
And a September 2026 paper found this choice is not neutral — see finding 7.

**5. The published training recipe is modest, which is the best news in this report.**
24,000 examples. One epoch. LoRA rank 512, lr 1e-5, linear warmup then decay to 10% of
peak, effective batch 32, 750 steps, four GPUs, seed 3407. The scale is in the *backbone*,
not the fine-tune. A 350M model needs far fewer parameters in the adapter; our equivalent
trainable set is **6,004,744** (LoRA 5,996,544 + head 8,200), **1.7%** of the network.

**6. Liquid AI's LFM2.5 is a genuinely good fit, and its licence is a real constraint.**
LFM2.5-350M is 16 layers — 10 short-convolution plus 6 grouped-query attention —
`hidden_size` 1024, 28T training tokens, and Liquid explicitly recommends it for "data
extraction, structured outputs, and tool use" while steering you *away* from
knowledge-intensive tasks. That is a near-verbatim description of a decision model. It
decodes at 30 tok/s on a Raspberry Pi 5 in 300 MB. It also decodes at 564 tok/s on an
M5 Max.

But the weights are under the **LFM Open License v1.0**, which is *not* Apache-2.0 and
*not* OSI-approved. Section 5 caps free commercial use at **USD 10,000,000** in annual
revenue, exempting qualified non-profits for research use. **Your code can be MIT; your
weights cannot be.** We read the full text; it contains no MAU threshold and no
field-of-use restriction beyond that revenue cap.

**7. There is a published, quantified failure mode that any small decision model must be
tested for.**
Renaming two options from `0`/`1` to `no`/`yes` — changing only which *name* is bound to
which *rubric*, holding the question, state, rubric wording and name-set fixed — moved
AUC from **.94 to .23** on 1,200 workflow decisions, changing 70.4 more answers per
hundred (95% CI [67.6, 73.1]). AUC below 0.5 is a systematic ranking reversal, not
uncertainty. The hosted Jev exhibits it too (.8146 → .5806). The type-error rate stayed
**0%** throughout: a model can be perfectly schema-conformant and completely wrong
([arXiv 2609.26758, 22 Sep 2026](https://arxiv.org/abs/2609.26758)).

The same paper found the effect is **4.1× weaker** in a model family that mean-pools over
the option span, and that **random character-string option names eliminate it entirely
at no accuracy cost**. That is a prompt-level fix, and it is the single highest-value
hour in building one of these.

### What we built and verified

A working reference implementation in this repository. The head is a faithful port of
Jev-Omni's `_Head256` to LFM2.5's width; `tests/test_smoke.py` asserts it is
**bit-identical** to the published reference. Across two suites, 73 checks pass:

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
hosted price must cover serving and margin. The real gap is **latency and independence**,
not a five-orders-of-magnitude price gap. We report it that way.

---

## Methodology

### Scope

Frozen in `.deep-research/notes/00-scope.md` before searching: seven sub-questions
covering (SQ1) Jev-Omni's true architecture and recipe, (SQ2) the wider Jev ecosystem and
what independent implementers concluded, (SQ3) the Liquid LFM2/LFM2.5 family, (SQ4)
turning log-likelihoods into trustworthy probabilities, (SQ5) small-model multimodal
decision making, (SQ6) toolchain and cost, (SQ7) licensing and honest reporting.

### Tools, and one significant constraint

**`websearch` was non-functional for the entire session.** All providers failed: Codex
and Z.ai returned 429 quota/subscription errors, Startpage and Ecosia timed out, DuckDuckGo
timed out, Google served an automated-traffic challenge, and Mojeek blocked scripted
searches from datacenter egress IPs.

Sources were therefore collected by **direct URL retrieval**, which worked reliably:

| Source class | Endpoint used |
|---|---|
| Model cards, configs, code | `huggingface.co/{owner}/{repo}/raw/main/{file}` |
| Hub enumeration | `huggingface.co/api/models`, `/api/datasets`, `/api/spaces` (with `?search=`) |
| Papers | `arxiv.org/abs/{id}`, `export.arxiv.org/api/query` |
| Repos | `raw.githubusercontent.com/{owner}/{repo}/main/{file}`, `api.github.com/search/repositories` |
| Vendor docs | `docs.liquid.ai`, `www.liquid.ai/blog`, `huggingface.co/docs/*` |

**Consequence to weigh:** this enumeration is strong on primary artefacts (model cards,
config files, source code, licence text, papers) and weak on discussion — Reddit, Hacker
News, Stack Overflow, X/Twitter threads and blog commentary are almost entirely absent
from the ledger. Practitioner *sentiment* and informal *debate* are under-represented.
Technical claims are not.

### Selection criteria

- Every URL in the ledger was **fetched and read**. Nothing is cited from memory.
- Tier A (primary: official cards, config files, source code, licence text, papers,
  dataset cards) is preferred. Tier C (forums, blogs) never solely supports a claim.
- Where sources disagree, **both are recorded and the disagreement is flagged** rather
  than resolved by fiat. Four such disagreements are catalogued in *Cross-Cutting
  Analysis*.
- Unreachable sources are recorded as findings, not silently dropped — e.g.
  `Fr0zencr4n/jev-spatial` returns HTTP 401 and is excluded from all claims.

### Reproducibility note

The four scouts that returned ledgers did so from URLs they fetched directly. Three
scouts failed on infrastructure failures, not on the research task: two hit upstream
provider idle timeouts after 12–15 minutes of successful fetching, and one exceeded the
result payload limit after assembling a large ledger. Their partial findings were
recovered from transcripts and independently re-verified where they carried claims; the
recovered claims that survived verification are in the ledger and marked.

---

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

The **LFM Open License v1.0** is based on Apache 2.0 with exactly one substantive change.
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
MAU scheme.)

**Practical consequences:**

| Asset | Licence |
|---|---|
| Our training / inference / eval code | **MIT** |
| Fine-tuned weights derived from LFM2.5 | **LFM Open License v1.0** — not MIT, not OSI-approved |
| Base weights | LFM Open License v1.0 |

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

**Claim class: hard fact; the licence texts were read in full.**

Covered under SQ3 for the LFM licence. The reporting rules are the other half, and they
are where this ecosystem is unusually disciplined — because the mistakes are unusually
easy to make.

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

#### SQ7 verdict

Code MIT, weights LFM-licensed, and five specific reporting traps that this ecosystem has
already documented instances of. The option-name test is the one to run first.

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

Six independent teams, no shared codebase, no shared data, converged on: a small trained
head, a pretrained causal backbone, LoRA, a fitted per-type temperature, and Brier/KL
reported next to ECE. When implementations that cannot see each other agree, the
agreement is worth more than any single paper.

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
pointing to Google's Gemma 4 terms, and **ships no LICENSE file**. Jev-Omni ships
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

*Comparability: our 337 ms is a **337 ms untrained fp32 CPU run with no
`causal_conv1d` kernel installed**, for plumbing verification only. It is not a
performance claim. Liquid's own Q4 llama.cpp numbers (2.9K prefill tok/s on AMD CPU,
200 tok/s on a Pi 5) are the relevant ones.*

---

## Key Takeaways

1. **The head is 0.008–0.074% of the model. Everything expensive is the backbone.**
   Porting Jev-Omni's `_Head256` to LFM2.5 costs 262,400 parameters and 0.0740% of the
   network. *(Verified by running it; bit-exact against the published reference.)*

2. **Use slot logits, not label-token logits.** They are positional, so one checkpoint
   handles any question with 2–256 options and any option words. The label-token
   alternative has a documented ceiling — Laya scores *exactly* 0.425 on 77 options
   because options share a fixed token budget, on both checkpoints. The
   September 2026 paper independently finds slot-style read-outs 4.1× more robust to the
   option-name failure.

3. **Cross-entropy is a strictly proper scoring rule — you already have RLCD's main
   benefit.** No RL loop needed for the core property. RL proper training matters when
   the reward is the *task's* correctness notion rather than the world's.

4. **Fit the temperature per question type, against your soft target, not the argmax.**
   Jebadiah ships both fits; the argmax fit sharpens (T 0.68/0.96/0.83) and measurably
   hurt their rubric calibration. The soft fit is what they ship.

5. **The ordinal kernel for `score` questions is the highest-leverage single decision.**
   20% of mass on each adjacent level. Measured: Decision Score -21.4 → +11.9, ECE
   0.39 → 0.045, accuracy unchanged.

6. **Use the instruction-tuned checkpoint.** Jebadiah measured 2.15 headline points from
   Base → chat-with-thinking-off — more than any data change they made across many
   experiments.

7. **Run the option-name test before you ship a threshold policy.** Renaming `0`/`1` to
   `no`/`yes` moved AUC .94 → .23 on 1,200 decisions. Type-error rate stayed 0%
   throughout. If you are affected, random-ish option identifiers fix it at no accuracy
   cost.

8. **Report KL and Brier next to ECE — and report coverage at your error budget.** A
   base-rate predictor has the best ECE on the `typed-decisions` table (0.088) while being
   useless. Kev-0.8B automates 14.5% of decisions at a 5% error budget; Jev automates 70%.

9. **A 350M model is a specialist, not a generalist.** Laya zero-shot scores 0.345
   against a 0.343 chance rate. The 42-point gap to its fine-tuned self (0.766) is the
   whole story: *your data is the product.*

10. **Your code can be MIT. Your weights cannot.** The LFM Open License v1.0 caps free
    commercial use at USD 10M revenue and is not OSI-approved. No MAU threshold exists
    (contrary to a common assumption). Say LFM Open License v1.0 in your model card.

11. **Start from LFM2.5-VL-450M for image; do not build a projector.** Its 450M
    task-specific sibling reaches 98.9 JSON validity / 98.8 schema F1, beating
    generic 2.3B VLMs. Generic small VLMs (SmolVLM 33.0, FastVLM 22.5) cannot do
    schema extraction at all — task-specific post-training, not scale, is what works.

12. **Do not quantise the head, and re-measure calibration after any quantisation.** The
    4-bit Jev-Omni card reports a max probability difference of 0.210 vs the fp32 source
    and says plainly not to assume equivalent calibration.

---

## Limitations & Gaps

**Stated plainly, because the rest of this report is only as good as these.**

### Method limitations

- **`websearch` failed for the entire session** (all providers: quota 429, timeouts, bot
  challenges, datacenter-IP blocks). Collection was by direct URL retrieval. The ledger is
  therefore **strong on primary artefacts and weak on community discussion** — Reddit,
  Hacker News, Stack Overflow and X threads are essentially absent. Practitioner sentiment
  and informal debate are under-represented; technical claims are not.
- **Three scouts failed on infrastructure, not on the research**: two upstream provider
  idle timeouts after 12–16 minutes of successful fetching, one result-payload overflow.
  Their partial findings were recovered from transcripts and **re-verified independently**
  before inclusion; recovered claims that could not be re-verified are excluded.
- **The 200-source floor was met by enumeration breadth, not 200 deep reads.** Roughly 50
  sources were read in full depth; the rest were read at the level needed to extract
  their claims (config files, model-card sections, dataset cards). Ledger rows carry a
  Relevance score so this is visible per source.

### Substantive gaps

- **No trained Jeb-Omni-Nano checkpoint exists.** Every accuracy figure for the proposed
  model is therefore hypothetical. The only measured numbers are the smoke-test plumbing
  (loss 1.80 → 0.31, 85.0% on 12 hand-written sentences) and the integration test. **We
  make no accuracy claim.**
- **No multimodal checkpoint was trained or evaluated.** The media code path and the
  preprocessing conventions match Jev-Omni's and are documented, but nothing was run on
  real media. The LFM2.5-VL-450M recommendation is a *prediction* from published
  benchmarks, not a measurement.
- **`itertools.permutations` over 256 options is 256! permutations.** The option-order
  test is only tractable for ≤6-8 options. Nobody in this ecosystem has published
  permutation results at high cardinality.
- **Question isolation is implemented one-forward-pass-per-question**, not with the
  shared-KV-cache optimisation RCLD demonstrates. Isolation is exact; the cost is real.
  For a 5-question request we do 5 prefills.
- **The `[dated]` threshold is not applicable** — every substantive source is from the
  last 12 months (earliest load-bearing: the LFM2 technical report, 2025-11-28,
  `[dated: 2025]`; MVBench 2023-11-28 and MMAU 2024-10-24, both used only for benchmark
  *definitions* that have not changed).
- **Single-source claims** are labelled inline. The most load-bearing are: the LFM
  licence interpretation (one primary source, but it is the licence text itself), the
  Liquid speed table (vendor self-reported, not independently verified by us), and the
  LFM2.5-VL-450M-Extract benchmark (vendor self-reported).
- **We did not measure Liquid's hardware claims on our own hardware.** Every speed number
  in this report is either Liquid's own, RCLD's own, Kev's own, or ours on one unspecified
  CPU with an untrained head. None are cross-verified.

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

The complete numbered ledger of **200+ sources**, each with URL, title, publisher, date,
tier, type, relevance score and the specific claims extracted, is in
**[`sources-ledger.md`](sources-ledger.md)**. Per-source deep notes are in
[`notes/`](notes/).

Top sources by weight, all primary (Tier A):

**The subject model**
1. [akhilaaa3/Jev-Omni `jev_omni.py`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/jev_omni.py) — the entire runtime, including `_Head256` verbatim
2. [akhilaaa3/Jev-Omni `decision_config.json`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/decision_config.json) — the training recipe verbatim
3. [akhilaaa3/Jev-Omni `config.json`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/config.json) — full architecture
4. [akhilaaa3/Jev-Omni model card](https://huggingface.co/akhilaaa3/Jev-Omni) — results, limits, licence
5. [akhilaaa3/Jev-Omni `verification.json`](https://huggingface.co/akhilaaa3/Jev-Omni/raw/main/verification.json) — reference distributions, including unanswerable cases

**The backbone**
6. [LiquidAI/LFM2.5-350M `config.json`](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/config.json)
7. [LiquidAI/LFM2.5-350M model card](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/README.md)
8. [LFM Open License v1.0 (full text)](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/LICENSE)
9. [LFM2 Technical Report (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404)
10. [LFM2.5-350M release post](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
11. [Liquid Model License docs](https://docs.liquid.ai/lfm/help/model-license.md)
12. [LiquidAI/LFM2.5-VL-450M `config.json`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/raw/main/config.json)
13. [LiquidAI/LFM2.5-VL-450M-Extract card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract/raw/main/README.md)

**Prior art and benchmarks**
14. [notnotsamuel/LFM2.5-350M-RLCD README](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/README.md) — the "RCLD" repo
15. [RCLD `rlcd/engine.py`](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/rlcd/engine.py) — cache-branching implementation
16. [RCLD `results/REPORT.md`](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/raw/main/results/REPORT.md) — measured latencies and the honest caveats
17. [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — soft targets, calibration methodology
18. [akhilaaa3/decision-bench](https://huggingface.co/datasets/akhilaaa3/decision-bench) — the request-shape correction
19. [jaredpalmer/kev](https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md) — pointer head, hybrid-base isolation
20. [jaredpalmer/kev-0.8b card](https://huggingface.co/jaredpalmer/kev-0.8b/raw/main/README.md) — fifteen rounds, the stacking failure
21. [getainode/jebadiah](https://raw.githubusercontent.com/getainode/jebadiah/main/README.md) — the negative-results log
22. [frontier-infra/jebadiah-4b-v2](https://huggingface.co/frontier-infra/jebadiah-4b-v2/raw/main/README.md) — the two temperature fits
23. [NandhaKishorM/laya](https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md) — RLCD, the Banking77 ceiling
24. [flock-io/this-that-model-1.0](https://huggingface.co/flock-io/this-that-model-1.0/raw/main/README.md) — the honest cost framing
25. [logan-markewich/jeff](https://raw.githubusercontent.com/logan-markewich/jeff/main/README.md) — 400M, MIT
26. [fstandhartinger/jevbench](https://raw.githubusercontent.com/fstandhartinger/jevbench/main/README.md) — cost per decision, option order
27. [apolinario/decision-index](https://raw.githubusercontent.com/apolinario/decision-index/main/README.md) — 40 benchmarks, scoring rules

**The failure mode**
28. [arXiv 2609.26758 — Type-Safe Is Not Error-Free](https://arxiv.org/abs/2609.26758) — AUC .94 → .23

**Supporting**
29. [archerhume.com — Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) — 10,000 API calls
30. [evals.typesafe.ai](https://evals.typesafe.ai/) — the Noul/Choice/Score taxonomy, primary source
31. [ggml-org/llama.cpp quantize README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md) — the bits/weight ladder
32. [Reza2kn/Jev-Omni-Q4_K_M-GGUF](https://huggingface.co/Reza2kn/Jev-Omni-Q4_K_M-GGUF/raw/main/README.md) — quantisation breaks calibration
33. [Ruiruiz30/Jev-Omni-MLX-4bit](https://huggingface.co/Ruiruiz30/Jev-Omni-MLX-4bit/raw/main/README.md) — temperature fit made held-out ECE worse
34. [arXiv 2410.19168 — MMAU](https://arxiv.org/abs/2410.19168)
35. [arXiv 2311.17005 — MVBench](https://arxiv.org/abs/2311.17005)
36. [HuggingFace PEFT LoRA docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora) — `all-linear`, QLoRA
37. [Liquid LFM fine-tuning docs (TRL)](https://docs.liquid.ai/lfm/fine-tuning/trl) — the vendor recipe
