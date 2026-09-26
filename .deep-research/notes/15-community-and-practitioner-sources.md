# Note 15 — Community and practitioner sources (fills the gap the Methodology admits)

**Date written:** 2026-09-26
**Why written:** the report's own Methodology said search coverage was *"strong on
primary artefacts and weak on community discussion… Reddit threads, Hacker News
discussions, Stack Overflow answers and news coverage are largely absent."* That
was true and it was my failure, not a property of the topic: `websearch` had
been non-functional for the whole project and I never found a working
substitute for the discussion layer.

The substitute that works is the **Hacker News Algolia API**
(`hn.algolia.com/api/v1/search` and `/api/v1/items/<id>`). It needs no API key,
returns full JSON including complete comment trees, and is not rate-limited the
way search engines were. Reddit's `search.json` endpoint returns HTTP 403 to
non-authenticated agents, so Reddit remains uncovered — recorded as a residual
gap below, not papered over.

## What the discussion layer actually says

### 1. The largest signal in this space was invisible to me

A 570-point, 137-comment **front-page** HN thread posted **2026-09-25** — one day
before this note — on **Ollaya**, an "Ollama for open-source decision models"
([ollaya.dev](https://ollaya.dev/), Apache-2.0,
[github.com/ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)).

Its benchmark table is the single best independent latency comparison I have
found, and it is measured on one machine by one party on five-question
requests through the HTTP API, RTX 4090:

| model | median latency | note |
|---|---:|---|
| `laya:multilingual` | **8.1 ms** | Convai Innovations, 322m/421m |
| `laya:en` | 9.6 ms | same family, English |
| `gliclass` | 14.7 ms | Knowledgator instruction-following zero-shot |
| `nli` | 20.4 ms | Moritz Laurer zero-shot entailment |
| `decider:0.8b` | 155 ms | Mapika, Qwen3.5, option-letter logits |
| `decider:2b` | 190 ms | same |
| **TypeSafe Jev (hosted API)** | **236–276 ms** | third-party hosted latency, includes network |

Ollaya's own caveat is the correct one to carry forward: *"Setups differ, so read
it as an order-of-magnitude comparison."* It is not a controlled benchmark
against Jev, and the Jev figure comes from third-party hosted measurements
([AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks),
[nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark)),
not from TypeSafe. **Both sides of that comparison are third-party, and the two
halves were measured on different hardware.** The report's existing cost section
already refuses to compare Jev's in-process 83 ms against a hosted API number;
this finding does not license a stronger claim, only a weaker one: local
decision models are plausibly an order of magnitude faster than a hosted Jev
round trip, and no open model in this table is *both* fast and accurate.

The ecosystem-list detail that matters most for design: **every model ships its
own calibration, and Ollaya's `Modelfile` refits it on your own labelled data.**
That is an independent implementation of exactly the per-bucket temperature
fitting this project does in `calibration.py`. Two parties arriving at
per-model calibration is corroboration, not novelty.

Ollaya also serves TypeSafe's `/v1/systemone` shape, so **the official TypeSafe
Python SDK 0.7.1 runs against it unchanged** (`TYPESAFE_BASE_URL=http://localhost:11435`).
That is a materially different deployment story from "ship your own weights":
it makes the *interface* the durable artefact and the *weights* replaceable,
which is the same architectural bet this project's `decision-interface`
contract makes.

### 2. The practitioner consensus is sceptical, and it agrees with my measurements

From the 137-comment thread:

- *"Are there many models that are comparable to Jev for generic decision
  making? **Smarter move if you have an eval set is to just train a classifier
  and call it a day.**"* — top-level, upvoted.
- *"The best open ones are close to Jev now, **but they're big models**."*
- *"In a benchmark with actual decisions — navigation, traffic, waypoints —
  **laya does only slightly better than a small classifier**."*
- The strongest open entry is *"trained by the perplexity CTO for $3k"*, per a
  commenter pointing at the [jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
  Space. **Single source, unverified, no budget breakdown** — recorded as
  colour, not as a cost claim.

That third bullet is the important one. It is an independent report of a **small
gap between a fast decision model and a plain classifier on real task-shaped
decisions**, on a benchmark I have never seen. This project's own measurement
reaches a compatible conclusion from the other direction: a frozen
LFM2.5-350M backbone with only the head trained scores 0.2667 accuracy and
**0.0000 risk-coverage at a 5% error budget**, while the LoRA arm reaches 0.5778
and 0.0967. The practitioner claim and the measured claim are not the same
claim, and I have not merged them — but they point the same way, and the
practitioner version has an unseen benchmark behind it.

### 3. The mechanism is confirmed by a third party, not just by me

A commenter on the thread, unprompted:

> *"Their marketing language is misleading. They must still use some transformer
> language model backbone to encode the text input (BERT or decoder-only LLM).
> The biggest difference is the output, instead of auto-regressively generating
> tokens, they produce probabilities over a bounded set of decisions (more
> flexible classification)."*

This is **exactly** the report's central architectural claim — a decision model
is a transformer backbone with the decode loop replaced by a bounded read-out —
reached independently by a practitioner reading the TypeSafe announcement. It
raises TypeSafe's marketing to a contested claim (see Contradictions below).

The same thread also has a top-level commenter asserting *"text classification
is equivalent to decision. This is exactly the same thing Jev does."* That is
**overstated** and should not be adopted: text classification predicts one label
from a fixed set, whereas a decision model must score arbitrary per-call option
sets, handle score and boolean question types, and calibrate across those sets.
The report's typing section makes the narrower, defensible version of this
point, and the evidence supports the narrower version — the three question types
behave differently in my own measurements.

### 4. Blink documents the floor of the size trade-off, honestly

[marcobambini/blink](https://github.com/sqliteai/blink) (Show HN, 2026-09-22) is
the smallest serious artefact in this space: C99, no dependency beyond libc and
libm, **zero allocations during scoring** — and the test suite checks that
mechanically rather than by inspection, which is the standard this project
should be held to. Same sources build to a 66 KB WebAssembly module.

On one core of an Apple M5 Pro: **406 µs** for a fresh decision, **54 µs** for
the same state with a new question, **18,211 decisions/s** with state reused.
The state-reuse number is the same caching insight as the Jev-Omni "one call per
question" pricing note, arrived at from the other direction.

What makes this valuable is its own honesty about the ceiling:

> *"Blink recognises the **form** of a decision; it does not read text the way a
> pretrained language model does. Where the answer is carried by form — which
> queue a ticket's wording points at, whether a claim's verb agrees with the
> state — blink-tiny is near perfect and well calibrated. **Where the answer
> requires reading — natural language inference, binding a name to the right
> sentence — it is at or a little above chance, and a frozen 4B model is far
> ahead.** Its confidence is calibrated enough to branch on: answer when it is
> sure, escalate when it is not."

This is the clearest public statement of the trade-off this project sits on. A
form-based model at chance for reading tasks, versus a frozen 4B model far
ahead — that is my 0.2667-vs-0.5778 gap, stated by someone who built the
smallest version and measured where it stops working. The "answer when it is
sure, escalate when it is not" formulation is also the correct product framing
for a calibrated decision head, and matches the risk-coverage curve this project
measured.

### 5. Three more independent attempts, same interface

- [privatemode.ai](https://www.privatemode.ai/blog/system-one-from-glm-flash) —
  retrofitting System One behaviour onto an existing instruction model (GLM-5.3-Flash)
  instead of pretraining a decision model. Small footprint; a technique note, not evidence.
- [lateos-ai/reflex](https://github.com/lateos-ai/reflex) — a local Jev-like
  runtime targeting a 16GB NVIDIA GPU. Name collision with the cascade system in
  ledger row 89; unrelated project, flagged in the ledger to prevent a wrong merge.
- [sshh12/nanojev](https://github.com/sshh12/nanojev) — the whole design in 200
  lines, and [ankit-aglawe/tinyjev](https://github.com/ankit-aglawe/tinyjev) in a
  HF repo. Evidence of how low the floor is, and useful references for the
  minimal version of this design.

Five unrelated teams converging on "bounded read-out head over a general
backbone" is a strong convergence signal, and it is worth recording that this
conclusion did not require a single one of them to have read the others.

## Contradictions this pass introduced

1. **"Jev does not use an LLM" is contested, not settled.** TypeSafe's
   announcement language implies something that is not an LLM. A practitioner on
   the HN thread states the opposite reading: a transformer backbone is
   unavoidable, and the difference is only in the output head. The report
   already prefers the narrower architectural description, so **this
   strengthens the report's framing rather than forcing a change** — but the
   project should stop describing decision models as "not LLMs" flatly, because
   at least one informed reader rejects that phrasing. *Tier C single source;
   the claim is plausible and consistent with my own measurements, but it is one
   person's reading of a marketing page.*
2. **"Fast" is claimed for models that are not accurate.** Ollaya's own table
   makes the fastest models (`laya` at 8.1 ms) sit alongside `decider:2b` at
   190 ms, and a commenter reports `laya` beating a small classifier only
   slightly. **Latency and capability are separate axes in this ecosystem and are
   routinely plotted as if they were one.** The report already refuses to read a
   2.4x latency gap as a 2.4x capability gap; this is a second, independent
   confirmation of that discipline.
3. **Two "Jev is a classifier" positions both appear in one thread**, and they
   are not the same claim. See §3. The report keeps the narrow version.

## What I did *not* get

- **Reddit remains uncovered** (`search.json` → HTTP 403 without auth). A second
  independent Reddit front end may be reachable, but I have not tried one and
  will not claim coverage I do not have. Recorded as a residual gap.
- **Stack Overflow** — not searched. The Algolia API covers HN only.
- **The jev-decision-index leaderboard contents** were not fetched. A commenter
  cited it and its leader; I recorded the leader as unverified single-source
  colour rather than fetching and laundering it into a claim.
- **No new experiment.** These are read-only network calls. The host is still at
  load ~89–92, so nothing GPU-bound was attempted and nothing new was measured.
  **Every number in this note is somebody else's measurement, and none of it is
  mine.**
