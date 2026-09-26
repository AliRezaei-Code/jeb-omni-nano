# Note 17 — The ecosystem census, and what 693 projects say about the speed claims

**Date written:** 2026-09-26
**Why written:** note 16 listed "the 287 projects were not individually read" as a gap.
The directory itself turned out to be the source, and it is worth reading as data rather
than as a list.

---

## 1. The directory is larger than the community post said

The r/LLMDevs survey (row 216) reported "**287** source-reviewed projects" on 2026-09-25.
The same repository's README today says "**693+** curated projects" across 18 categories.
That is a 2.4x difference in about a day. Either growth really is that fast — plausible
for a directory that was a week old and clearly filling a vacuum — or the two count
differently (entries vs projects vs repos). **Recorded as a discrepancy, not resolved.**
It is a small thing, but it is the kind of number a directory should pin.

## 2. The category census is the real finding

| Category | n | | Category | n |
|---|---:|---|---|---:|
| SDK & Decision Frameworks | **120** | | Creative Tools | 20 |
| Domain Tools (vertical) | 69 | | Code Navigation | 14 |
| CLI & Pipelines | 62 | | SDK Integrations | 6 |
| **Security & Guardrails** | **58** | | Voice & Conversation | 4 |
| Model Routing | 55 | | **Classification** | **2** |
| High-Frequency / Games | 52 | | | |
| Browser & OS Action | 50 | | | |
| Data & Search | 44 | | | |
| MCP & Integrations | 43 | | | |
| Context GC | 40 | | | |
| Evaluation & Observability | 29 | | | |
| Decision Tools | 25 | | | |
| **Total** | **693** | | | |

Grouping the categories:

- **Orchestration, plumbing, safety and evaluation — 387 projects, 56%.** Frameworks,
  routing, MCP, guardrails, context GC, CLIs, evaluation harnesses.
- **Classification-class work — 41 projects, 6%.** Classification (2), code navigation
  (14), decision tools (25).
- **Demos, verticals and toys — 145, 21%.** Domain tools, games, creative tools, voice.

**Only 2 of 693 projects are filed under Classification.** A field whose entire pitch is
"a small model that classifies" has spent its first weeks overwhelmingly building
**plumbing around classifiers** — adapters, routers, MCP servers, guardrails — rather
than classifiers.

That is the strongest possible corroboration of the practitioner consensus on the
largest thread: *"Smarter move if you have an eval set is to just train a classifier and
call it a day."* The census says the community already agrees and is spending its effort
accordingly. It is also the honest boundary on what this project is: a 350M decision
model is 6% of the field, and the 56% is the integration work that decides whether it is
usable at all.

## 3. The directory systematically refuses to verify the speed claims

This is the part that matters for the report's central argument. Every entry carries a
standardised provenance block, and two caveats recur verbatim across the corpus:

- **376 of 693 entries** (54%) are marked: *"performance and cost benefits have not been
  independently verified"*.
- **352 of 693 entries** (51%) say: *"consult the source for the exact decision policy"* —
  i.e. the reviewer could not confirm **what decision the project actually makes**.

So: in the largest census of this ecosystem, the reviewers could not substantiate the
performance claim in the majority of entries, and could not identify the decision point
in roughly half of them.

**In a category whose entire value proposition is latency, the biggest census says it
cannot substantiate the latency claims.** That is not a criticism of the directory — the
directory is being *more* honest than the field. It is a measurement of the field: the
"20-200x faster" number has essentially **not been independently reproduced by
independent people** in six weeks.

The directory's discipline is otherwise exemplary and worth borrowing outright. It also
notes, per project, things like:

- *"arbitrary-task generalization is not claimed"* (CUA-JEV)
- *"Author demo numbers were not retested here"* (jev-clerk)
- *"Supports persistent sessions and UI readback, which does not by itself prove database
  persistence"* (tontoko/jev-browser)
- *"GitHub SPDX is empty; the LICENSE file is MIT"* (jev-clerk)
- Licences recorded per entry, with **"Not declared"** used honestly in at least 3 cases.

The genuinely independent performance evidence in the entire corpus remains **one
number**: Vercel Labs' `json-render` at **3.21 s → 0.88 s**. Everything else is either
self-reported by the project author or explicitly unverified by the directory.

## 4. A fourth latency contradiction

The directory's own architecture table claims **"Sub-100ms Latency: Delivers decisions in
50–100ms"** for TypeSafe Jev.

`typed-decision-bench` measures **`jev-1.13.0` at 716.4 ms p50 and 778.8 ms p95** on an
H200 NVL.

That is a **~7x disagreement** between a community architecture summary and a controlled
benchmark. Contributing factors are visible and unresolvable from here: the directory does
not say which Jev version, which request shape, how many questions per call, or whether
network time is included; typed-decision-bench measures a specific multi-question shape
and includes HTTP latency, and states so.

**That is now four independent latency measurements in this report that disagree:**
Jev-Omni's own 83 ms in-process on an H200, Ollaya's 236–276 ms hosted, the directory's
50–100 ms claim, and typed-decision-bench's 716 ms. **No cross-source latency comparison
in this report is sound**, and the report should stop implying otherwise anywhere it
lines these numbers up. The honest statement is a range, with the request shape attached
to each end.

## 5. What this does *not* change

Nothing here displaces a measured result. The census is descriptive: it says the field is
building plumbing rather than classifiers, that its speed claims are unverified, and that
the one solid commercial datapoint is Vercel's. It does not change the architecture
finding, the LoRA result, the coverage-at-5% result, or the 0.766 bar.

It does change one thing about how the report should present latency, and it hardens
Takeaway 23 (*latency is not capability*) into something sharper: **latency claims in this
field are mostly unverified, and the ones that have been checked are slower than
advertised.**

## 5a. Cross-check: the machine-readable index agrees

The directory publishes a machine-readable `llms.txt`. It repeats the headline
figures, and **every one of the 18 category counts matches the README exactly** --
693 total, 120 SDK & Decision Frameworks, 69 Domain Tools, 62 CLI, 58 Security &
Guardrails, 55 Routing, 52 High-Frequency, 50 Browser, 44 Data, 43 MCP, 40 Context
GC, 29 Evaluation, 25 Decision Tools, 20 Creative, 14 Code Navigation, 6 SDK
Integrations, 4 Voice, 2 Classification.

**How much this is worth, stated precisely: not much, and it is a consistency check
rather than independent verification.** Both files come from the same publisher, so
agreement confirms the directory is internally consistent and that my transcription
from the README was correct. It does **not** independently confirm the counts are
accurate about the world -- only the publisher can do that, and note 3 is direct
evidence that the publisher's *verdicts* are conservative, not that their *counts* are
audited.

`llms.txt` also lists a `projects.json` and an `llms-full.txt`. **Neither was
fetched.** They would carry the per-project dataset behind the 376/352 caveat counts,
which are currently exact `grep -c` counts over the README rather than counts read
from structured data. That remains the honest gap in this note.

## 6. Method limits of this note

- Category counts are **as the README's own badge reports them**; I did not recount the
  entries by parsing, so an off-by-some in the badges would propagate.
- The two caveat counts are **exact `grep -c` counts** over the raw README: 376 and 352.
- Individual projects were **not** read. The directory is evidence about the *field*,
  not about any one project.
- The `llms.txt` / `llms-full.txt` agent-skill files the directory publishes were not
  fetched; they may carry a cleaner per-project dataset than the README.
