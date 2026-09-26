# Frozen Scope — Jeb-Omni-Nano Deep Research

**Topic.** How the "Jev" family of typed-decision models is built, and how to build a
much smaller, cheaper sibling of `akhilaaa3/Jev-Omni` on Liquid AI LFM2.5.

**Research cutoff.** 2026-09-26 (all freshness judgements are made against this date).

**Audience.** A competent engineer who has never trained a model and knows nothing
about RCLDs / decision heads / calibration / RCLD-style constrained decoding. The
deliverable guide assumes zero prior knowledge.

**Primary deliverable.** An extensive `.txt` build guide in the repo root.
**Secondary deliverable.** This research report plus a 200+ row source ledger.

---

## Sub-questions (7)

**SQ1 — What is Jev-Omni, exactly, and how was it built?**
Recover the true architecture and training recipe of `akhilaaa3/Jev-Omni`: backbone,
decision head shape and dtype, LoRA configuration, optimiser, schedule, data volume,
multimodal preprocessing, evaluation protocol. Also establish what the upstream
"TypeSafe AI / Jev" model is and is not, and which parts are independent
reimplementation rather than reproduction.

**SQ2 — What is the wider Jev ecosystem, and what did other people conclude?**
The open reimplementations (Kev, Jebadiah, jeff, Open-Jev, Tiny-Jev, JevK5, Laya,
GLiNER2, smalljev, djev…), what each concluded about the correct design, and which
design choices have converged vs diverged across independent implementations.

**SQ3 — What is Liquid AI's LFM2/LFM2.5 family, precisely?**
Architecture, exact layer composition, cache mechanics, licensing, supported runtimes,
and — critically — which Liquid checkpoints can accept image/audio/video at the 350M
scale, and what the LFM Open License actually permits.

**SQ4 — How do you turn log-likelihoods or a hidden state into trustworthy probabilities?**
Calibration theory and practice: temperature scaling, ECE definitions and binning
disagreements, Brier, KL, soft vs hard targets, ordinal targets, per-type temperatures,
conformal/selective prediction, and the measured effect of each.

**SQ5 — What actually works for small-model multimodal decision making?**
Encoders and projectors attachable to a sub-1B text backbone, image token budgets,
frame sampling for video, audio feature extraction at 16 kHz, and what MMAU/MVBench
measure. Plus a realistic assessment of whether multimodal is worth it at this scale.

**SQ6 — What is the toolchain and what does it cost?**
LoRA/QLoRA SFT, the TRL/Unsloth recipes Liquid documents, distillation, GGUF
quantisation ladder with bits-per-weight, the runtimes (llama.cpp, MLX, ONNX,
OpenVINO, vLLM, SGLang, ExecuTorch), and measured latency/memory on real hardware.

**SQ7 — What are the licensing and reporting constraints?**
LFM Open License v1.0 vs Apache-2.0 vs Gemma terms; what a public MIT-licensed
derivative may and may not claim; and the benchmarking pitfalls (cost-per-decision,
request shape, option order, specialist vs generalist) that a small model must avoid.

---

## Method notes and constraints

- `websearch` is non-functional in this environment (all providers return 429/timeout).
  Sources were therefore collected by direct URL retrieval: Hugging Face model and
  dataset APIs and raw files, arXiv, GitHub raw, and vendor documentation sites.
- Every URL in the ledger was fetched and read. No URL is included from memory.
- Where sources disagree, both are recorded and the disagreement is flagged rather
  than resolved by fiat.
- Claims resting on a single source are labelled `[single source]`.
- Claims older than 12 months are labelled `[dated: YYYY]`.
