# Does more data rescue the head-only model?

Every experiment so far used 240–360 training questions against a published norm of
10k–24k. That gap was the standing caveat on all of it. This tests the frozen arm
with **3.4× the data** and all four workflows, which also asks whether the result
transfers across workflow types.

## Result: more data helps the frozen model a lot, and still not enough

| arm | train Q | eval Q | accuracy | Brier | ECE | fitted T | cov@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| customer_service, frozen | 360 | 300 | 0.3233 | 0.1404 | 0.1467 | **3.300** | **0.0000** |
| invoice_processing, frozen | 240 | 200 | 0.4700 | 0.1502 | 0.0765 | **4.150** | 0.0300 |
| **all 4 workflows, frozen** | **1212** | 996 | **0.4357** | 0.1244 | 0.1040 | **1.800** | **0.0000** |

**3.4× the data cut the miscalibration by roughly half** — fitted T from 3.300 to
1.800, Brier 0.1404 → 0.1244 — and raised accuracy from 0.3233 to 0.4357. That is a
real, substantial improvement from data alone.

**And coverage at a 5% error budget is still exactly 0.0000.** The frozen model has no
usable operating point at 1212 questions, on four workflow types, with 996 evaluation
questions to measure it against.

So the honest reading is: **data quantity and backbone fine-tuning are not
substitutes.** More data moves a frozen head a long way toward being calibrated and
still does not make it deployable; LoRA reaches a deployable model at 360 questions
where the frozen model does not at 1212.

## Not measured

The LoRA arm on the 4-workflow split was stopped mid-evaluation. Its epoch-1 training
loss was 1.2288 against the frozen arm's 1.4161, so it was tracking ahead, but no
metrics were produced and **none are claimed**. The single-workflow and two-workflow
LoRA results stand as reported elsewhere.

## What this costs to finish

Each arm is ~1–2 hours on CPU at this data size, and the evaluation loop scores
questions one at a time, which dominates. Batching the eval, and running on a GPU,
would make a proper data-scaling curve (say 300 / 1.2k / 5k / 20k questions) a
half-hour job rather than a day. That is the next thing to do, and it needs hardware
this project does not have.
