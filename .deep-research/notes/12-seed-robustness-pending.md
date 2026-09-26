# Open check: is the headline claim seed-robust?

## The concern

Every headline number in this project comes from **one seed (17)**:

- LoRA 0.5633 vs frozen 0.3233 ("+24 accuracy points")
- fitted temperature 3.300 -> 1.050
- coverage@5% 0.0000 -> 0.0967

A single seed is not a result. A single seed is an observation. If the delta is mostly
seed noise, the project's central claim is overstated and must be corrected.

## What was launched

The identical run at **seed 42** — same data, same splits, same hyper-parameters, same
evaluation. Only the seed differs. If the claim is real it should reproduce within
noise; the accuracy delta here is +0.2400 and the temperature delta 2.25, both far
larger than typical seed variance, but "should" is not "does".

## Status: blocked by machine load, not by the experiment

Load average is **93** on this host, with several `nvidia-smi` processes each consuming
~83% CPU in a spin. The seed-42 run is receiving ~152% CPU and has taken more than
three times as long per epoch as the seed-17 run did on an idle machine.

The check is left running in the background. It costs nothing to let it finish, but
**no result is claimed until it does.**

## What the claim is worth in the meantime

Stated honestly in the absence of a replication:

- The **direction** of every effect reproduces across two workflows
  (`customer_service`, `invoice_processing`) and across a 3.4x change in training-set
  size. Three independent configurations agree.
- The **magnitude** is single-seed and should be read as an order of magnitude, not a
  point estimate.
- The **coverage gap** (0.0000 against 0.0967) is the claim least likely to be seed
  noise, because a zero is hard to produce by accident at n=300.

Until the seed-42 run lands, every accuracy and temperature figure in these notes
carries `single seed, not replicated` in its scope statement.
