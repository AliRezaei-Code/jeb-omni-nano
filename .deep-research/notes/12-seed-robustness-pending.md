# Open check: is the headline claim seed-robust? — NOT ANSWERED

## The concern

Every headline number in this project comes from **one seed (17)**:

- LoRA 0.5633 vs frozen 0.3233 ("+24 accuracy points")
- fitted temperature 3.300 → 1.050
- coverage@5% 0.0000 → 0.0967

A single seed is not a result, it is an observation. If the delta is mostly seed noise,
the project's central claim is overstated.

## What was attempted

The identical run at **seed 42** — same data, same splits, same hyper-parameters, same
evaluation, only the seed differs.

## Outcome: stopped after 48 minutes, no result

Host load average was **88–94** throughout, with three `nvidia-smi` processes and one
`nvidia-settings` burning 83–93% CPU continuously for 11+ hours. The run received
~153% CPU and **did not complete a single epoch in 48 minutes**, against ~13 minutes per
epoch for the same work on an idle machine.

It was stopped rather than left running, because it was adding load to an already
oversubscribed machine with no prospect of finishing in useful time.

**No seed-42 number is claimed. None was produced.**

## Consequence for the deliverables

Because the check did not land, the single-seed caveat is **permanent and prominent** —
stated next to the numbers in `README.md`, `GUIDE.txt` and the report, not in a
footnote. The report's finding 3 is labelled "our own measurement, single seed, not
replicated" in its claim-class line.

If this check is ever run on an unloaded machine and lands, the caveats can be removed
**on the evidence**. Until then they stand.

## What partially mitigates the exposure, stated rather than used as an excuse

- The **direction** of every effect reproduces across two workflows
  (`customer_service`, `invoice_processing`) and across a 3.4× change in training-set
  size — three independent configurations, all with the same seed, so this is
  consistency across *data*, not across seeds. It is weaker evidence than a seed
  replication and is not presented as a substitute.
- The **coverage gap** (0.0000 against 0.0967 at n = 300) is the least noise-prone of
  the claims: a zero at that sample size is hard to produce by accident, and the
  ordering is consistent across both workflows.
- The **temperature signature** is large and consistent (3.300 and 4.150 frozen; 1.050
  and 0.850 tuned), which is a two-cluster separation rather than a small offset.

None of these substitute for the seed replication. They bound how badly the claim
could be wrong, not whether it is right.
