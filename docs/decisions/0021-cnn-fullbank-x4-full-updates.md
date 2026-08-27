# Decision 0021: Compute-saturation DP4 batch profile

## Status

Accepted for a named CNN-FullBank Task 1 acquisition ablation. It does not
replace the fixed-global-batch or sample-matched protocols.

## Context

The sample-matched `x4-linear-lr` profile increased global batches by four but
reduced Adam updates and pretraining steps by four. It delivered high
throughput, but early MsPacman acquisition collapsed relative to the
fixed-global-batch run. Equal sampled-frame use did not preserve the optimizer
trajectory: only one quarter as many parameter updates were performed, and the
Adam learning rates were increased fourfold.

The immediate objective is instead to keep all four GPUs occupied without
removing optimization updates. This intentionally spends more compute and is
not a budget-matched acceleration result.

## Decision

Add `x4-full-updates` for CNN-FullBank DP4:

- world-model and pretraining global sequence batches are 64, or 16 per rank;
- actor context global sequence batch is 512, or 128 per rank;
- world-model, pretraining, and actor update counts remain 1,000 per epoch,
  30,000 initially, and 800 per epoch respectively;
- world-model and actor Adam learning rates remain `1e-4`;
- environment interaction, replay capacity, evaluation, and ARROW-50 replay
  selection remain unchanged; and
- manifests identify fourfold optimization sample use and unchanged update
  counts relative to fixed-global-batch DP4.

## Consequences

This profile processes four times as many replay and actor-context samples per
environment epoch. It may improve accelerator occupancy and learning per epoch,
but cannot be reported as an equal-compute or equal-sample speedup. The first
run remains a 90-epoch Task 1 acquisition gate; a six-task campaign is justified
only after the final deterministic MsPacman mean reaches the declared threshold.
