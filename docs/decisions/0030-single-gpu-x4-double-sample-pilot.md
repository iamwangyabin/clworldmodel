# 0030: Single-GPU X4 Double-Sample Pilot

## Status

Accepted for a seed-0 MsPacman speed/acquisition pilot on 2026-08-25.

## Context

The original single-GPU ARROW optimization batches preserve 1,000 world-model
and 800 actor updates per epoch, but the `N=16/128` batches underutilize a
24-GiB accelerator. The sample-matched x4 profile reduced wall time by about
half, but its one-quarter Adam-update schedule finished below the frozen
MsPacman reference. Repeating either endpoint would not test a new tradeoff.

Using BF16 for probability saturation points is not an acceptable throughput
shortcut. The continuation audit showed that confident BF16 sigmoid outputs
can round to one before the terminal target reaches BCE. The stable mixed
precision boundary therefore remains part of the method.

## Decision

Add `single-gpu-x4-double-sample-linear-lr` for a distinct 90-epoch Task 1
pilot:

- world-model batch `N=64`, 500 updates per epoch, and `wm_lr=2e-4`;
- actor-context batch `N=512`, 400 updates per epoch, and `ac_lr=2e-4`;
- 15,000 world-model pretraining updates at `N=64`;
- unchanged environment interaction and replay capacity;
- two times the original sampled replay and actor-context frame use;
- half the original Adam-update count;
- BF16 autocast for model kernels, with float32 master parameters, optimizer
  state, continuation probability/loss, categorical math, and returns; and
- the fixed validation cohort, disjoint final held-out cohort, and frozen
  task-specific ARROW reference.

## Consequences

This is an extra-sample speed/acquisition ablation. It is neither a matched
ARROW reproduction nor an all-BF16 experiment. Report wall time, GPU
utilization, memory, raw return, Adam updates, and sampled-frame multipliers
together. A single seed remains a pilot, and a favorable result cannot support
a fair-superiority claim without a matched-budget control.
