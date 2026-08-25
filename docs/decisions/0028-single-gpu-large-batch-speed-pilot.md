# 0028: Single-GPU Large-Batch Speed Pilot

## Status

Accepted for a seed-0 Task 1 systems/acquisition pilot on 2026-08-25.

## Context

The independent expert campaign preserves the original single-device ARROW
optimization batches (`N=16` for the world model and `N=128` for actor context).
Those batches reproduce the established optimization schedule but leave a
24-GiB RTX 4090 substantially underutilized. The 180-epoch expert protocol also
doubles the original 90-epoch task budget.

The existing DP4 `x4-linear-lr` profile demonstrates a sample-matched large-batch
rule, but its name and validation require four devices. Reusing that name on one
device would hide a material execution-protocol change.

## Decision

Add `single-gpu-x4-linear-lr` as a distinct CNN-FullBank Task 1 profile:

- one GPU and the original 90-epoch task duration;
- world-model `N=64`, 250 optimizer updates per epoch, and `wm_lr=4e-4`;
- actor context `N=512`, 200 optimizer updates per epoch, and `ac_lr=4e-4`;
- unchanged environment interaction and sampled replay/context-frame use;
- BF16 autocast with FP32 continuation logits/sigmoid/BCE;
- fixed periodic validation, independent final held-out evaluation, and the
  frozen task-specific ARROW reference matrix.

The larger batches and fewer optimizer steps are intended to reduce Python and
kernel-launch overhead and improve accelerator occupancy. Linear learning-rate
scaling follows the existing `x4-linear-lr` rule.

## Consequences

This is a throughput and acquisition ablation, not an original-ARROW
reproduction. Equal sampled-frame use does not make the optimization trajectory
equivalent: it has one quarter as many Adam updates, a four-times larger batch,
and a four-times larger learning rate. Report wall time, stage time, GPU
utilization, and raw return together. Do not claim a speed improvement without
also reporting acquisition quality.
