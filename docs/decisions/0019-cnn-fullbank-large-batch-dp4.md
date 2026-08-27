# Decision 0019: Sample-matched large-batch DP4 ablations

## Status

Accepted for controlled throughput and acquisition experiments. This does not
change the frozen fixed-global-batch CNN-FullBank baseline.

## Context

The CNN-FullBank DP4 run uses global world-model sequence batch 16 and actor
context batch 128. On four GPUs those become local batches 4 and 32, leaving
substantial accelerator capacity unused while paying one DDP synchronization
per optimizer step. Simply increasing the batch without accounting for sample
use would give the method additional optimization data and confound a speed or
quality comparison.

## Decision

Add named `x2-linear-lr` and `x4-linear-lr` profiles for CNN-FullBank on DP4.
Each profile:

- multiplies world-model and actor sequence batches by `x`;
- divides world-model, pretraining, and actor optimizer steps by `x`;
- multiplies world-model and actor learning rates by `x`;
- keeps total replay-frame and imagined-context-frame use unchanged;
- keeps collection, evaluation, replay capacity, and ARROW-50 sampling
  unchanged; and
- records all resolved values and multipliers in the run manifest.

These profiles are optimization ablations, not transparent runtime flags.
Linear learning-rate scaling is a testable initialization choice rather than a
claim of equivalence. A short run must first establish wall-clock improvement,
memory headroom, and finite metrics. The full continual campaign remains gated
by the final 90-epoch MsPacman raw return threshold defined in the protocol.

## Consequences

The number of Adam updates changes even though sampled-frame use is matched.
Results must use the profile-qualified protocol name and cannot be pooled with
fixed-global-batch runs. If throughput improves but acquisition fails, learning
rate scaling may be varied only through another named, documented ablation.
