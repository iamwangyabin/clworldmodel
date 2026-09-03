# 0050: Ablate current-task Q/F/P output regularization

## Status

Accepted for one separately named seed-0 pilot on 2026-09-03. This decision
authorizes implementation, a target-CUDA smoke, and a pilot launch. It is not a
performance claim.

## Context

The Dense-acquire adaptive-compression method inherits
`1e-4 * sum_c E[||A_k^c(x)||_2^2]` from the original Evolving-Core online
objective. That term biases the task-private recurrent/posterior/prior residual
functions toward small corrections. In this task-aware topology, however, each
private route is selected only for its owner task and freezes at the boundary.
The experiment hypothesis is that unconstrained private Q/F/P output may improve
task acquisition without weakening the existing replay, teacher, gradient-
projection, consolidation, compression, or return-gating safeguards.

## Decision

Add the separately named method
`evolving_atomic_rssm_adaptive_compression_shared_heads_no_atom_reg_arrow` and
protocol
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFP-SharedDistilledHeads-PrivateMLPAC-NoAtomOutputReg-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

The sole intended optimization change is
`task_atom_output_regularization: 1e-4 -> 0.0` during online current-task world-
model updates. The task-private Q/F/P output trace remains available for metrics
and compression distillation. All environment steps, task order and durations,
Replay, online/consolidation/compression update counts, batch splits, learning
rates, teacher losses, component-gradient projection, prediction-head sharing,
Actor-Critic ownership, compression candidates, validation cohorts, and return
gates remain identical to the adaptive-compression control.

The launcher exposes this only as
`--adaptive-qfp-compression --disable-atom-output-regularization`; the flag is
invalid outside that named ablation. Config validation rejects a nonzero atom
regularization scale under the new method key.

## Evidence

The first run is a seed-0 pilot. Compare acquisition curves, Q/F/P output norms,
shared-core drift, forgetting, selected compression widths, and raw per-task
returns to the matched adaptive-compression control. Do not infer superiority
from one seed.

The required target-CUDA smoke profile is
`adaptive_qfp_compression_no_atom_reg`; its JSON result records
`task_atom_output_regularization: 0.0` before any environment-interacting run
may start.
