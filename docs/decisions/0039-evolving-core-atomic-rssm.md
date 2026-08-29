# 0039: Evolving-Core Atomic RSSM

## Status

Accepted as a separately named task-aware, from-scratch pilot design on
2026-08-29. No performance claim exists until a multi-seed campaign completes.

## Context

MB-RSSM and REC-RSSM preserve Task 1 by freezing its CNN and base RSSM, then
giving later tasks private residual mechanisms. This provides clear retention
ownership but makes the first task a permanent representational anchor. A full
task bank removes interference at substantially higher parameter cost and does
not test whether replay-compatible knowledge can mature in one shared world
model.

## Decision

Keep exactly one CNN and base posterior/recurrent/prior RSSM plastic throughout
the curriculum. Give every task—including Task 0—the same zero-effect spatial
projector, four-atom recurrent/posterior/prior residual mechanisms, private
observation/reward/continuation heads, and independent Actor-Critic. Completed
private modules freeze; old atoms remain selectively reusable through
task-owned gates.

Later online updates split the unchanged 16-sequence batch into 12 current and
four task-homogeneous LTDM memory sequences. Old-task Dreamer replay is joined
by posterior, hidden-state, and frozen-Actor interface protection. Shared
current gradients are projected independently for encoder, posterior,
recurrent, prior, and latent-interface components only when they conflict with
the memory gradient. Current private gradients are never projected.

Use one persistent shared Adam rather than rebuilding an optimizer after task
activation. Private, route, and Actor-Critic optimizers have explicit task
ownership. At boundaries, save the complete completed state before attempting
1,000 small-learning-rate task-balanced shared-only updates. Fixed-cohort raw
return validation either accepts the core and boundary teacher or rolls back
both weights and shared Adam state.

## Consequences

- Task 0 and later tasks have symmetric topology and zero-effect initialization.
- Compatible new knowledge can enter the shared core; conflicting knowledge
  retains a full-capacity private residual path.
- Retention failures can be attributed to world-model/interface drift because
  completed Actor-Critics never update.
- Online sample count remains fixed, but boundary consolidation adds 3,000
  world-model optimizer steps in a three-task run and must be reported as extra
  compute.
- Task identity selects private routes and policies, so the method is not
  task-agnostic.
- Checkpoints are larger because exact Replay state, optimizer banks, RNG, and
  immutable mmap assets are required for an equivalent resume.
- A parameter-free current `zh_transform` is still an explicit shared interface
  group; it contributes diagnostics but no optimizer-owned tensor.

## Rejected Alternatives

- Freezing the Task-1 base preserves the existing MB/REC assumption rather than
  testing an evolving core.
- Summing current and memory losses without component projection gives no
  explicit protection against destructive current gradients.
- Updating old Actor-Critics confounds world-model retention with policy
  relearning.
- Mixing task IDs inside one RSSM batch violates the scalar task-route contract.
- Rejection sampling task-labelled Replay is inefficient and obscures sampling
  semantics; exact task-to-slot indices are maintained instead.
- Online atom pruning or route ablation can make serialization/evaluation errors
  affect formal training, so those analyses remain offline.

## Evidence Gate

Before a pilot launch, focused tests and a target-GPU smoke must cover symmetric
Task-0 topology, exact zero effect, task-pure Replay, private-gradient isolation,
component projection, persistent Adam steps, complete checkpoint round trips,
immutable mmap restore, consolidation rollback, finite current/memory losses,
and one successful shared/private/route optimizer step.
