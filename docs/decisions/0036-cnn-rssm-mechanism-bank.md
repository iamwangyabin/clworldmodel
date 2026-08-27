# 0036: Shared-Base RSSM Mechanism Bank

## Status

Accepted as a task-aware, Task-1-snapshot-seeded pilot method on 2026-08-28.

## Context

The full-bank pilot protects retention by copying an entire RSSM per task. The
projector/LoRA pilots instead keep one frozen Task-1 RSSM, but their later-task
plasticity is constrained to affine low-rank deltas. The desired method should
keep one immutable visual and dynamics base, give every later task an
unattenuated nonlinear residual path, and test whether frozen mechanisms from
earlier tasks can be reused without forcing them into the new task.

## Decision

Add the separately named, task-aware
`CNN-MechanismBank-RSSM-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware`
protocol. Task 0 is the completed MsPacman route imported from the same strong
CNN-FullBank boundary snapshot used by the LoRA pilots. Its CNN, recurrent RSSM,
posterior, and prior remain shared and immutable thereafter.

Each later task owns:

- one zero-effect spatial projector;
- one recurrent residual mechanism with hidden width 512;
- one posterior-logit residual mechanism with hidden width 512;
- one prior-logit residual mechanism with hidden width 256;
- private decoder, reward, and continuation heads; and
- a fresh deterministic Actor-Critic and optimizer.

Every mechanism uses LayerNorm, a down projection, SiLU, and a zero-initialized
up projection, with residual scale 0.1. The current task's mechanism always has
coefficient 1. For task `k`, each frozen older mechanism `m < k` has an
independent scalar `tanh(route[k,m])` gate. Routes initialize to zero; there is
no softmax competition. Posterior and prior corrections are added to raw flat
categorical logits and normalized once through the existing Dreamer uniform
mixture. The recurrent correction is added to the frozen base GRU output.

The primary method enables reuse. The capacity-matched `NoReuse` ablation keeps
the same mechanism and route tensors but leaves all reuse routes frozen at
zero. Both variants use only the existing Dreamer world-model and
Actor-Critic losses. They add no distillation, sparsity, orthogonality, or
consistency objective.

The three new mechanisms contain exactly 3,816,192 FP32 parameters per later
task (15,264,768 bytes). Task 2 also stores three scalar route parameters. This
is a plasticity-oriented mechanism-bank pilot, not a parameter-efficiency claim.

## Protocol Invariants

- Task order is MsPacman, Boxing, then CrazyClimber, 90 epochs each.
- Replay, interaction frames, sampled sequences, world-model updates,
  Actor-Critic updates, learning rates, BF16 compute, uint8 mmap replay, seed,
  and fixed evaluation cohorts match the snapshot-seeded LoRA pilots.
- Only current-task replay receives updates; evaluation transitions never enter
  Replay.
- Task identity selects the projector, mechanism route, private heads, and
  Actor-Critic. This protocol is not task-agnostic.
- New world-model heads copy the preceding task once. New Actor-Critics do not.
- Old mechanisms, old heads, old policies, the shared CNN, and the shared base
  RSSM must remain tensor-exact after their task ends.
- Boundary and evaluation snapshots remain inference artifacts, not resumable
  checkpoints; optimizers, Replay, RNG, and scheduler position are omitted.

## Evidence Gate

Before a pilot, focused tests must demonstrate exact zero effect, current-task
gradient isolation, frozen base and old-mechanism tensors after an optimizer
step, and nonzero gradient at a zero-initialized Task-2 reuse gate. A GPU smoke
on the target accelerator remains required before a formal run.

The first pilot is successful only as feasibility evidence if Boxing reaches a
raw return of at least 70, CrazyClimber reaches at least 40,000, the MsPacman
path remains unchanged, old mechanisms remain exact, reuse exceeds NoReuse,
and learned reuse gates are not all zero. A single seed cannot support a paper
claim.
