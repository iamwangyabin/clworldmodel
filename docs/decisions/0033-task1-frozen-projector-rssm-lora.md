# 0033: Adapt a Frozen Task-1 CNN/RSSM with Per-Task Projectors and LoRA

## Status

Accepted for a seed-0 snapshot-seeded incremental pilot on 2026-08-26.

## Context

The completed three-task CNN-FullBank pilot establishes a task-aware upper
bound but copies the full CNN and RSSM route for every task. Posthoc functional
distillation found that a frozen Task-1 CNN plus a small spatial projector and
Task-1 RSSM plus affine LoRA retained useful Boxing and CrazyClimber behavior.
That probe used privileged finished-task teachers and target-policy
initialization, so it established representational capacity rather than
incremental learnability.

The completed Task-1 boundary snapshot is inference-complete but not resumable:
it omits Replay, optimizers, RNG, and scheduler state. Because every post-Task-1
module and optimizer in this method is deliberately fresh and the Task-1 core
is frozen, it can seed a separately named acquisition pilot, but not an
equivalent continuation of the source run.

## Decision

Add `CNN-Projector-RSSM-LoRA-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware`.
Import only the completed Task-1 CNN, RSSM route, reconstruction/reward/
continuation heads, and Actor-Critic. Restart at Boxing with empty ARROW-50
Replay and new RNG/optimizer state.

For every later task, allocate:

- one zero-effect residual `256 x 4 x 4` spatial projector with a 64-channel
  bottleneck;
- one RSSM affine-LoRA route that shares the frozen Task-1 recurrent,
  representation, and transition parameters at ranks `128/128/32`, with exact
  task-specific one-dimensional bias and normalization deltas;
- fresh independent pixel decoder, reward head, continuation head, and
  Actor-Critic, each initialized deterministically (the world-model heads copy
  Task 1; behavior does not); and
- fresh optimizer state.

Train only the current task's new modules from its normal environment
interaction, ARROW Replay, Dreamer world-model loss, imagined rollouts, and
Actor-Critic loss. Do not use a finished later-task teacher or target-task
weights. Old task routes and policies remain frozen. Evaluation transitions
remain isolated from Replay.

## Consequences

This remains a task-aware upper bound. One LoRA/projector/Actor-Critic is stored
per later task; one shared LoRA is not claimed to cover all tasks. Model storage
grows linearly only in the adapters, policies, and private heads; the CNN and
RSSM base are stored once. Private reconstruction/reward/continuation heads are
retained because normal Dreamer training cannot learn new visual and reward
semantics using only a frozen Task-1 head.

The first run is a single-seed pilot and cannot support a general or
task-agnostic claim. Its decisive test is whether Boxing and CrazyClimber can
be acquired without privileged teachers while the imported Task-1 tensors stay
bitwise unchanged.
