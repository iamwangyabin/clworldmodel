# 0052: Separate current-task learning from never-clear rehearsal memory

## Decision

The paper-semantics Atari Dream Rehearsal pilot retains every collected
trajectory but filters ordinary world-model and actor-critic training to the
current scheduled task.  Old-task sequences are used only to initialize graded
dream rehearsal.  Replay task metadata and model routing are separate
interfaces; the shared model and policy never receive task identity.

## Why

The bounded v1 diagnostic used one global uniform reservoir for ordinary
DreamerV3 updates.  Later-task data therefore entered ordinary minibatches only
in proportion to its accumulated share, while old-task actor-only rehearsal
continued at the paper cadence.  That conflated a storage intervention with a
different acquisition protocol and produced a severe plasticity failure.

The paper's never-clear claim must be tested with its relevant data-flow
semantics before deciding whether dream grading transfers from MiniGrid to
Atari.

## Consequences

- Full original-order history retains 8,863,744 transitions and requires about
  109.7 GB of accounted tensors with uint8 observations.
- The run is deliberately not replay-capacity matched to ARROW-50 or the
  bounded Dream Rehearsal port.
- A result can isolate the effect of paper-style full history, but Atari and
  framework differences still prevent a faithful MiniGrid reproduction claim.
- Any later task-balanced real replay, real-action cloning, longer dream
  horizon, or reward-density adaptation is a separately named ablation.
