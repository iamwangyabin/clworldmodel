# Decision 0025: Preserve every completed CNN task bank

## Status

Accepted for `CNN-FullBank-ARROW-v1`. Every non-dry training launch now
requires a task-boundary snapshot directory and the exact 40-character project
Git commit.

## Problem

The conventional `save_wm.pt`, `save_ac.pt`, and `save_ac_bank.pt` names are
final or replaceable convenience artifacts. Periodic evaluation snapshots are
also conditional on an evaluation schedule. Neither contract guarantees an
immutable copy of the model immediately after every task's final update.

That gap prevents exact retention audits and makes it possible to finish a
six-task run with no preserved Task 1 through Task 5 model states.

## Decision

At each sequential task boundary, rank 0 saves a unique artifact after the
task's final world-model and Actor-Critic updates and before the environment
schedule advances. The artifact contains:

- the complete world-model state, including every allocated task route;
- the complete Actor-Critic inference bank and the just-completed task actor;
- the completed task identity and reward scale;
- completed epoch, world-model update, Actor-Critic update, and raw-frame
  counters;
- the resolved training config and exact project Git commit; and
- an explicit list of omitted state.

The `.pt` file, its `.sha256`, and `index.json` are written through temporary
paths and atomic replacement. Existing boundary paths or index entries are
never overwritten. The index refuses a project commit change within one run.
A normal six-task run must therefore produce six indexed snapshots.

These are inference and audit artifacts, not resumable checkpoints. They omit
optimizers, replay, RNG state, environment schedule state, and step schedulers.
No old run is relabeled as having this guarantee retroactively.

## Validation

Focused tests verify the complete state payload, counters, commit, checksum,
index, temporary-file cleanup, duplicate rejection, and launcher command and
manifest fields.
