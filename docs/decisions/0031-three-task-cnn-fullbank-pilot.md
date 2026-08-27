# 0031: Restrict the CNN-FullBank Continual Pilot to Three Tasks

## Status

Accepted for the seed-0 pilot on 2026-08-25.

## Context

The exploratory sequential launch trained only MsPacman and Boxing but kept
all six original curriculum environments configured for diagnostic evaluation.
That made its first periodic evaluation unnecessarily expensive and did not
cover the newly requested CrazyClimber stage. It was stopped after three
completed MsPacman epochs, before any task boundary, and remains preserved as
an interrupted, non-resumable run.

Neither that launch nor the completed standalone MsPacman acquisition pilot is
a valid continuation point: their snapshots omit optimizers, Replay, RNG, and
scheduler/task-position state.

## Decision

Add `three-task-single-gpu-x4-double-sample-pilot-v1` as a fresh sequential run
over MsPacman, Boxing, and CrazyClimber. Restrict both the resolved environment
list and evaluation cohorts to those three tasks. Allocate exactly three world
model expert slots and three task-boundary snapshots.

Keep the established single-GPU x4 double-sample optimization profile, BF16
autocast with float32 sensitive paths, uint8 mmap ARROW-50 Replay, fixed
periodic cohorts, and independent final held-out evaluation.

## Consequences

The run starts from epoch zero and cannot reuse the three partial epochs. It is
a single-seed extra-sample pilot: per-task environment interaction is matched,
Adam updates are halved, and sampled-frame use is doubled relative to original
single-GPU ARROW. Results must report raw taskwise acquisition, retention, and
forgetting and cannot support a fair-superiority claim.
