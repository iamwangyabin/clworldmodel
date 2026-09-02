# 0048: Use atomic Rank-128 later-task Q/F/P with shared plastic heads

## Status

Accepted for a separately named seed-0 boundary-bootstrap pilot on 2026-09-02.
This decision does not assert improved performance.

## Context

The learned-base Rank-32 experiment from decision 0047 failed its first later
task. At the epoch-150 fixed cohort it retained MsPacman at raw
`3456.25 +/- 1430.89`, but Boxing remained `-100 +/- 0`. Over the corresponding
late-Boxing window its world-model reconstruction loss was roughly two orders
of magnitude above the Dense-Q/F/P shared-head run. Its imagined return was
positive while real Boxing return was terminal-floor performance. Continuing
that pilot would therefore spend compute on an architecture that had not
acquired Task 1.

The negative result confounded two changes relative to the stronger experiment
A: a frozen Task-0 prediction-head base with private adapters, and a Rank-32
later-task Q/F/P delta that always included the Task-0 mechanism base. The
successful parts of A should be retained: one plastic replay-protected
decoder/reward/continue set, boundary output distillation, old-atom routing,
and private MLP behavior. Only the full Dense later-task Q/F/P growth should be
compressed.

Task 0 of the stopped Rank-32 run completed before the failure and has an
immutable post-consolidation resumable checkpoint. It contains only Task-0
Replay and one Task-0 Actor-Critic. Reusing this boundary avoids repeating
about 90 epochs while not introducing future-task data. This is a declared
cross-topology bootstrap, not an equivalent resume or a from-scratch C run.

## Decision

Add `evolving_atomic_rssm_atomic_lora_shared_heads_arrow` and protocol
`Evolving-Core-Task0BoundaryBootstrap-AtomicRank128QFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

- Restore the exact Task-0 shared CNN/RSSM/latent state, Dense Task-0 Q/F/P,
  shared prediction heads, Task-0 MLP Actor-Critic, FIFO/LTDM Replay, counters,
  environment position, and RNG streams from the immutable Task-0 checkpoint.
- Preserve A's one plastic shared decoder/reward/continue set and output
  distillation scale `0.1`; do not allocate private prediction adapters.
- Keep Task 0's full Dense Q/F/P mechanisms. Give Tasks 1-5 independent
  exact-zero Rank-128 nonlinear Q/F/P residuals, partitioned into four
  routable Rank-32 atoms.
- Do not make a later residual call or own the Task-0 base. Its output is the
  shared RSSM base plus its own low-rank delta plus explicitly gated older
  atoms. This prevents the forced Task-0-base behavior used by the failed
  Rank-32 experiment.
- Retain old-atom reuse, independent MLP Actor-Critics, 12/4 current/LTDM
  world-model batches, component conflict projection, boundary consolidation,
  rollback, ARROW-50 Replay, and the original six-task/90-epoch schedule.
- Reset world-model optimizers at the ownership transition. Restore the Task-0
  Actor-Critic optimizer for provenance, although that Actor-Critic is frozen
  once Task 1 starts. Record this asymmetry in the run manifest.

Rank 128 is fixed for v1. It is four times the failed delta rank while still
cutting each later Q/F/P allocation from `3,816,192` to `1,391,360` parameters.
Adaptive rank, rank sweeps, and a from-scratch C run require new experiments.

## Consequences

The exact six-task online total is `40,773,375` parameters: `30,477,465` world
model and `10,295,910` private behavior. This saves `12,124,160` parameters
(`22.92%`) versus A's Dense-Q/F/P shared-head total of `52,897,535`, while
using `3,617,280` more than the failed Rank-32/private-head-adapter pilot.

The composite run has a matched total interaction/update budget only when the
source Task-0 phase and C's Tasks 1-5 phase are accounted together. The C
launcher executes 450 new epochs, inherits 90 completed epochs, and must retain
both source and target provenance. It cannot be presented as a from-scratch
seed, and a promising result must be confirmed with a clean full run and
multiple seeds.

Task identity is exposed. Raw returns remain separate from reward-scaled and
ARROW-normalized metrics. The stopped negative run remains part of the
experiment record rather than being discarded.
