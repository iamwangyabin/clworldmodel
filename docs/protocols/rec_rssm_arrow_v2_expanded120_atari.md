# REC-RSSM ARROW v2 Expanded120 (Atari)

## Scope

`REC-RSSM-ARROW-v2-Task1SnapshotSeeded-Atari-TaskAware-Expanded120` is a
single-seed follow-up pilot to the matched v1 REC-RSSM run. It tests whether a
moderate mechanism expansion and a predeclared late-task Actor-Critic learning
rate decay stabilize later-task acquisition under longer training. It is
task-aware and is not a capacity-, interaction-, update-, or compute-matched
replacement for v1.

The run may start only after the v1 formal pilot finishes and passes its
artifact audit. It uses the same immutable Task-1 boundary snapshot, seed,
task order, preprocessing, Replay policy, evaluation cohorts, and isolation
rules as v1. No v1 validation result selects a v2 checkpoint or changes the v2
schedule after launch.

## Schedule And Budgets

Task durations are explicitly non-uniform:

- MsPacman: source snapshot for completed epochs 0 through 89;
- Boxing: epochs 90 through 209, 120 new epochs; and
- CrazyClimber: epochs 210 through 329, 120 new epochs.

The 240 new epochs contain 15,728,640 raw environment frames, 120,000
world-model updates, and 96,000 Actor-Critic updates. These are 4/3 of the v1
post-Task-1 interaction and update budgets. The resolved config records the
per-task duration list rather than treating one `swap_sched` value as if every
task had equal duration.

## Expanded Mechanisms

The recurrent, posterior, and prior mechanism widths are `640/640/320`, a
25-percent hidden-width increase from v1. Four lossless atoms have widths
`160/160/80`. Each later task receives exactly 4,766,784 mechanism parameters;
Task 3 adds the same 12 route parameters as v1. The shared Task-1 CNN and base
RSSM, old mechanisms, old heads, and old Actor-Critics remain immutable.

Reuse probing and consolidation retain the v1 semantics: one route-only local
epoch at the start of CrazyClimber, route learning rate five times the
world-model rate, eight Replay batches for contribution estimates, minimum
contribution `0.01`, and whole-route rollback when the same-cohort 16-rollout
mean falls by more than five percent.

## Actor-Critic Schedule

Each later task receives a fresh independent MLP Actor-Critic. Its learning
rate is `2e-4` through local epoch 60 and then follows a task-local cosine decay
to `5e-5` at local epoch 120. The entropy scale remains fixed at `3e-4` so the
follow-up changes optimization step size without adding an entropy ablation.
The schedule depends only on task age and never on evaluation performance.

Before the formal run, a separately classified three-epoch GPU smoke uses
post-Task-1 durations `1/2`. This reaches Boxing expansion, CrazyClimber's
route-only probe, and CrazyClimber full expansion without claiming formal
schedule equivalence. It uses the expanded mechanisms but a constant Actor
learning rate; unit tests cover the formal 120-epoch cosine schedule.

## Evidence And Claims

Periodic and final evaluation uses the preserved fixed 16-rollout cohorts and
raw returns. Reports include final results, the full learning curve, v1 deltas,
retention, forgetting, acquisition AUC, and exact resource ledgers. A favorable
single-seed result establishes only pilot evidence that the combined expanded
capacity and longer decayed training configuration merits a controlled
factorial ablation; it does not identify which change caused the difference.
