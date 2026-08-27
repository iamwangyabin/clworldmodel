# Decision 0022: Fixed-cohort evaluation and late actor stabilization

## Status

Accepted for a named CNN-FullBank Task 1 pilot. It does not replace the
constant-hyperparameter `x4-full-updates` result or change published baselines.

## Context

The seed-0 `x4-full-updates` pilot improved gradually through epoch 50, where a
16-rollout periodic evaluation reached a raw MsPacman mean of 2008.125, but the
90-epoch held-out evaluation finished at 1563.75. Actor entropy rose from about
1.19 near the peak to 1.74 at the end while imagined return continued rising,
which is consistent with late behavior drift or exploitation of world-model
error. It is not proof of either mechanism.

Periodic checkpoints in that run used successive environment seeds. Therefore
the old curve cannot cleanly separate policy regression from cohort variance,
and only final inference weights were retained. Selecting the old epoch-50
score after observing it would also violate the declared final gate.

## Decision

Add the opt-in `late-cosine-40-90` Actor-Critic stability profile:

- keep the existing `x4-full-updates` interaction, replay, batch, update, and
  sampled-frame budgets;
- keep Actor-Critic LR `1e-4` and entropy scale `3e-4` through task epoch 40;
- cosine decay LR to `2.5e-5` and entropy scale to `5e-5` by task epoch 90;
- restart the fixed schedule for each newly routed task and never condition it
  on evaluation return;
- reuse one deterministic periodic-validation seed cohort at all checkpoints;
- reserve a disjoint deterministic cohort for final evaluation; and
- atomically retain the exact task-banked inference weights corresponding to
  every periodic evaluation plus the held-out final evaluation.

The best-validation pointer is diagnostic model selection metadata. It cannot
be reported as the held-out final result, and the saved weight-only artifacts
cannot be presented as resumable checkpoints.

## Consequences

The next Task 1 run can distinguish within-policy change from changing Atari
seeds and cannot lose an interesting checkpoint merely because later updates
regress. The actor schedule is nevertheless a hyperparameter chosen after one
seed-0 pilot, so passing the held-out Task 1 threshold still requires an
independent seed confirmation before a continual campaign claim.

The extra snapshot writes add persistent storage and brief rank-0 I/O stalls.
They add no environment interaction or gradient update. The final held-out
evaluation budget remains 16 deterministic rollouts per evaluated task.
