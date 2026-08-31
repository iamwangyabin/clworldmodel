# 0042: Promote the Task-0 Shared-Core Learning Rate

## Status

Accepted by the operator on 2026-08-30 for the next formal Evolving-Core run.

## Context

The seed-0 EnvParallel16 diagnostic produced a sustained intermediate
MsPacman advantage for `first_task_shared_core_lr=3e-4` over the original
`2e-4` value. The observed runs were preempted before the preregistered
90-epoch selection boundary, so their artifacts are diagnostic rather than an
eligible sweep result.

## Decision

Introduce a separately named `fixed_v2` full-curriculum profile and make it the
formal launcher default. Change only `first_task_shared_core_lr` from `2e-4`
to `3e-4`. Preserve `shared_core_lr=1e-4` for later tasks and preserve every
other optimizer, loss, architecture, schedule, Replay, evaluation, checkpoint,
and budget setting.

Keep `fixed_v1` selectable and unchanged. Keep all preregistered Task-0 sweep
profiles anchored to `fixed_v1`; do not reinterpret their controls after this
decision. A v2 run starts from scratch and receives a new protocol name rather
than resuming or relabeling a v1 checkpoint.

## Consequences

- Future full-curriculum launches default to the prospectively chosen `3e-4`
  Task-0 shared-core rate.
- Historical v1 results remain exactly reproducible.
- The change is not a claim that the incomplete single-seed diagnostic found a
  general winner; confirmation seeds and complete evaluations remain required.
- The higher learning rate does not propagate to Boxing or CrazyClimber.
