# Evolving-Core Atomic RSSM ARROW v2 (Atari)

## Scope

`Evolving-Core-Atomic-RSSM-ARROW-v2-Atari-TaskAware` is the formal
full-curriculum successor to v1. It inherits the complete model ownership,
Replay, loss, gradient projection, evaluation isolation, checkpoint,
consolidation, task order, and compute-budget contract from
`evolving_core_atomic_rssm_arrow_v1_atari.md`.

The sole behavioral change is the Task-0 shared-core Adam learning rate:

| setting | `fixed_v1` | `fixed_v2` |
|---|---:|---:|
| Task-0 shared core | `2e-4` | `3e-4` |
| later-task shared core | `1e-4` | `1e-4` |
| task-private state | `2e-4` | `2e-4` |
| Actor-Critic | `1e-4` | `1e-4` |
| route optimizer | `1e-3` | `1e-3` |
| boundary consolidation | `2e-5` | `2e-5` |

In particular, `fixed_v2` does **not** set the general `shared_core_lr` to
`3e-4`; the higher value applies only while acquiring Task 0. All three tasks
remain 90 epochs, so the exact v1 budgets remain unchanged: 17,694,720 raw
frames, 270,000 online world-model updates, 3,000 extra consolidation updates,
and 216,000 Actor-Critic updates.

## Evidence And Claim Limit

The operator authorized this promotion after the seed-0, EnvParallel16 Task-0
diagnostic showed a large and sustained intermediate advantage for the
`3e-4` shared-core candidate. Those attempts were partial and ineligible for
the preregistered final selector. Consequently, v2 is a prospectively chosen
formal configuration, not a completed hyperparameter-sweep result and not
evidence of general or multi-seed superiority.

No v1 checkpoint may be relabeled or resumed as v2. A v2 result must start from
scratch and record `evolving_task0_profile=fixed_v2` in its resolved config and
the v2 protocol name in its launch manifest.

## Launch

Inspect the formal configuration without environment interaction:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order mspacman-boxing-crazyclimber \
  --task0-profile fixed_v2 \
  --seed 0 \
  --classification official \
  --dry-run
```

`fixed_v2` is the launcher default. Exact v1 reproduction remains available
only through the explicit `--task0-profile fixed_v1` option. Non-dry launches
retain the clean, committed, pushed, and upstream-synchronized Git gate.

## Metric reporting

A completed v2 run must be summarized under
`continual_evaluation_metrics_v1.md` and retain the entire raw evaluation
matrix. Report `ACC_1` through `ACC_3`, `min-ACC_2/3`, `WC-ACC_1/2/3`, and
three-task forgetting `F_3`, together with every task's raw return and
normalized score. Because this is a three-task, task-aware method evaluated
with deterministic routed actors, its metrics belong in the diagnostic
three-task/task-aware table and must not be compared directly with ARROW's
published stochastic six-task headline values. Parameter growth per task,
trainable parameters, parameter bytes, Replay bytes, and all extra
consolidation updates are mandatory companion columns.
