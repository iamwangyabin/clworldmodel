# Evolving-Core Task-0 Hyperparameter Sweep v1 (Atari)

## Scope

This is a seed-0 acquisition pilot for the fixed curriculum
**MsPacman, Boxing, CrazyClimber**. It addresses the specific risk that the
first task may be under-trained before its representation becomes the starting
point for continual learning. It does not change task order, interaction
budget, architecture, Replay, loss terms, evaluation cohort, or random seed.
It is not evidence that any setting is generally superior, and no finite sweep
can guarantee successful learning.

The already launched `fixed_v1` 270-epoch run supplies the fifth, unchanged
control. Four additional jobs train from scratch through only the first
90-epoch MsPacman phase:

| profile | Task-0 shared-core LR | private projector/atom/head LR | Actor-Critic LR |
|---|---:|---:|---:|
| `fixed_v1` control | `2e-4` | `2e-4` | `1e-4` |
| `task0_shared_lr_1e4` | `1e-4` | `2e-4` | `1e-4` |
| `task0_shared_lr_3e4` | `3e-4` | `2e-4` | `1e-4` |
| `task0_private_lr_3e4` | `2e-4` | `3e-4` | `1e-4` |
| `task0_actor_lr_2e4` | `2e-4` | `2e-4` | `2e-4` |

Each non-control profile changes exactly one declared value. Configuration
rejects any other drift and rejects a sweep profile whose total duration is not
exactly the first task duration. These LR values therefore apply only during
the Task-0 pilot; they do not silently change later-task optimization.

## Matched Budget

Every candidate uses seed index 0 and the published Task-0 collection stream:

- 90 epochs and 5,898,240 raw emulator frames;
- 90,000 online world-model updates with `T=32, N=16`;
- 72,000 Actor-Critic updates;
- unchanged ARROW-50 FIFO/LTDM capacity and selection probability;
- uint8 CPU mmap observations and the same BF16/TF32 execution profile; and
- the existing 1,000-update boundary consolidation, reported as extra compute.

The selection observation is taken before those 1,000 consolidation updates,
so consolidation cannot improve a candidate's ranking. Evaluation transitions
never enter Replay or optimization.

## Eligibility And Selection

A candidate is eligible only if it completes all 90 online epochs, writes both
resumable Task-0 boundary checkpoints, has a finite raw return, uses the fixed
validation seed, and produces no held-out-final evaluation. The trainer writes
`evolving_core_consolidation/task_00_pre_validation.json` immediately after
the fixed 16-rollout validation and before any consolidation gradient. This
artifact survives a safely rolled-back consolidation failure.

Rank the complete set of five profiles by descending
`validation.raw_mean[0]`. An exact-score tie is resolved by the smaller sum of
absolute natural-log LR ratios from `fixed_v1`, then by lexical profile name.
The selector reads no `final_evaluation.json`. The full-run control launched
before the standalone artifact existed is read compatibly from
`task_00_boundary.json.validation.pre_raw_mean[0]`, which is the same
pre-consolidation measurement.

Seed 0 is used for selection only. If `fixed_v1` wins, its already-running full
pilot may continue. If another profile wins, it must be launched from scratch
under a separately resolved 270-epoch config; the 90-epoch checkpoint cannot
be presented as an equivalent resume because its resolved config differs.
Fresh confirmation seeds are required before any performance claim.

## Launch And Selection

Inspect one profile without training:

```bash
python scripts/run_evolving_task0_sweep.py \
  --profile task0_shared_lr_1e4 \
  --seed 0 \
  --dry-run
```

Run the command once for each of the four non-control profiles. After all four
jobs and the control's Task-0 boundary exist, create the immutable ranking:

```bash
python scripts/select_evolving_task0_profile.py \
  --candidate-dir /path/to/fixed_v1 \
  --candidate-dir /path/to/task0_shared_lr_1e4 \
  --candidate-dir /path/to/task0_shared_lr_3e4 \
  --candidate-dir /path/to/task0_private_lr_3e4 \
  --candidate-dir /path/to/task0_actor_lr_2e4 \
  --output /path/to/task0_selection.json
```

Both training launchers enforce clean, committed, pushed, upstream-synchronized
Git provenance before any environment interaction or parameter update.
