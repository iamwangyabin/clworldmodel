# Evolving-Core Task-0 Duration Sweep v1 (Atari)

## Question And Scope

This seed-0 pilot tests whether the fixed 90-epoch MsPacman phase is too short
for first-task acquisition. The curriculum identity remains
**MsPacman, Boxing, CrazyClimber**, but every candidate stops at its Task-0
boundary before seeing Boxing. It is a resource-scaling diagnostic, not a
matched-budget comparison and not evidence of continual retention.

The active 270-epoch `fixed_v1` run supplies its unchanged 90-epoch Task-0
boundary as the control. Four additional from-scratch candidates use:

| profile | Task-0 epochs | raw frames | online WM updates | Actor-Critic updates |
|---|---:|---:|---:|---:|
| `fixed_v1` control | 90 | 5,898,240 | 90,000 | 72,000 |
| `task0_epochs_120` | 120 | 7,864,320 | 120,000 | 96,000 |
| `task0_epochs_150` | 150 | 9,830,400 | 150,000 | 120,000 |
| `task0_epochs_180` | 180 | 11,796,480 | 180,000 | 144,000 |
| `task0_epochs_240` | 240 | 15,728,640 | 240,000 | 192,000 |

Every candidate also attempts the same separately reported 1,000 shared-core
boundary-consolidation updates. Replay capacity remains 512 FIFO plus 512 LTDM
trajectories; the later profiles see more total trajectories but receive no
larger capacity. All learning rates retain `fixed_v1` values: Task-0 shared
core `2e-4`, private modules `2e-4`, and Actor-Critic `1e-4`.

## Configuration And Isolation

Duration profiles replace scalar `swap_sched=90` with exact declared durations
`[D, 90, 90]` and set total training epochs to `D`. Configuration rejects a
different Task-0 duration, any later-task duration other than 90, a total epoch
count that crosses the first boundary, or any optimizer drift. This ensures a
candidate cannot silently begin Boxing.

All candidates use seed index 0, the same fixed validation cohort, the same
Atari preprocessing/action/reward/reset semantics, BF16/TF32 execution, uint8
CPU mmap Replay, ARROW-50 sampling, model topology, loss coefficients, and
random-policy rule. Evaluation transitions never enter Replay.

## Eligibility And Selection

The trainer atomically writes
`evolving_core_consolidation/task_00_pre_validation.json` after the final
online update and fixed 16-rollout validation but before any consolidation
gradient. Held-out-final evaluation is disabled for every duration candidate.

A run is eligible only when it completes exactly its declared duration, writes
both resumable Task-0 checkpoints, records a finite pre-consolidation raw mean,
and uses the identical validation seed. Require all five durations. Let `m` be
their maximum raw mean and define the near-best floor as
`m - 0.05 * max(abs(m), 1)`. Select the shortest candidate at or above that
floor, then higher raw mean and lexical name. Persist both score ranking and the
duration-ordered learning curve. This rule seeks the first saturation point;
it does not assert equivalence or statistical significance from one seed.

If every observed score is scientifically inadequate, no profile should be
promoted merely because the selector must name a mathematical winner. A chosen
duration requires a fresh full-curriculum run and new confirmation seeds.

## Launch

Inspect a candidate without training:

```bash
python scripts/run_evolving_task0_sweep.py \
  --profile task0_epochs_180 \
  --seed 0 \
  --dry-run
```

After all four candidates and the 90-epoch control boundary exist:

```bash
python scripts/select_evolving_task0_profile.py \
  --family duration \
  --candidate-dir /path/to/fixed_v1 \
  --candidate-dir /path/to/task0_epochs_120 \
  --candidate-dir /path/to/task0_epochs_150 \
  --candidate-dir /path/to/task0_epochs_180 \
  --candidate-dir /path/to/task0_epochs_240 \
  --output /path/to/task0_duration_selection.json
```

Non-dry launches enforce a clean, committed, pushed, and upstream-synchronized
Git state before environment interaction or parameter updates.
