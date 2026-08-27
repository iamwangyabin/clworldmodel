# CNN-Projector-RSSM-CompactLoRA-ARROW-v2 Atari Task-Aware Protocol

## Claim Boundary

This is a Task-1-snapshot-seeded, task-aware, single-seed compactness ablation.
It tests whether the later-task acquisition of the matched high-capacity
projector/LoRA pilot survives a fixed 73.8 percent reduction in persistent RSSM
adapter parameters. It is not a task-agnostic ARROW-50 comparison and is not an
equivalent resume of the source Task-1 run.

## Matched Model

Task 0 loads and freezes the same completed MsPacman CNN, RSSM route,
reconstruction/reward/continuation heads, and Actor-Critic used by the
high-capacity pilot. Each later task uses:

```text
pixels -> frozen Task-0 CNN -> task spatial projector
       -> frozen Task-0 RSSM + task compact affine LoRA
       -> task-private decoder/reward/continue heads
       -> task-independent Actor-Critic
```

The spatial projector remains the zero-effect residual `256 x 4 x 4`,
64-channel-bottleneck module. The only changed model values are the RSSM LoRA
ranks:

| RSSM component | Capacity profile | Compact profile |
| --- | ---: | ---: |
| Recurrent | 128 | 32 |
| Representation | 128 | 32 |
| Transition | 32 | 16 |

Bias and normalization vectors retain exact task-specific additive deltas. The
compact route contains 643,648 FP32 RSSM adapter parameters, or 2,574,592
persistent bytes, per later task. Projectors, private heads, and independent
Actor-Critics are unchanged and accounted separately.

## Training And Evaluation

The strong Task-1 boundary contributes weights only. Replay, optimizers, RNG,
and schedule position restart at Boxing. Training uses current-task ARROW-50
Replay only; no old real or evaluation transition enters training.

All budgets match the high-capacity pilot:

- 90 epochs each for Boxing and CrazyClimber;
- world-model batch `N=64`, 500 updates per epoch at `2e-4`;
- Actor-Critic context `N=512`, 400 updates per epoch at `2e-4`;
- BF16 autocast with FP32 parameters, Adam state, and sensitive operations;
- uint8 mmap Replay and 12 configured CPU threads; and
- the same fixed 16-rollout periodic and held-out evaluation cohorts.

Task identity selects the route and independent Actor-Critic. Old routes and
policies freeze at task boundaries. Raw returns are reported separately by
task; periodic-cohort retention must not be mixed with held-out-final values.

## Required Evidence

- clean pushed Git provenance and exact source-boundary SHA256;
- a successful target-GPU smoke before the formal run;
- resolved ranks `32/32/16` and runtime parameter accounting equal to 643,648
  RSSM adapter parameters per later task;
- empty new Replay and fresh later-task optimizer/RNG evidence;
- finite losses and exactly one CUDA training context;
- bitwise unchanged Task-0 tensors and frozen old routes at boundaries;
- matched interaction, update, sample-use, precision, and evaluation budgets;
- fixed-cohort acquisition, retention, forgetting, snapshots, and checksums;
- final model/Actor bank artifacts and Replay/evaluation isolation evidence;
  and
- explicit `resumable=false` labels for snapshots omitting optimizers, Replay,
  RNG, and scheduler/task position.
