# CNN-Projector-RSSM-LoRA-ARROW-v1 Atari Task-Aware Protocol

## Claim boundary

This is a Task-1-snapshot-seeded, task-aware, single-seed incremental pilot.
The scheduler supplies a task index for route allocation and evaluation. It is
not directly comparable to task-agnostic ARROW-50 and is not an equivalent
resume of the source CNN-FullBank run.

## Model

Task 0 uses the completed MsPacman CNN, RSSM route, pixel/reward/continuation
heads, and Actor-Critic. They are frozen before Task 1 (Boxing) begins.

For each later task `k`:

```text
pixels -> frozen Task-0 CNN -> residual spatial projector P_k
       -> frozen Task-0 RSSM + affine LoRA L_k
       -> private decoder/reward/continue heads H_k
       -> independent Actor-Critic AC_k
```

`P_k` reshapes 4,096 features to `256 x 4 x 4`, applies GroupNorm and a
`256 -> 64 -> 64 depthwise -> 256` convolutional block, and adds the block to
its input. Its final convolution is zero initialized, making the initial
projector exactly the identity.

All Linear, GRUCell, and LayerNorm matrices in the task's recurrent,
representation, and transition modules receive LoRA deltas. Ranks are
`128/128/32`; biases and normalization vectors receive exact additive deltas.
The frozen originals are the same shared Task-0 parameters, not per-task
copies; only LoRA matrices and exact vector deltas are task-specific.

The private pixel decoder, reward head, and continuation head copy Task 0 once
and then train normally. `AC_k` uses fresh deterministic weights and a fresh
optimizer. This differs from the posthoc distillation probe, which initialized
behavior from the finished target task.

## Training and Replay

The source Task-1 boundary contributes weights only. Replay, optimizers, RNG,
and schedule position are reset and recorded as such. The environment schedule
starts at Boxing. New tasks begin with the same random collection rule as
CNN-FullBank.

ARROW-50 FIFO/LTDM capacity and sampling remain unchanged. One hundred percent
of world-model and Actor-Critic updates go to the current task. Evaluation
uses frozen deterministic policies and disjoint fixed seed cohorts;
evaluation transitions never enter Replay.

The first formal pilot inherits the source single-GPU x4 double-sample profile:
world-model `N=64`, 500 updates/epoch at `2e-4`; Actor-Critic context `N=512`,
400 updates/epoch at `2e-4`; BF16 autocast with FP32 parameters/Adam/sensitive
operations; uint8 mmap Replay; 90 epochs each for Boxing and CrazyClimber.

## Required evidence

- clean pushed Git provenance and exact source-boundary SHA256;
- empty new Replay and fresh optimizer/RNG evidence;
- finite world-model and Actor-Critic losses under actual Atari interaction;
- Task-0 CNN/RSSM/heads and Actor-Critic bitwise unchanged at every boundary;
- only the selected projector, LoRA deltas, private heads, and Actor-Critic
  receive gradients;
- fixed-cohort raw taskwise acquisition/retention/forgetting values;
- final weights, Actor-Critic bank, evaluation snapshots, boundary snapshots,
  manifests, TensorBoard events, parameter accounting, and checksums; and
- explicit `resumable=false` labeling for inference snapshots that omit Replay,
  optimizers, RNG, and scheduler/task position.

## Completed Seed-0 Pilots

Two three-task high-capacity (`128/128/32`) runs completed successfully. The
held-out final cohort used 16 deterministic rollouts per task:

- Initial Task-1 source (`7dd1bc66...`): MsPacman
  `1341.875 +/- 143.6671`, Boxing `79.6875 +/- 8.7373`, and CrazyClimber
  `21275.0015 +/- 803.4810`.
- Strong Task-1 source (`e3fef571...`): MsPacman
  `2595.625 +/- 491.7821`, Boxing `59.6875 +/- 12.2562`, and CrazyClimber
  `15900.0006 +/- 624.3371`.

Each completed model contained 41,604,207 world-model parameters and 5,147,955
parameters in its three-entry Actor-Critic bank. The source snapshots differ,
so these runs are not a controlled comparison of Task-1 acquisition quality.
They are single-seed, task-aware plasticity evidence only. The lightweight
result record is
[`references/cnn_incremental_seed0_results_20260827.json`](references/cnn_incremental_seed0_results_20260827.json);
no checkpoint, inference snapshot, or weight file is committed to Git.
