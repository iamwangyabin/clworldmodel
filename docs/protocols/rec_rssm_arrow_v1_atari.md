# REC-RSSM ARROW v1 (Atari)

## Scope

`REC-RSSM-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware` is a single-seed
three-task pilot of fine-grained mechanism reuse over one frozen Task-1 CNN and
base RSSM. It is not task-agnostic and cannot establish a paper claim by
itself. The architectural decision is recorded in
`docs/decisions/0038-rec-rssm.md`.

The curriculum is MsPacman, Boxing, and CrazyClimber for 90 epochs each. The
fixed strong MsPacman boundary snapshot supplies Task 1. Training then starts
with empty Replay and new optimizer, RNG, and schedule state. Boxing occupies
epochs 90 through 179 and CrazyClimber epochs 180 through 269.

## Fixed Architecture

- shared immutable Task-1 CNN and recurrent/posterior/prior RSSM;
- one private zero-effect spatial projector per later task;
- recurrent, posterior, and prior mechanism widths `512/512/256`;
- four lossless atoms of widths `128/128/64` in each mechanism;
- one full coefficient-one current mechanism per later task;
- independent per-old-task, per-atom `tanh` gates and binary masks;
- private decoder, reward, and continuation heads; and
- one fresh independent Actor-Critic and optimizer per later task.

The three mechanisms contain exactly 3,816,192 FP32 parameters per later task.
Task 3 adds 12 FP32 route parameters. Splitting a mechanism into atoms changes
neither its parameter count nor its full forward function beyond reduction
order.

## Training Phases

Boxing has no old mechanism and begins directly in expansion. The first local
CrazyClimber epoch is the reuse probe: current mechanisms remain at their
zero-effect initialization and frozen, old atoms and the shared base remain
frozen, and only current routes, projector, and private heads are plastic. The
following 89 epochs jointly train current mechanisms and routes. Route learning
rate is `1e-3`, exactly five times the fixed `2e-4` world-model learning rate.

The method adds no teacher, distillation, sparsity, orthogonality,
decorrelation, or extra replay objective. Actor-Critic learning remains
independent and follows the matched MB-RSSM protocol.

## Boundary Consolidation

For each currently enabled old atom, eight frozen replay batches estimate

```text
delta_loss = loss(atom disabled) - loss(full route)
contribution = mean(norm(routed atom)) / mean(norm(full correction))
```

The same sampled tensors and restored PyTorch RNG state are used for every
condition. An atom is proposed for masking when `delta_loss <= 0` or
`contribution < 0.01`. The full and proposed routes then use the same fixed
16-rollout validation cohort and deterministic policy. Proposed masks are
accepted only when the pruned mean is at least 95 percent of the full-route
mean. Consolidation restores training RNG state, writes no evaluation
transition to Replay, performs no optimization, and never changes mechanism
weights.

The boundary artifact records every candidate, losses, contribution ratios,
full/pruned returns, accepted masks, rollback state, and the private/shared atom
manifest. A shared label means only that a second task retained the atom after
this test.

## Matched Budgets

The strong Task-1 snapshot, task order and duration, environment frames,
world-model and Actor-Critic updates, replay capacity and sampling, observation
preprocessing, BF16/FP32 execution, uint8 mmap Replay, CPU-thread limit, and
fixed evaluation cohorts match the whole-gate MB-RSSM pilot. Evaluation data
never enters Replay. Task identity selects the projector, mechanism route,
private heads, and Actor-Critic.

## Execution Gate

Training may start only from a clean pushed commit that is synchronized with
its upstream. First run a dry-run manifest check, then a unique no-compile GPU
smoke. The smoke must establish one intended CUDA context, finite losses,
phase-specific gradients, exact frozen tensors, Replay isolation, migration
parity, and return code zero. A failed smoke is preserved and not automatically
resumed or overwritten.

Formal results must retain raw periodic and final 16-rollout returns, snapshot
SHA-256 values, Task-1/Boxing retention, route manifests, atom ablations,
complete parameter accounting, and all boundary artifacts. Every saved analysis
snapshot is non-resumable because it omits optimizers, Replay, RNG, and schedule
position.
