# Evolving-Core Learned-Base Low-Rank Adapters v1 (Atari)

## Scope and protocol name

This document defines the separately named, from-scratch, task-aware seed-0
pilot:

`Evolving-Core-LearnedTask0Base-LowRank32QFP-PrivatePredictionAdapters-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

The v1 experiment uses a **fixed rank of 32**. It is not an adaptive-rank or
boundary teacher-compression result, and must not be described as either. The
fixed-rank pilot isolates whether a learned Task-0 mechanism basis can replace
five full Dense Q/F/P copies without repeating the failed random frozen-basis
hypothesis.

## Fixed topology

The continually plastic, replay-protected core remains the CNN, base posterior,
base recurrent transition, base prior, and latent interface. Every task retains
one zero-effect spatial projector. Task identity selects the projector,
mechanism route, prediction adapters, Replay slice, and Actor-Critic; therefore
the method is task-aware.

### Learned Q/F/P mechanism base

Task 0 owns and learns the original full-width Dense recurrent/posterior/prior
residual mechanisms with hidden widths `512/512/256` and four atoms. At the end
of Task 0 those three mechanisms are preserved as the learned base. They remain
frozen for Tasks 1-5.

Each later task owns a zero-effect Rank-32 residual on top of that learned base:

`LayerNorm -> in-to-32 -> 32-to-hidden -> SiLU -> hidden-to-32 -> 32-to-out`.

Only the final `32-to-out` projection is initialized to zero. Consequently a
new route is exactly the learned Task-0 mechanism before its first update, while
the final projection receives a nonzero first-step gradient. The private-output
regularizer sees only the task's low-rank delta, not the frozen Task-0 base.

Old-atom routing is disabled. The unused route tensors retained by the common
bank container are never called and never enter an optimizer; their 180 scalar
parameters are nevertheless included in the online ledger.

### Prediction heads

Task 0 learns the one decoder, reward head, and continuation head already
contained in the ARROW world model. After Task 0 these base heads are frozen.
For each of Tasks 1-5, each of the three heads receives an independent
zero-effect Rank-32 feature adapter:

`state + 0.1 * up(SiLU(down(LayerNorm(state))))`.

The final projection is zero initialized. The adapted state is passed to the
frozen base head. This avoids image-size-specific adapter code and avoids a full
decoder/reward/continue copy.

Each task retains an independent standard DreamerV3 MLP Actor and Critic. This
pilot deliberately does not share behavior and does not use FastKAN.

## Training and protection

Task 0 keeps the original-six Dense `fixed_v1` optimizer profile. The shared
CNN/base RSSM uses `2e-4` on Task 0 and `1e-4` later. Task-private state uses
`2e-4`; private Actor-Critics use `1e-4`. Task-0 Q/F/P and base prediction heads
belong to the Task-0 private optimizer. For later tasks, only the selected
projector, Rank-32 Q/F/P delta, and three prediction adapters enter the current
private optimizer. No route optimizer is created because reuse is disabled.

Later online updates preserve the fixed 16-sequence budget: 12 current-task
sequences and four task-homogeneous LTDM sequences from one uniformly selected
completed task. Real old observations, rewards, and continuation targets retain
the complete Dreamer loss. The frozen cumulative boundary teacher also supplies
the existing posterior/hidden/old-Actor interface losses and observation,
symlog-reward, and continuation-output distillation at scale `0.1`. No extra
teacher copy, sequence, environment transition, or optimizer step is added.

Component conflict projection applies to the plastic shared CNN/base RSSM and
latent interface. Frozen base heads and old private adapters are not optimizer
groups. Current task-private gradients are unprojected and isolated.

At each boundary, the existing 1,000 task-balanced LTDM consolidation steps
update only the shared CNN/base RSSM/latent interface at `2e-5`. Fixed-cohort
raw returns gate acceptance; rollback restores both shared weights and their
persistent Adam state. Prediction base heads and adapters are excluded from
consolidation because they are task-private frozen state.

## Curriculum, Replay, and compute budgets

The order is MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and Enduro,
with 90 epochs per task and resolved seed `123456789` for seed index 0. Replay
is ARROW-50 with FIFO/LTDM capacity `512/512` and 50/50 sub-buffer selection.
Observations use uint8 CPU mmap storage. Evaluation transitions never enter
Replay.

| quantity | value |
|---|---:|
| raw emulator frames | `35,389,440` |
| online world-model updates | `540,000` |
| boundary-consolidation updates | `6,000` |
| total world-model optimizer steps | `546,000` |
| Actor-Critic updates | `432,000` |
| online current sequences | `6,840,000` |
| online memory sequences | `1,800,000` |
| consolidation sequences | `96,000` |

## Exact online parameter ledger

The learned Dense Task-0 Q/F/P base contains `3,816,192` parameters. A later
Rank-32 Q/F/P delta contains `359,168` parameters. One prediction-feature
adapter contains `102,912` parameters, so the three adapters add `308,736` per
later task. Each projector adds `34,240`; the common inactive route container
adds `12 * task_id` scalars for task `task_id`.

| component | parameters |
|---|---:|
| base ARROW world model | `19,498,853` |
| six projectors | `205,440` |
| Task-0 Dense Q/F/P base | `3,816,192` |
| five Rank-32 Q/F/P deltas | `1,795,840` |
| five sets of prediction adapters | `1,543,680` |
| registered inactive route scalars | `180` |
| **world model** | **`26,860,185`** |
| six private MLP Actor-Critic pairs | `10,295,910` |
| **online total** | **`37,156,095`** |
| FP32 parameter bytes | `148,624,380` |

Per-task world-model additions are `3,850,432`, `702,156`, `702,168`,
`702,180`, `702,192`, and `702,204`. The online total is `58,554,585`
parameters (`61.1787%`) below six-task Dense Evolving-Core and `15,741,440`
(`29.7584%`) below the shared-plastic-head/Dense-QFP pilot. It is still
`15,941,257` parameters (`75.1420%`) above single-policy ARROW-50 because six
independent Actor-Critics and task routing are intentional.

Optimizer state, gradients, activations, Replay, boundary checkpoints, and the
one training-only boundary teacher are outside this online inference ledger and
must be reported separately.

## Metrics and evidence boundary

Fixed evaluation reports episodic return. Raw per-task mean and standard
deviation remain primary; reward-scaled means and ARROW-paper normalization are
separate derived metrics. Early epoch evaluations are not final results.
Final reporting requires the six-task boundary matrix, average performance,
forgetting, minimum/worst-case performance, and valid transfer metrics.

A completed seed-0 run remains a pilot. It can reject a broken architecture or
justify a multi-seed campaign, but cannot establish superiority over ARROW or
Dense Evolving-Core.

## Launch gate

Inspect without environment interaction or updates:

```bash
python scripts/run_evolving_learned_base_adapters.py --seed 0 --dry-run
```

A real launch requires a clean commit already pushed and synchronized with the
configured GitHub upstream, sufficient disk space for live Replay and rolling
atomic checkpoints, and a successful CUDA smoke on the target 3090.
