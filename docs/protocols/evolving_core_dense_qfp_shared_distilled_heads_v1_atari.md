# Evolving-Core Dense Q/F/P + Shared Distilled Heads v1 (Atari)

## Scope and protocol names

This is a separately named, from-scratch, task-aware pilot with two declared
curriculum variants:

- `Evolving-Core-DenseQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-ThreeTask-Atari-TaskAware-Pilot`;
- `Evolving-Core-DenseQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

It changes only prediction-head ownership and its required protection path
relative to the corresponding full-width Dense Evolving-Core configuration.
It does not replace the Dense, compact, Shared-Frozen-Down, or shared-FastKAN
methods. No performance result exists yet.

## Fixed topology

The world model retains one continually plastic CNN and base
posterior/recurrent/prior RSSM. Every task, including Task 0, retains exactly
the original Dense Evolving-Core private state:

- one symmetric zero-effect spatial projector;
- four-atom recurrent Q/F/P routes with learned full-width Dense mechanisms at
  `512/512/256`; and
- one independent DreamerV3 MLP Actor and Critic.

The method deliberately does **not** use Shared-Frozen-Down matrices or
FastKAN behavior. It allocates only one decoder, one reward head, and one
continuation head. Every task route calls those same modules. They remain
plastic throughout training and belong to the protected shared parameter set.

Task identity selects the projector, Q/F/P route, replay slice, and private
Actor-Critic. It is not appended to latent input, but the method is still
task-aware.

## Online world-model update

Task 0 uses the unchanged 16-sequence current-task update. Later tasks keep the
same fixed 16-sequence budget: 12 current sequences and four task-homogeneous
LTDM sequences from one uniformly selected completed task. No additional
sequence or optimizer update is introduced.

On the four real old-task sequences, the student still minimizes the complete
Dreamer world-model loss against stored observations, rewards, and continuation
targets. It also retains the existing posterior, hidden-state, and frozen-old-
Actor interface losses. For the shared prediction heads, the already existing
frozen boundary world-model supplies:

\[
\begin{aligned}
L_D &= \operatorname{mean}\|\hat{x}_{student}-\hat{x}_{teacher}\|_2^2,\\
L_R &= \operatorname{mean}\|\widehat{\operatorname{symlog}r}_{student}
       -\widehat{\operatorname{symlog}r}_{teacher}\|_2^2,\\
L_C &= \operatorname{mean}\,\mathrm{KL}\left(
  \mathrm{Bernoulli}(\sigma(c_{teacher}))\;\|\;
  \mathrm{Bernoulli}(\sigma(c_{student}))\right).
\end{aligned}
\]

The additional term is `0.1 * (L_D + L_R + L_C)`. All three reductions are
means so the coefficient is independent of image and minibatch size. Teacher
outputs are stopped. This reuses the teacher and student forward passes already
required for Evolving-Core interface protection; no second teacher or forward
is allocated.

Component projection treats `observation_head`, `reward_head`, and
`continue_head` as three independent protected groups in addition to encoder,
posterior, recurrent, prior, and latent interface. Current task-private Q/F/P,
projector, route, and Actor-Critic gradients remain unprojected and isolated.

## Learning rates, consolidation, and rollback

The shared optimizer has explicit ownership groups. The CNN/base RSSM follows
the selected Dense profile (`3e-4` for three-task `fixed_v2` Task 0, `2e-4` for
original-six `fixed_v1` Task 0, then `1e-4`). The shared prediction heads keep
the original Dense task-private-head LR `2e-4` during every online task. Private
Q/F/P state uses `2e-4`, routes `1e-3`, private Actor-Critics `1e-4`, and
boundary consolidation `2e-5`.

At every boundary, 1,000 task-balanced LTDM updates expose only the shared CNN,
base RSSM, latent interface, and the three shared prediction heads. Fixed-
cohort raw returns gate acceptance. A rejected or failed consolidation restores
all of those weights and the complete persistent shared Adam state. The next
boundary teacher is the accepted cumulative world model, not a per-task head
bank.

## Curriculum and budgets

The three-task variant uses MsPacman, Boxing, and CrazyClimber for 90 epochs
each and the accepted `fixed_v2` Task-0 core LR. The original-six variant uses
MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and Enduro for 90 epochs
each and preserves the preregistered original-six `fixed_v1` Task-0 profile.

The original-six ledger remains:

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

Replay remains ARROW-50: equal FIFO/LTDM capacity and equal sub-buffer
selection probability. Observation storage stays uint8 CPU mmap under the
named Evolving-Core runtime profile. Evaluation transitions never enter Replay.

## Parameter and byte accounting

One decoder/reward/continue set has `8,562,629` parameters and is included once
in the base `19,498,853`-parameter world model. The shared-head method adds no
later head copies.

| curriculum/topology | world model | behavior | online total | FP32 bytes |
|---|---:|---:|---:|---:|
| three-task shared heads + Dense Q/F/P + 3 private MLP pairs | `31,050,185` | `5,144,883` | `36,195,068` | `144,780,272` |
| three-task Dense reference | `48,175,443` | `5,144,883` | `53,320,326` | `213,281,304` |
| six-task shared heads + Dense Q/F/P + 6 private MLP pairs | `42,601,625` | `10,289,766` | `52,891,391` | `211,565,564` |
| six-task Dense reference | `85,414,770` | `10,289,766` | `95,704,536` | `382,818,144` |

The six per-task world-model additions are `3,850,432`, `3,850,444`,
`3,850,456`, `3,850,468`, `3,850,480`, and `3,850,492`. The six-task reduction
relative to Dense is `42,813,145` parameters (`44.7347%`). It removes repeated
heads but does not eliminate linear Q/F/P or private-Actor-Critic growth.

The common full boundary world-model teacher contains one frozen copy of the
shared heads during later-task training. This was already required by
Evolving-Core and is training-only, constant-count state; there is no extra
head-only teacher. Runtime accounting must still report optimizer state,
gradients, activations, Replay, checkpoint assets, and actual peak bytes
separately from online inference parameters.

## Checkpoint and metrics contract

Complete boundary checkpoints retain the shared heads, shared optimizer groups,
all task-private Q/F/P and Actor-Critic banks, boundary teacher, Replay and mmap
provenance, RNG streams, schedule position, and all counters. The existing
original-six `latest_boundary` retention policy remains unchanged.

Logs must preserve raw per-task returns and separate scaled/normalized metrics.
Final reporting requires average performance and forgetting, plus forward or
backward transfer only when the required reference points are valid. Early
epoch evaluations are not final results. A seed-0 completion is still a pilot.

## Dry run and target-CUDA smoke

Inspect the original-six resolved protocol without interaction or updates:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --prediction-head-profile shared_distilled \
  --seed 0 \
  --classification pilot \
  --dry-run
```

Exercise one synthetic old/current update on the target accelerator:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --prediction-head-profile shared_distilled \
  --device cuda:0
```

A real run remains gated on a clean commit that has been pushed and verified
synchronized with its upstream. The implementation alone does not authorize a
performance claim.
