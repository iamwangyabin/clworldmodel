# ARROW-KANActor-50 Atari actor ablation

## Status

Completed but not advanced. The seed-0 `relu_kan` T2 pilot exposed a fixed-grid
interface failure: the unbounded first KAN output could leave the second
layer's compact support, so its poor returns cannot be interpreted as a fair
test of KAN trainability or continual learning. The implementation remains
available for checkpoint compatibility under its original name. The corrected
successor is documented in
[`arrow_kan_actor_bounded_atari.md`](arrow_kan_actor_bounded_atari.md).

The method is named `ARROW-KANActor-50`: ARROW-50 replay remains the continual
learning strategy, while only the Dreamer actor function approximator changes.
It must not be described as a full KAN-Dreamer implementation because the
world model, critic, and all other heads remain unchanged.

## Research question

The first question is intentionally sharp:

> Under matched interaction, update, replay, and actor-parameter budgets, does
> a fixed-grid local KAN actor retain the first-task policy after learning the
> second task without sacrificing acquisition or current-task performance?

The motivating mechanism is local basis activation: a new task may update a
smaller functional region than a dense MLP. This is a hypothesis, not a
guarantee. Different tasks can occupy overlapping basis supports, the latent
interface can drift, and the unchanged world model or critic can still be the
dominant source of forgetting.

## Frozen method definition

The matched control is the MLP actor in `ARROW-50`. Both methods retain:

- 512 FIFO trajectories and 512 LTDM trajectories, each 512 steps long;
- whole-minibatch buffer selection with probability 0.5 FIFO and 0.5 LTDM;
- identical replay dtype, device, interaction, world-model update, and
  actor-critic update budgets;
- the same encoder, RSSM, reconstruction decoder, reward head, continue head,
  critic, optimizer, action handling, evaluation schedule, and task labels;
- no task identity exposed to the actor or world model.

The unchanged float32 replay allocates exactly 25,813,843,968 persistent tensor
bytes (24.041 GiB), of which 25,769,803,776 bytes are observations. LTDM also
holds 512 Python priority-index entries; their interpreter overhead is not
included in the tensor total and is identical between actor methods. The launch
manifest records these values and a zero replay-byte difference.

Python, NumPy, and PyTorch use the frozen run seed, including Python `random`
for ARROW's FIFO/LTDM buffer choice. Dedicated collection and evaluation seed
streams deterministically seed every Atari reset and action space without
letting evaluation change later training seeds. Evaluation also restores the
parent Python, NumPy, and PyTorch CPU/CUDA RNG states before training resumes.
CUDA is not restricted to deterministic-only kernels, so the T2 result remains
a stochastic pilot rather than a bitwise-identical paired run; this limitation
is explicit in `launch.json`.

Only the actor changes:

```text
ARROW-50 actor:          1536 -> 512 -> 18, LayerNorm + SiLU
ARROW-KANActor-50 actor: 1536 ->  64 -> 18, fixed-grid ReLU-KAN
```

The actor input is the flattened categorical latent `z` with 1,024 binary
features concatenated with the 512-dimensional GRU state `h`. KAN receives `z`
unchanged and maps the bounded GRU state from `[-1, 1]` to `[0, 1]` by
`(h + 1) / 2`. This preprocessing is part of the named KAN actor and must be
reported as such.

For grid size `g = 5` and spline order `k = 3`, each edge has `g + k = 8`
fixed compact-support basis functions. For support endpoints `s_i, e_i`, the
independently implemented ReLU-KAN basis is

```text
R_i(x) = [ReLU(e_i - x) ReLU(x - s_i) 4 / (e_i - s_i)^2]^2.
```

Only the edge coefficients and output biases are trainable; grid locations are
persistent buffers. The width `64` is fixed because `64 x 8 = 512`, matching
the dominant coefficient budget of the original hidden layer:

| Actor | Trainable parameters | Difference from MLP |
| --- | ---: | ---: |
| MLP | 797,202 | 0 |
| fixed-grid ReLU-KAN | 795,730 | -1,472 (-0.185%) |

The implementation under `src/clworldmodel/models/relu_kan.py` is
project-owned and written independently from the formula in Qiu et al., 2024:
`https://arxiv.org/abs/2406.02075`. No third-party KAN source code is vendored.

## Two-task screening protocol

The first run stops after one task switch in the frozen original order:

| Stage | Environment | Epochs | Snapshot |
| ---: | --- | ---: | --- |
| 1 | `ALE/MsPacman-v5` | 0-89 | epoch 89 |
| 2 | `ALE/Boxing-v5` | 90-179 | epoch 179 |

This is the named `T2Pilot`, not a shortened official ARROW result. Task
duration remains 90 epochs and no per-task budget is reduced. The launcher
saves both boundary states and a final frozen-policy evaluation over the two
seen tasks only. Evaluation uses 16 stochastic rollouts per task, matching the
vendored ARROW evaluation behavior; those transitions never enter replay or
affect updates.

The regular evaluation at epoch 90 occurs before any Task 2 model or actor
update and estimates Task 1 return at acquisition. The explicit evaluation
after epoch 179 estimates Task 1 retention and Task 2 current-task performance.
The vendored environment scales rewards for optimization (`0.05` for
MsPacman), so the trainer now reports both the scaled value and the recovered
raw game return; raw return is the primary result. Report at minimum:

```text
F_MsPacman = R_MsPacman(epoch 90) - R_MsPacman(after epoch 179)
```

Also report the two absolute returns. A small forgetting value is not useful if
the KAN actor never learned MsPacman, and high retained MsPacman return is not
enough if Boxing acquisition collapses.

## Decision rule

The seed-0 pilot advances only if all three qualitative conditions hold:

1. KAN reaches comparable first-task acquisition to the matched MLP control.
2. KAN has materially smaller MsPacman forgetting after Boxing.
3. KAN retains comparable Boxing return, without extra samples, updates, or
   replay bytes.

The strongest outcome would be near-zero Task 1 forgetting with unchanged Task
2 performance. Before making that claim, repeat the frozen pair on at least
three seeds and show per-seed returns with episode-level uncertainty. A single
favorable seed, actor KL alone, or stable parameters without high environment
return cannot support a "solves incremental learning" conclusion.

If T2 passes but the conclusion remains ambiguous, `--task-prefix-length 3`
adds CrazyClimber as a second switch and stops after epoch 269. It is a second
pilot rung, not permission to inspect arbitrary prefix lengths until one looks
favorable.

If the pilot is positive, the next diagnostic is the existing fixed-input actor
audit: symmetric action KL, top-1 agreement, old-action margin, and the latent
to actor interface. Only after the actor result is stable should the pre-GRU
mapping and RSSM prior be replaced in separate named ablations.

## Launches

Inspect both matched commands without starting environment interaction:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network mlp \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run

python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run
```

After the exact commit is clean, pushed, and synchronized on the target GPU
host, launch into distinct persistent directories:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network mlp \
  --task-prefix-length 2 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_ar50_t2_mlp_s0

python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan \
  --task-prefix-length 2 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_ar50_t2_kan_s0
```

`launch.json` records the actor definition and unchanged replay allocation.
`actor_critic_parameter_accounting.json` records actual parameter, buffer, and
byte counts. `final_evaluation.json` contains reward scales plus final raw and
scaled per-task means and standard deviations. Analysis snapshots are
non-resumable because they omit replay, optimizers, RNG state, and the
environment scheduler.
