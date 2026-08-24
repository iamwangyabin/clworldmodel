# CNN-FullBank-ARROW-v1 Atari Task-Aware Protocol

## Status and claim boundary

`CNN-FullBank-ARROW-v1-BF16AMP-Uint8Replay-Atari-TaskAware` is an
implementation-ready experimental protocol. It restores the original ARROW
Dreamer CNN and pixel reconstruction objective, then banks every trainable
world-model and behavior component by scheduled task. It has no performance or
retention claim until the named pilots and multi-seed campaign complete.

Task identity is privileged scheduler input. Results belong in a task-aware
upper-bound table and are not directly comparable to task-agnostic ARROW-50.

## Complete task route

For task `k`, the scheduler selects:

```text
o_t [3,64,64]
  -> CNN encoder E_k
  -> e_t [4096]
  -> posterior q_k(z_t | h_t, e_t)

(z_(t-1), a_(t-1), h_(t-1))
  -> recurrent dynamics R_k
  -> latent prior p_k(z_t | h_t)

[z_t, h_t]
  -> pixel decoder D_k
  -> reward head r_k
  -> continuation head c_k
  -> independent Actor-Critic AC_k
```

The CNN is the vendored four-layer Dreamer encoder with channel widths
`32, 64, 128, 256`, `4 x 4` kernels, stride 2, channel LayerNorm, and SiLU. Its
output is `4 x 4 x 256 = 4,096` and it has 691,104 parameters. There is no DINO
model, resize, frozen feature cache, patch projection, or shared adapter.

The posterior, recurrent dynamics, prior, decoder, reward and continuation
heads, KL terms, pixel loss, imagined rollouts, and MLP Actor-Critic otherwise
retain the vendored ARROW formulas and fixed budgets.

## Boundary and update semantics

Task 0 uses route 0. When task `k > 0` first arrives, route `k-1` is copied once
into the complete world-model route `k`, including the CNN. The new
Actor-Critic uses fresh deterministic initialization and fresh optimizer state;
it does not copy the previous game's policy. Every new task begins with the
configured random collection.

After activation, only route `k` and `AC_k` require gradients. Old CNNs,
posteriors, dynamics, priors, decoders, heads, policies, and their behavior are
isolated from later updates. One hundred percent of the unchanged update totals
go to the current task through task-conditional ARROW sampling. FIFO/LTDM
capacity, selection probability, overwrite and retention rules remain ARROW-50.

## Storage and precision

Replay stores the source Atari pixels as uint8 in run-local mmap files and
decodes a sampled batch to the prior float32 `[0,1]` model interface. This
reduces observation storage from 25,769,803,776 to 6,442,450,944 bytes without
changing capacity or sampled indices. Actions, rewards, continuation flags,
reset flags, and task IDs retain their existing dtypes.

CUDA model kernels run under BF16 autocast. Parameters, Adam state, categorical
sampling and KL, symlog/symexp, reconstruction and behavior losses, returns,
and value targets remain float32. BF16 uses no gradient scaler.

The continuation head emits autocast logits but applies its sigmoid and
Bernoulli loss in float32 using `binary_cross_entropy_with_logits`. This is a
runtime-correctness requirement, not a changed objective: applying sigmoid in
BF16 can round a confident continue prediction to exactly one before a terminal
target reaches the loss, producing a clamped loss of 100 and a misleading
quantized training trace.

The launcher accepts `--devices 1|2|4`. Two and four devices use native PyTorch
DDP with one fixed global replay draw split on the sequence axis. The regular
world-model global batch remains `T=32, N=16`; local `N` is 8 or 4. Actor
context remains globally `T=4, N=128`; local `N` is 64 or 32. Rank 0 alone owns
collection, replay, logging, and artifacts. Evaluation tasks are partitioned
across ranks and reduced to rank 0.

## Large-batch DP4 ablations

The optional `--batch-profile` flag defines named four-GPU throughput
ablations. These are not the fixed-global-batch baseline. The sample-matched
profiles keep environment interaction, replay capacity, replay selection, and
total sampled world-model and actor context frames fixed. They increase the
sequence batch, reduce optimizer steps by the same factor, and apply linear
learning-rate scaling. The compute-saturation profile instead retains the
original optimizer steps and learning rates while increasing sampled-frame use:

| Profile | WM global/local `N` | WM steps/epoch | Actor global/local `N` | Actor steps/epoch | WM/AC LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x2-linear-lr` | 32 / 8 | 500 | 256 / 64 | 400 | `2e-4` / `2e-4` |
| `x4-linear-lr` | 64 / 16 | 250 | 512 / 128 | 200 | `4e-4` / `4e-4` |
| `x4-full-updates` | 64 / 16 | 1,000 | 512 / 128 | 800 | `1e-4` / `1e-4` |

The hypothesis is that fewer DDP synchronization points and larger local
matrix operations improve wall-clock throughput. Equal sampled-frame use does
not make the optimization trajectory identical: there are fewer Adam updates,
different gradient noise, and scaled learning rates. Speed, finite losses, and
the final acquisition score must therefore all be measured. A larger batch is
not assumed to improve return before the 90-epoch gate passes.

`x4-full-updates` is deliberately not sample matched. It also keeps 30,000
world-model pretraining updates. Relative to the fixed-global-batch DP4
profile, it performs the same number of Adam updates with four times the replay
and actor-context samples per update schedule. Each GPU therefore receives the
original single-device local batches (`N=16` world model and `N=128` actor), but
the run consumes roughly four times the optimization FLOPs and sample uses.
Measure examples per second and acquisition alongside epoch wall time; this is
additional training compute, not a transparent four-GPU speedup.

### Late actor-stability pilot

The first `x4-full-updates` seed-0 Task 1 pilot reached a periodic MsPacman raw
mean of `2008.125 +/- 397.794` after 50 completed epochs, then finished at
`1563.75 +/- 433.732` after 90. That run did not pass the predeclared final
gate. Its periodic evaluations also advanced to a new environment-seed cohort
at every checkpoint, so the peak-to-final difference combines policy change
with evaluation-cohort noise.

`--actor-stability-profile late-cosine-40-90` is a separately named controlled
pilot. It preserves environment interaction, replay, world-model updates,
Actor-Critic updates, global batches, and sampled-frame use. During each
90-epoch task it keeps the Actor-Critic learning rate and entropy coefficient at
`1e-4` and `3e-4` through task epoch 40, then cosine decays them to `2.5e-5`
and `5e-5` at task epoch 90. The schedule depends only on task-local update
position, never on evaluation return.

This profile also makes evaluation auditable:

- every periodic checkpoint reuses one fixed validation seed cohort;
- final evaluation uses a disjoint held-out seed cohort;
- the two cohort seed lists are persisted in `evaluation_seed_manifest.json`;
- exact world-model and complete Actor-Critic-bank inference weights for each
  evaluated checkpoint are atomically saved under `evaluation_snapshots/`; and
- `best_validation_snapshot.json` points to the maximum mean raw return over
  seen tasks, but does not turn that validation result into a final test claim.

These snapshots are diagnostic inference artifacts, not resumable checkpoints:
they omit replay, optimizers, RNG, task-scheduler state, and counters required
to continue training equivalently. The held-out final score remains the Task 1
acquisition gate.

### Extended-training evaluation audit

The separate `--evaluation-audit-profile fixed-cohort-snapshots` option enables
the same fixed validation cohort, disjoint held-out final cohort, exact
evaluation snapshots, and best-validation pointer without changing Actor-Critic
hyperparameters. It is the correct profile for testing the narrower hypothesis
that the large-batch model is undertrained rather than unstable.

The next controlled seed-0 audit combines `x4-full-updates`,
`--task-duration-multiplier 2`, and this evaluation audit. It trains MsPacman
for 180 epochs while keeping WM/Actor-Critic learning rates at `1e-4`, entropy
scale at `3e-4`, and all per-epoch batch and update settings unchanged. Relative
to the failed 90-epoch run it doubles environment interaction, Adam updates,
and optimization sample use. It is therefore an extended-budget acquisition
ablation, not a replacement result under the 90-epoch protocol.

### Extended-duration acquisition

`--task-duration-multiplier 2` is a separately named task-1 acquisition
ablation. It changes the MsPacman task duration from 90 to 180 epochs. This
doubles environment interaction and total optimization sample use; with the
`x4-linear-lr` batch profile it performs half as many Adam updates as the
90-epoch fixed-batch baseline. It is not a result under the 90-epoch protocol.

Large mmap replay files should use node-local storage rather than a network
FUSE run directory. `--replay-mmap-root /dev/shm/clworldmodel-replay` creates a
unique backing directory and places only a symlink in the persistent run
directory. Logs, manifests, TensorBoard events, evaluations, and final weights
remain persistent. Replay is not checkpointed in either layout and the backing
directory is recorded in `launch.json`.

## Task-boundary artifacts and early diagnostics

Every CNN-FullBank launch saves an immutable complete-bank snapshot immediately
after each task's final model and behavior updates and before the scheduler
advances. The snapshot includes the complete world-model bank, complete
Actor-Critic bank, completed task actor, counters, resolved config, and exact
project Git commit. A SHA256 sidecar and atomic `index.json` record every
boundary. Existing files and index entries are never overwritten; a six-task
run must finish with six indexed boundary snapshots.

These artifacts are sufficient for exact per-task inference and retention
audits but are explicitly non-resumable because replay, optimizers, RNG, and
schedule state are omitted. Final convenience weights and fixed-cohort
evaluation snapshots remain separate artifacts.

The replacement Task 1 pilot also uses the predeclared diagnostic in
`tests/fixtures/arrow_ar50_original_s0_early_metrics.json`. After at least
three aligned points in world-model steps 1,000 through 5,000, continuation,
reconstruction, KL, and gradient norm must remain within the documented broad
ratio envelope and finite. This catches order-of-magnitude numerical failures;
it is not a pure comparison with original ARROW because precision, replay,
device count, task routing, and batch sample use differ. A failed guard permits
a recorded diagnostic stop, never deletion or an unrecorded restart.

## Evaluation gates

The first run is a one-task, 90-epoch MsPacman pilot with the published data,
interaction, and update budgets. It passes acquisition only if the final
16-rollout deterministic raw mean is at least 2,000. Intermediate peaks do not
pass the gate. A failed pilot remains recorded and is not extended into a
continual run.

After that gate, a two-task pilot must evaluate every seen task and report the
raw return matrix, final seen-task average, and computable forgetting. A full
six-task claim requires the fixed seed set; no single run establishes a method
claim.

## Launcher

```bash
python scripts/run_moe_arrow_atari.py \
  --method cnn-fullbank \
  --devices 4 \
  --batch-profile x4-full-updates \
  --task-duration-multiplier 2 \
  --evaluation-audit-profile fixed-cohort-snapshots \
  --early-progress-guard arrow-original-s0-v1 \
  --replay-mmap-root /dev/shm/clworldmodel-replay \
  --seed 0 \
  --task-prefix-length 1 \
  --dry-run
```

No `DINOV3_MODEL_PATH` is required or consumed.
