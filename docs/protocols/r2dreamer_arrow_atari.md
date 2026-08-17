# R2Dreamer-ARROW-50 Atari protocol

## Status

Implementation-ready. The first run is a one-seed, single-task acquisition
sanity check, not a retention result or an R2-Dreamer reproduction claim.

## Method definition

`R2Dreamer-ARROW-50` combines the native R2-Dreamer `size12M` world model and
actor-critic update with ARROW-50 trajectory replay:

- R2-Dreamer source: `https://github.com/NM512/r2dreamer` at
  `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`;
- decoder-free CNN encoder with 1,024-dimensional embeddings;
- discrete RSSM with `deter=2048`, `stoch=32`, and `discrete=16`, for a
  2,560-dimensional feature;
- R2 Barlow objective with `B=16`, `T=64`, 1,024 flattened samples,
  loss scale `0.05`, and redundancy scale `5e-4`;
- LaProp at `4e-5`, AGC `0.3`, a 1,000-update linear warm-up, and the
  upstream PyTorch AMP `GradScaler` initial scale of `65,536`;
- FIFO: 512 trajectories x 512 steps;
- LTDM: 512 trajectories x 512 steps;
- whole-minibatch buffer selection: 0.5 FIFO and 0.5 LTDM.

ARROW's full-action-space Atari adapter, reward transformations, four-frame
action repeat, task schedule, and 50/50 retention policy are retained. The
R2-Dreamer model, controller, optimizer, latent replay state, and update ratio
are native. This is therefore not a faithful upstream R2-Dreamer benchmark or
a compute-matched ARROW ablation.

The project target is PyTorch 2.3 while the pinned R2 source targets a newer
runtime. The vendored route uses an equivalent affine RMSNorm fallback when
needed and deliberately disables `torch.compile`; both facts are saved in the
launch artifact. They are runtime compatibility deviations, not model or
objective changes.

## Replay integration and storage accounting

ARROW stores 512-step trajectory columns. The adapter samples 65 steps, uses
the first posterior state as R2 context, shifts actions once, and learns on the
following 64 observations. It retains Gymnasium's next-step terminal handling:
the terminal observation is marked `is_last`, while the following reset
observation is marked `is_first`. Reward scaling remains optimization-only;
the trainer writes both scaled and derived raw per-task returns to
`metrics.jsonl` and TensorBoard.

It stores R2 posterior stochastic and deterministic states in a project-owned
float32 CPU sidecar, plus one boolean `is_last` label per stored transition,
and writes posterior rollouts back after each update. ARROW's accepted
FIFO/LTDM slots and selected sub-buffer are unchanged.

The latent sidecar adds 5 GiB for the frozen 1,024-trajectory ARROW capacity;
the boolean terminal metadata is an additional 0.5 MiB. The
trainer writes `replay_storage_accounting.json` with actual transition bytes,
sidecar bytes, dtypes, and devices. It must be included in all later method
comparisons; equal trajectory capacity alone is not equal byte usage.

## Single-task acquisition sanity

The default launcher uses the first original-order task, `ALE/MsPacman-v5`,
and the native R2 Atari train ratio of 128 sampled model transitions per agent
decision. One ARROW collection block holds 16,384 trajectory positions and a
nominal 65,536 raw frames. Seven full blocks have a nominal 458,752-frame
budget, the nearest whole-block budget to R2-Dreamer's 410,000-frame
Atari-100k configuration. Terminal/autoreset positions do not advance Atari,
so realized agent decisions and raw frames are written separately per epoch.

At a full, uninterrupted block, the native ratio corresponds to 2,048 R2
updates of `16 x 64` samples. The trainer carries fractional sample credit
across blocks and schedules updates from realized agent decisions, preserving
the 128-to-1 native ratio despite terminal/autoreset positions. The expected
check is finite training, a non-trivial R2 objective, and rising deterministic
return. It is not comparable to ARROW's original 90-epoch task duration or its
model update budget.

## Continual exploratory scope

`--scope continual` retains the original six-task order and 90-epoch switch
schedule, including the final epoch 540. It keeps R2's native train ratio:
up to 2,048 updates per epoch, or a nominal 2,097,152 sampled model
transitions per epoch. The corresponding ARROW source budget is 512,000
samples per epoch. Label any such run `native-R2-compute exploratory`; it is
useful for feasibility and mechanistic diagnosis but not a fair ARROW
comparison.

## Execution

Inspect the resolved single-task run first:

```bash
python scripts/run_r2dreamer_arrow_atari.py --seed 0 --dry-run
```

After the exact committed branch is pushed and the target CUDA environment is
verified, run a smoke job before the seven-block acquisition run. It performs
twelve updates: the initial mixed-precision scale can intentionally skip early
updates while it calibrates, so a valid smoke requires a later finite gradient
and a completed optimizer step rather than treating that expected calibration
as representation collapse.

```bash
python scripts/run_r2dreamer_arrow_atari.py \
  --seed 0 \
  --smoke \
  --profile-stages \
  --output-dir /persistent/path/r2dreamer_arrow50_smoke_original_s0
```

Then launch the single-task sanity run:

```bash
python scripts/run_r2dreamer_arrow_atari.py \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/r2dreamer_arrow50_single-task_original_s0
```

The output directory receives `launch.json`, ARROW and R2 resolved configs,
parameter and replay-byte accounting, TensorBoard events, `train.log`,
`run_status.json`, and non-resumable analysis snapshots. Evaluation data does
not enter replay or updates.
