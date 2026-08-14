# DreamerV3/FIFO Atari matched-control protocol

## Status

Matched control for the frozen ARROW-50 Atari protocol. A single seed is a
pilot; the five published seeds are required before making a baseline claim.

## Method definition

The control uses the same vendored DreamerV3-style world model, actor-critic,
curriculum, interaction budget, update budget, evaluation schedule, and total
replay capacity as ARROW-50. Its only replay store is a FIFO buffer:

- FIFO: 1,024 trajectories x 512 time steps;
- total: 1,024 trajectories and 524,288 observations;
- image observations: 64 x 64 RGB stored as float32 on CUDA;
- world-model minibatch: 32 time steps x 16 sequences.

ARROW-50 divides the same 1,024-trajectory allowance between FIFO and LTDM.
This control therefore isolates replay retention policy without changing the
trajectory capacity or raw observation-byte allowance.

## Frozen curriculum

The primary control uses the same original-order sequence as ARROW-50:

1. `ALE/MsPacman-v5`
2. `ALE/Boxing-v5`
3. `ALE/CrazyClimber-v5`
4. `ALE/Frostbite-v5`
5. `ALE/Seaquest-v5`
6. `ALE/Enduro-v5`

Tasks switch every 90 epochs. Seeds map to `123456789`, `1337`, `31337`, `42`,
and `987654321`. Preserve the upstream reward scales, full action space, frame
repeat, update budgets, and stochastic evaluation policy.

The upstream configuration contains 541 epochs so it can evaluate at epoch
540. Because the sequential schedule advances after each epoch, epoch 540 has
already returned to the first task for one training epoch. The analysis
snapshot after completing the sixth task is therefore epoch 539; the launcher
also preserves the distinct final epoch-540 weights.

## Analysis snapshots

The canonical launcher saves world-model and actor-critic weights after epochs
89, 179, 269, 359, 449, and 539, plus the final weights after epoch 540. Each
snapshot contains:

- CPU-portable world-model and actor-critic state dictionaries;
- resolved configuration;
- seed, epoch, distinct world-model and actor-critic update counts, raw-frame
  count, and task metadata;
- an adjacent SHA-256 checksum.

These are explicitly analysis snapshots, not resumable checkpoints. They omit
optimizers, replay contents, RNG states, and environment-schedule state. They
are sufficient for fixed-data checkpoint differencing but must not be used to
claim equivalent training resume.

## Canonical launch

Inspect the resolved seed-0 command without launching:

```bash
python scripts/run_dv3_fifo_atari.py --seed 0 --dry-run
```

Run it into a persistent directory:

```bash
python scripts/run_dv3_fifo_atari.py \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/dv3_fifo_original_s0_analysis
```

The output directory must not already exist. It receives `launch.json`, the
resolved config and TensorBoard events, `train.log`, `run_status.json`, and the
`analysis_snapshots/` directory. On VirtAI offline training, place the run
under the mounted result-output directory such as `/gemini/output`; do not use
temporary container storage.

The required environment and GPU sizing are identical to
`docs/protocols/arrow_ar50_atari.md`.
