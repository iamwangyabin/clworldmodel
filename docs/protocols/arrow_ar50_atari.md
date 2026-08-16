# ARROW-50 Atari primary-method protocol

## Status

Primary method. A seed-0 analysis repeat preserves task-boundary snapshots;
no single run is a paper reproduction until the complete multi-seed protocol
has finished.

## Method definition

ARROW-50 combines the upstream DreamerV3-style world model and actor-critic
with two replay buffers under a fixed total budget:

- FIFO: 512 trajectories x 512 time steps;
- LTDM: 512 trajectories x 512 time steps;
- total: 1,024 trajectories and 524,288 observations;
- minibatch buffer selection: 0.5 FIFO and 0.5 LTDM;
- image observations: 64 x 64 RGB;
- world-model minibatch: 32 time steps x 16 sequences.

The name `ARROW-50` freezes both capacity allocation and sampling allocation.
Changing either value creates a named ablation.

## Upstream reference

- Repository: `https://github.com/Cerenaut/ARROW`
- Commit: `cb05e7d97ed83c3cf6e528960db0da6868e29232`
- Source: `Code/ARROW_and_DV3/Atari/`
- Configs: `Configs/Atari configs/CL-task configs/`

Runs execute the project-maintained source under `third_party/arrow` directly.
Its base commit, compatibility changes, runtime optimizations, and current file
manifest are recorded in `third_party/arrow/UPSTREAM.md`.

## Frozen first curriculum

The first official campaign uses the original-order task sequence:

1. `ALE/MsPacman-v5`
2. `ALE/Boxing-v5`
3. `ALE/CrazyClimber-v5`
4. `ALE/Frostbite-v5`
5. `ALE/Seaquest-v5`
6. `ALE/Enduro-v5`

Tasks switch every 90 epochs. Preserve the upstream reward scales, full action
space, frame repeat of 4, update budgets, evaluation schedule, and seeds.

Published seed IDs map to `123456789`, `1337`, `31337`, `42`, and `987654321`.

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

## Target environment

- Linux x86_64;
- Python 3.10;
- one NVIDIA A40 48 GB or A100 80 GB; use at least 48 GB VRAM;
- NVIDIA driver compatible with CUDA 11.8 wheels;
- at least 64 GB system RAM and 100 GB free local storage;
- the Atari ROMs bundled in the pinned `ale-py==0.11.1` wheel.

Install the exact reference dependencies in an isolated environment:

```bash
conda create -n arrow python=3.10 -y
conda activate arrow
python -m pip install --upgrade pip
python -m pip install -r third_party/arrow/requirements.txt
```

Do not run `AutoROM` for this protocol. `ale-py==0.11.1` packages the ROMs in
its wheel, while AutoROM 0.6.1 attempts a separate GitHub Gist download. Verify
the six required environments directly after installation:

```bash
python - <<'PY'
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
for game in ("MsPacman", "Boxing", "CrazyClimber", "Frostbite", "Seaquest", "Enduro"):
    env = gym.make(f"ALE/{game}-v5")
    observation, _ = env.reset(seed=0)
    print(game, observation.shape, env.action_space.n)
    env.close()
PY
```

On a mainland China server, obtain `ale-py==0.11.1` from an accessible PyPI
mirror rather than redirecting AutoROM to an unofficial ROM mirror. ROM files
must not be copied into this repository.

The two float32 image replay tensors alone occupy about 24 GiB. A 24 GB GPU is
therefore not suitable for the frozen full-capacity run. The paper reports one
A40 or A100 per experiment and approximately 50 hours for one continual Atari
ARROW/DV3 setting. Treat that as scheduling guidance, not a runtime guarantee.

## Execution ladder

1. Run `python scripts/run_arrow_ar50_atari.py --seed 0 --dry-run` and inspect
   the resolved launch record.
2. Run the frozen original-order curriculum for seed 0 as a pilot.
3. Run all five frozen seeds.

Only step 3 supports an official reproduction claim. Preserve failed runs and
resource measurements from every step.

## Canonical launch

From the repository root:

```bash
python scripts/run_arrow_ar50_atari.py \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_ar50_original_s0_analysis
```

The output directory must not already exist. It receives `launch.json`, the
resolved config and TensorBoard events, `train.log`, `run_status.json`, and the
`analysis_snapshots/` directory. On VirtAI offline training, place the run
under the mounted result-output directory such as `/gemini/output`; do not use
temporary container storage.

On quota-limited hosts, use `--cpu-threads N` to cap OpenMP, MKL, OpenBLAS,
and NumExpr thread pools. This does not change the interaction or update
budgets, and the launcher records the setting in its launch manifest. Choose
`N` from a documented performance pilot rather than silently changing it
during an official run.

Enable stage timing when profiling the default optimized runtime:

```bash
python scripts/run_arrow_ar50_atari.py \
  --seed 0 \
  --profile-stages
```

The launcher always enables the documented tensor categorical kernels,
compiled world-model loss, fused Adam, TF32, and set-to-none gradients. Results
must therefore be labeled `vendored-optimized`, not bitwise-faithful upstream.

Use `--curriculum reversed` or `--curriculum two-cycle` only for their named
paper protocols. The primary default remains original order.
