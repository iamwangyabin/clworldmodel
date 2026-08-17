# clworldmodel

`clworldmodel` is a research workspace for continual world-model reinforcement
learning. Its current base is the paper's canonical ARROW-50 method: a
DreamerV3-style agent with an equally split FIFO and long-term
distribution-matching (LTDM) replay budget.

## Current status

The repository contains the maintained baseline stack plus two named
decoder-free research routes:

- the maintained ARROW source based on a pinned upstream commit;
- the canonical ARROW-50 Atari launcher;
- the matched DreamerV3/FIFO Atari control launcher;
- the completed `ARROW-R2Rep-50` partial-port pilot, preserved as a failed
  single-task acquisition check rather than a retention result;
- the implementation-ready native `R2Dreamer-ARROW-50` route, which uses the
  upstream R2-Dreamer size12M model and optimizer with ARROW-50 replay;
- the exact GPU/container environment;
- protocol, provenance, and runtime-optimization records.

The reusable R2 projector and loss are project-owned components under
`src/clworldmodel/`. The native R2 route owns its integration trainer and
replay adapter while vendoring the checked R2-Dreamer model primitives. This is
not yet a general clean-room Dreamer implementation.

## Setup

The reference environment uses Python 3.10, PyTorch 2.3.0, and CUDA 11.8.
Install the vendored requirements in an isolated environment:

```bash
conda create -n arrow python=3.10 -y
conda activate arrow
python -m pip install -r third_party/arrow/requirements.txt
python -m pip install -e . --no-deps
python scripts/verify_arrow_environment.py
```

For the pinned headless GPU container, build the root `Dockerfile`. It uses
PyTorch 2.3.0 with CUDA 11.8, a configurable base-image registry, a mainland
China PyPI mirror, and build-time verification of all six bundled Atari ROMs.
See `docs/environment/docker.md`.

## Primary research method: ARROW-50

The first research-grade target is ARROW-50 on the paper's continual Atari
curriculum. `50-50` means that both replay capacity and replay-buffer sampling
are split equally between the short-term FIFO and LTDM buffers. DV3/FIFO is
retained only as a matched control. See `docs/protocols/arrow_ar50_atari.md`.

On a Linux CUDA machine with Atari ROMs installed, inspect the canonical
original-order seed-0 ARROW-50 run, then launch it into a new persistent run
directory:

```bash
python scripts/run_arrow_ar50_atari.py --seed 0 --dry-run
python scripts/run_arrow_ar50_atari.py \
  --seed 0 \
  --output-dir /persistent/path/arrow_ar50_original_s0_analysis
```

The launcher executes the maintained source under `third_party/arrow`
directly. Its base commit and local changes are recorded in `UPSTREAM.md`.

Inspect the resolved reference build, config, and command without starting
training:

```bash
python scripts/run_arrow_ar50_atari.py --seed 0 --dry-run
```

The launcher pins `--arrow-replay-ratio 50-50`, validates the official replay
and curriculum parameters, enables the documented optimized runtime, and saves
portable analysis snapshots at task boundaries and training end. Those
snapshots support offline checkpoint differencing but are not resumable
training checkpoints because replay, optimizers, RNG, and schedule state are
not included.

## Matched control: DreamerV3/FIFO

The DreamerV3 control uses one 1,024-trajectory FIFO buffer, matching
ARROW-50's total trajectory and raw observation-byte capacity. Its canonical
launcher also preserves portable world-model and actor-critic analysis
snapshots at every task boundary and at training end:

```bash
python scripts/run_dv3_fifo_atari.py --seed 0 --dry-run
python scripts/run_dv3_fifo_atari.py \
  --seed 0 \
  --output-dir /persistent/path/dv3_fifo_original_s0_analysis
```

The snapshots support offline checkpoint differencing but are not resumable
training checkpoints because replay, optimizers, RNG, and schedule state are
not included. See `docs/protocols/dv3_fifo_atari.md` for the frozen protocol
and artifact semantics. The component-level research questions, diagnostic-set
rules, interpretation matrix, and planned result tables are defined in
`docs/protocols/component_forgetting_audit.md`.

## Representation-objective ablation: ARROW-R2Rep-50

`ARROW-R2Rep-50` keeps ARROW-50 replay, budgets, curriculum, RSSM, reward and
continue heads, and actor-critic fixed. It removes the pixel decoder and trains
a bias-free latent projector with the R2-Dreamer Barlow Twins objective. Inspect
the seed-0 command without starting training:

```bash
python scripts/run_arrow_ar50_atari.py \
  --observation-objective r2 \
  --seed 0 \
  --dry-run
```

The completed seed-0 pilot is a negative acquisition result, not a reproduced
R2-Dreamer baseline and not evidence about retention. It is retained because
the failed port constrains interpretation. See
`docs/protocols/arrow_r2rep_atari.md` for the exact objective and protocol.

## Native R2-Dreamer with ARROW replay

`R2Dreamer-ARROW-50` replaces the partial port with the pinned R2-Dreamer
size12M architecture, `16 x 64` batches, native LaProp/AGC optimization, and
R2 latent-state replay context. ARROW remains responsible for FIFO/LTDM
trajectory retention and 50/50 sub-buffer selection. The default command is a
single-task, native-R2 Atari-100k-style acquisition check; it is deliberately
not compute-matched to ARROW:

```bash
python scripts/run_r2dreamer_arrow_atari.py --seed 0 --dry-run
python scripts/run_r2dreamer_arrow_atari.py \
  --seed 0 \
  --output-dir /persistent/path/r2dreamer_arrow50_single-task_original_s0
```

Run `--smoke` first on a target GPU. The adapter adds a byte-accounted CPU
posterior-state sidecar, so future comparisons must report both trajectory
capacity and actual storage bytes. See `docs/protocols/r2dreamer_arrow_atari.md`
for the frozen native-R2 configuration and scope labels.

Project-wide research and engineering constraints are defined in `AGENTS.md`.
