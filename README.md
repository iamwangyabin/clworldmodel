# clworldmodel

`clworldmodel` is a research workspace for continual world-model reinforcement
learning. Its current base is the paper's canonical ARROW-50 method: a
DreamerV3-style agent with an equally split FIFO and long-term
distribution-matching (LTDM) replay budget.

## Current status

The repository contains the maintained baseline stack plus named method
ablations at different stages of evidence:

- the maintained ARROW source based on a pinned upstream commit;
- the canonical ARROW-50 Atari launcher;
- the matched DreamerV3/FIFO Atari control launcher;
- an opt-in decoder-free R2 representation-objective ablation with ARROW-50
  replay;
- fixed-grid ReLU-KAN actor pilots, including a completed bounded-interface T1
  trainability result;
- an opt-in trainable-anchor ReLU-KAN actor for the next fresh T1 screen;
- the exact GPU/container environment;
- protocol, provenance, and runtime-optimization records.

The reusable R2 projector and loss are project-owned components under
`src/clworldmodel/`, alongside the project-owned ReLU-KAN layers and actor. The
trainer remains the documented vendored ARROW runtime; this is not yet a clean
project-owned Dreamer implementation.

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

This is an unrun representation-objective ablation, not a reproduced R2-Dreamer
baseline and not evidence of improved retention. See
`docs/protocols/arrow_r2rep_atari.md` for the exact objective, frozen comparison,
accounting, and experiment ladder.

## Actor ablation: ARROW-KANActor-50

The direct `ARROW-KANActor-50` pilot exposed an out-of-support second KAN layer
and is retained only for reproducibility. `ARROW-KANActorBounded-50` keeps the
ARROW-50 FIFO/LTDM replay strategy, world model, critic, budgets, and
curriculum unchanged, but inserts a LayerNorm--sigmoid adapter between the two
fixed-grid KAN layers. Its completed 90-epoch MsPacman trainability pilot
reached raw return `1597.5`. Its fresh 180-epoch extension was deliberately
stopped before final evaluation.

The current trainability screen is `ARROW-KANActorAdaptive-50`: it retains the
bounded interface but makes every per-input ReLU-KAN support start and width
trainable. It has 821,458 actor parameters, so it is not parameter matched to
the MLP. The new one-task 180-epoch protocol is:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan_adaptive \
  --task-prefix-length 1 \
  --task-duration-epochs 180 \
  --seed 0 \
  --dry-run
```

This is a 2x training-budget actor trainability extension, not evidence that
KAN prevents forgetting. See
`docs/protocols/arrow_kan_actor_adaptive_atari.md` for the trainable-anchor
definition and `docs/protocols/arrow_kan_actor_bounded_atari.md` for the
fixed-grid history and interface diagnosis.

The separate `ARROW-FastKANAC-KDAligned-50` pilot replaces both actor and
critic with the fixed-Gaussian FastKAN behavior architecture reported by
KAN-Dreamer. It uses width 34, eight centers over `[-2, 2]`, and the directly
portable behavior-training settings (LaProp, AGC, `4e-5`, horizon 15, actor
unimix, persistent return normalization, and critic EMA) while preserving the
ARROW-50 replay strategy and world-model training. Its 68 epochs map to
1,114,112 agent decisions, the first whole-epoch boundary at or above the
paper's 1.1M-step endpoint:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac \
  --task-prefix-length 1 \
  --task-duration-epochs 68 \
  --seed 0 \
  --dry-run
```

This is an independently reconstructed Atari pilot, not a reproduction of
KAN-Dreamer's DMC experiment. Exact alignments and unavoidable deviations are
listed in `docs/protocols/arrow_fastkan_ac_kd_aligned_atari.md`.

The follow-up `ARROW-FastKANAC-ParamMatchedRepVal-50` protocol keeps both
FastKAN heads, increases their width to 53 (1,700,670 combined parameters,
0.83% below the MLP pair), adds a `0.3` replay critic objective over the
already-sampled four-frame context, and doubles the one-task budget to 136
epochs. It writes an explicit 68-epoch midpoint evaluation and snapshot:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac_param_matched \
  --task-prefix-length 1 \
  --task-duration-epochs 136 \
  --seed 0 \
  --dry-run
```

Actor and critic diagnostics always go to TensorBoard. SwanLab mirroring is an
optional launcher flag and never accepts an API key. The controlled changes and
claim limits are documented in
`docs/protocols/arrow_fastkan_ac_param_matched_repval_atari.md`.

Project-wide research and engineering constraints are defined in `AGENTS.md`.
