# clworldmodel

`clworldmodel` is a research workspace for continual world-model reinforcement
learning. Its current base is the paper's canonical ARROW-50 method: a
DreamerV3-style agent with an equally split FIFO and long-term
distribution-matching (LTDM) replay budget.

## Current status

The repository is intentionally at the baseline stage. It currently contains:

- the maintained ARROW source based on a pinned upstream commit;
- the canonical ARROW-50 Atari launcher;
- the exact GPU/container environment;
- protocol, provenance, and runtime-optimization records.

There is no project-owned method under `src/` yet. That package will be created
only when implementation of the new method begins; placeholder toy models are
deliberately not kept in the repository.

## Setup

The reference environment uses Python 3.10, PyTorch 2.3.0, and CUDA 11.8.
Install the vendored requirements in an isolated environment:

```bash
conda create -n arrow python=3.10 -y
conda activate arrow
python -m pip install -r third_party/arrow/requirements.txt
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

On a Linux CUDA machine with Atari ROMs installed, launch the canonical
original-order, seed-0 ARROW-50 run from the repository root:

```bash
python scripts/run_arrow_ar50_atari.py --seed 0
```

The launcher executes the maintained source under `third_party/arrow`
directly. Its base commit and local changes are recorded in `UPSTREAM.md`.

Inspect the resolved reference build, config, and command without starting
training:

```bash
python scripts/run_arrow_ar50_atari.py --seed 0 --dry-run
```

The launcher pins `--arrow-replay-ratio 50-50`, validates the official replay
and curriculum parameters, and enables the documented optimized runtime.

Project-wide research and engineering constraints are defined in `AGENTS.md`.
