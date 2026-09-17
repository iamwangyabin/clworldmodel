# clworldmodel

`clworldmodel` is a research workspace for continual world-model reinforcement
learning. Its current base is the paper's canonical ARROW-50 method: a
DreamerV3-style agent with an equally split FIFO and long-term
distribution-matching (LTDM) replay budget.

## Current method scope

The sole formal project-owned paper method is **AWM-AutoRoute — Accumulative
World Modeling with Automatic Routing**, formerly D-AutoRoute. ARROW-50,
DreamerV3/FIFO, native R2-Dreamer and Dream Rehearsal remain baselines or
reference integrations, not project methods. The AWM/D oracle path remains an
internal training, parity and diagnostic reference; it is not a second paper
method. See [Decision 0065](docs/decisions/0065-awm-autoroute-only-paper-method.md).

The authoritative paper experiment backlog and progress counter is
[the AWM-AutoRoute paper plan](docs/experiments/AWM_AUTOROUTE_PAPER_PLAN.md).

AWM-AutoRoute accumulates learned prediction and control functionality in a
continually trainable world model, then selects the retained route without a
task ID at inference. Existing launcher paths, config keys, protocol IDs and
historical records stay unchanged; Decisions 0060 and 0065 record the naming
and final paper scope separately.

The early representation/KAN, KARROW, MoE/full-bank, frozen-first-task
adaptation and other Evolving-Core implementations have been retired by user
decision. Their protocol and experiment records below are historical evidence,
not a claim that their commands still work on the current checkout.

**Cleanup is not yet fully validated:** retained launch dry runs and fixed-input
model parity pass, but migration of old method-specific tests is pending explicit
approval. Do not launch a campaign from this intermediate worktree.
See [Decision 0057](docs/decisions/0057-retire-historical-method-code.md) and
[Decision 0058](docs/decisions/0058-retire-stabletargets-and-d-autokan.md).

**StableTargets and F / D-AutoKAN were retired on 2026-09-07.** Their
execution paths and FastKAN implementation are removed. Historical protocols,
raw results and provenance remain; they require the recorded Git revision.

## Implementation and experiment history

The repository contains the maintained baseline stack plus the retained
reference integrations:

- the maintained ARROW source based on a pinned upstream commit;
- the canonical ARROW-50 Atari launcher;
- the matched DreamerV3/FIFO Atari control launcher;
- the implementation-ready bounded Dream Rehearsal baseline, which keeps the
  paper's actor-only graded self-imitation but caps its replay at the same
  524,288 transitions as ARROW-50;
- the implementation-ready native `R2Dreamer-ARROW-50` route, which uses the
  upstream R2-Dreamer size12M model and optimizer with ARROW-50 replay;
- the AWM-AutoRoute v4 trainer and its capacity-organization controls;
- protocol, provenance, and runtime-optimization records.

Retired method implementations (ReLU-KAN actors, ARROW-R2Rep, KARROW v1–v4,
MoE-ARROW, the DINO/CNN task banks, the Evolving-Core variants, and the
D-AutoRoute v1/v2 routers) were removed from the working tree. Their design
docs, protocols, and curated records remain recoverable from Git history; see
Decisions 0057–0065 for the retirement and naming chain.

The reusable R2 projector and loss are project-owned components under
`src/clworldmodel/`. The trainer remains the documented vendored ARROW
runtime; this is not yet a clean project-owned Dreamer implementation. The
native R2 route owns its integration trainer and replay adapter while
vendoring the checked R2-Dreamer model primitives.

## Experiment records

The single entry point for preserved experiment evidence is
[`docs/experiments/README.md`](docs/experiments/README.md). Its generated
[`RESULTS.md`](docs/experiments/RESULTS.md) index currently covers the Atari
ARROW-50/DV3 seed-0 runs, the CoinRun five-seed baseline archives, and the
eighteen single-task normalization anchors.

The repository keeps structured provenance, raw per-task boundary results,
source hashes, and small evaluation-log excerpts. It deliberately excludes
weights, checkpoints, Replay storage, TensorBoard events, full logs, and other
generated run data. Rebuild and validate the index with:

```bash
python scripts/experiment_registry.py write
python scripts/experiment_registry.py check
```

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

## Bounded Dream Rehearsal baseline

`Bounded-Dream-Rehearsal-v1-Atari` ports the Dream Rehearsal actor-only graded
self-imitation update into the maintained DreamerV3 trainer. Unlike the
reference artifact's never-clear phase libraries, it uses one fixed random-key
reservoir with 1,024 trajectories x 512 transitions. uint8 mmap storage makes
the run practical but does not increase that matched sample capacity. Task IDs
exist only as replay-filter metadata and are not inputs to the shared world
model or actor.

```bash
python scripts/run_bounded_dream_rehearsal_atari.py --seed 0 --dry-run
python scripts/run_bounded_dream_rehearsal_atari.py \
  --seed 0 \
  --output-dir /persistent/path/bounded_dream_rehearsal_original_s0
```

This method is storage matched, not compute matched: the reference cadence adds
actor-only optimization for every prior task, and its manifest reports those
updates separately. No target-CUDA run has yet validated the implementation.
See `docs/protocols/bounded_dream_rehearsal_atari.md` for formulas, provenance,
declared deviations, and the required comparison matrix.

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

## Retired methods

ReLU-KAN actors, ARROW-R2Rep, KARROW v1-v4, MoE-ARROW, the DINO/CNN task
banks, the Evolving-Core variants, the Task-2 snapshot diagnostic, the
KANActor ablation, and the D-AutoRoute v1/v2 routers are retired. Their
README sections, protocols, decision records, and curated experiment
records were removed from the working tree on 2026-09-14 and remain
recoverable from Git history. See Decisions 0057-0065.

## Dream Rehearsal: official-code Atari reference

The requested **full-history vs bounded-history pair** now uses one entry:

```bash
python scripts/run_dream_rehearsal_memory_pair_atari.py --history full --dry-run
python scripts/run_dream_rehearsal_memory_pair_atari.py --history bounded --dry-run
```

Both arms have identical learning, ordinary update budgets, rehearsal and
evaluation. The only config difference is history capacity: all transitions
versus a uniform reservoir of **1,024 × 512 = 524,288** transitions. Ordinary
training and rehearsal both read the same retained pool; old-phase views and
disk archives cannot preserve a hidden full-history backup in the bounded arm.
See the [memory-only pair protocol](docs/protocols/dream_rehearsal_memory_pair_v1_atari.md).
CPU parity/storage fixtures pass; target-GPU smoke and both new runs are still
pending. Capacity matching to ARROW is not a claim of full compute/byte matching.

`Dream-Rehearsal-OfficialCode-v1-Atari` directly executes pinned, unmodified
Dream Rehearsal `tunnel_update` and NM512 DreamerV3 learning components. Ordinary
world-model and actor/critic training use the **shared full history**, and the
normal batch is **16 × 64**, not the author's smoke preset. This separate path
does not change the old ARROW ports or relabel their results.

```bash
python scripts/run_dream_rehearsal_official_atari.py --dry-run
```

The [protocol](docs/protocols/dream_rehearsal_official_code_v1_atari.md) describes
its isolated torch 2.4.1 environment, source-parity tests, mandatory clean/pushed
launch provenance, target-GPU smoke and explicit Atari adaptations. CPU
fixtures pass; no corrected Atari or paper reproduction result is claimed.
The [fidelity audit](docs/experiments/dream_rehearsal_fidelity_audit_20260905.md)
records why the earlier never-clear scores are not official-method results.

### Capacity-organization pilot controls

The five user-authorized task-aware controls have a separate prospective
[protocol](docs/protocols/capacity_organization_v1_atari.md) and
`python scripts/run_capacity_control_atari.py --control shared --seed 0
--output-dir /persistent/run --dry-run` launcher. They are not reproduced
results or compute-matched single-switch AWM ablations.
