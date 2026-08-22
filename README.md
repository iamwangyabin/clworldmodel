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
- the completed `ARROW-R2Rep-50` partial-port pilot, preserved as a failed
  single-task acquisition check rather than a retention result;
- the implementation-ready native `R2Dreamer-ARROW-50` route, which uses the
  upstream R2-Dreamer size12M model and optimizer with ARROW-50 replay;
- the completed negative `KARROW-FrozenCore-v1` two-task pilot and the
  implementation-ready spatial-patch posterior correction in v2;
- the experimental `KARROW-ReplayConsolidated-v3` incremental KAN path and its
  fixed-checkpoint DINO/RSSM task-region audit;
- the implementation-ready `KARROW-InputAligned-v4` path, whose KAN or matched
  MLP branches consume each corrected module's original input;
- the completed negative task-aware `MoE-ARROW-v1` two-task pilot, preserved as
  a failed acquisition result;
- the implementation-ready `DINO-FullBank-ARROW-v2` correction with complete
  per-task world models and Actor-Critics;
- the stopped negative/inconclusive `DINO-PatchBank-ARROW-v3` Task-1 pilot and
  the implementation-ready `DINO-ConvBank-ARROW-v4` lightweight posterior
  interface;
- the exact GPU/container environment;
- protocol, provenance, and runtime-optimization records.

The reusable R2 projector and loss are project-owned components under
`src/clworldmodel/`, alongside the project-owned ReLU-KAN layers and actor. The
trainer remains the documented vendored ARROW runtime; this is not yet a clean
project-owned Dreamer implementation. The native R2 route owns its integration
trainer and replay adapter while vendoring the checked R2-Dreamer model
primitives.

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

## New method: KARROW-FrozenCore-v1

`KARROW-FrozenCore-50` freezes a local DINOv3 ViT-S/16 encoder, replaces pixel
reconstruction with one-step frozen-feature prediction, and trains the standard
ARROW core jointly with small zero-initialized residual adapters on task 1. At
the first task boundary, the shared RSSM, latent transition, reward/continue
heads, and actor-critic MLPs are frozen. Only one fixed set of residual adapters
continues learning. The KAN arm uses fixed-grid local corrections; its control
swaps each KAN core for an exactly parameter-matched MLP core.

The launcher supports `dino`, `mlp`, and `kan` arms and never downloads model
weights during training:

```bash
python -m pip install -e '.[dinov3]'
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_karrow_ar50_atari.py \
  --variant kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run
```

The completed seed-0 two-task pilot is a negative diagnostic: its CLS cosine
target admitted an almost constant solution and Task 1 acquisition was weak.
Its equations and claim limits remain frozen in
`docs/protocols/karrow_v1_atari.md`.

## Corrected visual path: KARROW-SpatialFrozenCore-v2

V2 excludes CLS and register tokens, pools the final DINOv3 patch grid to
`4 x 4`. Before the first world-model update, it fits a 384-to-64 PCA channel
projection on 512 uniformly sampled frames from the initial random Task-1
collection, then freezes that projection permanently. The RSSM posterior
reconstructs the resulting 1,024 frozen spatial features. Batch-standardized
SmoothL1 makes a constant feature prediction an explicit nonzero baseline. The
original Dreamer KL trains the prior; v2 does not reuse the collapsed prior-only
cosine target.

The shared KARROW launcher keeps v1 as its default. Select v2 explicitly for
the new acquisition screen:

```bash
python scripts/run_karrow_ar50_atari.py \
  --visual-version v2 \
  --variant dino \
  --task-prefix-length 1 \
  --seed 0 \
  --dry-run
```

The float16 spatial sidecar occupies `1,073,741,824` bytes, so a target-GPU
memory smoke is required before training. See
`docs/protocols/karrow_spatial_v2_atari.md`.

## Incremental method: KARROW-ReplayConsolidated-v3

V3 keeps the v2 visual path, but turns the residual KAN into an explicit
continual-learning mechanism. Before each new task, it estimates the functional
importance of every Gaussian RBF coefficient from unchanged ARROW replay and
short deterministic imagination. Task-1 coordinate maps are frozen; important
coefficients receive smaller future gradients and an anchor penalty, while cold
coefficients remain plastic. Inference still uses one shared fixed-capacity KAN
with no task ID or router.

```bash
python scripts/run_karrow_ar50_atari.py \
  --visual-version v3 \
  --variant kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run
```

The offline latent audit evaluates all games at one fixed checkpoint and
reports held-out task decodability, normalized region separation, PCA
artifacts, and per-module RBF support overlap. See
`docs/protocols/karrow_replay_consolidated_v3_atari.md` for the equations,
collection command, and claim limits.

## Input-aligned method: KARROW-InputAligned-v4

V4 fixes a plasticity problem in the earlier residual topology. The dynamics,
posterior, prior, actor, and critic corrections no longer consume only the
output of a frozen base trunk. Each correction is a parallel function of the
same state variables as its base module and directly predicts that module's
output residual. Reward, continuation, and feature prediction already followed
this pattern through the full `[z,h]` model state.

Task 1 includes both the original ARROW base and the residual branches in
optimization. Every residual output projection starts at zero and its scale is
`0.1`, so initialization is exactly the base model and the original path keeps
a direct learning signal. Residual construction also preserves the matched
base initialization and global training RNG state. At the first task boundary,
the base is frozen and the same fixed-capacity residuals continue learning. V4
adds no router, task ID, adapter expansion, or replay consolidation.

```bash
python scripts/run_karrow_ar50_atari.py \
  --visual-version v4 \
  --variant kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run
```

This is an untrained experimental protocol, not a performance claim. Its exact
topology and matched controls are defined in
`docs/protocols/karrow_input_aligned_v4_atari.md`.

## Task-aware method: MoE-ARROW-v1

`MoE-ARROW-50` replaces the fixed KAN capacity with one routed recurrent
dynamics, latent prior, reward/continue head, and Actor-Critic per scheduled
game. DINOv3 spatial features, the posterior representation, and feature head
remain shared. ARROW stores task labels and supplies task-homogeneous replay;
half of each fixed update budget targets the current game and half rehearses
replay-available old games. No extra gradient or environment steps are added.

The visual target uses a seeded fixed orthogonal 384-to-64 patch projection, so
it keeps the `4 x 4 x 64` spatial target without fitting anything on Task 1.
This protocol explicitly exposes scheduler task identity and spends parameters
per task. It is therefore a task-aware upper bound, not a direct replacement
for task-agnostic ARROW-50.

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --seed 0 \
  --task-prefix-length 2 \
  --dry-run
```

The completed seed-0 two-task pilot was negative: final deterministic raw
returns were 700.0 on MsPacman and 9.4375 on Boxing. Its prior cosine feature
loss collapsed while KL reached the free-bits floor, and Task 1 itself remained
weak. See
`docs/protocols/moe_arrow_v1_atari.md` for routing equations, fixed budgets,
storage accounting, the pilot record, and claim limits.

## Corrected task-aware method: DINO-FullBank-ARROW-v2

`DINO-FullBank-ARROW-50` keeps frozen DINOv3 but removes the remaining shared
trainable bottlenecks. Every task owns its posterior representation, recurrent
dynamics, latent prior, posterior feature head, reward/continue heads, and a
fresh independent MLP Actor-Critic. Task 1 activates only expert 0. At a later
boundary the complete previous world-model expert is copied once, all old
parameters are frozen, and every fixed update goes to the current task.

The observation target is the current stopped `4 x 4 x 64` DINO patch feature.
It is reconstructed from the current posterior with batch-standardized
SmoothL1, including first and reset observations. This directly grounds the
posterior and logs a constant-prediction baseline instead of relying on the
failed prior cosine objective. The first collection on each new task is random;
the new Actor-Critic does not inherit the preceding game's policy.

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --method dino-fullbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dry-run
```

This is a task-aware, storage-expanding reference and is currently untrained.
Its first gate is MsPacman acquisition, not continual retention. See
`docs/protocols/dino_fullbank_arrow_v2_atari.md` for the exact routing,
resource accounting, and execution order.

## Full-patch method: DINO-PatchBank-ARROW-v3

`DINO-PatchBank-ARROW-50` removes the fixed `4 x 4 x 64` visual bottleneck.
The frozen DINOv3 ViT-S/16 supplies every `16 x 16 x 384` patch coordinate to
the task-routed DreamerV3 posterior, and the original 64-pixel reconstruction
decoder is restored. There is no spatial pooling, channel projection, or DINO
feature-prediction head. RSSM dynamics, reward/continue prediction, latent
imagination, and MLP Actor-Critic training retain their existing algorithms.

Replay retains only ARROW's unchanged float32 observations. Each sampled batch
recomputes frozen DINO patches on the accelerator, rounds them through float16
to preserve the fixed feature interface, and feeds them to the RSSM. There is
no persistent feature sidecar. The observations use run-local file-backed mmap
tensors because the target container cannot hold ARROW's 24-GiB image replay
plus model working memory anonymously.

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --method dino-patchbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dry-run
```

See `docs/protocols/dino_patchbank_arrow_v3_atari.md` for the fixed protocol,
paper-derived motivation, stopped seed-0 pilot, and claim limits. That pilot
reached a raw MsPacman return of `511.25 +/- 123.89` at epoch 10 and was stopped
after epoch 14 rather than being presented as a completed seed result.

## Lightweight full-patch method: DINO-ConvBank-ARROW-v4

`DINO-ConvBank-ARROW-50` keeps V3's frozen complete `16 x 16 x 384` DINO grid
and original Dreamer pixel/KL/reward objectives, but replaces the direct
98,304-coordinate posterior input with one shared learned adapter:

```text
Conv2d(384,64,kernel=3,stride=2,padding=1)
  -> ChannelLayerNorm
  -> SiLU
  -> 8 x 8 x 64
  -> flatten 4096
```

The adapter has 221,376 parameters and is shared across task routes. Each task
still owns its posterior, recurrent dynamics, prior, decoder, reward/continue
heads, and Actor-Critic. Under the fixed Atari RSSM shape, the posterior drops
from 151,784,960 parameters per task in V3 to 7,081,472 in V4. Sharing the
trainable adapter is intentionally not strict task isolation; later tasks can
move the visual coordinates consumed by frozen old experts, so retention must
be measured rather than assumed.

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dry-run
```

This protocol is implemented but untrained. See
`docs/protocols/dino_convbank_arrow_v4_atari.md` for the exact gradient path,
resource accounting, routing semantics, and acquisition gates.

## Task-2 snapshot acquisition diagnostic

The Task-2 snapshot protocol starts from the completed Task-1 analysis
snapshot and runs Boxing alone for the matched 90-epoch budget. The primary
`kan_only` arm freezes the original shared core but leaves the complete KAN
residual modules plastic. The `kan_plus_heads` arm additionally opens only
small latent and behavior readouts. Replay, optimizer, RNG, and schedule state
are reset, so this is a trainability diagnostic rather than a resumable
continual run:

```bash
python scripts/run_karrow_task2_from_snapshot.py \
  --snapshot /path/to/boundary_01_task_00_epoch_0089.pt \
  --adaptation-mode kan_only \
  --dry-run
```

See `docs/protocols/karrow_task2_snapshot_acquisition_atari.md` for the
adaptation arms and reporting contract.

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

For the poor FastKAN actor/critic training dynamics, the corrected follow-up is
`ARROW-FastKANAC-StableTargets-50`. It keeps both width-53 FastKAN heads and the
same per-epoch budgets, but uses the existing EMA critic for imagination
targets, replay-value bootstraps, and the detached actor baseline. It also
bootstraps lambda returns from the actual post-transition horizon state instead
of reusing the preceding state. The 90-epoch screen matches the historical MLP
and bounded KAN acquisition duration:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac_stable \
  --task-prefix-length 1 \
  --task-duration-epochs 90 \
  --seed 0 \
  --dry-run
```

This is a predeclared trainability correction, not an established performance
improvement. Its invariant, comparison budget, and historical-behavior
isolation are documented in
`docs/protocols/arrow_fastkan_ac_stable_targets_atari.md`.

Project-wide research and engineering constraints are defined in `AGENTS.md`.
