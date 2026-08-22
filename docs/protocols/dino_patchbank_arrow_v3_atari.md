# DINO-PatchBank-ARROW-v3 Atari protocol

## Status and purpose

This document freezes the opt-in `dino_patchbank_arrow` protocol. Its reported
ARROW-50 name is `DINO-PatchBank-ARROW-50`. It is a task-aware, high-memory
research method, not a published ARROW reproduction and not a storage-matched
replacement for ARROW-50.

The acquisition hypothesis is that DINO-FullBank-v2 discarded control-relevant
information before the RSSM by reducing `16 x 16 x 384` frozen patch tokens to
`4 x 4 x 64` with fixed average pooling and an orthogonal channel projection.
V3 tests the smallest paper-derived correction: replace only the Dreamer visual
encoder with complete frozen DINO patches and restore Dreamer's pixel
reconstruction objective.

The design follows the encoder substitution evaluated by Jones, Zhou, and
Pattinson, *Pre-trained Visual Representations Generalize Where it Matters in
Model-Based Reinforcement Learning*, arXiv:2509.12531. Their positive evidence
concerns severe visual distribution shifts in two control domains, not Atari
sample efficiency. This protocol therefore makes no performance claim before
the local Atari gates pass.

## Visual and latent path

Stored Atari observations remain float32 `3 x 64 x 64`. The frozen local
DINOv3 ViT-S/16 internally resizes them to `256 x 256` and returns all 256 patch
tokens:

```text
o_t [3,64,64]
  -> frozen DINOv3 resize/normalization
  -> U_t [16,16,384]
  -> flatten without pooling or projection
  -> e_t [98,304]
  -> task-k DreamerV3 posterior q_k(z_t | h_t, e_t)
```

The recurrent update remains action-conditioned and does not consume the image
embedding directly:

```text
h_t = GRU_k(z_(t-1), a_(t-1), h_(t-1))
```

The selected posterior state `[z_t,h_t]` trains the original DreamerV3 pixel
decoder against the unmodified `64 x 64` replay observation. Reward,
continuation, free-bit KL, latent imagination, and Actor-Critic losses retain
their existing formulas and weights. There is no feature predictor and no DINO
fine-tuning in V3.

## Continual routing

The scalar scheduler task ID is privileged information. It hard-routes one
complete task bank containing:

- posterior representation;
- recurrent dynamics;
- latent prior;
- pixel decoder;
- reward and continuation heads;
- independent DreamerV3 MLP Actor-Critic and optimizer.

The frozen DINOv3 encoder is the only shared model module. Task 0 starts from
the seeded initialization. At a later boundary, the previous complete world
model is copied once into the new task route, while the new Actor-Critic uses
fresh deterministic weights. Old routes are frozen. Every fixed world-model
and Actor-Critic update is assigned to the current task. The first collection
for every new task is random.

## Replay and resources

FIFO/LTDM trajectory capacities and 50/50 selection semantics remain ARROW-50.
Replay stores task IDs and samples task-homogeneous sequences. The frozen
feature sidecar follows the exact ARROW write and sample indices.

For each of two `512 x 512` sub-buffers, float16 feature storage is:

```text
512 * 512 * 98,304 * 2 bytes = 51,539,607,552 bytes
```

Total feature storage is `103,079,215,104` bytes, excluding base observations,
task IDs, allocator overhead, gradients, activations, parameters, and optimizer
state. This byte difference must accompany every comparison.

The target VirtAI container exposes a 32-GiB memory cgroup even though the host
reports substantially more RAM. Therefore V3 stores both the unchanged
float32 replay observations and float16 feature sidecar as shared, run-local
file-backed mmap tensors under `mmap_replay/`. The tensor shapes, dtypes, FIFO
overwrite order, LTDM retention, write maps, and sampled indices do not change.
Only the storage backing changes from anonymous CPU memory to persistent
`/gemini/code` files. Logical mapped storage is 25,769,803,776 observation bytes
plus 103,079,215,104 feature bytes; auxiliary replay tensors remain in CPU RAM.
The operating system may cache active pages, but the complete logical stores
must still be byte-accounted and must not be described as memory matched.

All environment interactions, world-model updates, Actor-Critic updates,
evaluation points, action repeat, reward transformation, and per-task duration
remain inherited from the resolved published ARROW configuration. New-task
random pretraining collections are explicitly recorded as extra interactions.

## Execution and gates

The first run is the seed-0, one-task, 90-epoch MsPacman pilot:

```bash
python scripts/run_dino_patchbank_arrow_atari.py \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages
```

The run must start from a clean pushed commit and preserve its generated
`launch.json`, resolved config, parameter accounting, feature-cache accounting,
replay mmap accounting, raw evaluation returns, final model, and Actor-Critic
bank. A successful smoke or single seed establishes execution or acquisition
evidence only. Task 2 must not be interpreted until Task 1 clearly learns under
a matched evaluation schedule.
