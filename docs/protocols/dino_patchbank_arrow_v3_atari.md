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
tokens. Replay stores only the observations; every sampled minibatch recomputes
its frozen DINO features on the accelerator:

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
fine-tuning in V3. Before RSSM consumption, on-the-fly encoder outputs round
through float16 and return to float32, preserving the numerical interface of
the superseded float16 sidecar implementation without retaining its values.

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
Replay stores task IDs and samples task-homogeneous sequences. There is no
frozen-feature replay sidecar: the sampled ARROW observations are the sole
persistent visual source, and the corresponding complete patch tensor is
computed only for the active minibatch.

The target VirtAI container exposes a 32-GiB memory cgroup even though the host
reports substantially more RAM. V3 therefore stores the unchanged float32
ARROW replay observations as shared, run-local file-backed mmap tensors under
`mmap_replay/observations/`. Tensor shape, dtype, FIFO overwrite order, LTDM
retention, and sampled indices do not change. Logical mapped observation
storage is `25,769,803,776` bytes; frozen-feature storage is zero bytes and
auxiliary replay tensors remain in CPU RAM. This keeps ARROW's image replay
semantics while avoiding the failed `103,079,215,104`-byte feature sidecar.

The superseded commit `d52bb17` cached every patch feature on the network-backed
`/gemini/code` mount. Its seed-0 Task-1 pilot reached epoch 3 but became dominated
by mmap major faults once the active FIFO/LTDM feature working set exceeded the
container memory limit. That interrupted run is a runtime failure, not a method
performance result.

All environment interactions, world-model updates, Actor-Critic updates,
evaluation points, action repeat, reward transformation, and per-task duration
remain inherited from the resolved published ARROW configuration. New-task
random pretraining collections are explicitly recorded as extra interactions.

## Execution and gates

The first run is the seed-0, one-task, 90-epoch MsPacman pilot:

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-patchbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages
```

The run must start from a clean pushed commit and preserve its generated
`launch.json`, resolved config, parameter accounting, zero-byte feature-source accounting,
replay mmap accounting, raw evaluation returns, final model, and Actor-Critic
bank. A successful smoke or single seed establishes execution or acquisition
evidence only. Task 2 must not be interpreted until Task 1 clearly learns under
a matched evaluation schedule.
