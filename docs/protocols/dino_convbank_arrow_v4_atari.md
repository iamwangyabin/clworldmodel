# DINO-ConvBank-ARROW-v4 Atari protocol

## Status and purpose

This document freezes the opt-in `dino_convbank_arrow` protocol. Its reported
ARROW-50 name is `DINO-ConvBank-ARROW-50`. It is a task-aware experimental
method, not a published ARROW reproduction and not yet a performance result.

The preceding DINO-PatchBank-v3 pilot retained every frozen DINO coordinate but
sent a 98,304-dimensional vector directly into each posterior. That made the
posterior projection extremely large and did not show convincing Task-1
acquisition by the first formal evaluation. V4 tests one controlled change:
insert a single learned spatial convolution between frozen DINO patches and the
otherwise unchanged Dreamer posterior.

## Visual and latent path

Replay stores only the unchanged float32 Atari observations. A sampled batch is
encoded on the accelerator by the frozen local DINOv3 ViT-S/16. Under the
canonical `fp32-tf32` profile, encoder outputs round through float16 and return
to float32, as in V3. The separately named `bf16-amp` execution profile keeps
DINO output and adapter input in bfloat16 and therefore removes that redundant
round trip. The new adapter then runs inside the RSSM:

```text
o_t [3,64,64]
  -> frozen DINOv3 resize/normalization
  -> U_t [16,16,384]
  -> detach and explicit precision profile
  -> permute [384,16,16]
  -> Conv2d(384,64,kernel=3,stride=2,padding=1)
  -> per-location ChannelLayerNorm(64, eps=1e-3)
  -> SiLU
  -> A_t [64,8,8]
  -> flatten
  -> e_t [4,096]
  -> task-k posterior q_k(z_t | h_t, e_t)
```

`ChannelLayerNorm` normalizes only the 64 channels at each spatial coordinate;
it does not mix the `8 x 8` positions. There is exactly one standard
convolution, one stride-2 spatial reduction, and no autoencoder, pooling layer,
depthwise factorization, or second compression stage.

The adapter has 221,376 trainable parameters: 221,184 convolution weights, 64
convolution biases, and 128 LayerNorm scale/bias parameters. Its convolution is
approximately 14.2 million multiply-accumulates per frame. The embedding width
falls by 24 times, from 98,304 to 4,096. The latter equals the original vendored
Dreamer CNN interface, `4 x 4 x 256`, although V4 obtains it from the more
spatially resolved `8 x 8 x 64` tensor.

With the fixed Atari RSSM shape `(32,32)`, 512 recurrent units, and the existing
two-layer width-512 posterior including its encoder skip projection, the
posterior has 7,081,472 parameters per task. The unadapted V3 posterior has
151,784,960 parameters per task. These counts cover the posterior
representation only, not the rest of a task expert or optimizer state.

The recurrent equation remains unchanged:

```text
h_t = GRU_k(z_(t-1), a_(t-1), h_(t-1))
```

The selected posterior state `[z_t,h_t]` trains the existing task-specific
pixel decoder against the original `64 x 64` observation. Pixel reconstruction,
reward, continuation, free-bit KL, latent imagination, and Actor-Critic losses
retain their V3 formulas and weights. V4 adds no DINO feature target and never
fine-tunes DINO.

## Continual routing

The scheduler task ID remains privileged input and hard-routes the same complete
per-task bank as V3:

- posterior representation;
- recurrent dynamics and latent prior;
- pixel decoder;
- reward and continuation heads;
- independent DreamerV3 MLP Actor-Critic and optimizer.

There is one adapter for the whole run, not one adapter per task. It is trained
on Task 0 and remains plastic on every later task. New task experts copy the
preceding complete task expert once, but do not copy or reset the shared
adapter. Old task experts and policies are frozen; evaluation of an old task
uses its frozen route together with the current shared adapter.

This is a deliberate stability risk. A changing adapter can move the input
coordinates seen by every frozen old posterior, so V4 does not provide strict
functional isolation even though its task experts are frozen. The first gate is
Task-1 acquisition. A later two-task run must report both current-task learning
and old-task retention before the shared-adapter hypothesis is accepted.

## Replay and resources

FIFO/LTDM capacities, 50/50 buffer selection, task-homogeneous sampling, update
counts, action repeat, reward processing, and evaluation timing remain the
same as V3. Replay observations remain run-local file-backed mmap tensors under
`mmap_replay/observations/` to respect the target container's 32-GiB memory
cgroup. There is no DINO feature mmap or sidecar; frozen features and adapted
features exist only for the sampled minibatch.

The launch manifest records both visual widths, exact adapter parameters,
posterior parameters per task, replay bytes, and the fact that the adapter is
shared and plastic across task boundaries.

## Execution precision profiles

The default `fp32-tf32` profile is unchanged and remains the V4 reference. The
opt-in `bf16-amp` profile is named
`DINO-ConvBank-ARROW-v4-BF16AMP-Atari-TaskAware`; it changes numerical
execution, so its results must not be silently merged with the FP32 profile.
It makes only the following runtime changes:

- CUDA matrix and convolution kernels in DINO, RSSM, decoder, actor, and critic
  run under bfloat16 autocast.
- Model parameters, gradients owned by FP32 parameters, and Adam states remain
  float32. BF16 does not use a gradient scaler.
- Categorical sampling and KL, symlog/symexp, pixel/reward/continue losses,
  lambda returns, actor log probabilities, and value targets explicitly compute
  in float32.
- On-the-fly DINO features stay bfloat16 through the shared convolution adapter
  instead of allocating a float16-to-float32 conversion.
- The DINO execution chunk grows from 128 to 512 frames. The optimization batch
  remains exactly `T=32, N=16`, or 512 frames per world-model update; this is
  execution batching, not more samples or updates.

Environment interactions, replay capacity and sampling, update counts, loss
weights, task routing, evaluation, and checkpoints are unchanged. The profile
requires a CUDA accelerator with native BF16 support and fails before model
allocation otherwise.

## Execution and gates

The first run is the seed-0 one-task 90-epoch MsPacman pilot:

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages
```

The BF16 execution pilot adds the explicit profile flag:

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --precision-profile bf16-amp \
  --profile-stages
```

Training must start from a clean commit already synchronized with its configured
GitHub branch. Preserve `launch.json`, the resolved config, model parameter
accounting, replay mmap accounting, raw evaluation returns, stage timings, the
final world model, and the Actor-Critic bank. A smoke run proves execution only;
one seed can supply acquisition evidence but not a reproduction or general
continual-learning claim.
