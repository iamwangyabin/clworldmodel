# DINO-ConvBank-ARROW-v4 Atari protocol

## Status and purpose

This document freezes the opt-in `dino_convbank_arrow` protocol. Its current
reported ARROW-50 name is
`DINO-ConvBank-ARROW-50-BF16AMP-Uint8Replay`. It is a task-aware experimental
method, not a published ARROW reproduction and not yet a performance result.
The launcher requires BF16 execution and uint8 observation replay for this
method; the superseded FP32/TF32 V4 profile is not a valid current run group.

The preceding DINO-PatchBank-v3 pilot retained every frozen DINO coordinate but
sent a 98,304-dimensional vector directly into each posterior. That made the
posterior projection extremely large and did not show convincing Task-1
acquisition by the first formal evaluation. V4 tests one controlled change:
insert a single learned spatial convolution between frozen DINO patches and the
otherwise unchanged Dreamer posterior.

## Visual and latent path

Replay stores the unchanged discrete `64 x 64` RGB Atari pixels as uint8. After
the selected pixels move to the training device, replay converts them to
float32 and divides by 255, reproducing the model-facing `[0,1]` values used by
the prior float32 replay. The sampled batch is then encoded by the frozen local
DINOv3 ViT-S/16 under required BF16 autocast. DINO output and adapter input stay
bfloat16, so there is no intermediate float16-to-float32 feature round trip.
The adapter then runs inside the RSSM:

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

ARROW-50 stores 524,288 observation frames. Uint8 observation mmap storage is
therefore `6,442,450,944` bytes instead of the prior float32
`25,769,803,776` bytes. The unchanged float32 action, reward, continuation, and
reset tensors add `44,040,192` bytes. Quantization occurs only at the replay
storage boundary: Atari observations originate as uint8 pixels, and focused
tests require exact sampled-value, FIFO wraparound, and LTDM-retention parity
after float32 division by 255.

The launch manifest records both visual widths, exact adapter parameters,
posterior parameters per task, replay bytes, and the fact that the adapter is
shared and plastic across task boundaries.

## Required execution precision

The only current execution profile is named
`DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-Atari-TaskAware`. Omitting
`--precision-profile` selects `bf16-amp` for `dino-convbank`; explicitly asking
for `fp32-tf32` is an error. Historical FP32/TF32 pilot artifacts remain a
separate superseded result group and must not be merged with this protocol.

- CUDA matrix and convolution kernels in DINO, RSSM, decoder, actor, and critic
  run under bfloat16 autocast.
- Model parameters, gradients owned by FP32 parameters, and Adam states remain
  float32. BF16 does not use a gradient scaler.
- Categorical sampling and KL, symlog/symexp, pixel/reward/continue losses,
  lambda returns, actor log probabilities, and value targets explicitly compute
  in float32.
- On-the-fly DINO features stay bfloat16 through the shared convolution adapter
  without allocating a float16-to-float32 conversion.
- The DINO execution chunk is 512 frames. The optimization batch
  remains exactly `T=32, N=16`, or 512 frames per world-model update; this is
  execution batching, not more samples or updates.

Environment interactions, replay capacity and sampling, update counts, loss
weights, task routing, evaluation, and checkpoints are unchanged. The profile
requires a CUDA accelerator with native BF16 support and fails before model
allocation otherwise. Model parameters and Adam state deliberately remain
float32 master state; "BF16 training" here refers to model kernel execution,
not unsafe BF16 optimizer accumulation.

## Fixed-global-batch data parallel execution

The launcher accepts `--devices 1`, `--devices 2`, or `--devices 4`. Values 2
and 4 use one process per CUDA device through `torch.distributed.run`, NCCL, and
native PyTorch DistributedDataParallel. They create separately named
`DP2`/`DP4` execution groups. Multi-GPU execution is currently rejected for all
other methods.

This profile partitions the existing independent sequence axis; it does not
increase the research batch:

| Stage | Global shape | 2-GPU local shape | 4-GPU local shape |
| --- | --- | --- | --- |
| Regular world model | `T=32, N=16` | `T=32, N=8` | `T=32, N=4` |
| Pretrain world model | `T=32, N=16` | `T=32, N=8` | `T=32, N=4` |
| Actor replay context | `T=4, N=128` | `T=4, N=64` | `T=4, N=32` |

Rank 0 is the sole owner of the file-backed FIFO/LTDM buffers. On every model
or Actor-Critic update it makes exactly one global ARROW sub-buffer choice and
one global sequence draw, preserving whole-minibatch FIFO/LTDM semantics. The
sampled tensor is split contiguously along `N` and scattered to the ranks. Each
rank then recomputes DINO features only for its local observations. DDP averages
equal-sized local mean-loss gradients, and Actor-Critic return quantiles are
computed after gathering the local returns into the original global batch.

Environment collection remains on rank 0 because the fixed `n_sync=2` collector
is small and changing it would alter the interaction protocol. Evaluation is
parallelized by assigning task index `k` to rank `k mod world_size`; raw and
scaled task statistics are reduced before rank-0 logging. Rank 0 alone writes
TensorBoard, SwanLab, mmap accounting, evaluations, and checkpoints.

Each rank holds a full frozen DINO, world model, active Actor-Critic, optimizer,
and optimizer state. This is data parallelism, not model sharding, so every GPU
must independently fit the complete local state. Replay scatter and rank-0
collection limit scaling; neither 2x nor 4x speedup is a protocol claim.

## Execution and gates

The first run is the seed-0 one-task 90-epoch MsPacman pilot:

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --precision-profile bf16-amp \
  --profile-stages
```

The initial seed-0 DP4 continual run failed the Task-1 acquisition gate. Its
raw MsPacman mean was `1139.375` after 90 completed training epochs, below the
fixed `2000` threshold. The largest periodic-checkpoint mean was `1717.5` at
epoch 40, but a transient peak is not the acquisition result. The run was
aborted after 122 completed epochs rather than spending the remaining
continual-training budget on an unqualified representation and policy.

### Task-1 acquisition tuning pilot

Two named seed-0 ablations test the hypothesis that the default Actor-Critic
optimization is too aggressive for the frozen-DINO representation. They run on
disjoint two-GPU groups so both candidates can be evaluated without changing
the fixed global batch, 90-epoch task duration, environment interactions,
world-model updates, Actor-Critic updates, replay capacity, or evaluation
schedule.

| Profile | Actor-Critic LR | Entropy scale | Controlled purpose |
| --- | ---: | ---: | --- |
| `aclr5e5` | `5e-5` | `3e-4` | Test whether halving LR prevents the observed post-peak policy regression |
| `aclr5e5-ent1e4` | `5e-5` | `1e-4` | Conditional test of lower entropy regularization after halving LR |

The frozen DINO encoder, shared convolution adapter, world-model learning rate
`1e-4`, seed, collection stream, and all other resolved settings remain fixed.
The acquisition gate is the 16-rollout `raw_return_mean` in
`final_evaluation.json` after all 90 completed epochs. It passes only at
`>=2000`; intermediate peaks do not pass. Seed 0 is used for candidate
selection only. A selected profile requires fresh confirmation seeds before it
can support a method claim.

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --devices 2 \
  --seed 0 \
  --task-prefix-length 1 \
  --task1-tuning-profile aclr5e5 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages

python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --devices 2 \
  --seed 0 \
  --task-prefix-length 1 \
  --task1-tuning-profile aclr5e5-ent1e4 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages
```

The corresponding 2- and 4-GPU dry runs are:

```bash
python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --devices 2 \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages \
  --dry-run

python scripts/run_moe_arrow_atari.py \
  --method dino-convbank \
  --devices 4 \
  --seed 0 \
  --task-prefix-length 1 \
  --dinov3-model-path /absolute/local/model/path \
  --profile-stages \
  --dry-run
```

Training must start from a clean commit already synchronized with its configured
GitHub branch. Preserve `launch.json`, the resolved config, model parameter
accounting, replay mmap accounting, raw evaluation returns, stage timings, the
final world model, and the Actor-Critic bank. A smoke run proves execution only;
one seed can supply acquisition evidence but not a reproduction or general
continual-learning claim.
