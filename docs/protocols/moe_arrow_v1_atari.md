# MoE-ARROW-v1 Atari Task-Aware Protocol

## Status and claim boundary

`MoE-ARROW-v1-Atari-TaskAware` is implemented but untrained. It is a
storage-unconstrained, task-aware continual DreamerV3 method built on ARROW-50
replay. A smoke run establishes only execution correctness. No performance or
reproduction claim exists until matched multi-seed experiments finish.

The code identifier is `moe_arrow`; the reported method name is
`MoE-ARROW-50`. Because the scheduler exposes task identity to the model and
policy, its results belong in a separate task-aware table. ARROW-50 and KARROW
remain task-agnostic unless their own protocol says otherwise.

## Fixed task interface

For scheduled task index `k`, the scheduler supplies scalar `task_id=k` during
collection, replay sampling, imagination, and evaluation. One minibatch may
contain only one task. The ID selects modules and is never concatenated to the
DINO feature, stochastic latent, deterministic state, action, or reward.

V1 allocates exactly one expert and one Actor-Critic per configured Atari game.
It does not infer boundaries, learn a router, merge experts, or add experts at
runtime. Those changes require a new protocol name.

## Frozen observation target

Pixels are resized to `256 x 256`, ImageNet-normalized, and encoded by a local
frozen DINOv3 ViT-S/16. CLS and register tokens are excluded. Final patch tokens
are adaptively pooled to a `4 x 4` grid. A local generator with seed 0 creates a
`384 x 64` Gaussian matrix; reduced QR and canonical column signs produce a
fixed orthogonal channel projection `P`.

For pooled patch vector `p`, the cached target is

```text
y = flatten(p P),        y in R^(4*4*64) = R^1024.
```

No projection is fitted on Task 1. Targets are cached as float16 alongside the
existing FIFO and LTDM write decisions and converted to float32 for loss
computation. There is no pixel decoder or VAE reconstruction loss.

The RSSM prior predicts the stopped next-observation feature with cosine loss.
First sequence positions and reset transitions are masked exactly as in the
existing DINO prior-feature path.

## Routed world model

The stochastic latent has the existing 32-by-32 categorical shape and the
deterministic hidden state retains 512 units. DINOv3, posterior representation,
and feature-prediction head are shared. Task `k` selects four complete modules:

```text
h_t       = GRU_k(MLP_k([z_(t-1), a_(t-1)]), h_(t-1))
p_k(z_t)  = Transition_k(h_t)
r_t       = RewardHead_k([z_t, h_t])
c_t       = ContinueHead_k([z_t, h_t]).
```

Only the selected recurrent, prior, reward, and continuation experts receive
gradients on an update. Shared posterior and feature modules receive gradients
from the selected task batch. KAN and MLP residual corrections are disabled.

Expert 0 uses the normal initialization. On the first arrival of task `k>0`,
its four modules copy task `k-1` once. The target expert has no optimizer state
at that point. Returning to a previously seen task selects its preserved expert
without copying again.

## Actor-Critic bank

Each task owns an independent original DreamerV3 MLP actor, critic, optional
slow critic, return-normalization state, and optimizer. A new Actor-Critic copies
the preceding task's network and EMA weights but starts with a fresh optimizer.
Collection and evaluation select the bank entry using the same task ID as the
world model.

Actor updates imagine exclusively from task-filtered replay and route every
world-model transition and reward prediction through that task's expert. No
Actor-Critic parameter or optimizer state is shared after initialization.

## Replay and update budgets

FIFO and LTDM retain the source ARROW-50 trajectory capacities and nominal
`0.5/0.5` sub-buffer selection. Every trajectory slot stores one signed int64
task ID. Sampling a selected task is uniform over eligible sequence slots.
When both sub-buffers contain that task, selection remains `0.5/0.5`; if one
does not, weights are renormalized over eligible sub-buffers rather than failing.

For an epoch update budget `U`, current task `k`, available tasks `S`, and
`rho=0.5`:

```text
U_k = round(rho U)
U_j = (U - U_k) / |S - {k}|,  j != k,
```

with deterministic integer remainders assigned in task-index order. If only the
current task is available, it receives all `U` updates. The world-model schedule
is shuffled by an owned NumPy generator. Actor-Critic banks are independent, so
their allocated blocks may run in task-index order.

The sum remains exactly `steps_per_batch` for the world model and
`ac_train_steps` for Actor-Critic training. Environment decisions, frame repeat,
task durations, trajectory capacity, and evaluation cadence are not increased.

The launcher changes replay storage from CUDA to CPU explicitly so the published
float32 pixel tensors plus the 1,024-dimensional feature sidecar fit a 24 GiB
GPU. This changes transfer cost and storage device, not trajectory capacity.
Both base replay bytes, feature-cache bytes, and task-ID bytes are recorded.

## Evaluation and artifacts

Periodic evaluation keeps evaluation transitions out of replay and preserves
training RNG state. MoE-ARROW uses modal categorical latents and deterministic
argmax actions. Seen tasks use their preserved Actor-Critic and matching
world-model expert. Future tasks use the existing random-policy path until their
bank entry exists.

Final artifacts include the resolved config, launch manifest, raw TensorBoard
metrics, world-model state, and all Actor-Critic inference states. The artifact
is explicitly non-resumable because replay, all optimizers, scheduler position,
and RNG states are not yet serialized together. Interrupted runs must restart
and cannot be combined with uninterrupted results.

## Commands

Inspect a two-task pilot without creating a run directory:

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --seed 0 \
  --task-prefix-length 2 \
  --dry-run
```

After committing and pushing a clean synchronized branch, launch by removing
`--dry-run` and supplying a new persistent `--output-dir`. Run a target-GPU
smoke before any pilot or official campaign.

## Required comparisons

Report at least ARROW-50, the shared-DINO ARROW control, and MoE-ARROW under
matched environment and update budgets. Parameter count, optimizer state,
replay bytes, feature-cache bytes, task-ID privilege, current-task return,
final average return, and forgetting must all be shown. Per-task RSSM plus
per-task Actor-Critic is the storage-unconstrained reference; smaller expert
sharing or learned routing is evaluated against this v1 reference, not silently
folded into it.
