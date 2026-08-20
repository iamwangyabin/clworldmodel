# KARROW-SpatialFrozenCore-v2 Atari protocol

## Status and question

This is an implementation-ready correction to the failed v1 observation path.
No performance or forgetting claim exists yet. V1 remains reproducible through
`scripts/run_karrow_ar50_atari.py`; v2 is launched through
`scripts/run_karrow_spatial_ar50_atari.py`.

The first question is acquisition, not retention: does a frozen spatial DINOv3
target make the posterior RSSM state useful enough to approach the original
ARROW-50 MsPacman learning curve under the same interaction and update budget?

## Frozen spatial observation path

For each `64 x 64` RGB observation, frozen DINOv3 receives the same bicubic
`256 x 256` ImageNet-normalized input as v1. V2 discards CLS and register tokens.
The 256 patch tokens are reshaped to `[16, 16, 384]`, average-pooled to
`[4, 4, 384]`. V2 uniformly samples 512 frames from the first random Task-1
collection and fits PCA over their 8,192 pooled spatial cells. The leading 64
components form one channel projection shared by every spatial cell. The mean
and basis are frozen before the first world-model update and remain fixed for
all later Task-1 and continual updates. Each method arm fits from its identical
seed-matched initial collection. The flattened `[4, 4, 64]` target has 1,024
dimensions.

The observation objective follows the Dreamer posterior observation model:

```text
P, mu     = PCA64(initial_random_Task1_patches)  # fit once, then freeze
e_t       = stop_gradient((pool_4x4(DINO_patch_tokens(x_t)) - mu) @ P)
s_post_t  = concat(z_post_t, h_t)
e_hat_t   = feature_head(s_post_t) + feature_residual(s_post_t)
L_obs     = SmoothL1(standardize_TN(e_hat), standardize_TN(e))
```

Standardization is per flattened spatial-feature coordinate over the sampled
time and batch axes, with standard-deviation floor `0.05`. Prediction and target
use their own statistics. A zero standardized prediction is evaluated against
the same target and logged as `Metric/dinov3_constant_feature_loss`; the model
to constant ratio is also logged. All sequence positions are valid, including
the first and reset positions.

There is no v2 prior-feature loss. The original Dreamer dynamics and
representation KL terms remain unchanged and align the prior with the posterior.
Reward and continuation objectives remain unchanged. No pixel decoder is used.

PCA is the closed-form optimum of a linear autoencoder. It is used instead of
jointly optimizing the projection through `L_obs`: the latter admits a trivial
constant or zero target. PCA calibration uses existing initial Task-1 frames,
adds no environment interaction or gradient update, and writes its explained
variance and fit timing to `dinov3_patch_projection.json`.

## Replay and resources

ARROW-50 still uses 512 FIFO and 512 LTDM trajectories and chooses each buffer
with probability 0.5. Frozen features follow the exact write and sample indices
of each sub-buffer.

```text
2 buffers * 512 time * 512 trajectories * 1,024 features * 2 bytes
= 1,073,741,824 feature-cache bytes
```

The original replay tensors and their `25,813,843,968` allocated bytes remain.
Capacity, selection, interaction, world-model update, and actor update budgets
are unchanged, but v2 is not memory-matched to v1 or original ARROW.

## Residual and continual schedule

The KAN and parameter-matched MLP residual definitions and placements are
unchanged from v1. Task 1 trains the shared RSSM and behavior bases together
with one residual set. At the first task boundary, the shared core is frozen and
the same residual set remains plastic. There is no task ID, router, reset,
expansion, or task-specific parameter set.

## Controlled sequence

Run diagnostics in this order:

1. A short v2 DINO-only acquisition smoke validates finite losses, feature-cache
   alignment, model-to-constant improvement, and posterior sensitivity.
2. A 90-epoch `ARROW-DINOSpatial-50` MsPacman run measures the corrected visual
   path without residuals or freezing.
3. Only if acquisition is viable, run matched MLP-residual and KAN-residual
   single-task screens.
4. Only after those screens, run the two-task Frozen-Core comparison.

The historical `2109.375` raw ARROW-50 epoch-90 value is a diagnostic reference,
not a publication-grade paired result. V1's `521.875` is the failed-path
reference. A low feature loss is not sufficient: the model must beat its logged
constant baseline and improve environment return.

## Dry run

```bash
python scripts/run_karrow_spatial_ar50_atari.py \
  --variant dino \
  --task-prefix-length 1 \
  --seed 0 \
  --dinov3-model-path /absolute/path/to/dinov3-vits16-pretrain-lvd1689m \
  --dry-run
```

No long run should begin before a target-GPU smoke confirms the 1,024-feature
cache and posterior loss fit the declared memory budget.
