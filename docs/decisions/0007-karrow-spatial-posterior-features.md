# Decision 0007: spatial posterior feature grounding

## Context

The completed seed-0 `KARROW-FrozenCore-v1` two-task pilot learned only
`521.875` raw MsPacman return at the first task boundary. This failure occurs
before the shared core is frozen, so post-boundary stability cannot explain it.

The v1 observation target is a single DINOv3 CLS token. On 224 seeded random
policy MsPacman frames, the mean pairwise CLS cosine similarity was `0.9833`,
the adjacent-frame similarity was `0.9962`, and a constant mean-feature
predictor achieved cosine loss `0.00836`. The centered 384-dimensional sample
had effective rank about `6.09`. During training, the feature loss fell from
`0.941` to `0.012` after one epoch and to about `0.004` at the task boundary,
while roughly 97 percent of posterior KL values remained under the free-bits
threshold. The prior-only cosine objective therefore admitted an almost
constant shortcut and did not ground the posterior in the current frame.

## Decision

Preserve v1 unchanged and introduce `KARROW-SpatialFrozenCore-v2`.

1. Read the final DINOv3 patch tokens, excluding CLS and register tokens.
2. Reshape the 256 tokens to the native `16 x 16` grid, average-pool it to
   `4 x 4`. Uniformly sample 512 frames from the first random Task-1 collection
   and fit one shared 384-to-64 PCA channel projection over their 8,192 spatial
   cells. This is the closed-form optimum of a linear autoencoder, so it learns
   a data-dependent bottleneck without a jointly trainable target that could
   collapse. Freeze it before the first world-model update and flatten its
   output to 1,024 features. Its mean, basis, and explained variance are stored
   in checkpoints.
3. Predict the current frozen feature from the posterior RSSM state
   `(z_t, h_t)`. Include first and reset observations because each has a valid
   current-frame target.
4. Standardize prediction and target variation independently over the sampled
   time/batch axes for every spatial-feature coordinate, then apply SmoothL1.
   This removes the shared feature direction: a constant prediction is an
   explicit logged baseline rather than a near-zero solution.
5. Do not add a separate prior-feature objective in v2. The existing Dreamer
   dynamics KL continues to train the prior toward the grounded posterior.
6. Keep the residual placements, ARROW-50 replay decisions, per-epoch budgets,
   and post-task-1 Frozen-Core schedule unchanged.

The DINOv3 backbone remains frozen. Feature reconstruction is still necessary:
it trains the RSSM posterior, not the target encoder. A stable target prevents
visual drift but does not by itself force the latent state to preserve visual
information.

## Consequences

The float16 feature sidecar grows from `402,653,184` bytes in v1 to
`1,073,741,824` bytes in v2. The projected v2 target is one sixth the size of
the uncompressed 6,144-dimensional design while preserving its `4 x 4` spatial
layout. PCA adds one deterministic calibration computation but no environment
steps or gradient updates. This is an explicit resource difference and must be
reported. The first experiment is a single-task acquisition diagnostic;
continual retention is not evaluated until the spatial observation path learns
MsPacman substantially better than v1.
