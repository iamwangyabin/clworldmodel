# KARROW Replay-Consolidated v3 Atari Protocol

## Status

Experimental method protocol. Results must be labeled smoke, pilot, ablation,
or official according to the evidence actually produced.

## Hypothesis

Frozen spatial DINO features and the Task-1-fitted RSSM coordinate system place
different Atari games in distinguishable regions. A fixed-grid KAN can then use
input-local RBF coefficients as an implicit task-agnostic routing mechanism. If
replay-important coefficients are protected at task boundaries, later tasks can
preferentially update cold coefficients and reduce cross-task interference.

## Base

The base is `KARROW-SpatialFrozenCore-v2` on ARROW-50. Environment order,
task duration, interaction budget, world-model updates, actor-critic updates,
ARROW FIFO/LTDM capacity, and ARROW minibatch selection remain unchanged.

The visual path remains:

- frozen DINOv3 ViT-S/16;
- pooled `4 x 4` patch tokens, excluding CLS and register tokens;
- one Task-1 PCA projection from 384 to 64 channels per patch;
- a frozen 1,024-dimensional spatial feature target;
- posterior-state batch-standardized SmoothL1 feature prediction;
- no pixel decoder.

## Incremental KAN

Task 1 trains the base and all zero-initialized KAN residuals jointly. At the
first boundary, freeze the shared world-model and actor-critic base exactly as
in v2. Also freeze each residual's down projection, RMSNorm scale, and up
projection. Only its `64 x 64 x 8` Gaussian RBF coefficient tensor remains
trainable.

At every task boundary, before collecting the next game:

1. Draw 16 minibatches of `32 x 16` sequences from the unchanged ARROW replay
   mixture.
2. Run deterministic teacher-forced RSSM posterior inference.
3. Run eight deterministic argmax-policy imagination steps from each replay
   context endpoint.
4. Estimate the squared local output Jacobian of every RBF coefficient.
5. Normalize importance by the module's positive 99th percentile and merge it
   with earlier importance using a coefficient-wise maximum.
6. Anchor a coefficient at the current value only when the new estimate exceeds
   its prior maximum.
7. Scale its future gradient by
   `(1 - importance)^2 * 0.99 + 0.01` and add an importance-weighted anchor loss
   with scale 1.0.

After every Adam step, scale the realized RBF parameter delta by the same
importance-dependent factor. This optimizer-aware projection is required
because Adam's second-moment normalization can otherwise cancel a simple
gradient rescaling.

The boundary pass restores all training RNG states. It adds no environment
steps and no optimizer steps. Its extra forward compute is recorded separately.
Across eight residuals, persistent importance, anchor, and update-scale buffers
add 3,145,832 bytes; the boundary accumulator peaks at another 1,048,576 bytes.

## Inference Contract

Inference receives no task identity. All tasks use the same frozen DINO, RSSM
base, actor-critic base, and one fixed-capacity KAN residual per module. Gaussian
basis activations provide input-conditioned routing. No grid grows and no
task-specific parameters are created.

## Required Comparisons

- ARROW-50;
- ARROW plus frozen spatial DINO;
- spatial DINO plus frozen core and parameter-matched MLP residuals;
- KARROW spatial v2 with unconsolidated KAN residuals;
- KARROW replay-consolidated v3.

All comparisons preserve the behavioral budgets. The consolidation forward
pass is compute overhead and must be reported, but is not an extra training
update.

These comparisons can establish that KAN locality plus consolidation works;
they do not by themselves establish that KAN is uniquely necessary. A later
generic-adapter EWC or update-projection control is required before making that
stronger claim.

## Task-Region Audit

First collect held-out per-task chunks with the component-audit collector. These
transitions remain isolated from replay. Then evaluate every task at one fixed
checkpoint:

```bash
python scripts/component_forgetting_audit.py collect \
  --run-dir RUN_DIR \
  --output-dir RUN_DIR/latent_audit_data \
  --dinov3-model-path /absolute/path/to/dinov3-vits16 \
  --collection-policy uniform_random \
  --event-chunks 0 \
  --label karrow_v3_latent_regions

python scripts/latent_region_audit.py \
  --run-dir RUN_DIR \
  --audit-dir RUN_DIR/latent_audit_data \
  --output-dir RUN_DIR/latent_region_results \
  --dinov3-model-path /absolute/path/to/dinov3-vits16 \
  --checkpoint final
```

The audit reports, separately for frozen DINO features, RSSM posterior
probabilities, deterministic hidden state, and their combined state:

- held-out nearest-centroid task accuracy with a Wilson interval;
- held-out 5-nearest-neighbor task accuracy;
- a nearest-centroid label-permutation p-value;
- pairwise centroid distance divided by pooled within-task RMS radius;
- a two-dimensional PCA artifact and figure;
- pairwise weighted-Jaccard overlap of every KAN residual's mean RBF support.
- pairwise correlation between RSSM region distance and KAN support
  disjointness.

Task labels exist only in this offline evaluator. A positive result establishes
task decodability in the representation, not disjoint support and not causal
protection from forgetting. The paper claim requires the support-overlap result
and continual retention result to agree across seeds. Matched uniform-random
collection is the primary game-region test because it avoids attributing a
checkpoint actor's different action distribution to the environment itself;
checkpoint-actor collection is a secondary on-policy sensitivity analysis.

## Launch

After committing and pushing a clean revision:

```bash
python scripts/run_karrow_ar50_atari.py \
  --visual-version v3 \
  --variant kan \
  --seed 0 \
  --curriculum original \
  --dinov3-model-path /absolute/path/to/dinov3-vits16
```
