# Decision 0013: preserve complete DINO patches before the RSSM

## Context

The DINO-FullBank-v2 first-task pilot did not show MsPacman acquisition even
though its posterior feature head beat a constant predictor. That objective can
only verify reconstruction of its own `4 x 4 x 64` target. It cannot recover
information already removed by `16 x 16` to `4 x 4` average pooling and the
fixed 384-to-64 channel projection.

Direct DreamerV3 work with DINOv2 instead retains all spatial patch embeddings,
replaces only the CNN encoder, and leaves the RSSM and image reconstruction
path intact. DINO-WM independently reports a large planning gap between spatial
patches and a global DINO CLS representation. Neither result proves an Atari
improvement, but both make the current fixed bottleneck the next controlled
variable to remove.

## Decision

Add the separately named `DINO-PatchBank-ARROW-v3-Atari-TaskAware` protocol.
It sends the complete frozen `16 x 16 x 384` DINOv3 patch tensor to each task's
existing Dreamer posterior projection and restores a task-routed pixel decoder.
It removes the fixed spatial pooling, channel projection, and feature head.

The frozen DINO backbone, task-isolated RSSM/heads, fresh per-task Actor-Critic,
ARROW-50 replay decisions, and update budgets are otherwise retained. V2
remains selectable so its negative pilot is reproducible.

## Consequences

- The posterior receives 98,304 visual coordinates instead of 1,024.
- The RSSM's existing learned projection, pixel reconstruction, reward,
  continuation, and KL losses decide which coordinates matter.
- The float16 feature sidecar grows from 1 GiB to 96 GiB under the published
  two-buffer geometry; comparisons are not storage matched.
- Posterior parameter count and matrix-multiply cost increase substantially.
- A frozen backbone preserves continual stability but does not test end-to-end
  or partial DINO fine-tuning. Task-specific LoRA is a later named ablation,
  contingent on successful Task-1 acquisition.
