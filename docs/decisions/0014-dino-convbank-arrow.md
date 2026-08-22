# Decision 0014: adapt complete DINO patches with one shared convolution

## Context

DINO-PatchBank-v3 removed the fixed `4 x 4 x 64` projection and retained the
complete `16 x 16 x 384` DINO patch grid. Its direct flattening produced a
98,304-dimensional posterior input. Under the standard Atari RSSM dimensions,
that expanded each task's posterior representation to 151,784,960 parameters.
The interrupted seed-0 Task-1 pilot did not show convincing acquisition at its
first formal evaluation and became progressively slower as replay filled.

Returning immediately to fixed pooling would recreate the information-loss
hypothesis V3 was meant to remove. A complete autoencoder would change both the
visual objective and optimization problem before the simple capacity question
was answered.

## Decision

Add the separately named
`DINO-ConvBank-ARROW-v4-Atari-TaskAware` protocol. Preserve V3 and change only
the interface between frozen full-grid DINO features and the posterior:

```text
[B,16,16,384]
  -> permute to NCHW
  -> Conv2d(384,64,3,2,1)
  -> ChannelLayerNorm(eps=1e-3)
  -> SiLU
  -> flatten [B,4096]
```

Use a standard convolution rather than a depthwise-separable convolution so the
first test can learn joint local spatial and channel selection. Stop at
`8 x 8 x 64`; do not add a second stride, pooling, an autoencoder, or a new
feature-prediction loss. Keep the V3 pixel decoder, KL, reward, continuation,
imagination, Actor-Critic, replay, and update semantics.

Own exactly one trainable adapter in the RSSM. Share it across all task routes
while keeping DINO frozen and the remaining world-model and policy modules
task-specific. Raw DINO outputs are detached before the adapter, so gradients
reach the adapter but never the frozen encoder or replay observations.

## Consequences

- DINO-to-posterior width falls from 98,304 to 4,096, matching the original
  Dreamer CNN's flattened interface while retaining an `8 x 8` spatial map.
- The adapter adds 221,376 parameters once per run, not once per task.
- Each task posterior falls from 151,784,960 to 7,081,472 parameters under the
  fixed Atari configuration.
- DINO extraction and image replay costs remain unchanged from V3; no feature
  mmap is introduced.
- The shared adapter creates transfer capacity but also a direct forgetting
  path. Frozen old experts can change function when later tasks update their
  shared input transform.
- V3 remains selectable and its stopped pilot remains part of the negative
  experiment record. V4 begins with a fresh Task-1 acquisition pilot and makes
  no performance claim until raw evaluation returns support one.
- A depthwise-separable adapter is reserved for a later named compute ablation
  only after the standard convolution demonstrates acquisition.
