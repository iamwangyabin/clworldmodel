# Decision 0005: test fixed-capacity KAN residual corrections on frozen features

- Status: accepted for implementation and controlled pilots
- Date: 2026-08-19

This decision records the original trainable-core residual design. Its continual
schedule is superseded by Decision 0006 before training begins.

## Context

Replacing the complete ARROW actor or actor-critic with KAN changes acquisition,
optimization, and function class at the same time. The existing KAN actor pilots
therefore cannot isolate whether local KAN updates reduce cross-task interference.
They also leave the learned pixel encoder as a separate source of representation
forgetting.

The research question is narrower: under identical ARROW replay, interaction,
and update budgets, does a fixed-capacity local residual learn new tasks with less
drift than an exactly parameter-matched dense residual?

## Decision

Add the separately named `KARROW-v1` method. It freezes a local DINOv3 ViT-S/16
visual encoder, removes pixel reconstruction, and trains the RSSM to predict the
current frozen feature from its one-step prior state. The existing `nn.GRUCell`,
MLP actor, and MLP critic remain intact.

Three independent zero-initialized corrections are added after the GRU hidden
update and to the actor and critic logits. Every correction uses a 64-dimensional
bottleneck and a fixed scale of `0.1`. The KAN arm uses a basis-only Gaussian RBF
core with eight fixed centers over `[-2, 2]`. Its control replaces only that core
with a bias-free `64 -> 256 -> 64` MLP. Both cores contain exactly 32,768
parameters, and the surrounding adapters are identical.

The method owns one fixed correction per module from task 1 through task N. It
does not reset or expand corrections, change grids, route by task, or expose task
identity. ARROW continues updating every base parameter and retains its original
FIFO/LTDM behavior.

## Consequences

- Four arms are required: ARROW-50, frozen DINO only, frozen DINO plus matched MLP
  residuals, and frozen DINO plus KAN residuals.
- The DINO and residual arms are new named protocols, not ARROW reproductions.
- Frozen features are stored in a byte-accounted replay sidecar that follows
  ARROW's existing write and sample indices. It changes storage and representation
  compute, but not replay capacity or selection.
- A result supports KAN locality only if the KAN residual outperforms the matched
  MLP residual, not merely the original ARROW model.
- GPU smoke testing and acquisition checks must precede any continual-retention
  claim. A single seed or task-prefix run remains a pilot.
