# Decision 0016: Require BF16 and uint8 replay for DINO-ConvBank

Scope note: Decision 0018 later applies the same explicit storage and precision
profile to the separately named CNN-FullBank method; prior protocols remain
unchanged.

## Status

Accepted. This decision supersedes Decision 0015's FP32/TF32 default for the
current DINO-ConvBank launcher. Historical FP32 artifacts retain their original
name and are not part of the new result group.

## Context

DINO-ConvBank recomputes frozen DINOv3 features for 512 replay frames in every
regular world-model update and for another 512 context frames in every
Actor-Critic update. The original FP32 execution path added avoidable arithmetic
and feature-conversion cost. It also retained ARROW's float32 observation replay
on a CPU file-backed mmap. Those observations originate as discrete uint8 Atari
pixels, so float32 storage consumes four times the required mmap capacity and
host-to-device bandwidth.

Changing compute or replay dtype is behavior-affecting execution metadata. It
must be explicit, tested, and reported rather than silently folded into V4.

## Decision

The current `dino-convbank` launcher selects and requires `bf16-amp`. CUDA model
kernels run under BF16 autocast with a 512-frame DINO execution chunk. FP32
parameters, parameter gradients, Adam state, categorical and KL operations,
symlog transforms, losses, returns, and value targets remain deliberate master
or sensitive state. Native BF16 accelerator support is required.

Add a typed `replay_observation_dtype` configuration field whose default is
`float32`, preserving every published ARROW, DV3, KARROW, MoE, FullBank, and
PatchBank configuration. The current DINO-ConvBank protocol alone requires
`uint8`. Replay rounds incoming normalized Atari pixels back to their exact
discrete byte values, stores them under the unchanged FIFO/LTDM write maps, and
after sampling transfers the bytes to the requested device before converting
to float32 and dividing by 255. All model-facing observation tensors therefore
retain the prior shape, dtype, and value convention.

The reported protocol is
`DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-Atari-TaskAware`.

## Consequences

- Observation mmap storage falls from `25,769,803,776` to
  `6,442,450,944` bytes; unchanged auxiliary replay tensors add
  `44,040,192` bytes.
- FIFO/LTDM capacities, random decisions, task labels, update budgets,
  interaction budgets, and evaluation cadence do not change.
- Focused tests require exact decoded-value parity, FIFO wraparound parity,
  LTDM random-key retention parity, mmap byte accounting, config isolation,
  and FP32-launch rejection.
- BF16 reduces arithmetic cost but does not remove repeated DINO encoding. A
  frozen compressed feature cache remains a separate future method decision.
