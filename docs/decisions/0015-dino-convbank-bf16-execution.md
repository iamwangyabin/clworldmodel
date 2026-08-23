# Decision 0015: Name and isolate DINO-ConvBank BF16 execution

## Status

Accepted for implementation and pilot validation. No speed or return claim has
been established yet.

## Context

DINO-ConvBank V4 recomputes frozen full-patch DINO features for 512 sampled
frames in every regular world-model update and another 512 context frames in
each Actor-Critic update. The FP32 path splits each set into four 128-frame DINO
calls and converts full 98,304-wide features through float16 and back to
float32 before the shared adapter. The target accelerator has native BF16
Tensor Cores and ample unused memory, while measurements identify compute, not
memory capacity, as the active bottleneck.

Changing arithmetic precision can change optimization behavior. It therefore
cannot silently replace the V4 reference even when interaction, replay, and
update budgets are identical.

## Decision

Keep `fp32-tf32` as the default and add an explicit `bf16-amp` profile named
`DINO-ConvBank-ARROW-v4-BF16AMP-Atari-TaskAware`.

The BF16 profile autocasts CUDA model kernels, keeps parameters and optimizer
state in float32, and uses no gradient scaler. Numerically sensitive
probability, KL, return, target, and loss calculations promote to float32. DINO
features remain bfloat16 through the on-the-fly source and convolution adapter,
and the DINO execution chunk is 512 frames. The `T=32, N=16` optimization batch,
all update counts, and all data budgets remain unchanged.

The launcher records the profile, every relevant dtype, sensitive FP32
operations, DINO chunk size, and unchanged optimization batch. It rejects BF16
for methods outside DINO-ConvBank until each path receives focused validation.

## Consequences

The profile should reduce DINO and model arithmetic cost and eliminate one
large temporary conversion without pretending that a larger execution chunk is
a larger training batch. Its speed, peak memory, loss stability, and Atari
returns require a matched pilot. FP32 and BF16 runs remain separate result
groups even if their learning curves later agree.
