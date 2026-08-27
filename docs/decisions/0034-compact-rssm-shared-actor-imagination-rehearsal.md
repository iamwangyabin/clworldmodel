# Decision 0034: Compact RSSM Routes and One Rehearsed Actor

## Status

Accepted for a task-aware pilot. Not yet accepted as the project default.

## Decision

Replace per-task Actor growth with one shared Actor protected by policy
distillation on states imagined by frozen old world-model routes. Replace
rank-128 recurrent matrix LoRA with a bottleneck-32 GRU-output correction and
reduce representation LoRA from rank 128 to rank 32. Keep transition LoRA at
rank 32 and retain the existing spatial projector.

At each task boundary, keep exactly one frozen copy of the previous shared
Actor as a temporary teacher. Do not preserve one teacher per task. Old routes
generate synthetic states from zero initialization; no old real Replay is
stored or sampled for Actor retention.

## Why

The preceding task-banked pilot removed world-model forgetting but paid for a
new Actor and large RSSM adapters per task. A shared Actor tests whether that
growth is necessary. Output-only recurrent correction targets the final hidden
state directly and is much smaller than adapting every GRU and pre-GRU affine.

## Consequences

The method adds model-only imagination compute, so comparisons must report it.
The world-model router remains task-aware. A successful three-task pilot does
not prove task-ID-free operation or scale to longer curricula. Zero-state
imagination can miss old states reachable from real observations; a later
ablation may compare replay-conditioned imagination under a separately named
protocol.
