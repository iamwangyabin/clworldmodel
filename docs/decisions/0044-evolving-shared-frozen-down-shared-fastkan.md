# 0044: Share a FastKAN Actor-Critic across Shared-Frozen-Down tasks

## Status

Accepted for implementation on 2026-08-31. No performance result exists for
this combination. The decision authorizes a separately named prospective
three-task pilot, not a superiority or continual-learning claim.

## Context

The Shared-Frozen-Down Task-0 pilot retained the full `512/512/256` Q/F/P
mechanism widths while replacing repeated task-private down matrices with one
frozen basis per bank and private LayerNorm/FiLM/up state. Its single seed is
an acquisition signal only. Separately, the local
`ARROW-FastKANAC-StableTargets-50` work established an existing width-53
FastKAN Actor and FastKAN Critic implementation with stabilized value targets.

The inherited Evolving-Core behavior topology still allocates one independent
MLP Actor-Critic per task. That prevents behavior interference but grows
linearly and cannot test transfer through one behavior model. The requested
method combines the Shared-Frozen-Down world model with one task-shared KAN
Actor-Critic.

This combination must not silently change interaction, Replay capacity,
world-model batch composition, consolidation compute, or the number of
Actor-Critic optimizer updates. Sharing, FastKAN, StableTargets, and replay
rehearsal are behavior changes, so the result receives a new protocol name.

## Decision

Add the task-aware
`Evolving-Core-SharedFrozenDown-SharedFastKANAC-StableTargets-ARROW-v1-ThreeTask-Atari-TaskAware-Pilot`
profile. It uses the fixed-v2 Evolving-Core optimizer and loss contract, the
`shared_frozen_down_film` Q/F/P parameterization, and exactly one persistent
width-53 FastKAN Actor plus one persistent width-53 FastKAN Critic.

Keep the Actor-Critic budget fixed at 800 optimizer updates per epoch. Task 0
uses all 800 on its route. On later tasks, allocate 600 updates to the current
route and 200 uniformly across completed routes, then shuffle the exact route
schedule. Each update draws task-homogeneous context from the existing
task-conditioned ARROW mixed Replay. A separately seeded RNG prevents this
schedule from perturbing Evolving-Core's world-model memory-task sampling.

Use the existing `fast_kan_ac_stable` bundle: LaProp at `4e-5`, three hidden
FastKAN layers of width 53 with eight Gaussian centers, persistent return
normalization, an EMA slow critic, replay-value loss scale `0.3`, slow value
targets, and the corrected terminal imagination bootstrap. Both shared heads
update on current and rehearsal routes.

Before each new task, retain one transient frozen copy of the preceding
boundary Actor. It protects the old world-model policy interface and is not a
second online policy. The older actor-only imagination-distillation objective
is disabled. Resumable checkpoint schema v2 stores the shared pair, LaProp
state, slow critic, return EMAs, future-task Actor teacher, and behavior-route
RNG in addition to the existing complete Evolving-Core state.

## Parameter consequences

For the fixed three-task Atari topology:

- Shared-Frozen-Down world model: `42,675,539` online parameters.
- One online FastKAN pair: `1,700,670` (`793,692` Actor and `906,978` Critic).
- New-method online total: `44,376,209` parameters.
- The same Shared-Frozen-Down world model with three private MLP pairs would
  use `47,823,494`, so sharing the FastKAN pair removes `3,447,285` online
  parameters.
- Dense Evolving-Core v2 with three private MLP pairs uses `53,323,398`; the
  combined world-model and behavior change is `8,947,189` parameters smaller.
- Plain ARROW-50 uses `21,214,838`, so the new method remains `23,161,371`
  parameters larger.

The StableTargets slow critic adds `906,978` training-only parameters and the
transient Actor teacher adds at most `793,692`. Peak behavior state before
optimizer state is therefore `3,401,340` parameters. Replay, gradients,
optimizer state, activations, and the existing boundary world-model teacher
remain separately byte-accounted.

## Consequences and required controls

- Later tasks add no behavior parameters, but retain private projector,
  LayerNorm/FiLM/up mechanism, route, and world-model head state.
- The method remains task-aware: orchestration selects a world-model route,
  although task identity is not concatenated to Actor-Critic input.
- The comparison that isolates behavior sharing is Shared-Frozen-Down with
  private MLP Actor-Critics under the same three-task protocol. Dense v2 is a
  secondary total-capacity reference, not an architecture-only control.
- A result cannot attribute an effect to KAN alone. Shared-MLP StableTargets
  and private-FastKAN controls are needed to separate sharing, architecture,
  and target stabilization.
- Smoke and one-seed runs remain pilot evidence. Preserve raw task returns,
  retention, forgetting, update routing, parameter and byte accounting, and
  all negative or rolled-back boundaries.
