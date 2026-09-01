# 0047: Use learned Task-0 bases and private low-rank adapters

## Status

Accepted for a separately named seed-0 pilot on 2026-09-02. This decision does
not assert that the method improves performance.

## Context

Dense Evolving-Core acquired tasks more reliably than the random
Shared-Frozen-Down experiment, but its six-task online topology grows to
`95,710,680` parameters. Sharing plastic prediction heads reduced that to
`52,897,535`, yet the remaining full Dense Q/F/P copy per task still grows by
roughly 3.82 million parameters and exposed prediction heads to cross-task
interference.

The failure of Shared-Frozen-Down does not show that every shared basis is bad.
It shows that a random fixed full-width projection is a poor substitute for a
task-acquired function. Task 0 provides a learned, behaviorally grounded Q/F/P
basis. Later tasks can preserve that basis and learn small private corrections.
Likewise, a frozen learned prediction-head base plus private feature adapters
offers isolation without repeating full decoders.

Actor-Critic sharing is excluded from this experiment because its previous
pilot confounded capacity reduction with behavior interference. Independent
MLP behavior is retained as the conservative acquisition path.

## Decision

Add `evolving_atomic_rssm_learned_base_adapters_arrow` and protocol
`Evolving-Core-LearnedTask0Base-LowRank32QFP-PrivatePredictionAdapters-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

- Learn full Dense `512/512/256` Task-0 Q/F/P residual mechanisms, then freeze
  them as the shared learned mechanism base.
- Give every later task an exact-zero Rank-32 nonlinear Q/F/P residual on that
  base. Disable old-atom reuse so the Task-0 base is not duplicated through
  multiple routed atoms.
- Learn the decoder/reward/continuation base on Task 0, freeze it thereafter,
  and give every later task three independent exact-zero Rank-32 input-feature
  adapters.
- Retain one independent standard MLP Actor and Critic per task.
- Preserve the Dense original-six order, 90 epochs per task, ARROW-50 Replay,
  optimizer/update budgets, real old-task losses, interface/output
  distillation, boundary consolidation, rollback, and fixed evaluation cohorts.
- Treat the base prediction heads as Task-0 private state. Later consolidation
  updates only the shared CNN/base RSSM/latent interface.

V1 fixes rank 32. Automatic rank growth and temporary full Dense teacher
compression are separate hypotheses and require separately named protocols.

## Consequences

The exact six-task online total is `37,156,095` parameters: `26,860,185` world
model and `10,295,910` private behavior. This is `61.1787%` below Dense and
`29.7584%` below the shared-plastic-head/Dense-QFP pilot, while keeping learned
rather than random Q/F/P capacity.

Task identity remains exposed by the named protocol. The method is not
task-agnostic. The frozen Task-0 bases may be biased toward MsPacman, Rank 32
may underfit later tasks, and frozen prediction heads may limit reconstruction
adaptation. Those are experimental risks rather than silently adjustable
hyperparameters.

Runtime accounting must include dormant common-bank route scalars even though
they are not called or optimized, and must separate online weights from Replay,
optimizer state, activations, checkpoints, and the training-only boundary
teacher. Raw returns remain separate from reward-scaled and normalized scores.
