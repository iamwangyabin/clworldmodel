# 0035: Compact RSSM LoRA with Independent Actors

## Status

Accepted for a matched seed-0 snapshot-seeded ablation on 2026-08-27.

## Context

The high-capacity Task-1-seeded projector/LoRA pilot uses RSSM LoRA ranks
`128/128/32` for recurrent, representation, and transition modules. Its
independent Actor-Critic restores later-task plasticity while the frozen Task-1
route retains MsPacman exactly. Runtime parameter accounting reports 2,455,808
FP32 RSSM adapter parameters, or 9,823,232 persistent bytes, for each later
task. This is useful as a capacity upper bound but is too large for the intended
compact continual method.

The earlier compact shared-Actor pilot changed both RSSM capacity and policy
topology. Its weak acquisition therefore cannot identify small RSSM adapters as
the cause. A controlled ablation must retain the strong Task-1 seed and
independent Actor-Critics while changing only LoRA capacity.

## Decision

Add the named profile `compact-r32-r32-r16` under the separately reported
`CNN-Projector-RSSM-CompactLoRA-ARROW-v2-Task1SnapshotSeeded-Atari-TaskAware`
protocol. It keeps the projector, private world-model heads, independent
Actor-Critics, replay behavior, environment interaction, optimizer updates,
sample use, learning rates, precision, evaluation cohorts, and task schedule
identical to the matched `128/128/32` pilot.

Only the later-task RSSM LoRA ranks change:

- recurrent: 32;
- representation: 32; and
- transition: 16.

Exact one-dimensional bias and normalization deltas remain uncompressed. The
result is 643,648 FP32 RSSM adapter parameters, or 2,574,592 persistent bytes,
per later task. This is a 73.8 percent reduction from the matched capacity
profile.

## Consequences

The first compact run remains a task-aware, single-seed feasibility ablation.
It cannot establish a general compression frontier or a task-agnostic result.
Its decisive comparison is later-task acquisition and frozen-task retention
against the matched high-capacity run. If `32/32/16` loses material acquisition,
the next predeclared intermediate capacity is `64/64/16`; it is not selected
from held-out-final results.
