# 0043: Compact Evolving-Core mechanism capacity pilot

- Status: accepted
- Date: 2026-08-30

## Context

The original-six Evolving-Core pilot allocates one private recurrent,
representation/posterior, and transition/prior residual mechanism per task.
At the fixed Atari RSSM interfaces, its `512/512/256` bottlenecks contribute
`22,897,332` parameters across six tasks and their reuse routes. The posterior
branch is the largest contributor. These mechanisms remain on the action-time
inference path, unlike the decoder, reward, and continuation heads.

The first capacity question should be isolated from Actor sharing, head
sharing, replay, optimizer, or curriculum changes. A shared or LoRA Actor would
introduce a separate forgetting and distillation hypothesis while saving less
capacity than the mechanism bank.

## Decision

Run one separately named seed-0, original-six capacity pilot with mechanism
bottlenecks:

- recurrent: `128`;
- representation/posterior: `128`; and
- transition/prior: `64`.

The external RSSM interfaces remain `512 -> 512`, `4608 -> 1024`, and
`512 -> 1024`. Four atoms remain in every mechanism, so their respective atom
widths become `32/32/16`. Zero-effect initialization, reuse routing, private
heads, independent Actor-Critics, all learning rates, replay semantics,
sequence batches, task durations, evaluation cohorts, consolidation, and
checkpoint retention remain identical to the high-capacity original-six
control.

The launcher exposes this only as the explicit
`compact_128_128_64` profile on `arrow-original-six`; its existing default
remains `matched_512`.

## Capacity consequence

One task's private mechanisms drop from `3,816,192` to `964,416` parameters.
Across six tasks, mechanisms plus the 180 reuse-route scalars drop from
`22,897,332` to `5,786,676`, a reduction of `17,110,656` parameters
(`74.7%`) in the targeted component. The independent Actor bank and private
training-only heads are deliberately unchanged. Therefore this experiment is
not a general compression endpoint: it isolates whether the principal
action-time growth term was over-wide.

## Evidence limits

This is a one-seed pilot and a task-aware method. It can diagnose the capacity
tradeoff against the concurrently executed high-capacity control, but cannot
establish multi-seed reliability, task-agnostic performance, or superiority.
The trainer's emitted parameter-accounting artifacts are authoritative if an
implementation-derived total differs from this preregistered formula.
