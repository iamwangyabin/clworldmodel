# 0041: Task-0 Acquisition-Duration Sweep

## Status

Accepted as a seed-0 pilot design on 2026-08-29. No general performance claim
exists until the selected duration is rerun from scratch on fresh seeds.

## Context

The primary concern is that 90 MsPacman epochs may end before the Evolving-Core
world model and Actor-Critic have fitted the first task. Learning-rate-only
profiles do not directly test this hypothesis. More online epochs increase both
environment interaction and optimizer updates, so duration is a deliberately
unmatched resource intervention rather than a fair fixed-budget ablation.

## Decision

Keep the active 90-epoch `fixed_v1` full run as the control. Reallocate GPUs
1--4 to 120, 150, 180, and 240 Task-0 epochs. Keep all learning rates,
architecture, losses, Replay capacity and sampling, seed, task order, and
evaluation cohorts fixed. Each candidate stops exactly at its extended
MsPacman boundary and never trains on Boxing or CrazyClimber.

Use the fixed 16-rollout pre-consolidation raw MsPacman mean to build the
duration curve. Let `m` be the maximum observed mean across 90/120/150/180/240.
Choose the shortest duration whose mean is at least
`m - 0.05 * max(abs(m), 1)`. Break any remaining tie by higher mean and then
lexical profile name. This favors the first practical saturation point rather
than automatically selecting the largest compute budget.

## Consequences

- The result directly tests whether extra joint sample/update budget improves
  Task-0 acquisition, but it cannot distinguish which resource caused it.
- Raw frames, world-model updates, Actor-Critic updates, and wall time differ
  across candidates and must be reported.
- Boundary consolidation remains an extra 1,000 updates for every candidate,
  but ranking uses the measurement taken before those updates.
- A non-90 winner changes the curriculum protocol and requires a fresh full
  run with Task-0 duration `D` and later durations `90/90`.
- If all candidates remain weak, the sweep provides no guarantee and the
  architecture/loss diagnosis must be reopened rather than selecting a result
  merely because it is the least poor.

## Rejected Alternatives

- Continuing the LR-only sweep would spend the available GPUs on a secondary
  hypothesis.
- Selecting the largest duration unconditionally would confuse compute with
  evidence of saturation.
- Resuming a 90-epoch run into an extended profile would change the resolved
  config and would not be equivalent to a from-scratch declared candidate.
- Using held-out-final or post-consolidation return would contaminate selection.
