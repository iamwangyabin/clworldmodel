# 0038: Reuse-Expand-Consolidate RSSM

## Status

Accepted as a task-aware, Task-1-snapshot-seeded pilot method on 2026-08-28.

## Context

The whole-gate mechanism bank established a useful plasticity floor: every
later task receives a full nonlinear recurrent, posterior, and prior residual
mechanism while the Task-1 CNN and base RSSM remain frozen. Its scalar reuse
route, however, can only accept or reject an entire old mechanism. A
teacher-based shared/private decomposition would introduce an extra model,
non-identifiable branches, and distillation behavior before fine-grained reuse
has been established.

## Decision

Add the separately named
`REC-RSSM-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware` protocol. It preserves
the complete `512/512/256` mechanism capacity and original Dreamer losses, and
adds three explicit phases: reuse, expansion, and route consolidation.

Each mechanism's hidden axis is split into four contiguous atoms. The recurrent
and posterior atoms have width 128; prior atoms have width 64. An atom owns its
slice of the output weight and one quarter of the output bias. Summing all four
atom outputs therefore reproduces the original mechanism, apart from floating
point reduction order, with no additional parameter and no distillation.

For task `k`, every old task and atom receives an independent zero-initialized
`tanh` route gate and a persistent binary mask. The current task's full new
mechanism always has coefficient one. Old mechanisms remain frozen. The first
local epoch of CrazyClimber freezes its zero-effect new mechanisms and trains
only the old-atom routes plus the current projector and private heads. Boxing
skips this probe because there is no earlier mechanism. Expansion then trains
the full current mechanisms and routes together. Route parameters use five
times the world-model learning rate; the normal world-model loss and all
interaction/update budgets remain unchanged.

At each later-task boundary, eight fixed replay minibatches measure the
world-model loss increase from ablating one old atom at a time and the atom's
mean routed-output norm relative to the complete correction. A candidate is
masked when its loss increase is non-positive or its contribution ratio is
below `0.01`. The full and proposed routes are then evaluated with the same
fixed 16-rollout cohort. All proposed pruning is rolled back if the pruned
return is below 95 percent of the full-route return. Consolidation changes only
the current route masks. It performs no gradient update, environment
interaction, or replay write.

An atom is called shared only after a later task keeps it through functional
ablation and fixed-cohort validation. Otherwise it remains private to its owner.
No mechanism weight is merged into the shared base.

## Compatibility

Legacy scalar route logits are repeated across all four atoms. A missing
legacy mask becomes all ones. The migration must preserve recurrent output,
posterior logits, prior logits, and deterministic rollout behavior within the
declared floating point tolerance. REC-RSSM snapshots store atom masks and
routes but remain inference-only; they do not become resumable checkpoints.

## Consequences

- REC-RSSM retains the whole-gate method as its no-reuse functional floor
  because the current mechanism is never narrowed or attenuated.
- Fine-grained reuse costs 12 FP32 route parameters on Task 3, while each new
  task still owns exactly 3,816,192 mechanism parameters.
- The four contiguous hidden groups are functional atoms, not a claim of
  semantic disentanglement. A single seed can only establish pilot evidence.
- This protocol is task-aware and must not be reported as task-agnostic.
- Atom dropout, orthogonality, output decorrelation, teachers, and
  shared/private distillation are intentionally excluded from v1.

## Evidence Gate

Before formal training, tests must cover lossless atom summation, scalar-gate
migration, recurrent/posterior/prior parity, probe/expand gradient ownership,
hard-mask persistence, strict configuration, and launch budgets. A target-GPU
smoke must additionally verify finite optimized losses, nonzero route and
current-mechanism gradients in their respective phases, exact frozen Task-1 and
old-mechanism tensors, replay/evaluation isolation, deterministic 16-rollout
migration parity, and normal process exit.

## Expanded120 Follow-Up

The v1 pilot later exhibited substantial within-Boxing policy variation after
reaching a stronger intermediate checkpoint. The separately named v2
Expanded120 pilot therefore increases mechanism hidden widths to `640/640/320`,
extends only the two snapshot-seeded later tasks to 120 epochs each, and uses a
task-age-only Actor-Critic cosine learning-rate decay from local epoch 60 to
120. It does not modify or supersede the running v1 protocol. Its additional
capacity, frames, updates, and compute are explicit, so its results cannot be
described as matched evidence. The protocol is specified in
`docs/protocols/rec_rssm_arrow_v2_expanded120_atari.md`.
