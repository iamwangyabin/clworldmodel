# Decision 0006: freeze the shared KARROW core after task 1

- Status: accepted for implementation, before training
- Date: 2026-08-19

## Context

The first KARROW residual design still allowed the shared GRU, latent transition,
reward and continuation heads, and actor-critic MLPs to update on every task.
Residual adapters could therefore reduce only part of the interference: the
shared functions still moved under new-task gradients and changed the latent
coordinates seen by every module.

## Decision

Use the `KARROW-FrozenCore-v1` schedule. During task 1, train the original ARROW
shared functions and one fixed set of residual adapters together. At the first
task boundary, freeze the shared RSSM, reward/continue, feature-prediction, and
actor-critic MLP bases. From task 2 onward, only the same residual adapters are
optimized; their optimizer state is preserved, while frozen parameters are
removed from optimizer groups.

To keep new-task learning possible, residual adapters cover the post-GRU hidden
state, posterior logits, latent-prior logits, reward, continuation, feature
prediction, actor logits, and critic logits. The KAN and control arms use the
same placements, fixed capacity, alpha, replay, schedule, and update budgets.
The DINOv3 encoder is frozen from initialization and is not part of the
task-boundary transition.

## Consequences

- This is a new protocol, not a change to ARROW-50 or the earlier trainable-core
  residual pilot.
- Task 1 is the representation/core acquisition phase; tasks 2 through N test
  adapter plasticity without shared-core drift.
- A residual-only arm that cannot learn a new task is a valid negative result,
  not a reason to unfreeze the shared core mid-run.
- The first experiment must compare fixed-capacity KAN adapters with matched MLP
  adapters and record both core drift and adapter support overlap.
