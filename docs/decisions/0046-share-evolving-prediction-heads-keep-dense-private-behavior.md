# 0046: Share prediction heads while retaining Dense Q/F/P and private behavior

## Status

Accepted for implementation on 2026-09-01. The decision authorizes a
separately named pilot implementation; it does not assert improved performance.

## Context

The full-width Dense Evolving-Core pilot showed substantially better task
acquisition than the Shared-Frozen-Down plus shared-FastKAN pilot, but its
online state grows in two independent places. Every task owns a learned
`512/512/256` Q/F/P mechanism and an independent MLP Actor-Critic, and every
later task also duplicates the large decoder/reward/continue prediction-head
set. The six-task Dense topology has `95,704,536` online parameters.

The evidence does not justify compressing the parts that isolate acquisition
and behavior. The requested hypothesis is narrower: the full-width learned
task-private Q/F/P routes and private MLP Actor-Critics carry the useful
task-specific capacity, while decoder/reward/continue can remain one plastic
set if real old-task replay and a frozen cumulative teacher protect their
outputs.

This is a behavioral change. It must not silently redefine the existing Dense,
compact, Shared-Frozen-Down, or shared-FastKAN protocols.

## Decision

Add `evolving_atomic_rssm_shared_heads_arrow` and the named
`Evolving-Core-DenseQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1` pilot
profiles. Preserve the Dense Evolving-Core model and optimizer contract except
for prediction-head ownership:

- retain one full-width learned Dense Q/F/P mechanism and zero-effect spatial
  projector per task;
- retain one independent DreamerV3 MLP Actor and Critic per task;
- allocate exactly one plastic decoder, reward head, and continuation head for
  the complete curriculum;
- move those three shared heads from the task-private optimizer into explicit
  shared-optimizer groups while retaining their Dense head LR `2e-4` during
  online acquisition;
- on old-task LTDM sequences, keep the ordinary Dreamer reconstruction,
  reward, and continuation losses and additionally match observation output,
  symlog reward, and continuation probability to the existing frozen boundary
  world-model teacher at total scale `0.1`;
- include the three heads as separate component-gradient-projection groups;
  and
- include them in task-balanced boundary consolidation and whole-state/Adam
  rollback.

The output loss uses mean observation MSE, mean symlog-reward MSE, and mean
Bernoulli KL from teacher continuation probability to student logit. The
ordinary replay loss remains unchanged and continues to use real targets. The
new loss reuses the already required old-task teacher and student forwards: it
adds no model copy, environment transition, replay capacity, sequence, optimizer
step, or teacher forward.

## Parameter consequences

The one prediction-head set contains `8,562,629` parameters and is already part
of the base ARROW world model. Removing its five later-task copies reduces the
six-task topology by `42,813,145` parameters:

| topology | world model | behavior | online total |
|---|---:|---:|---:|
| Dense Evolving-Core, six private head sets and six private MLP pairs | `85,414,770` | `10,289,766` | `95,704,536` |
| Shared prediction heads, Dense Q/F/P, six private MLP pairs | `42,601,625` | `10,289,766` | `52,891,391` |

The new total occupies `211,565,564` FP32 parameter bytes before buffers,
gradients, optimizer state, activations, Replay, or the common boundary teacher.
It is `44.7347%` smaller than the Dense six-task online topology. World-model
growth still remains linear because every task keeps its projector and Dense
Q/F/P route; behavior growth also remains linear by design.

## Consequences and evidence required

- Task identity still selects Q/F/P routes and the private Actor-Critic. The
  method is task-aware and makes no task-agnostic claim.
- Prediction-head sharing can introduce interference. LTDM replay, output
  distillation, component projection, consolidation, and rollback reduce that
  risk but do not prove retention.
- The clean attribution control is the existing full Dense Evolving-Core with
  private heads and private MLP behavior under the same order, seed, budgets,
  reward scaling, and evaluation cohorts.
- Preserve raw per-task returns separately from scaled/normalized metrics and
  report final average performance, forgetting, and valid transfer metrics.
- Smoke and seed-0 results remain pilot evidence. A superiority claim requires
  a matched multi-seed campaign.
