# 0049: Acquire Dense Q/F/P, then select a compact width by raw return

## Status

Accepted for a separately named implementation on 2026-09-02. The decision
authorizes code and tests; a target-CUDA smoke or seed-0 pilot requires a
separate launch instruction. It does not assert that compression preserves
performance.

## Context

The shared-prediction-head experiment from decision 0046 retains the strongest
observed acquisition behavior of the current Evolving-Core variants, but every
task permanently keeps a `512/512/256` Dense Q/F/P mechanism. Fixed Rank-32 and
Rank-128 replacements failed to acquire Boxing reliably. Those failures show
that choosing a small width before seeing the task is unsafe; they do not show
that the full acquired function must remain full forever.

The requested hypothesis is therefore **dense acquisition followed by
task-specific compression**. Capacity is selected only after the task has been
learned. Shared decoder/reward/continue heads and independent MLP behavior from
experiment A remain unchanged so the experiment isolates post-boundary Q/F/P
compression.

This changes optimizer steps, evaluation use, serialization topology, and
parameter accounting. It must not redefine experiment A, the compact-width
ablation, or the failed low-rank methods.

## Decision

Add method
`evolving_atomic_rssm_adaptive_compression_shared_heads_arrow` and protocol
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

For every task:

1. acquire the task with experiment A's complete Dense private Q/F/P widths
   `512/512/256`, four atoms, private projector, route, and private MLP
   Actor-Critic;
2. run the unchanged 1,000-update shared-core/head consolidation and rollback;
3. freeze that accepted world model as a training-only Dense teacher;
4. independently construct structured-pruned Q/F/P candidates at fractions
   `0.75`, `0.5`, `0.25`, and `0.125`, corresponding to widths
   `384/384/192`, `256/256/128`, `128/128/64`, and `64/64/32`;
5. give **every** candidate exactly 250 Adam updates at `2e-4` on the completed
   task's LTDM sequences, with identical replay indices and stochastic latent
   draws across widths;
6. keep only Q/F/P trainable during recovery and add mean Q/F/P output MSE at
   scale `1.0` to the existing real-target Dreamer, interface, and shared-head
   protection losses;
7. evaluate the Dense teacher and all four candidates on a dedicated fixed
   16-rollout pruning-validation cohort; and
8. install the smallest candidate with relative raw-return drop at most five
   percent, otherwise retain the Dense teacher function.

The signed gate is

\[
d = \frac{R_{dense}-R_{candidate}}
         {\max(|R_{dense}|,1)},\qquad d \le 0.05.
\]

This definition behaves correctly for negative-return games and near-zero
teacher returns. Selection uses raw episodic return, never reward-scaled or
ARROW-normalized values. The final held-out evaluation cohort is not inspected
by the selector.

Structured pruning ranks hidden channels by incoming-norm times outgoing-norm
inside each atom and retains an equal number per atom. The selected rows and
columns are copied into newly allocated smaller Dense modules. A full-width
masked tensor or random projection is not retained. The four-atom route
interface therefore remains valid even when older tasks have different widths.

All candidates are attempted even when an earlier candidate passes. This makes
the extra optimization and validation budget independent of observed
performance. The full Dense teacher is discarded after the boundary. The
completed task's stale Dense Adam optimizer is also retired because its
Parameter objects no longer belong to the installed module.

## Budget and parameter consequences

The original-six run adds `6 * 4 * 250 = 6,000` compression optimizer updates
and `96,000` LTDM sequences. Together with `540,000` online updates and `6,000`
shared-consolidation updates, the exact world-model optimizer-step budget is
`552,000`. The selector performs `6 * 5 * 16 = 480` additional evaluation
rollouts. Evaluation transitions never enter Replay.

One task's Q/F/P parameter count at each candidate is:

| retained width | Q/F/P parameters |
|---|---:|
| Dense `512/512/256` | `3,816,192` |
| `384/384/192` | `2,865,600` |
| `256/256/128` | `1,915,008` |
| `128/128/64` | `964,416` |
| `64/64/32` | `489,120` |

Experiment A's six-task maximum remains `52,897,535` online parameters. The
outcome-dependent final model lies between that Dense fallback and
`32,935,103` parameters if all six tasks accept the smallest candidate, a
maximum reduction of `19,962,432` parameters (`37.74%`). Private MLP
Actor-Critics are intentionally not compressed in v1.

For attribution and deterministic initialization, v1 predeclares all six
full-width task slots exactly as experiment A does, then physically replaces
completed slots. Thus its maximum allocation is still the A topology, while
boundary checkpoints and the final inference model reflect actual selected
widths. Allocator-reserved GPU bytes need not fall immediately after a module
is replaced and must not be confused with live parameter count.

## Consequences and evidence required

- The method remains task-aware: task identity selects projector, Q/F/P route,
  replay slice, and private Actor-Critic.
- Different tasks may finish with different physical widths. Width buffers are
  part of every checkpoint and rebuild the module topology before strict tensor
  loading.
- A failed compression phase stops the run; it is not silently accepted. The
  pre-consolidation checkpoint and an explicit failure artifact remain.
- Candidate validation is model selection and must be reported separately from
  periodic validation and final held-out results.
- The clean control is experiment A under the same order, seed, online budget,
  Replay, and evaluation protocol, with the additional compression compute
  reported rather than hidden.
- A seed-0 run is pilot evidence. Any performance or compression claim requires
  matched multi-seed results, raw per-task returns, final average performance,
  forgetting, and valid transfer metrics.
