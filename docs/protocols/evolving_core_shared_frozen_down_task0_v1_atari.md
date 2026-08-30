# Evolving-Core shared-frozen-down Task-0 pilot v1

## Status

- Classification: `pilot`
- Method name: `Evolving-Core Shared Frozen Down + Private LayerNorm/FiLM/Up`
- Launcher: `scripts/run_evolving_shared_down_task0.py`
- Benchmark allocation: ARROW original-six routes, in the published order
- Training scope: Task 0 (`ALE/MsPacman-v5`) only, seed index 0, 90 epochs
- Claim scope: acquisition feasibility only; this is not a continual-learning result

The pilot is a controlled response to the negative compact-width result. The
compact `128/128/64` Q/F/P mechanisms reduced private storage but also reduced
Task-0 acquisition sharply. This protocol restores the original
`512/512/256` hidden widths while reducing repeated per-task matrices.

## Hypothesis

A full-width frozen random feature basis can preserve more of the nonlinear
mechanism's acquisition capacity than reducing its hidden dimension. A private
task readout and private feature-wise modulation may then adapt that basis with
substantially less persistent growth than a complete private residual MLP.

This is an empirical hypothesis. A frozen random basis is not assumed to be as
expressive as a learned private down projection.

## Mechanism

For each of the recurrent (F), posterior/representation (Q), and prior/
transition (P) banks, one down projection is allocated and frozen at its seeded
initialization:

\[
  D_b: \mathbb{R}^{d_{in}} \rightarrow \mathbb{R}^{d_h}.
\]

Each task `t` owns a LayerNorm, feature-wise scale and shift, and zero-effect
up projection:

\[
  m_{b,t}(x) = 0.1\,U_{b,t}\,\mathrm{SiLU}
  (\gamma_{b,t}\odot D_b\,\mathrm{LN}_{b,t}(x)+\beta_{b,t}).
\]

`U` weight and bias are initialized to zero, so every new task route has exactly
zero initial effect. The shared `D` matrices never receive gradients. They are
registered once per bank in checkpoints and parameter accounting; task modules
hold only a non-owning reference. Existing four-atom routing and consolidation
semantics are unchanged.

This parameterization is task-symmetric: Task 0 does not donate a learned
projection to later tasks. Every task receives the same type and amount of
private state.

## Exact capacity

The Q/F/P interfaces and hidden widths remain those of the dense control.

| Persistent mechanism state | Parameters |
| --- | ---: |
| Shared frozen down matrices, once | 2,753,792 |
| Private state per task | 1,064,960 |
| Six tasks of private state | 6,389,760 |
| Six-task atom-route gates | 180 |
| Six-task mechanism + route total | 9,143,732 |

The dense full-width control uses 3,816,192 private mechanism parameters per
task and 22,897,332 mechanism-plus-route parameters for six tasks. Thus this
variant reduces per-task mechanism growth by about 72.1% and the six-task total
by about 60.1%. It does **not** reduce the full-width forward matrix multiplies,
so the claim is parameter/storage efficiency rather than compute efficiency.

All registered parameters count toward storage whether trainable or frozen.
Optimizer-state accounting must separately reflect that the shared down
matrices are frozen.

## Controlled pilot

The resolved config is composed from the dense original-six Evolving-Core
control. Before the stop condition it differs in exactly two top-level fields:

1. `task_mechanism_parameterization=shared_frozen_down_film`;
2. `epochs=90`, to stop after Task 0.

It retains six task routes so Task-0 model allocation matches the later full
curriculum. Environment preprocessing, ARROW-50 replay, step/update budgets,
actor-critic, learning rates, fixed validation cohort, and consolidation are
unchanged. Held-out-final evaluation is deliberately not launched and cannot
be used for architecture selection.

The primary pilot value is the Task-0 pre-consolidation fixed-validation raw
return mean. Post-consolidation return, parameter accounting, memory, throughput,
and failure artifacts remain required diagnostics. The dense and compact
single-seed values are controls, not statistical conclusions.

## Promotion rule

Do not launch or claim a complete six-task result merely because this seed is
favorable. A full original-six run is justified only if Task-0 acquisition is
competitive with the dense full-width control under the fixed cohort. Any
continual claim still requires complete-task retention evaluation and
confirmation seeds.

