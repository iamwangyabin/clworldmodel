# Decision 0010: use input-aligned parallel residuals

- Status: accepted for implementation, before training
- Date: 2026-08-21

## Context

KARROW v1-v3 attach several residuals to a base module's internal output. In
particular, actor and critic residuals consume the frozen MLP trunk feature, and
the recurrent correction consumes the new GRU hidden state. After Task 1, those
frozen transformations can discard distinctions needed by a later task. A
residual downstream of that bottleneck cannot reconstruct information it never
receives, regardless of whether its core is KAN or MLP. This confounds KAN
plasticity with the adequacy of frozen base features.

## Decision

Preserve v1-v3 and add the separately named `KARROW-InputAligned-v4` protocol.
For every corrected function, feed the residual branch the corresponding base
module input and make it predict a residual in that module's output space:

```text
y_m = F_m(x_m) + alpha * R_m(x_m).
```

The recurrent branch receives `[z_(t-1), a_(t-1), h_(t-1)]`; posterior receives
`[e_t, h_t]`; prior receives `h_t`; reward, continuation, feature prediction,
actor, and critic receive the full model state `[z_t, h_t]`. Branches remain
independent and fixed-capacity.

Task 1 trains the original base and every residual jointly. Residual output
projections are initialized to zero and `alpha=0.1`, giving exact base-model
forward parity at initialization while allowing residual adaptation from Task
1. V4 constructs residual parameters in private RNG forks, so adding them does
not change same-seed base initialization or the subsequent training RNG state.
At the first task boundary, base functions freeze and complete residual
branches remain trainable. V4 does not inherit v3 coefficient consolidation.

## Consequences

- Later tasks can form a new output correction from the full module state
  instead of relying on a frozen trunk's chosen features.
- Inference remains task-agnostic: both branches always execute and their
  outputs are added. There is no task ID, router, expansion, or adapter choice.
- This topology does not itself guarantee non-interference. The KAN claim still
  requires a parameter-matched MLP residual control and support-overlap/drift
  diagnostics.
- Actor/critic and RSSM adapter input projections become wider, but KAN and MLP
  arms remain matched at every placement. Parameter and byte accounting must
  report the complete adapters, not only their 32,768-parameter cores.
