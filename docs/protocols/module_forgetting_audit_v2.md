# Input-Fixed Module Forgetting Audit (V2)

## Status and scope

This protocol supersedes the component-attribution interpretation of the P1
parameter-swap analysis.  It answers a narrower descriptive question:

> For a fixed old task and a fixed old-task coordinate system, how much does
> the output of each DreamerV3 functional module drift at every later task
> boundary?

It is an offline checkpoint-differencing protocol.  It does not retrain a
component, collect new environment interaction, insert data into replay, or
change any training budget.  The completed DreamerV3/FIFO P1 run remains a
single-seed pilot because its source training worktree was dirty at launch;
this protocol can make the analysis reproducible and statistically explicit,
but cannot turn that training run into an official multi-seed baseline.

The P1 frozen decoder swap is retained only as an exploratory readout test. It
is not used to claim that a decoder causes planning or control forgetting.

## Checkpoints and fixed audit data

For each task `T_i`, use the task-boundary snapshot `C_i` and the held-out
diagnostic set `D_i`.  Evaluate every later boundary snapshot:

```text
C_i, C_(i+1), ..., C_6
```

`Cfinal_e540` is excluded from continual-learning headline results because it
contains an additional Task 1 update after the sixth-task boundary.  All
measurements use deterministic categorical posterior mode and the exact same
chunks, actions, resets, rewards, continuations, and episode IDs from `D_i`.

For each chunk, `C_i` produces a frozen reference activation trace:

```text
e_i(t)       encoder/image-embedder output
z_i(t), h_i(t)  posterior RSSM state
s_i(t)       actor state formed from z_i(t), h_i(t)
u_i(t)       model-head state formed from z_i(t), h_i(t)
```

Every later module is evaluated on the matching `C_i` reference input.  This
is the essential control: a downstream module is not allowed to receive a
different current encoder/RSSM state and then be mislabeled as having changed
on its own.

## Primary module measurements

| Functional module | Frozen old-task input | Primary retention measurement | Interpretation |
| --- | --- | --- | --- |
| Encoder | Raw old observations `x(t)` | linear CKA and Procrustes residual of `e_i(t)` vs `e_j(t)` | Feature geometry/coordinate drift caused by the image encoder itself. |
| Posterior/representation | `e_i(t), h_i(t)` | symmetric KL of posterior categorical distributions | Drift of the observation-to-latent recognition mapping under the old coordinate system. |
| RSSM recurrent transition | `z_i(t-1), h_i(t-1), a(t), reset(t)` | normalized hidden-state RMSE | Drift of deterministic state transition under the old coordinate system. |
| RSSM prior | resulting reference transition state | symmetric KL of prior categorical distributions | Drift of the one-step latent prior. |
| Reward head | `u_i(t)` | output drift plus fixed old reward error | Reward-readout retention, not environment return. |
| Continue head | `u_i(t)` | probability drift, BCE, Brier score | Continuation-readout retention. Terminal discrimination needs the event subset. |
| Actor head | `s_i(t)` | symmetric action KL and top-1 disagreement | Policy-head drift on exactly the same old state. |
| Critic head | `s_i(t)` | critic-distribution KL, value drift, anchored historical-return error | Value-head drift on exactly the same old state. |

The decoder is not a primary forgetting target.  It may later be used as a
*frozen downstream probe* of whether encoder/RSSM features still contain old
visual information, but a changing decoder output must never be reported as
the encoder's direct forgetting score.

## Reporting rules

For task `i`, checkpoint `j`, and metric `m`, preserve absolute values and
the boundary-relative difference:

```text
Delta(i, j, m) = metric(C_j, D_i) - metric(C_i, D_i)
```

For similarity/agreement metrics, render the corresponding loss of similarity
or agreement so positive values always mean less retention.  Do not combine
pixel MSE, categorical KL, and policy KL into an invented universal scalar.
The headline result is a per-task, per-checkpoint **forgetting profile**, with
each module reported in its native, interpretable units.

Use an episode-cluster bootstrap for per-chunk metrics.  Linear CKA is also
reported as a global geometry statistic; its per-chunk CKA values provide a
clustered uncertainty interval.  Full raw per-chunk arrays, input checksums,
script hash, snapshot hashes, command line, and resolved metric definitions
must accompany every report.

## What the audit can and cannot establish

The audit estimates *where and how strongly module outputs drift* over a
continual-learning trajectory.  It does not by itself prove which module
causes a final-return decline, nor does it demonstrate that protecting a
module during training improves retention.  Those are separate causal
intervention experiments that should be motivated only after this descriptive
audit is complete.
