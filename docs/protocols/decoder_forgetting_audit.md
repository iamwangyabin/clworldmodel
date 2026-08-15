# Fixed-Input Decoder Forgetting Audit

## Scope

This is a supplementary, offline diagnostic for the completed DreamerV3/FIFO
P1 pilot. It measures decoder parameter drift only. It does not revise the
primary module-forgetting result, which intentionally excludes the decoder
from the planning and control path.

For old task `T_i`, the boundary checkpoint `C_i` produces a deterministic,
held-out trace:

```text
u_i(t) = zh_transform(z_i(t), h_i(t))
```

Both the old decoder and each later decoder receive the exact same `u_i(t)`.
The audit therefore compares:

```text
D_i(u_i(t))  versus  D_j(u_i(t))
```

and never reports a mixed current path such as `D_j(u_j(t))` as decoder-only
forgetting.

## Measurements

The held-out natural diagnostic chunks, action trace, reset flags, and episode
cluster bootstrap are identical to the input-fixed module audit. `Cfinal_e540`
is excluded by default because it follows one additional Task 1 update after
the sixth-task boundary.

| Measurement | Meaning |
| --- | --- |
| Decoder output normalized RMSE | Direct decoder-output drift relative to the scale of `D_i(u_i)`. |
| Decoder output pixel MSE | Raw image-space drift between `D_i(u_i)` and `D_j(u_i)`. |
| Reconstruction-target pixel MSE delta | Change in reconstruction error against the old observation while `u_i` stays fixed. |

All metrics are decoder readout diagnostics. They cannot establish that decoder
drift causes planning, policy, or return forgetting because actor-critic
optimization does not consume decoded pixels.

## Reporting limits

The P1 source launch had a dirty worktree and uses a single seed. Results are
pilot evidence only. They must not be used to rank pixel MSE numerically against
latent KL, actor KL, or return without a separate, justified normalization.
