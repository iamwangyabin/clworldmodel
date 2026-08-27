# 0008: Replay-Consolidated Incremental KAN

Status: experimental

## Decision

Add a separately named KARROW v3 method that turns each fixed-grid KAN
residual into an incremental continual-learning module after Task 1.

Task 1 trains the existing spatial-DINO KARROW v2 model normally. At the first
task boundary, the shared RSSM and actor-critic bases are frozen as in v2. The
KAN input projection, RMS normalization, and output projection are also frozen,
leaving only the Gaussian RBF coefficients trainable. This fixes the coordinate
system in which locality is interpreted.

Before each new task starts, sample the unchanged ARROW replay mixture and run
deterministic posterior inference plus short deterministic imagination. For a
residual

```text
r(x) = alpha * U * SiLU(C * phi(q(x))),
```

estimate each coefficient's squared local output-Jacobian importance:

```text
I[o,i,k] = E[alpha^2 * ||U[:,o]||^2 * SiLU'(v[o])^2 * phi[i,k]^2].
```

Normalize within each adapter by its positive 99th percentile. Preserve the
coefficient-wise maximum importance across boundaries. An importance-dependent
gradient scale protects used coefficients, and the realized post-Adam parameter
delta is scaled by the same factor so adaptive normalization cannot cancel the
protection. A quadratic anchor loss further limits drift. Coefficients with low
replay importance remain plastic.

The importance pass performs no environment interaction and no gradient update.
Python, NumPy, and Torch RNG states are restored afterward, so replay sampling
for subsequent training is unchanged. Scheduler task boundaries trigger the
pass, but task identity is not passed to the world model, actor, critic, or KAN.

## Inference

Inference uses exactly one fixed-capacity residual per module. There is no task
ID, router, task-specific adapter, grid expansion, or checkpoint selection. The
current frozen DINO/RSSM state selects RBF support through ordinary Gaussian
basis activation.

## Evidence Required

The method must be compared against spatial KARROW v2 without consolidation,
the parameter-matched MLP correction, and ARROW/DINO controls under unchanged
interaction, replay, and gradient-update budgets. The latent-region audit is a
mechanism diagnostic, not a substitute for continual return and forgetting.
