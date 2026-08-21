# KARROW-InputAligned-v4 Atari protocol

## Status and question

This is an implementation-ready experimental protocol with no result claim.
It preserves the ARROW-50 curriculum, replay capacity and sampling, interaction
budget, world-model updates, actor-critic updates, evaluation schedule, and v2
frozen spatial-DINO observation path.

The primary question is whether an input-aligned KAN branch has enough Task-2
plasticity after the base is frozen, while retaining Task-1 behavior better than
an exactly matched MLP branch. V1-v3 remain selectable with their historical
`base_output` residual topology.

## Parallel correction topology

For module `m`, v4 computes

```text
y_m = F_m(x_m) + R_m(x_m)
R_m(x) = alpha * U_m * SiLU(K_m(RMSNorm(D_m x)))
alpha = 0.1
```

`D_m` projects the module input to 64 dimensions. `K_m` is either the
fixed-grid Gaussian-RBF KAN core or its exactly core-parameter-matched MLP
control. `U_m` maps back to the corrected output space and is initialized to
zero. Each placement owns an independent branch.

The inputs and output spaces are:

| Module | Residual input | Residual output |
| --- | --- | --- |
| recurrent dynamics | `[flatten(z_(t-1)), a_(t-1), h_(t-1)]` | `delta h_t` |
| posterior | `[e_t, h_t]` | posterior-logit delta |
| latent prior | `h_t` | prior-logit delta |
| feature predictor | `[flatten(z_t), h_t]` | DINO-feature delta |
| reward | `[flatten(z_t), h_t]` | reward delta |
| continuation | `[flatten(z_t), h_t]` | continuation-logit delta |
| actor | `[flatten(z_t), h_t]` | action-logit delta |
| critic | `[flatten(z_t), h_t]` | value-logit delta |

This differs from merely applying `KAN(F_m(x_m))`: the residual sees the state
variables available before the frozen base transformation. It can therefore
represent a new correction even when a Task-1 trunk omits a later-task
distinction.

## Task-1 acquisition and boundary

From the first Task-1 update, the base RSSM, observation/reward/continuation
heads, actor-critic MLPs, and all residual branches are trainable. KAN is not
inserted after Task 1 and is never reset. Because every `U_m` starts at zero,
the initial network is exactly the corresponding DINO/base model. The small
fixed `alpha` makes the residual contribution grow gradually rather than
replacing the original learning path at initialization. Residual construction
uses a private RNG fork, preserving same-seed base parameters and the global
training RNG state used by the matched control.

At the first sequential task boundary, the trainer freezes the base RSSM,
feature/reward/continuation heads, and actor/critic MLPs and removes them from
the optimizers. The complete residual branches, including input/output
projections and KAN or MLP core, remain trainable with preserved optimizer
state. DINO and the Task-1 PCA projection are frozen from their original fit.

## Inference

Inference always evaluates `F_m(x_m) + R_m(x_m)`. There is one branch per
module for the entire curriculum. No task identity, task-specific parameters,
router, checkpoint selection, grid expansion, or explicit task detector is
used. Any task-selective behavior must arise from different model states
activating different local KAN support.

## Controlled arms

| Variant | Method | Residual |
| --- | --- | --- |
| `dino` | `ARROW-DINOSpatial-v4Control-50` | none; trainable base throughout |
| `mlp` | `ARROW-DINOSpatial-InputAligned-MLPRes-50` | input-aligned matched MLP |
| `kan` | `KARROW-InputAligned-50` | input-aligned Gaussian-RBF KAN |

V4 deliberately disables v3 replay-functional consolidation. The first
comparison isolates residual topology and KAN locality from coefficient
protection. KAN versus MLP must use identical branch placements, 64-dimensional
bottlenecks, output initialization, alpha, and budgets. Complete parameter and
byte counts must be reported because module-input adapters have wider input
projections than v1-v3.

## Required evidence

1. Verify exact zero-initialization parity with the base outputs and confirm
   nonzero Task-1 gradients reach both base and residual parameters.
2. Establish Task-1 acquisition against original ARROW-50 and the DINO-only
   control before interpreting continual retention.
3. Report current-task return, final average return, per-task forgetting, and
   raw per-task evaluation returns for KAN and matched MLP arms.
4. At fixed old replay states, report RSSM transition drift, actor-distribution
   drift, action sensitivity, residual magnitude, and KAN support overlap.
5. Treat a failure to acquire Task 2 with the frozen base as a plasticity
   failure; do not silently unfreeze modules or add capacity under this name.

## Dry run

```bash
python scripts/run_karrow_input_aligned_ar50_atari.py \
  --variant kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dinov3-model-path /absolute/path/to/dinov3-vits16-pretrain-lvd1689m \
  --dry-run
```

Run the one-task DINO, MLP, and KAN acquisition screens before a full continual
campaign. A smoke run establishes execution only; multiple seeds are required
for a method claim.
