# KARROW-FrozenCore-v1 Atari protocol

## Status and question

The seed-0 original-order two-task pilot completed successfully as an execution
run and produced a negative method result. At the Task 1 boundary it reached
`521.875 +/- 74.265` raw MsPacman return. After Task 2 it reached
`464.375 +/- 76.809` on MsPacman and `-32.0625 +/- 23.7315` on Boxing over 16
stochastic rollouts. This does not support a continual-learning claim.

The failure begins before the shared core is frozen. The CLS cosine feature
loss fell to about `0.004` while a constant feature predictor is already near
that scale, and most posterior KL values remained below the free-bits threshold.
Decision 0007 therefore preserves this protocol for reproduction and introduces
the separately named spatial posterior-feature v2 correction.

KARROW asks whether local, fixed-capacity KAN residual updates reduce RSSM and
policy interference relative to an exactly parameter-matched MLP residual, once
visual representation drift has been removed and the shared core is no longer
allowed to drift after task 1.

## Controlled arms

All arms use the canonical ARROW-50 Atari task order, FIFO/LTDM capacities and
50/50 whole-minibatch selection, interaction budget, world-model updates,
actor-critic updates, action handling, reward scaling, and evaluation schedule.

| Arm | Visual model and objective | Corrections |
| --- | --- | --- |
| `ARROW-50` | learned CNN plus pixel reconstruction | none |
| `ARROW-DINO-50` | frozen DINOv3 plus feature prediction | none |
| `ARROW-DINO-FrozenCore-MLPRes-50` | frozen DINOv3 plus feature prediction | matched MLP residuals, frozen core |
| `KARROW-FrozenCore-50` | frozen DINOv3 plus feature prediction | fixed-grid KAN residuals, frozen core |

The last two arms use the same Frozen-Core schedule and differ only in the
residual core. The DINO-only arm separates the effect of freezing visual
representations from residual capacity.

## Frozen observation path

The encoder is `facebook/dinov3-vits16-pretrain-lvd1689m`, loaded only from an
explicit local Hugging Face model directory. Its parameters remain frozen and in
evaluation mode. A `64 x 64` RGB observation in `[0, 1]` is resized to `256 x 256`
with bicubic antialiasing and normalized with ImageNet mean and standard
deviation. The 384-dimensional CLS feature is the RSSM observation embedding.

There is no pixel decoder. For aligned action `a_(t-1)` and observation `x_t`, the
posterior consumes frozen feature `e_t`, while the one-step prior state predicts
the same stop-gradient target:

```text
h_t       = dynamics(z_(t-1), a_(t-1), h_(t-1))
p(z_t)    = transition(h_t)
e_hat_t   = Linear(concat(E[p(z_t)], h_t))
L_feature = mean(1 - cosine(e_hat_t, stop_gradient(e_t)))
```

The first sequence position and reset positions are excluded because they lack a
valid within-sequence predecessor. KL, reward, and continuation losses remain
unchanged.

Frozen features are encoded once after collection and stored as float16 sidecar
tensors on the same devices and at the same slots as each ARROW replay buffer.
World-model batches and actor imagination context gather the sidecar with the
existing sampled buffer, time, and sequence indices. For ARROW-50 this adds
402,653,184 bytes: 201,326,592 bytes each for FIFO and LTDM. The original replay
tensors and their 25,813,843,968 allocated bytes remain unchanged.

## Residual architecture

The standard ARROW `nn.GRUCell` is not replaced or manually reimplemented. The
shared core is retained as the task-1 base, while residuals provide the only
plastic path after the first task:

```text
h_base = GRUCell(MLP(concat(z_(t-1), a_(t-1))), h_(t-1))
h_t    = h_base + correction_dyn(h_base)

post_logits = post_base(e_t, h_t) + correction_post(post_base(...))
prior_logits = prior_base(h_t) + correction_prior(prior_base(...))
reward_symlog = reward_base(zh_t) + correction_reward(zh_t)
continue = sigmoid(continue_base_logits(zh_t) + correction_continue(zh_t))
feature_hat = feature_base(zh_t) + correction_feature(zh_t)
```

The original actor and critic MLP trunks produce 512-dimensional features. Their
original final linear layers remain, with corrections added before log-softmax:

```text
actor_logits  = actor_base(phi_actor) + correction_actor(phi_actor)
critic_logits = critic_base(phi_critic) + correction_critic(phi_critic)
```

Each independent correction is:

```text
alpha * Linear_zero(SiLU(Core(RMSNorm(Linear(input)))))
```

`alpha=0.1`, the bottleneck width is 64, and the output linear is initialized to
zero. Consequently every arm begins with exactly the base RSSM, reward/continue
heads, actor, and critic function under a shared seed.

The KAN core has weights of shape `64 x 64 x 8` over eight fixed Gaussian centers
uniformly spaced in `[-2, 2]`; it has no global base branch and no trainable grid.
It contains 32,768 parameters. The matched control is a bias-free
`64 -> 256 -> 64` SiLU MLP, also with exactly 32,768 parameters. Thus all three
complete correction pairs are exactly parameter matched at each placement.

## Continual constraints

- Task 1 trains the shared base and one residual adapter at every listed
  placement together.
- At the first sequential boundary, after task 1 completes, the shared RSSM
  input MLP, GRUCell, posterior and prior MLPs, feature predictor base,
  reward/continue bases, and actor/critic MLP trunks and output layers are
  frozen. The DINOv3 encoder is frozen from initialization.
- From task 2 onward, the same single residual set is the only trainable path.
  The optimizer drops frozen parameters and keeps adapter optimizer state.
- No correction reset, grid extension, task-specific module, task ID, or router is
  permitted.
- No latent consistency, distillation, or per-knot protection is part of v1.
- DINO artifacts are external, immutable run inputs. Every file, size, and SHA-256
  digest is recorded in `launch.json`; no network download occurs during launch.

## Measurements

Report final average raw return and per-task forgetting from preserved evaluation
matrices. Also evaluate fixed diagnostic inputs at task boundaries:

1. RSSM transition drift on old `(h, z, a)` tuples.
2. Actor distribution drift on old latent states.
3. Weighted overlap of KAN Gaussian-basis activations between task datasets.
4. Parameter counts, replay bytes, feature-cache bytes, and wall-clock throughput.

Support overlap is diagnostic, not proof of causality. The primary comparison is
KAN residual versus matched MLP residual under the same seed and budget.

## Launch

Install the optional local-model runtime and make the DINOv3 artifact available:

```bash
python -m pip install -e '.[dinov3]'
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
```

Inspect a two-task seed-0 pilot without starting it:

```bash
python scripts/run_karrow_ar50_atari.py \
  --variant kan \
  --task-prefix-length 2 \
  --seed 0 \
  --dry-run
```

Run the DINO-only and matched-MLP controls by changing `--variant` to `dino` and
`mlp`. A real launch requires a clean commit already pushed and synchronized with
its configured upstream. Run a short target-GPU smoke before the 90- or 180-epoch
pilot; a task-prefix pilot is not an official continual result.

## External model

DINOv3 source and weights are not vendored. They are distributed separately by
Meta under the DINOv3 License. The launcher records the local artifact rather
than assuming that a model name uniquely identifies its bytes.
