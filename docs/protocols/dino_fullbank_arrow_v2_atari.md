# DINO-FullBank-ARROW-v2 Atari Task-Aware Protocol

## Status and claim boundary

`DINO-FullBank-ARROW-v2-Atari-TaskAware` is an implementation-ready correction
to the failed MoE-ARROW-v1 acquisition path. Its code identifier is
`dino_fullbank_arrow` and its reported ARROW-50 name is
`DINO-FullBank-ARROW-50`. No performance or forgetting claim exists until the
target-GPU smoke and matched runs complete.

The scheduler exposes task identity to the agent. Results therefore belong in
a task-aware table and must not be presented as a direct task-agnostic ARROW-50
improvement. V1 remains available through `scripts/run_moe_arrow_atari.py`.

## Frozen observation target

Each `64 x 64` RGB observation is bicubic-resized to `256 x 256`, normalized for
ImageNet, and encoded by a local frozen DINOv3 ViT-S/16. CLS and register tokens
are discarded. The final `16 x 16 x 384` patch grid is average-pooled to
`4 x 4 x 384`. A private generator with seed 0 creates a fixed orthogonal
`384 x 64` channel projection. No game, replay sample, or Task-1 fit changes
that projection. The flattened stopped target has 1,024 coordinates:

```text
e_t = stop_gradient(flatten(pool_4x4(DINO_patches(x_t)) P)).
```

Targets are cached as float16 using the exact FIFO/LTDM write slots. They are
converted to float32 for loss computation. There is no pixel decoder.

## Complete task expert

For task `k`, one complete expert owns the posterior and dynamics path:

```text
h_t       = Recurrent_k(z_(t-1), a_(t-1), h_(t-1))
q_k(z_t)  = Representation_k(e_t, h_t)
p_k(z_t)  = Transition_k(h_t)
s_t       = concat(z_t, h_t)
e_hat_t   = FeatureHead_k(s_t)
r_hat_t   = RewardHead_k(s_t)
c_hat_t   = ContinueHead_k(s_t).
```

The observation loss uses every current posterior position, including first
and reset positions:

```text
L_obs = SmoothL1(
  standardize_TN(e_hat),
  standardize_TN(e)
).
```

Standardization is independent for prediction and target, per feature
coordinate, with a standard-deviation floor of 0.05. The trainer logs
`Metric/dinov3_constant_feature_loss` and
`Metric/dinov3_model_to_constant_ratio`. The original Dreamer dynamics and
representation KL terms still align the prior to the now-grounded posterior.
There is no separate prior feature loss.

Frozen DINOv3 is the only shared neural module. `ZhToModelState` performs only a
parameter-free concatenation. The protocol does not use KAN, LoRA, residual
adapters, a learned router, or a pixel VAE.

## Initialization and plasticity

All configured task modules exist in one fixed bank, but only one route is
active. During Task 1, only expert 0 and Actor-Critic 0 receive gradients. Future
experts are dormant, not ensembled.

When task `k>0` first arrives:

1. Copy the complete world-model state of expert `k-1` into expert `k` once.
2. Keep fresh Adam state for the newly active world-model parameters.
3. Construct Actor-Critic `k` from its own deterministic seed without copying
   Actor-Critic `k-1` or its return statistics.
4. Freeze every old world-model expert and Actor-Critic.
5. Use a random policy for the first collection on the new task.

Subsequent epochs select the existing task entry without copying again. Task ID
is a scalar router input and is never concatenated to DINO features, latent
state, action, or reward.

## Replay and fixed budgets

The protocol preserves ARROW-50's 512 FIFO and 512 LTDM trajectory slots and
nominal 0.5/0.5 buffer selection. Each trajectory slot stores one int64 task ID.
Sampling is conditional on the current task and remains uniform over eligible
sequences; buffer weights are renormalized only when one sub-buffer lacks the
selected task.

For each epoch with world-model budget `U_wm` and Actor-Critic budget `U_ac`:

```text
updates_wm(k_current) = U_wm
updates_ac(k_current) = U_ac
updates_wm(k_old)     = 0
updates_ac(k_old)     = 0.
```

Thus `steps_per_batch`, `ac_train_steps`, task duration, and total gradient
updates are unchanged. The original Atari source config has
`pretrain_enabled=false`, so random first collections at later task boundaries
do not multiply environment steps. The launch manifest records the actual
decision and raw-frame totals rather than assuming this.

Replay pixels remain float32 on CPU. The frozen feature cache uses 1,024
float16 coordinates per slot, and task IDs use int64. Base replay bytes,
feature-cache bytes, task-ID bytes, model parameters, and per-task Actor-Critic
parameters must all be reported.

## Evaluation and artifacts

Periodic and final evaluation use modal categorical latents and deterministic
argmax actions. Each task selects its matching frozen world-model expert and
Actor-Critic. Evaluation transitions never enter replay and training RNG state
is restored afterward.

Final artifacts include the resolved config, launch manifest, raw metrics,
complete world-model bank, and all Actor-Critic inference states. They are
explicitly non-resumable because replay, every optimizer, scheduler position,
and all RNG states are not serialized together.

## Controlled execution order

1. Run a short GPU smoke and verify finite posterior feature, KL, reward, and
   Actor-Critic losses plus exact selected-expert gradients.
2. Run the 90-epoch MsPacman prefix and compare raw return with the historical
   ARROW-50 and failed V1 diagnostics.
3. Require the learned predictor to beat its logged constant baseline; a small
   raw feature loss alone is insufficient.
4. Start the two-task MsPacman-to-Boxing pilot only after Task-1 acquisition is
   credible.
5. Run multiple seeds before making a method claim.

Inspect the first-task acquisition configuration without creating a run:

```bash
export DINOV3_MODEL_PATH=/absolute/path/to/dinov3-vits16-pretrain-lvd1689m
python scripts/run_moe_arrow_atari.py \
  --method dino-fullbank \
  --seed 0 \
  --task-prefix-length 1 \
  --dry-run
```

Remove `--dry-run` only from a clean commit already pushed to its configured
upstream, supply a new persistent output directory, and first run the target-GPU
smoke. A fresh-world-model initialization arm is a separate ablation, not a
silent v2 change.
