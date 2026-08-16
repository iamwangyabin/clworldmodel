# ARROW-R2Rep-50 Atari representation-objective ablation

## Status

Implementation-ready, unrun ablation. A smoke run establishes execution only;
one seed is a pilot, and no retention or performance conclusion is valid until
the frozen multi-seed comparison is complete.

This protocol tests whether replacing pixel reconstruction with the
decoder-free self-supervised representation objective from R2-Dreamer reduces
encoder or latent-interface forgetting under ARROW replay. It is named
`ARROW-R2Rep-50`, not `R2-Dreamer`, because the ARROW world-model architecture,
KL scales, actor-critic, replay, and continual-learning protocol remain fixed.

## Hypothesis

The pixel decoder requires the recurrent latent to preserve detailed visual
information and adds a large trainable readout. Removing it may reduce
task-specific pixel pressure and parameter/activation cost. The R2 objective
instead asks the RSSM feature to predict the encoder embedding while preventing
the embedding target from receiving gradients through the target branch.

Decoder removal does not imply lower peak memory or wall time. The default
encoder width makes the cross-correlation matrix `4096 x 4096` (16,777,216
entries), and its matrix multiply and backward pass may offset decoder savings.
Parameter bytes, peak accelerator memory, and stage time must therefore be
measured rather than inferred.

This does not guarantee stable geometry. Both branches still use the current
encoder, and Barlow Twins constrains minibatch cross-correlation rather than
absolute feature coordinates. Fixed-input encoder, posterior, recurrent,
prior, actor, and critic audits remain necessary.

## Frozen comparison

The matched control is `ARROW-50` from `arrow_ar50_atari.md`. The following are
identical between the two methods:

- FIFO capacity: 512 trajectories x 512 time steps;
- LTDM capacity: 512 trajectories x 512 time steps;
- minibatch buffer selection: 0.5 FIFO and 0.5 LTDM;
- world-model minibatch: 32 time steps x 16 sequences;
- task order, task duration, seeds, environment interaction, update budgets,
  evaluation schedule, reward handling, and action handling;
- encoder, RSSM, reward head, continue head, actor, and critic architectures;
- KL dynamics scale 0.5 and representation scale 0.1.

The only intended behavioral change is the observation objective and its head:
the reconstruction decoder is absent and a bias-free linear R2 projector is
present. Replay observation dtype, capacity, device, and byte usage do not
change.

## R2 objective

For each replay minibatch, the encoder runs once. Let `e = encoder(o)` have
shape `[T, N, E]`, and let `f = concat(flatten(z), h)` have shape `[T, N, F]`.
The current Atari setting has `E = 4096` and `F = 1536`. A bias-free linear
projector maps `p = P(f)` from `F` to `E`. Time and batch are flattened to
`S = T * N` samples, while the target `stop_gradient(e)` is detached.

Each feature coordinate is centered and divided by its sample standard
deviation plus `1e-8`. With normalized matrices `p_hat` and `e_hat`,

```text
C = p_hat^T e_hat / S
L_invariance = sum_i (C_ii - 1)^2
L_redundancy = sum_{i != j} C_ij^2
L_R2 = 0.05 * (L_invariance + 5e-4 * L_redundancy)
```

No image augmentation is added. The complete world-model loss is the existing
scaled RSSM dynamics/representation KL plus `L_R2`, reward symlog MSE, and
continue BCE. Pixel reconstruction loss is absent.

The formula and defaults follow the official R2-Dreamer implementation:

- paper: `https://arxiv.org/abs/2603.18202`;
- repository: `https://github.com/NM512/r2dreamer`;
- inspected commit: `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`.

## Artifacts and accounting

The resolved `config.json` records the objective and every R2 coefficient.
`launch.json` records the method name, decoder status, exact R2 source commit,
and unchanged replay allocation. `model_parameter_accounting.json` records the
actual world-model and observation-head parameter counts and parameter bytes.
This accounting excludes gradients, optimizer state, and activations; runtime
peak memory must still be measured on the target accelerator.

Analysis snapshots contain the R2 projector instead of decoder weights. The
full visual component audit is undefined for these snapshots. Use fixed-input
encoder and latent/control-path audits for the matched forgetting comparison;
decoder reconstruction and visual-rollout metrics are reported as not
applicable, never as zero.

## Execution ladder

From the repository root, first inspect the frozen command:

```bash
python scripts/run_arrow_ar50_atari.py \
  --observation-objective r2 \
  --seed 0 \
  --dry-run
```

After the committed branch is pushed and the target GPU environment is
verified, run a short smoke job using an explicitly named smoke config. Do not
alter the official config in place. If the smoke passes, launch the seed-0
pilot:

```bash
python scripts/run_arrow_ar50_atari.py \
  --observation-objective r2 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_r2rep_ar50_original_s0_analysis
```

Compare against a matched `ARROW-50` run at the same seed and protocol. Preserve
all failed or interrupted runs. Five frozen seeds are required for an official
comparison; seed selection may not depend on observed outcomes.
