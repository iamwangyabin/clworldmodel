# CapacityOrganization-v1-Atari (task-aware, single-seed pilot)

## Status and scope

Prospective protocol authorized by Decision 0064. Five controls use the common
Atari ARROW/Dreamer trainer and injected `WorldModel` factory. The project
adapter only owns routes/parameters and calls the pinned `compute_loss`, RSSM
initial-state/forward, reward/continue, decoder, and activation APIs. No copied
RSSM, trainer, retired method or alternative Dreamer loss is introduced.

| Control key | World-model organization |
| --- | --- |
| shared | One standard WM; all WM parameters remain plastic. |
| wide | One WM with MLP hidden width 1536 (base 512), unchanged GRU/CNN/latent and AC dimensions. Actual counts must be reported; this is a wider control, not an unverified parameter match. |
| fullbank | Six full, storage-independent WMs allocated upfront from identical **untrained** initialization. Only the current WM is plastic; never clone a previous trained WM. |
| frozen | One shared CNN/RSSM/prediction core with symmetric per-task image projectors and four dense-private atoms per recurrent/representation/transition component. Historical atom reuse is enabled. After Task 0, all shared parameters including prediction heads are frozen; only current private atoms/projector/routes are plastic. |
| independent | Same residual organization but historical atom reuse disabled, shared parameters remain plastic. |

Residual widths are 512/512/256, residual scale 0.1 and image projector bottleneck
64, as declared by the typed config. New residuals/projectors reset through the
retained initializer, not a trained-weight copy. Old private tensors are frozen.
The named mode resolves these architecture switches; irrelevant historical
AWM config fields do not enable protection or compression in this protocol.
Record total, trainable, allocated and retained WM counts plus private AC counts;
allocated upfront modules must not be described as acquired knowledge.

## Shared protocol

- S0 = 123456789; six tasks: MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest,
  Enduro, in that fixed order, 90 epochs each (540 total; no wraparound epoch).
- Unchanged source RGB64 preprocessing, 18-action handling, repeat 4, reward
  scales, reset/termination behavior and model losses.
- 16,384 decisions/epoch: 8,847,360 decisions and 35,389,440 raw frames total.
- 1,000 WM and 800 AC updates/epoch: 540,000 WM and 432,000 AC updates.
  WM Adam LR 1e-4, batch time32, batch size16; BF16 compute, FP32 parameters,
  fused Adam/TF32 enabled, compilation disabled. AC uses the unchanged config.
- Per-task oracle WM route and independent oracle private AC for collection,
  imagination and deterministic evaluation. No task-agnostic claim.
- Task 0 WM update: 16 sequences from current task. Later WM updates: 12
  current-task sequences from mixed replay plus 4 sequences from one uniformly
  selected previous task's LTDM. Sum of mean Dreamer current and old losses,
  memory weight1. Sampling and optimizer-step count are common across controls.
  Isolated/frozen old losses may be constant with respect to plastic parameters;
  do not unfreeze old parameters simply to force a replay gradient. Current
  private AC alone gets all 800 steps, using current-task replay contexts.
- ARROW-50: 512 FIFO + 512 unbiased LTDM trajectory slots, time512, equal buffer
  selection probability where both contain the task. uint8 CPU observations
  (6,442,450,944 bytes), other replay tensors FP32. Other tensors add 44,040,192
  bytes; task-id metadata/index overhead is reported separately by runtime
  accounting. No GPU replay, compression, mmap or disk sampling in this profile.
- Fixed validation seeds every ten epochs, all seen tasks only; additionally
  evaluate all seen tasks after each boundary's last update. Held-out fixed
  final evaluation covers all six tasks. Nominal 16 rollouts/task under the
  retained legacy episode-count mode (not falsely exactly16). Preserve raw
  return means/stds and boundary matrices for final average and forgetting.
  Evaluation never writes replay, steps optimizers, or advances training RNG.
  No forward-transfer claim: future oracle actors are not yet initialized.

## Fairness limits

These five have matched interaction, sequence-capacity, online step and evaluation
budgets **with each other**, not equal FLOPs or parameter counts. Wider/shared
and retained residual/FullBank counts must be measured, never assumed matched.
They omit functional protection, gradient projection, boundary consolidation
and RCC. AWM uses 12,000 extra boundary WM updates. Frozen and independent are
organization controls, not exact one-switch AWM ablations; this campaign alone
cannot support a compute-matched AWM superiority claim. Baseline float32 replay
has four times the observation bytes and must be disclosed separately.

## Execution and artifacts

Use `scripts/run_capacity_control_atari.py --control KEY --seed 0 --python
/PINNED/PYTHON --output-dir /PERSISTENT/RUN`. Before every launch, fetch upstream,
verify a clean pushed 0-ahead/0-behind commit, and save controller/target Git
verification, dependency/runtime and accelerator evidence. Disconnected targets
may fetch a SHA-verified bundle freshly created after the controller's upstream
verification; preserve the bundle digest and GitHub upstream identity.

The launcher refuses an existing output directory and saves resolved config,
provenance, versions, budgets, logs and exit status. Trainer writes raw metrics,
model/AC accounting, boundary evaluations, held-out final evaluation, and atomic
SHA-indexed inference snapshots of WM and private actors. These snapshots omit
replay/optimizers/RNG and **are not resumable**. Failures remain failed runs; no
automatic retry or equivalent continuation. No performance claim from smoke/S0.

Validation gate: typed-config rejection, all five two-task CPU WM/imagined-AC
updates, storage/frozen ownership, model/optimizer serialization, retained
baseline/AWM parity, and production-width target GPU two-task smoke before each
pilot. Record failed attempts without overwriting them.
