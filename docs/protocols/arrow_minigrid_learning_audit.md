# ARROW MiniGrid first-task learning audit

Protocol: `ARROW-50-MiniGrid-DoorKey-AutoresetAudit-v1`

## Purpose and evidence status

This is a **pilot diagnostic**, not a replacement for the recorded
`ARROW-DV3RS-MiniGrid-3Task-v1` campaign or a paper reproduction. The historical
campaign ran from project commit
`2e9b7f6ff5ece5d3fccc27b1535d2d309097c72b`. Preserve its artifacts unchanged.
Completing the old runs did not validate their environment/collector semantics.

The confirmed implementation defect is an autoreset contract mismatch:

- Gymnasium 1.1.1 `AsyncVectorEnv` defaults to `NEXT_STEP`: a completed step
  returns the terminal observation; the following call resets without executing
  its supplied action.
- ARROW's collector assumes a completed step already returns the next episode's
  initial observation. It flags that row as a reset and moves the reward and
  continuation terminal to the preceding row.
- Consequently the historical collector resets RSSM memory on the terminal
  observation, then consumes the actual initial observation without a reset.
  The same collector is used for evaluation.

The synthetic fixture tests replay the exact observation/reset trace through the
real collector. They establish the defect and the corrected flag alignment,
**not its causal contribution to low returns**. The old training event streams
also contain first-task runs with very few positive-reward collection batches;
therefore later-task forgetting alone cannot explain their poor performance.
The old `Perf/rews_eps_mean` is a batch reward/reset-count proxy, not an exact
completed-episode return or success rate.

## Isolated change and invariants

Set `collection_autoreset_mode="same_step"` for both collection and evaluation.
The legacy default is explicitly named `legacy_next_step`; it remains available
for fixtures and diagnostic comparisons and is not silently redefined.

Do not change the model, losses, actor/critic targets, entropy, replay sampling,
reward transformation, image preprocessing, action space, task geometry or
episode limit. The diagnostic uses only the first task, actual
`DoorKeyEnv(size=9)`, with partial RGB resized to 64x64, no mission/direction
input, seven native actions, repeat one, and the historical 100-step limit.
It supplies no task label, demonstration, action mask or exploration bonus.

The following historical deviations are **not fixed in this pilot**:

- omission of terminal observations and previous-row terminal reward assignment;
- treating time limits as continuation terminals;
- four-frame context for imagination and historical actor-critic targets;
- scalar symlog-MSE reward head rather than claiming exact canonical DreamerV3;
- deterministic action argmax **and latent-mode** evaluation.

These are follow-up hypotheses, not established causes. If this isolated fix
does not restore acquisition, use the new diagnostics before changing another
factor. Increasing update ratio or entropy is a separately named ablation, not
a repair retroactively applied to the formal v1 results.

## Predeclared pilot budget

Use seed index zero (`123456789`), selected by campaign order rather than by
observed success. There is no automatic ten-run restart and no performance-based
seed selection. Formal claims still require all five predeclared seeds.

- 750,000 **stored rows**, including 3,000 initial-reset rows, in 741 epochs.
- 747,000 actually executed training actions in SAME_STEP mode.
- Random prefill: ten collections of 1,000 rows / 996 executed actions each.
- Subsequent collection: 1,000 rows / 996 actions per epoch.
- 610 initial and 61 subsequent world-model updates: 45,750 total, batch 16x32.
- 49 actor-critic updates per epoch: 36,309 total.
- ARROW-50 whole-minibatch selection; FIFO and LTDM each have 20,000 slots of
  50 rows. Total capacity remains 2,000,000 rows.
- CPU uint8 replay remains 24,656,000,000 tensor bytes, excluding allocator and
  indexing overhead. Sampled images become float32. No GPU replay allocation.
- One selected GPU; unchanged TF32/fused-Adam runtime settings.

The old v1's fields called “environment decisions” counted stored rows, including
initial resets and ignored NEXT_STEP actions. Historical true interaction counts
cannot be reconstructed exactly from its sparse batch logging. Do not claim
this pilot has an exactly matched **executed-action** budget against those runs.
Its row/update/capacity budgets match their first task, while the action-count
difference caused by the repair is explicit. Diagnostic evaluation and snapshot
training counters now count executed actions, not reset rows.

Evaluation remains separate from replay/updates. It uses the existing fixed
16-rollout validation cohort after prefill and every ten regular epochs
(9,960 executed actions), plus the boundary and a separate held-out final
cohort. Only DoorKey is evaluated; there is no future task in this diagnostic.
Snapshots are analysis-only and do not contain replay; they are not resumable.

## Diagnostic data flow

Enable `learning_diagnostics=true`. Diagnostics do not use random sampling,
perform an additional model pass, change losses, or feed information to agents.

1. `collection_diagnostics.jsonl`: all collection calls, including all prefill
   calls; actual/ignored actions, raw positive reward events, executed-action
   histograms, and completed episode returns. Exclude incomplete episodes from
   the completed-return list. This is not held-out evaluation accuracy.
2. `world_model_diagnostics.jsonl`: aggregate **every** world-model minibatch in
   each epoch. Count positive and zero reward targets and report conditional
   reward prediction means/errors. A missing positive subset has `null` error,
   never a misleading zero-error claim. Repeated sampled targets are not new
   environment experiences or distinct successful episodes.
3. TensorBoard actor/critic metrics include mean imagined reward and the fraction
   above zero, alongside the existing entropy and losses. Tiny positive model
   outputs are not a calibrated imagined success rate.
4. Existing raw taskwise evaluation artifacts retain fixed-cohort raw returns.

Interpretation: no real positive rewards points to exploration/collection;
positive collection but no sampled positives points to replay/data handling;
sampled positives with poor conditional reward prediction points to reward-model
learning; plausible reward prediction without policy improvement motivates an
actor/critic or train/evaluation-state audit. None of these alone proves a cause.

## Execution and verification

On 2026-09-05 the isolated repair working tree passed **28 CPU tests** in the
target server's existing PyTorch 2.3.0 / Gymnasium 1.1.1 runtime: 8 learning-audit
fixtures, 11 environment-seeding/evaluation contracts, 5 formal-campaign
config/dry-run contracts, and 4 uint8-replay parity tests. CUDA was hidden and
the collector used synthetic vector fixtures; no real environment interaction
or optimizer update occurred. This is not a GPU integration or learning result.
The repair has not yet been launched as a training experiment.

Dry-run (no environment interaction or parameter update):

```bash
python scripts/run_arrow_minigrid_learning_audit.py --dry-run
```

Pure CPU fixtures, schema, and dry-run regression checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p test_minigrid_learning_audit.py -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p test_environment_seeding.py -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p test_minigrid_formal_campaign.py -v
```

Launch only after scope approval and from a clean, pushed commit whose fetched
upstream is neither ahead nor behind. The launcher enforces that gate and saves
resolved config, dependency/runtime/GPU details, seeds, byte/update/interaction
budgets, command, and the full commit in `launch.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_arrow_minigrid_learning_audit.py \
  --seed-index 0 --variant same_step --nominal-rows 750000 \
  --evidence-level pilot --profile-stages --output-dir runs/arrow_doorkey_autoreset_v1_s0
```

A target-accelerator correctness check may use the same launcher with
`--nominal-rows 12000 --evidence-level smoke` (3 epochs, 732 world-model and
147 actor-critic updates), separately from the pilot budget. It establishes
execution only. It must not be reported as learned performance, and no later
official campaign may skip the required target correctness gate.
