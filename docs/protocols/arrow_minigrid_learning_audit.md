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

## Completed pilot result

The seed-0 SAME_STEP pilot completed on 2026-09-05 at project commit
`a9175dd1dbfe85b36d6ccd5d49f620b00e5c01bd`. It executed 747,000 training
actions and completed 6,000 training episodes. Collection contained **zero**
positive-reward events and zero positive-return episodes. Consequently all
23,424,000 sampled reward targets were zero. The final fixed held-out
evaluation return was `0.0 +/- 0.0` over 16 rollouts. Core losses remained
finite and the process exited successfully.

This is a negative acquisition result. It rules out the autoreset repair alone
as a sufficient fix. Because no real positive reward entered replay, it does
not diagnose reward-head fitting, reservoir retention, forgetting, or held-out
evaluation as the primary failure in this run.

## Post-run source-fidelity audit

The comparison premise used to define the historical MiniGrid campaign was too
strong. The following facts come from the published paper and the authors'
released code at `77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6`:

- The paper's MiniGrid Figure 1 compares DreamerV2 and DreamerV2 +
  Plan2Explore. It does not report the reservoir method named
  Continual-Dreamer as a MiniGrid curve; that name and replay comparison are
  introduced for the MiniHack experiments.
- In Figure 1, median plain DreamerV2 DoorKey success remains at zero through
  almost all of the first 0.75M-interaction task region and rises only after the
  curriculum switches to the second task. DreamerV2 + Plan2Explore is the curve
  that acquires DoorKey early during Task 1. Therefore zero at the end of a
  standalone 0.75M no-Plan2Explore screen is not, by itself, inconsistent with
  the published plot.
- The released `MiniGrid-DoorKey-9x9-v0` registration supplies no constructor
  arguments, while its vendored `DoorKeyEnv` defaults to `size=8`. The released
  executable task is thus 8x8 despite the 9x9 name and paper label. The
  historical project campaign instead deliberately constructed a literal 9x9
  task.
- The released plain-DreamerV2 command uses `--minlen=5`; the MiniGrid script
  sets actor entropy to `3e-3`, uses one collection environment, trains every
  ten actions, and resizes 56x56 images with nearest-neighbor interpolation.
  The historical project route uses four collection workers, fixed 50-row
  retention units, DreamerV3 entropy scale `3e-4`, a different update schedule,
  and OpenCV area interpolation.
- The released DreamerV2 time-limit wrapper keeps a timeout nonterminal for
  model discount learning and preserves the final observation. The current
  vendored ARROW collector treats timeout as a continuation terminal, drops the
  final observation, and moves terminal reward to the preceding stored row.

Accordingly, `ARROW-50-MiniGrid-DoorKey-AutoresetAudit-v1` is an informative
ARROW/DreamerV3 diagnostic, but it is not a controlled reproduction of the
paper's MiniGrid result. Re-running its other seeds would not resolve this
protocol mismatch.

The project adapter now names the two DoorKey interpretations explicitly:
`paper_label_9x9` preserves all historical project results, while
`released_source_8x8` exposes the authors' executable geometry for a new,
separately named source-fidelity protocol. No existing config is silently
changed.

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
The environment-contract follow-up is deliberately separate from training. It
uses a privileged shortest-path oracle only to verify that native actions 0-6,
key pickup, door toggle, goal reward, pixel wrapping, and the 100-step limit
work. It also measures a fixed-seed uniform-random baseline for both the
released 8x8 and literal 9x9 geometries. The oracle score is never an agent
score and never enters replay.

```bash
PYTHONPATH=src python scripts/audit_minigrid_doorkey_exploration.py \
  --oracle-seeds 100 --random-episodes 6000 \
  --output runs/diagnostics/doorkey_exploration_contract.json
```

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
