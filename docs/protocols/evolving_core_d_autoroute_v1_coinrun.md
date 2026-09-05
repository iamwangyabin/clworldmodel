# D-AutoRoute v1 — CoinRun original six, 541 epochs including terminal revisit

## Identity and invariant

- Entry point: `scripts/run_evolving_atomic_rssm_d_autoroute_coinrun.py`.
- Protocol: `Evolving-Core-D-AutoRoute-v1-OriginalSix-ProcgenCoinRun-541EpochRevisit-TaskAwareTraining-TaskIDFreeInference-Pilot`.
- Method key remains `evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`;
  the resolved `benchmark=procgen_coinrun` distinguishes the protocol.
- Classification: **pilot**, not an ARROW reproduction or a routing-accuracy claim.
- The user explicitly selected 541 rather than 540 epochs on 2026-09-06.
  The six original tasks each retain 90 epochs; epoch 540 (zero based) revisits
  task 0 for one ordinary collection/WM/AC update epoch. No budget is shortened.

Hypothesis: D's shared plastic world model, protected shared heads, adaptive
private Q/F/P mechanisms, and private MLP policies can serve the same
reconstruction-based inference rule on Procgen variants. Routing quality on
Atari or another historical model does not establish CoinRun routing quality.

This composes the existing [D-AutoRoute](evolving_core_d_autoroute_v1_atari.md)
launcher, trainer, model, replay and exact evaluator. No CoinRun copy of the
model/trainer is added. The published CoinRun baseline and running Atari commit
remain unchanged. This is **task-aware training and task-ID-free inference**,
not full task-agnostic training or automatic task discovery.

## Fixed environments and adapter

Published source presets remain under `third_party/arrow/Configs/CoinRun configs/CL-task configs/Original Order/`.
The five presets differ only by the predeclared seed. Order:

1. CoinRun
2. CoinRun+NB
3. CoinRun+NB+RT
4. CoinRun+NB+RT+GA
5. CoinRun+NB+RT+GA+MA
6. CoinRun+NB+RT+GA+MA+CA

Use the existing OpenAI Procgen source pin
`5e1dbf341d291eff40d1f9e0c0a0d5003643aebf` (`0.10.7+5e1dbf3`). The project-owned
`CoinRunFactory` resolves the same visual flags as the published EnvConfig:
NB removes backgrounds, RT restricts themes, GA uses generated assets, MA uses
monochrome assets, CA disables agent centering. Other native options are
explicit: hard distribution, unrestricted levels (`num_levels=0`), start level
0, nonsequential levels, no painted velocity, one native environment and zero
native threads per worker. Options are saved in the launch manifest.

Native RGB is already uint8 HWC `[64,64,3]`; there is no Atari preprocessing,
resize, grayscale conversion, reward scaling, or action repetition. All 15
native actions are preserved. The dummy previous action at reset/probe is the
native no-op index **4**, not Atari's index 0. The same generic collector
converts frames to CHW float32 `[0,1]` and writes uint8 CPU/mmap replay.

The adapter lazily constructs Procgen only on an explicitly seeded reset.
Seeds come from D-AutoRoute's independent collection, validation, pruning and
held-out streams, with independently derived worker/episode seeds reduced
modulo `2**31` for the native constructor. A seeded evaluation reset creates a
fresh constructor so episode seeds do not depend on worker completion order.
This is an explicit seeding/evaluation deviation from the released baseline.

Procgen emits a new episode's first observation on a terminal step. The adapter
caches that observation and consumes it exactly once when Gymnasium SameStep
calls reset; it never advances an extra step or skips an episode. Native done
maps to terminated, with no fabricated truncation or terminal image. The
collector preserves the inherited ARROW terminal-reward/preceding-transition
convention. Exact evaluation sums actual step rewards and accepts only complete
episodes. Adapter parity is tested against the pinned native Gym wrapper under
fixed identical actions/seeds, including raw rewards and termination.

## D learning and final revisit

Unchanged D settings include Dense Q/F/P widths 512/512/256, four-atom reuse,
private projectors and full private MLP Actor/Critic pairs, shared prediction
heads protected at scale 0.1, BF16 compute, and component gradient projection.
The first acquisition uses 16 current sequences; subsequent learning uses 12
current and four uniformly selected old-task LTDM sequences. Training and replay
labels always come from the scheduler, never the router. Initial random
collection at each newly acquired task follows D's `random_policy=new`, rather
than the released CoinRun baseline's first-task-only option.

The first task's shared core LR is 2e-4; later/revisit shared core LR is 1e-4.
Private Q/F/P/head LR is 2e-4, route LR 1e-3 and AC LR 1e-4, all inherited from
D fixed_v1 rather than presenting this as the baseline optimizer.

At the terminal task-0 revisit:

- The schedule has acquired all six routes; action selection and periodic
  evaluation continue to consider **all six**, not just route 0.
- Training reactivates task 0's existing compact Q/F/P and private MLP policy;
  it does not allocate a new slot, grow back Dense, or reset the actor.
- Each WM update samples a non-current memory route uniformly from 1–5 with the
  last boundary teacher and the ordinary D 12/4 protection. Old private policies
  remain frozen. Updating a reused task-0 atom can still affect another route;
  preservation of other task returns is **not guaranteed**.
- There is no seventh acquisition boundary, consolidation, pruning selection,
  or compression recovery. Final held-out evaluation occurs after epoch 541.

First-frame MSE routing, deterministic posterior/argmax evaluation, episode
locks, future-route exclusion, all-seen return gates, four independently tested
Q/F/P compression fractions and 16 exact episodes per selection/task remain D's.

## Budgets and accounting

Five fixed seed values: `123456789, 1337, 31337, 42, 987654321`.
Four environment workers collect 4,096 trajectory positions each per epoch.
The first position in each worker is a reset record, **not** an executed action:

| Quantity | Per full seed |
|---|---:|
| Epochs | 541 |
| Collected trajectory positions, including initial resets | 8,863,744 |
| Actual training agent decisions / native frames (repeat=1) | 8,861,580 |
| Online WM updates | 541,000 |
| Six boundary consolidations | 6,000 |
| Q/F/P recovery updates | 6,000 |
| Total WM updates | 553,000 |
| AC updates | 432,800 |
| Task-0 AC updates / each other task | 72,800 / 72,000 |
| AC compression updates | 0 |
| Q/F/P selector episodes | 1,680 exact |

No training budget is converted to extra steps to compensate for reset records.
CoinRun's new `interaction_counter_mode=environment_steps` records real step
calls; Atari's historical trajectory-position counters are not silently renamed
or changed. Evaluation decisions are separate and never enter replay or updates.

Replay retains the full ARROW-50 capacity: 512 FIFO plus 512 LTDM trajectories,
each 512 positions, half/half sub-buffer selection and CPU uint8 observation
storage. Observation storage is 6 GiB; 15-action auxiliary tensors add
37,748,736 bytes, plus explicitly recorded task IDs/reservoir/index metadata.
This is storage-optimized and not the baseline's float32 CUDA byte usage.

Dense online parameter count is 52,886,765 (no learned router weights); the
15-action input/output layers are analytically accounted for and checked at
runtime. Optimizers, activations, replay and boundary teachers are excluded.
Final parameter count depends on return-gated selected Q/F/P widths. The six
private Actor/Critic pairs are not shared or compressed.

Keep rolling pre/post boundary checkpoints and immutable replay assets under
`latest_boundary` retention. The last resumable boundary is at 540 epochs;
resuming it executes the pending revisit with saved replay/RNG/optimizer state.
Final 541-epoch inference weights and raw evaluation remain distinct from that
checkpoint. Per-run output preflight retains the existing 48-GiB minimum.

## Reporting, checks and execution

The raw matrix and complete episode returns remain in periodic routing records
and `final_evaluation.json`. `raw-retention-v1` reports final average raw return,
per-task acquisition-minus-final raw forgetting and its negative as raw backward
transfer. It also preserves the 540-epoch first-pass summary separately.
Acquisition uses the fixed validation cohort, final uses the held-out cohort;
final retention estimates therefore include cohort sampling noise. No Atari
normalization or fabricated random/single-task constants are applied. Forward
transfer is not claimed without matching reference curves.

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute_coinrun.py --seed 0 --dry-run
PYTHONPATH=src:tests:scripts python -m unittest test_d_autoroute_coinrun test_d_autoroute test_fastkan_autoroute
# The following perform real simulation or optimizer updates: clean, pushed,
# freshly fetched and synced provenance is required before either smoke.
python scripts/smoke_d_autoroute_coinrun_env.py
python scripts/smoke_evolving_atomic_rssm.py --method-profile d_autoroute --benchmark procgen_coinrun --device cuda:0
python scripts/run_evolving_atomic_rssm_d_autoroute_coinrun.py --seed 0 --cpu-threads 4 --output-dir runs/coinrun_s0_pilot
python scripts/summarize_continual_metrics.py --raw runs/coinrun_s0_pilot --output runs/coinrun_s0_pilot/continual_metrics.json
```

Use the pinned constraints and record full package/native runtime inventory.
Each host must pass environment and CUDA smokes before launch. The declared
campaign adds one CoinRun seed beside each existing Atari seed: 4090-1 GPU0,
4090-2 GPU0, and 4090x4 GPUs1/2/3. Never use 4090x4 GPU0 or duplicate result
seeds merely to fill capacity. Dual concurrency is a throughput experiment, not
a guarantee that every individual run becomes faster.
