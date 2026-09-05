# D-AutoRoute v1: D with private MLP behavior and task-ID-free inference

## Identity and scope

- Entry point: `scripts/run_evolving_atomic_rssm_d_autoroute.py`.
- Method: `evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`.
- Protocol:
  `Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-FirstFrameRouter-v1-OriginalSix-Atari-TaskAwareTraining-TaskIDFreeInference-Pilot`.
- Status: implementation and deterministic/mock tests, **not a performance
  result**. No CUDA smoke, training pilot, or routing accuracy is claimed here.

This is a separate from-scratch variant of [D](evolving_core_dense_acquire_adaptive_qfp_compression_v1_atari.md),
not a redefinition of D or [D-AutoKAN](evolving_core_fastkan_autoroute_v1_atari.md).
Task boundaries and true task IDs remain available during training. Interaction
and evaluation infer the executed route from observations. There is no task
discovery, unknown-task detector, or within-episode change detector.

## Unchanged D learning and capacity

- MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, Enduro; 90 epochs each,
  original-six `fixed_v1` optimizer profile and original action/reward settings.
- One plastic CNN/base RSSM and one plastic shared decoder/reward/continue set;
  shared prediction-output protection remains at scale `0.1`.
- Each task acquires Dense `512/512/256` Q/F/P with four atoms and its private
  projector/reuse gates, then physically compacts accepted Q/F/P candidates.
- One **independent full MLP Actor and Critic per task**, with D's `1e-4` AC LR
  and current-task-only AC updates. Old AC parameters remain frozen.
- No FastKAN, shared AC base, behavior residuals, AC compression, or AC
  compression-recovery updates.
- ARROW-50 CPU uint8 Replay; Task 0 uses 16 current sequences, later tasks use
  12 current and four old-LTDM sequences. Component gradient projection,
  interface protection, and 1,000-update boundary consolidation remain.

Model updates, imagined trajectories and Replay grouping always use training
labels. Collected trajectories are labelled by the scheduler, **never by the
router's prediction**, even when action selection inferred a wrong route.
Preserving update rules does not imply identical trajectories: inferred routing
intentionally changes the behavior policy used to acquire data.

## Inference algorithm and private policy ownership

The eligible registry contains completed task slots plus the currently acquiring
slot. Future preallocated slots are excluded. The same registry is supplied for
every environment evaluated at a given checkpoint, independently of its label.

For each worker's first RGB frame, normalize to `[0,1]` and independently probe
each eligible route `k` from zero recurrent state and dummy no-op previous
action, using the deterministic posterior mode:

```text
score[k] = mean_pixels((shared_decoder(RSSM_posterior(first_frame, route=k))
                       - first_frame) ** 2)
route = argmin(score)
```

The MSE reduction is float32. Lowest eligible ID wins exact ties; nonfinite
scores abort. Probes cannot advance the executed trajectory state or sample
latent noise. Lock the selected route for the episode and rescore on reset.

RSSM forwards are grouped by each worker's inferred route. The corresponding
private Actor then receives that group's latent/hidden features; logits are
scattered back to the original worker order before action selection. This is
not the scheduler's current Actor, and no true evaluation ID reaches selection.
The Critic remains private and is used for its task's training, not action
selection. A temporary `RoutedActorBank` references existing Actors without
allocating additional parameter tensors or entering any optimizer/checkpoint.

Collection samples stochastic actions/latents; evaluation uses action argmax and
latent mode. The inherited initial random collection remains random and emits
no route-accuracy samples. Labelled confusion/margins are post-inference audits,
not router inputs or training targets.

## Reset and evaluation changes relative to legacy D

As in D-AutoKAN, explicitly use Gymnasium 1.1 `SameStep` vector autoreset so the
router observes the new episode's reset frame. Reset previous actions to the
dummy no-op. Legacy D's default autoreset handling is not silently changed.

Evaluation runs exactly 16 independently seeded **complete** episodes per task
condition. Episode seeds depend on cohort/episode index, not worker completion
order. Do not step surplus episodes. A 32,768-agent-decision cap hit or nonfinite
reward aborts rather than reporting a partial return. Restore training RNG and
each module's original train/eval mode; evaluation does not update parameters,
Replay, or optimizer state. Close all environment resources on failure too.

These exact-episode results are not evaluator-matched to old D's trajectory-
budget/possibly-partial returns. A future oracle-routing control must use the
same exact evaluator and declared reset semantics.

## Routing-aware consolidation and compression

After shared consolidation, probe all four Q/F/P candidates `.75/.5/.25/.125`
independently from the same frozen Dense teacher. Each receives 250 Adam
updates at `2e-4` using only completed-task LTDM and identical restored sampling
streams. Only that task's Q/F/P tensors are trainable during recovery; private
policies, the shared core/heads, projector, and routes do not update.

Evaluate the Dense teacher and every candidate on **every seen task**, under
automatic routing and the same acquired registry. Select the smallest candidate
such that each seen task `j` satisfies:

```text
(R_dense_auto[j] - R_candidate_auto[j]) / max(abs(R_dense_auto[j]), 1) <= .05
```

A current-task gain cannot offset an old-task drop. Retain Dense when none
passes. The gate is relative to the teacher's automatic-policy return, so it
does not certify high recognition accuracy or preserve oracle returns if the
teacher itself misroutes. Shared-core consolidation likewise uses all-seen
automatic-policy return with rollback; exceptions abort after safe rollback.

Compression cohorts use the separate pruning seed domain, not the final held-
out cohort. Evaluation transitions never enter training. Log actual raw task
returns, route scores/margins/confusion, selected widths, and failures.

## Explicit budgets and storage

| Quantity | D-AutoRoute | Legacy D |
|---|---:|---:|
| Online WM updates | 540,000 | 540,000 |
| Consolidation WM updates | 6,000 | 6,000 |
| Q/F/P recovery updates | 6,000 | 6,000 |
| Total WM updates | 552,000 | 552,000 |
| Online AC updates | 432,000 | 432,000 |
| AC compression updates | 0 | 0 |
| Compression selection episodes/rollouts | **1,680 exact** | 480 nominal |
| Six private MLP AC pairs | 10,295,910 | 10,295,910 |
| Learned router parameters | 0 | 0 |
| Dense online model parameters | 52,897,535 | 52,897,535 |
| All-smallest Q/F/P online parameters | 32,935,103 | 32,935,103 |

Selection costs `5 * 16 * (1+2+3+4+5+6) = 1,680` episodes, plus one
posterior/decoder probe per eligible route at reset. Matching optimizer updates
does **not** match total compute. Episode lengths and decisions are logged.
Actual final size depends on accepted widths; changing selection can change
selected sizes. FP32 parameter-byte bounds are `131,740,412`–`211,590,140`,
excluding Replay, optimizers, buffers, activations and training-only teachers.
Peak allocation may remain Dense; this method claims no extra AC savings.

## Checkpoints, entry point and verification

Existing private-bank resumable schema v1 stores both policies/critics and their
optimizers, targets/EMA state, WM/teacher topology, RNG, Replay provenance and
counters. Routing metadata persists eligible IDs and validates them against
acquisition state. Boundary resume resets environments, so episode locks are
not resumed mid-episode. Inference snapshots retain the full private bank and
explicit eligibility; unacquired slots may never be selected.

From the repository root, inspect without environment interaction or updates:

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 0 --dry-run
```

The entry point exposes only seed/pilot classification, runtime/output paths,
CPU thread count, and dry-run. It fixes all architecture/curriculum selectors.
Relative paths resolve against the repository root. It delegates provenance,
manifest creation, resource preflight, checkpoint checks and reporting to the
existing launcher; it does not duplicate the trainer.

Focused CPU checks (fixed tensors and mocked environments only):

```bash
PYTHONPATH=src:tests python -m unittest test_d_autoroute test_fastkan_autoroute
```

The separately selectable CUDA execution smoke is
`scripts/smoke_evolving_atomic_rssm.py --method-profile d_autoroute --device cuda:0`.
**Do not launch it or training from uncommitted/local-only code.** A real run
requires explicit authorization, a clean pushed commit, fetched/synced upstream
verification, and the protocol manifest. Tests/dry runs establish contracts,
not Atari performance or historical FullBank routing-accuracy transfer.
