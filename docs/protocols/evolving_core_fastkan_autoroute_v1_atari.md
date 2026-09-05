# D-AutoKAN v1 (方案 F): D + shared FastKAN + reconstruction routing

## Status and claim boundary

Method key:
`evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow`.
Protocol:
`Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-SharedFastKAN-FirstFrameRouter-v1-OriginalSix-Atari-TaskAwareTraining-TaskIDFreeInference-Pilot`.

This is a new, from-scratch research variant, not a redefinition of D or E.
It implements **task-aware training and task-ID-free interaction/evaluation**.
Training still knows task boundaries and attaches true task IDs to Replay.
It is not fully task-agnostic continual learning, automatic task discovery, or
within-episode change detection. No Atari performance or routing-accuracy
result exists for this combination. Tensor/config/mock tests are not a pilot.

Hypothesis: D's route-conditioned reconstruction can identify the appropriate
latent interface, allowing a single fixed-capacity FastKAN controller to serve
all tasks while D compacts only world-model private mechanisms.

## Inherited world model and training

The world-model architecture and update rule come from
[D's protocol](evolving_core_dense_acquire_adaptive_qfp_compression_v1_atari.md):

- MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, Enduro; 90 epochs each.
- Original-six `fixed_v1`: Task-0 shared LR `2e-4`, not the three-task
  SharedFrozenDown/FastKAN `fixed_v2` profile.
- One plastic CNN/base RSSM and one shared decoder/reward/continue set;
  shared prediction-output distillation scale `0.1`.
- Per-task spatial projector, Dense-acquire Q/F/P mechanisms at
  `512/512/256`, four atoms, reuse gates, and physical adaptive compaction.
  There is **no SharedFrozenDown substitution and no private decoder bank**.
- Task 0: 16 current sequences. Later: 12 current plus four old LTDM sequences;
  conflict projection and posterior/hidden/frozen-actor interface protection.
- ARROW-50 retains its FIFO/LTDM capacity, selection semantics, CPU uint8
  observation storage, and labeled training partitions.
- 1,000 shared-only consolidation updates per boundary, with per-seen-task
  five-percent raw-return rollback. Validation uses the automatic policy below.
  An evaluation/consolidation exception aborts this new method after rollback;
  it must not silently continue with an unfulfilled fixed update budget.

Replay labels are always training-scheduler metadata, **never the inferred
route**. Model training and imagined trajectories use those labels. Evaluations
are frozen, restore training RNG streams, and cannot enter Replay or updates.

## Shared behavior: no growing Actor-Critic bank

Reuse the existing [FastKAN StableTargets bundle](arrow_fastkan_ac_stable_targets_continual_atari.md)
without importing the old SharedFrozenDown world-model topology:

- Exactly one Actor and one Critic; each has three width-53 hidden layers,
  eight fixed Gaussian centers over `[-2,2]`, RMSNorm epsilon `1e-4`, and a
  SiLU base branch. Actor output scale and unimix are both `0.01`.
- LaProp `4e-5`, epsilon `1e-20`, betas `.9/.999`, 1,000 warmup steps,
  AGC `.3`, no global gradient clip; imagination horizon 15, discount
  `1 - 1/333`, lambda `.95`, entropy `3e-4`, return-normalization decay `.99`.
- Slow-critic regularization `1.0`, EMA decay `.98`, replay critic loss `.3`,
  slow targets and corrected post-transition bootstrap.
- The unchanged 800 AC updates/epoch are 800 current on Task 0; later 600
  current plus 200 distributed across completed routes. Integer remainders
  are assigned by the existing balanced schedule, then shuffled by its own RNG.
- A single cumulative frozen boundary Actor protects the old WM interfaces.
  There are no per-task actors, critics, behavior adapters, task embeddings, or
  learned router. This reduces AC growth; it does not remove actor-critic RL.

## First-frame route selection and episode lock

The caller declares the acquired route registry: completed slots plus the
currently acquiring slot. Future preallocated slots are ineligible. This
registry is acquisition state, not the identity of the environment being
evaluated; the same registry is used for every task in a validation condition.

For each resetting worker independently, normalize its first RGB frame to
`[0,1]`. For every eligible route `k`, reset the latent/hidden state, use a
dummy no-op previous action, and infer the **deterministic posterior mode**:

```text
(z_k, h_k) = RSSM_posterior(x_first, zero_state, no_op, route=k)
score_k = mean_pixels((shared_decoder(concat(z_k, h_k)) - x_first)^2)
selected_route = argmin_k score_k
```

MSE reduction is float32; network forwards retain the configured compute
precision. Ties select the lowest eligible ID. Nonfinite scores fail closed.
Probes do not mutate the executed recurrent state or sample random latents.
Only observations and candidate IDs reach this selection boundary: no rewards,
environment names, true task labels, or policy scores.

Keep that route for the whole episode. On each real reset, discard the lock
and score the new first frame. Workers may choose different routes, so RSSM
forwards are grouped by **inferred** ID and scattered back to their original
worker positions. The shared Actor receives only the resulting latent/hidden
features, not a task ID. Collection uses stochastic action/latent sampling;
evaluation uses action argmax and latent mode. Random acquisition collections
remain random and produce no inferred-route accuracy sample.

### Explicit reset/evaluation deviations from legacy D

The pinned Gymnasium vector default is next-step autoreset, whereas the legacy
collector assumes same-step reset observations. **Only this new protocol**
explicitly requests `SameStep`. At a reset observation it stores a dummy no-op
action and clears the policy's previous action, matching the routing probe's
zero context. Legacy D's other trajectory packing/reward conventions remain;
this is not a general collector cleanup or a claim of unchanged trajectories.

The legacy D evaluator has a trajectory budget and can return fewer completed
episodes or a partial-return estimate. This new protocol instead evaluates
exactly 16 complete, independently seeded episodes per condition. Episode
reset/action seeds derive from the fixed task-cohort seed and episode index;
they do not depend on worker completion order. At most `n_sync` policies are
batched; only active episodes are stepped, and no surplus episode is collected.
The evaluator uses homogeneous per-task factories, steps local environments
synchronously, and closes all resources even on failure. A 32,768-decision
per-episode safety-cap hit **fails** rather than accepting a partial return.

These are named protocol differences, not silent fixes to D. Exact-episode
raw returns should not be presented as evaluator-matched to old D reports.
Record actual episode lengths and evaluation decisions to quantify this cost.

## Compression must preserve automatic routing

After consolidation, independently prune every candidate from the same frozen
Dense teacher. Width fractions `.75/.5/.25/.125` each receive exactly 250
Adam `2e-4` updates using completed-task LTDM and the same restored sampling
streams. Only that task's private Q/F/P is trainable; the shared FastKAN is
frozen during recovery. D's recovery losses and width lattice are unchanged.

Evaluate the Dense teacher and each candidate on **every seen task** using the
same eligible route set and the same 16 fixed episode seeds per task. Accept a
candidate only if, for every seen task `j`,

```text
(R_dense_auto[j] - R_candidate_auto[j]) / max(abs(R_dense_auto[j]), 1) <= .05
```

This catches a compressed current route attracting observations from an older
task. A current-task improvement cannot compensate for an old-task drop. Try
all four candidates; retain the smallest passing width or restore Dense.
This gate preserves the teacher's automatic-policy return, not oracle return;
it cannot guarantee high task-ID accuracy if the Dense teacher already routes
poorly. Report both closed-loop raw returns and routing confusion/margins.

Compression uses seed-sequence domain 3, separate from collection (0), periodic
validation/consolidation (1), and final evaluation (2). Final data are not used
for model updates or width selection. "Held out" describes within-run selection
isolation, not a claim that no related seeds have ever been inspected before.

## Resource ledger

| Quantity | D | D-AutoKAN |
|---|---:|---:|
| Online WM updates | 540,000 | 540,000 |
| Consolidation WM updates | 6,000 | 6,000 |
| Compression WM updates | 6,000 | 6,000 |
| Total WM optimizer steps | 552,000 | 552,000 |
| AC optimizer steps | 432,000 | 432,000 |
| Nominal compression validation episodes | 480 | **1,680 exact** |
| Dense-acquisition WM parameters | 42,601,625 | 42,601,625 |
| Online behavior parameters | 10,295,910 | **1,700,670** |
| Dense-acquisition total parameters | 52,897,535 | **44,302,295** |

The all-seen selector budget is `5 conditions * 16 episodes * (1+...+6)`.
It adds 1,200 nominal validation episodes versus D, plus route-probe inference
compute. Equal update counts **do not mean equal total compute**. The same
nominal online collection budget and Replay capacity/bytes are inherited;
resolved launch and runtime ledgers retain their detailed accounting.

The shared Actor is 793,692 parameters; the shared Critic is 906,978. Against
six private MLP pairs, this saves 8,595,240 online parameters (83.48% of D's
behavior bank), independently of the selected Q/F/P widths. Per-task AC growth
is zero. **Most savings come from sharing, not intrinsically from KAN**: a
single existing MLP pair is 1,715,985 parameters, only 15,315 more than this
FastKAN pair. A matched shared-MLP control is needed to attribute a performance
benefit specifically to FastKAN. The all-smallest-width total would be
24,339,863 parameters; actual
compression is outcome-dependent, not a prediction of the new method's result.
The Dense-acquisition total occupies 177,209,180 FP32 parameter bytes before
buffers, Replay, optimizer state, gradients, and activations.

A training-only slow Critic adds 906,978 parameters and one frozen boundary
Actor adds 793,692; both are constant-count copies. Dense WM teachers and
independent candidates temporarily increase peak memory. All task slots are
preallocated, so final compression does not remove the Dense acquisition peak.

## Artifacts, checkpoint and validation

- `task_routing/collection_epoch_*_batch_*.json`: acquisition epoch/counter,
  candidate registry, selected IDs, MSE vectors, margins, and posthoc label audit.
- `task_routing/periodic_epoch_*.json`, `final_evaluation.json`: exact seeded
  episode returns/lengths/routes and confusion. Future-task diagnostics explicitly
  mark that the true task has no eligible route; do not average them into seen-task
  recognition accuracy.
- Consolidation and compression artifacts retain pre/post/candidate per-task
  auto-routed results, selected widths, fallback/rollback and fixed update counts.
- Schema-v2 resumable checkpoints retain the shared AC/LaProp, slow Critic,
  return EMAs, cumulative Actor teacher, independent rehearsal RNG, replay and
  all existing WM/optimizer/RNG/counter state. Physical Q/F/P widths reconstruct
  before strict loading. An explicit eligible-ID registry must agree with the
  saved acquisition position. Inference-only snapshots also persist this registry.
- Router episode locks are transient: boundary checkpoint resume starts new
  environments and new locks. This is not a mid-episode resumable collector.

Local checks cover typed configuration/serialization, D topology isolation,
actual FastKAN counts and no behavior growth, per-worker routing/locking/reset,
no oracle input/future-route access, RNG preservation, mock exact-episode/timeout
handling, negative-return all-seen gates, and compact shared-decoder reload.
No Atari interaction or optimizer-update smoke was launched for this change.
Target-GPU update/integration tests remain necessary before a pilot campaign.

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --prediction-head-profile shared_distilled \
  --adaptive-qfp-compression \
  --behavior-profile shared_fastkan_autoroute \
  --seed 0 --classification pilot --dry-run
```

Training and the provided `smoke_evolving_atomic_rssm.py --method-profile
fastkan_autoroute` entry may run only after a clean commit is pushed and its
upstream relation is fetched/verified, following `AGENTS.md`. The smoke does
not establish Atari routing accuracy. Before a paper claim, run multiple
predeclared seeds and report raw per-task returns, retention/forgetting,
recognition confusion, final/peak bytes, and inference/evaluation compute.

## Historical motivation, not transferred evidence

The recovered 2026-08-26 three-task CNN-FullBank diagnostics reported perfect
classification in 192 random-policy windows, 192 trained-policy windows, and
192 independently reset first frames; tested window prefixes included length 1.
Those checkpoints had private encoders/RSSMs/decoders and actors. Trained-policy
data were collected with oracle routing, and many windows shared trajectories.
They did **not** demonstrate this shared-decoder model's closed-loop return.

The historical checkpoint commit was
`bcfd89abba16ae9d9f17a339b22c9d5a99883519`; the inline probe was not versioned
under that commit. The recovered, ignored local diagnostic is
`runs/analysis/recovered_fullbank_task_routing_20260826/routing_diagnostic.json`.
This is motivation to test the simplest router, not evidence to claim 100%
recognition for D-AutoKAN. Matched exact-evaluator oracle-routing and shared-MLP
controls are proposed follow-up ablations, not completed results or alternate
settings of this frozen named protocol.
