# Evolving-Core Shared-Frozen-Down + Shared FastKAN ARROW v1 (Atari)

## Scope and claim limit

`Evolving-Core-SharedFrozenDown-SharedFastKANAC-StableTargets-ARROW-v1-ThreeTask-Atari-TaskAware-Pilot`
is a separately named, from-scratch three-task pilot. It combines the existing
Shared-Frozen-Down world-model mechanism with one task-shared FastKAN Actor and
FastKAN Critic. It does not replace or retroactively redefine Evolving-Core v2,
the Shared-Frozen-Down Task-0 pilot, or the prior ARROW FastKAN experiments.

The method is task-aware because task identity selects the projector, Q/F/P
route, private heads, and replay slice. Task identity is not concatenated to
the shared Actor-Critic input. Neither the existing single-seed SharedDown
acquisition result nor the earlier FastKAN result proves that their combination
improves continual learning. This protocol is an implementation and evaluation
plan, not a performance claim.

## Fixed curriculum

The pilot uses the predeclared three-task order and 90 epochs per task:

1. `ALE/MsPacman-v5`, epochs 0--89;
2. `ALE/Boxing-v5`, epochs 90--179;
3. `ALE/CrazyClimber-v5`, epochs 180--269.

Task-0 shared-core Adam uses the accepted `fixed_v2` learning rate `3e-4`;
later tasks use `1e-4`. Task-private, route, and boundary learning rates remain
`2e-4`, `1e-3`, and `2e-5`.

## Evolving Shared-Frozen-Down world model

One CNN and one base posterior/recurrent/prior RSSM remain plastic throughout
the curriculum. Each recurrent (F), posterior/representation (Q), and
prior/transition (P) mechanism bank owns one full-width down projection
`D_b`, frozen at its seeded initialization. Each task `t` owns its LayerNorm,
hidden FiLM scale/shift, zero-effect up projection, four-atom partition, and
reuse route:

\[
  m_{b,t}(x)=0.1\,U_{b,t}\,\mathrm{SiLU}
  \left(\gamma_{b,t}\odot D_b\mathrm{LN}_{b,t}(x)+\beta_{b,t}\right).
\]

`U` weight and bias are initialized to zero, so every task route has exactly
zero initial effect. The shared `D` matrices never receive gradients and are
serialized once per bank. The mechanism widths stay `512/512/256`; this is a
parameter/storage reduction, not a narrower or cheaper matrix-multiply path.
Every task also owns a zero-effect spatial projector and private observation,
reward, and continuation heads.

Later online world-model updates keep the 16-sequence batch fixed: 12 current
sequences plus four task-homogeneous LTDM sequences from a uniformly selected
completed task. Posterior, hidden-state, and frozen-Actor interfaces protect
old routes. Conflicting current gradients are projected independently for the
shared encoder, posterior, recurrent, prior, and latent-interface groups;
current private gradients are not projected. Each boundary attempts 1,000
extra task-balanced shared-only updates and rolls back shared weights and the
persistent shared Adam state if the fixed-cohort raw return gate fails.

## Shared FastKAN behavior model

Exactly one Actor-Critic is built before first collection and persists across
all tasks. Both heads are independent width-53 FastKAN networks with three
hidden layers, eight fixed Gaussian centers over `[-2, 2]`, per-layer RMSNorm
(`1e-4` epsilon), a SiLU base branch, Actor output scale `0.01`, and Actor
unimix `0.01`.

| Component | Online parameters |
|---|---:|
| Shared FastKAN Actor | 793,692 |
| Shared FastKAN Critic | 906,978 |
| Shared online pair | 1,700,670 |

The fixed StableTargets bundle uses LaProp LR `4e-5`, epsilon `1e-20`, betas
`0.9/0.999`, 1,000 warmup steps, AGC `0.3`, no global gradient clip, imagination
horizon 15, discount `1 - 1/333`, lambda `0.95`, entropy `3e-4`, persistent
return-normalization decay `0.99`, slow-critic regularizer `1.0`, EMA decay
`0.98`, and replay critic loss scale `0.3`. The EMA critic supplies imagined
and replay value targets and the detached Actor baseline. Lambda returns use
the actual post-transition horizon state.

## Fixed-budget cross-task behavior rehearsal

The total remains exactly 800 Actor-Critic optimizer updates per epoch:

| Current acquisition task | Current-route updates/epoch | Completed-route updates/epoch |
|---|---:|---:|
| Task 0 | 800 | 0 |
| Task 1 | 600 | 200 on Task 0 |
| Task 2 | 600 | 100 on Task 0 + 100 on Task 1 |

Across three 90-epoch tasks, route totals are Task 0 `99,000`, Task 1 `63,000`,
and Task 2 `54,000`, summing to the unchanged `216,000` optimizer updates.
The exact route multiset is shuffled by an independently seeded RNG. Every
update samples task-homogeneous context from task-conditioned ARROW mixed
Replay. FIFO/LTDM capacity and 50/50 selection weights are unchanged and are
renormalized only when one sub-buffer lacks the requested task. Evaluation
transitions never enter Replay.

Both the shared Actor and shared Critic update on current and old routes.
Before a new task, the preceding boundary Actor is copied once and frozen. This
single cumulative teacher protects old world-model policy interfaces; it is
not persistent per-task state and receives no update. Actor-only imagination
distillation is disabled.

## Parameter and byte accounting

The analytic online inference ledger for the three-task topology is:

| Topology | World model | Behavior | Online total |
|---|---:|---:|---:|
| ARROW-50 | 19,498,853 | 1,714,961 | 21,213,814 |
| Dense Evolving-Core v2 + 3 private MLP pairs | 48,175,443 | 5,144,883 | 53,320,326 |
| Shared-Frozen-Down + 3 private MLP pairs | 42,675,539 | 5,144,883 | 47,820,422 |
| **New method: Shared-Frozen-Down + 1 shared FastKAN pair** | **42,675,539** | **1,700,670** | **44,376,209** |

The world-model total contains `2,753,792` frozen shared-down parameters once
and `1,064,960` private mechanism parameters per task. Including projectors,
route gates, and later private heads, per-task world-model additions excluding
the shared bases are `1,099,200`, `9,661,841`, and `9,661,853`.

The new method is `3,444,213` online parameters smaller than the matched
Shared-Frozen-Down/private-MLP topology and `8,944,117` smaller than dense v2,
but `23,162,395` larger than ARROW-50. Its FP32 online parameters occupy
`177,504,836` bytes before buffers, gradients, optimizer state, Replay, or
activations.

StableTargets adds a training-only `906,978`-parameter slow critic; after the
first boundary the transient Actor teacher adds `793,692`. Peak behavior state
excluding optimizer is `3,401,340` parameters. Runtime artifacts must verify
all analytic counts and separately report Replay trajectory capacity and
actual bytes.

## Checkpoint and artifact contract

Boundary checkpoint schema v2 stores:

- shared Actor-Critic weights and LaProp state;
- EMA slow critic and return-normalization EMAs;
- future-task frozen Actor teacher and independent route-schedule RNG;
- model and boundary-teacher weights;
- shared, task-private, and route optimizer states;
- complete Replay, retention indices, and immutable mmap provenance;
- Python, NumPy, PyTorch CPU/CUDA, environment-seed, and sampling RNG states;
- environment schedule position, total frame/update counters, and the exact
  cumulative Actor-Critic update count for every task route.

Every complete run must retain `model_parameter_accounting.json`,
`actor_critic_parameter_accounting.json`,
`shared_behavior_replay_accounting.json`, every pre/post-consolidation
checkpoint, raw evaluation vectors, derived continual metrics, and all rollback
or failure artifacts.

## Budgets

The run uses 17,694,720 raw Atari frames, 270,000 online world-model updates,
3,000 explicitly extra consolidation updates, and 216,000 Actor-Critic
updates. Replay capacity, preprocessing, task durations, action semantics,
evaluation cadence, and validation/held-out cohorts are unchanged from the
three-task Evolving-Core contract. Behavior rehearsal reallocates updates; it
does not add them.

## Launch

Inspect the resolved config without interaction or optimization:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order mspacman-boxing-crazyclimber \
  --task0-profile fixed_v2 \
  --behavior-profile shared_fastkan_stable \
  --seed 0 \
  --classification pilot \
  --dry-run
```

A non-dry run remains gated on a clean, committed, pushed, and
upstream-synchronized Git state. Before a campaign, run focused tests and the
target-CUDA smoke:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --behavior-profile shared_fastkan_stable \
  --device cuda:0
```

The legacy `evolving_atomic_rssm_arrow` name strictly retains
`dense_private`; the shared-frozen-down topology is valid only under this new
method name. A complete seed-0 run is still a pilot. Multi-seed evidence and
controls separating shared/private behavior, KAN/MLP architecture, and
StableTargets are required for a method claim.
