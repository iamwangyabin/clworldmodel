# Adaptive Q/F/P + Shared-Base Actor-Critic Compression v1 (Atari)

## Scope

The protocol name is
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFPAC-SharedDistilledHeads-SharedResidualMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.
Its config method key is
`evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow`.

This is a separately named, from-scratch pilot. It extends method D but does
not redefine it, and it has no performance result yet.

## Fixed online protocol

The task sequence is MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and
Enduro, for 90 epochs each. Seed indices, environment preprocessing, interaction
budget, ARROW-50 FIFO/LTDM capacity and 50/50 selection, shared prediction
heads, Dense Q/F/P acquisition, continual losses, gradient projection, and
boundary consolidation are identical to method D.

Q/F/P still acquire at `512/512/256` and use method D's independent physical
candidate widths, 250-update recovery, dedicated 16-rollout raw-return gate,
and Dense fallback.

## Behavior topology and online learning

For state `s` and explicit task route `t`:

```text
log pi_t(.|s) = log_softmax(Actor_shared(s) + DeltaActor_t(s))
log V_t(s)     = log_softmax(Critic_shared(s) + DeltaCritic_t(s))
```

`Actor_shared` and `Critic_shared` are the unchanged Dreamer MLP categorical
heads and are stored once. Every `Delta` is
`0.1 * up(SiLU(down(LayerNorm(s))))`, is exactly zero at initialization, has
full acquisition hidden width 512, and is partitioned into four atoms. Learned
routes may add gated frozen atoms from older tasks, exactly as the RSSM
mechanism bank does.

At each online update:

- shared Actor/Critic bases stay trainable;
- only the current task's private Actor/Critic residuals and reuse routes are
  trainable; and
- old task residuals/routes are frozen even when their replay route is sampled.

The fixed 800 online behavior updates per epoch do not increase. Task 0 uses
all updates. Later epochs assign 75 percent to the current task and split 25
percent uniformly over replay-available older tasks using an independently
seeded exact-count shuffled schedule. Task-conditioned ARROW replay initializes
the corresponding task-routed world model and behavior head.

## Actor-Critic boundary compression

After the accepted world-model boundary and Q/F/P selection, freeze a copy of
the completed full-width Actor-Critic as teacher. Construct all four residual
pair candidates from that one teacher:

| fraction | Actor residual width | Critic residual width |
|---:|---:|---:|
| `0.75` | `384` | `384` |
| `0.50` | `256` | `256` |
| `0.25` | `128` | `128` |
| `0.125` | `64` | `64` |

Hidden-channel importance is incoming row/bias norm times outgoing column norm,
ranked separately inside each atom. Selected rows and columns are copied into
new, physically smaller Linear tensors; no full-width mask remains.

Every candidate receives exactly 250 Adam updates at `2e-4`. Each update:

1. samples four context frames for 16 sequences from completed-task LTDM only;
2. infers the task-routed latent state and generates a 16-step trajectory with
   the frozen Dense teacher;
3. minimizes `KL(pi_teacher || pi_candidate)` for Actor logits; and
4. minimizes the same categorical KL over the Critic's 255 symlog bins.

Both KL scales are 1.0. Shared bases, old residuals, and routes are frozen.
Python, NumPy/replay, CPU torch, and CUDA torch states are restored before each
candidate, so widths see identical LTDM indices and imagination draws. The
compression phase restores the online training RNG state on exit.

## Return gate and isolation

Behavior compression uses seed-sequence spawn index 4, distinct from collection
(0), periodic validation (1), held-out final evaluation (2), and Q/F/P
selection (3). The Dense teacher and every candidate use the same fixed
16-rollout deterministic-policy cohort.

For raw episodic returns `R_dense` and `R_candidate`, the candidate passes when

```text
(R_dense - R_candidate) / max(abs(R_dense), 1.0) <= 0.05
```

All candidates run before choosing the smallest passing width. No pass means
Dense fallback. Reward-scaled return is auxiliary only. Validation transitions
never enter Replay, and held-out final data never controls capacity.

## Exact extra compute

| quantity | value |
|---|---:|
| online world-model updates | `540,000` |
| shared consolidation updates | `6,000` |
| Q/F/P compression updates | `6,000` |
| online Actor-Critic updates | `432,000` |
| Actor-Critic compression updates | `6,000` |
| Actor-Critic compression imagined states | `1,536,000` |
| Q/F/P selector validation rollouts | `480` |
| Actor-Critic selector validation rollouts | `480` |

The 6,000 behavior-compression updates are extra compute and use a counter
separate from online Actor-Critic updates.

## Parameter bounds

| behavior component | parameters |
|---|---:|
| shared MLP Actor + Critic | `1,715,985` |
| one full-width residual pair | `1,720,081` |
| one width-64 residual pair | `220,625` |
| all six behavior reuse routes | `120` |

The behavior subsystem ranges from `12,036,591` parameters (all Dense fallback)
to `3,039,855` (all width 64). Joint Q/F/P and behavior outcomes range from
`54,638,216` to `25,679,048` online parameters. Mixed selected widths must be
reported directly from runtime artifacts; neither bound is a performance
claim. Peak acquisition allocation can remain Dense even after physical
modules are replaced.

## Checkpoints and artifacts

Actor and Critic residual banks persist a per-task hidden-width buffer. Strict
loading rebuilds heterogeneous modules before tensor and optimizer restoration.
Each post-boundary checkpoint stores the cumulative, separate behavior
compression update count.

Every successful task boundary writes
`adaptive_behavior_compression/task_XX_boundary.json`, containing Dense and
candidate raw/scaled returns, selection hashes, Actor/Critic KL summaries,
fixed compute counters, pass decisions, selected layouts, and physical
parameter deltas. Q/F/P artifacts remain separate. A compression exception
restores the Dense Actor-Critic topology and optimizer, writes a failure record,
and stops the run.

## Dry run

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --prediction-head-profile shared_distilled \
  --adaptive-qfp-compression \
  --behavior-profile shared_adaptive_residual_mlp \
  --seed 0 \
  --classification pilot \
  --dry-run
```

The dry run performs no interaction or gradient update. A real run additionally
requires a clean pushed commit and this target-CUDA smoke:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --method-profile adaptive_qfp_ac_compression \
  --device cuda:0
```

The smoke uses synthetic data only. It verifies one normal shared/residual
Actor-Critic update, physical Q/F/P and Actor/Critic compaction, and strict
dynamic-topology checkpoint reload; it is not a performance result.
