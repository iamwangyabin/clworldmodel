# Dense-Acquire Adaptive Q/F/P Compression v1 (Atari)

## Scope

The protocol name is
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.
Its config method key is
`evolving_atomic_rssm_adaptive_compression_shared_heads_arrow`.

This is a separately named, from-scratch, seed-0-capable pilot derived from
experiment A. It is not a redefinition of A and has no validated performance
result yet.

## Fixed curriculum and online acquisition

The task order is MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and
Enduro, with 90 epochs per task. Replay remains ARROW-50 with equal FIFO/LTDM
capacity and selection probability. The Task-0 optimizer profile is the
original-six `fixed_v1` profile.

Online learning before every boundary is identical in topology to the shared
distilled-head experiment:

- one plastic shared CNN and base posterior/recurrent/prior RSSM;
- one plastic shared decoder/reward/continue set, protected with output
  distillation scale `0.1`;
- one full Dense private Q/F/P route per active task at `512/512/256`, split
  into four atoms;
- one private spatial projector and one independent MLP Actor-Critic per task;
- 16 current sequences for Task 0 and a 12-current/4-old-LTDM split later;
- component conflict projection, old-interface protection, and task-balanced
  1,000-update shared consolidation with five-percent rollback.

Task identity is available to select these routes. The protocol is task-aware.

## Boundary compression algorithm

Compression runs after accepted shared consolidation and before the cumulative
boundary teacher/checkpoint is committed.

The accepted full-width world model is frozen as one Dense teacher. Four
candidates are then attempted in this fixed order:

| fraction | recurrent Q | posterior F | prior P |
|---:|---:|---:|---:|
| `0.75` | `384` | `384` | `192` |
| `0.50` | `256` | `256` | `128` |
| `0.25` | `128` | `128` | `64` |
| `0.125` | `64` | `64` | `32` |

For a Dense residual `up(silu(down(LN(x))))`, channel importance is the product
of the channel's incoming row/bias norm and outgoing column norm. Ranking is
performed independently within each of the four atoms. Every candidate keeps
the same number of channels per atom, copies the selected Dense rows/columns,
and materializes physically smaller `Linear` tensors.

Each candidate starts independently from the same Dense teacher and receives:

- 250 Adam updates;
- learning rate `2e-4`;
- 16 completed-task LTDM sequences per update;
- the exact same restored Python, NumPy/replay, CPU torch, and CUDA torch
  sampling state as every other width; and
- only its recurrent/posterior/prior private parameters enabled.

The recovery objective is the existing real-target Dreamer old-task loss,
posterior/hidden/frozen-actor interface protection, and shared-head output
distillation, plus mean squared teacher/student Q/F/P output matching at total
scale `1.0`. It does not update the shared core, projector, atom route,
prediction heads, or Actor-Critic.

## Selection and evaluation isolation

A fourth deterministic seed-sequence domain, separate from collection,
periodic validation, and final held-out evaluation, supplies one fixed
16-rollout pruning cohort per task. The Dense teacher and every candidate use
the same task seed.

For raw episodic returns `R_dense` and `R_candidate`, a candidate passes when

```text
(R_dense - R_candidate) / max(abs(R_dense), 1.0) <= 0.05
```

After all candidates have consumed their fixed compute, the smallest passing
width is installed. If none passes, the full Dense Q/F/P is restored. Reward-
scaled means are logged only as auxiliary values and do not drive selection.
The final held-out cohort is never used for width selection, and no evaluation
transition enters Replay.

## Exact extra compute

| quantity | value |
|---|---:|
| online world-model updates | `540,000` |
| shared-consolidation updates | `6,000` |
| adaptive-compression updates | `6,000` |
| total world-model optimizer steps | `552,000` |
| compression LTDM sequences | `96,000` |
| compression validation rollouts | `480` |
| Actor-Critic updates | `432,000` |

Compression compute is explicitly extra relative to experiment A. All four
candidate budgets are paid even if a larger candidate already passes.

## Parameter bounds and runtime accounting

The full experiment-A maximum is `52,897,535` parameters. Physical selections
make the final count outcome dependent:

| all-task outcome | final online parameters | FP32 parameter bytes |
|---|---:|---:|
| all Dense fallback | `52,897,535` | `211,590,140` |
| all `0.75` | `47,193,983` | `188,775,932` |
| all `0.50` | `41,490,431` | `165,961,724` |
| all `0.25` | `35,786,879` | `143,147,516` |
| all `0.125` | `32,935,103` | `131,740,412` |

Mixed outcomes lie between the bounds. V1 preallocates the same full task bank
as A to preserve acquisition initialization, then replaces completed modules.
Consequently peak allocation remains Dense even when the saved final model is
smaller. Logs must report actual live parameter counts and selected widths at
every boundary, separately from CUDA allocator reserved bytes, optimizer state,
activations, Replay, and training-only teachers.

## Checkpoint and artifact contract

Each bank persists an integer `mechanism_hidden_features` buffer. Strict loading
uses it to rebuild heterogeneous recurrent/posterior/prior modules before their
weights are loaded. Post-boundary checkpoints omit completed-task Q/F/P and
route Adam state because the curriculum never trains them again.

Every successful boundary writes
`adaptive_qfp_compression/task_XX_boundary.json` with teacher/candidate raw and
scaled returns, channel-selection hashes, loss summaries, pass decisions,
selected widths, update counters, and parameter deltas. A failure writes an
explicit failure record and stops rather than falling through as a successful
Dense run.

Final reporting keeps raw per-task returns, reward-scaled means, and ARROW
normalization separate. Final average performance and forgetting are required;
transfer is reported only when its reference points are valid. Early or
pruning-validation results are not final performance.

## Commands

Inspect the fully resolved protocol without interaction or updates:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --prediction-head-profile shared_distilled \
  --adaptive-qfp-compression \
  --seed 0 \
  --classification pilot \
  --dry-run
```

Exercise the online update, structured compaction, Q/F/P recovery loss, and
dynamic state-dict topology on a target CUDA host:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --method-profile adaptive_qfp_compression \
  --device cuda:0
```

A real pilot remains gated on a clean pushed commit and successful target-CUDA
smoke. The implementation itself is not a performance claim.
