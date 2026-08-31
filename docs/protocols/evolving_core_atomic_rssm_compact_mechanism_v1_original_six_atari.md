# Evolving-Core Compact-Mechanism v1 Original-Six Pilot (Atari)

## Scope

`Evolving-Core-Atomic-RSSM-CompactMechanism-128-128-64-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`
is a from-scratch, seed-0 capacity ablation of
`Evolving-Core-Atomic-RSSM-ARROW-v2-OriginalSix-Atari-TaskAware-Pilot`.
It asks whether the private RSSM residual mechanisms can be substantially
narrower without changing any learning or evaluation budget.

The fixed order is MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and
Enduro. Each task lasts 90 epochs; training ends at epoch 540. Seed index 0 is
`123456789`. The run classification is fixed to `pilot`.

## Single controlled change

| mechanism | fixed input/output | control width | compact width | atoms | compact parameters/task |
|---|---:|---:|---:|---:|---:|
| recurrent | `512 -> 512` | 512 | 128 | 4 | 132,736 |
| representation/posterior | `4608 -> 1024` | 512 | 128 | 4 | 731,264 |
| transition/prior | `512 -> 1024` | 256 | 64 | 4 | 100,416 |
| **total** | | | | | **964,416** |

Every branch remains `LayerNorm -> Linear -> SiLU -> Linear`, retains residual
scale `0.1`, and has an exactly zero-effect output layer at task creation. Only
the bottleneck dimensions change. The four atoms are partitions of that
bottleneck and become widths `32/32/16`.

Across all six tasks, private mechanisms plus reuse-route scalars contain
`5,786,676` parameters instead of `22,897,332`, reducing the targeted capacity
by `17,110,656` (`74.7%`). The fixed projector, independent Actor-Critic bank,
and private decoder/reward/continue heads are unchanged. Training-time total
capacity therefore remains larger than ARROW; action-only deployment can prune
the training-only heads and critics but cannot prune these mechanisms.

## Inherited protocol

All other fields are inherited unchanged from the original-six control:

- one continually plastic shared CNN and base RSSM;
- task-aware projectors, routes, private heads, and independent Actor-Critics;
- ARROW-50 replay with 512 FIFO plus 512 LTDM trajectories;
- 16 Task-0 sequences and a later `12 current + 4 old LTDM` split;
- the same interface losses and component-wise conflict projection;
- 1,000 task-balanced shared-only consolidation updates per boundary with a
  five-percent fixed-cohort rollback limit;
- BF16 compute, uint8 CPU mmap replay, fused Adam, TF32, and no world-model
  compilation; and
- `latest_boundary` complete-checkpoint retention with the same 48 GiB launch
  storage gate.

Consequently the interaction and optimizer-step budgets remain:

| quantity | value |
|---|---:|
| raw emulator frames | `35,389,440` |
| online world-model updates | `540,000` |
| boundary-consolidation updates | `6,000` |
| total world-model optimizer steps | `546,000` |
| Actor-Critic updates | `432,000` |

The narrower matrices reduce per-update arithmetic and parameter/optimizer
bytes, so wall time and FLOPs are not asserted to be matched. Update counts,
sample counts, and environment interaction are matched.

## Execution

Inspect the resolved config and manifest without interaction or updates:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --mechanism-profile compact_128_128_64 \
  --seed 0 \
  --classification pilot \
  --dry-run
```

A non-dry launch requires a clean, committed, pushed, upstream-synchronized
Git state, a target-CUDA smoke, at least 48 GiB free, and a unique output path.
The profile-specific synthetic smoke is:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --mechanism-profile compact_128_128_64 \
  --device cuda:0
```

It performs no environment interaction and supports only execution evidence.

## Reporting and claim limits

Report the complete raw evaluation matrix, acquisition returns, final average
return, forgetting, consolidation decisions, exact parameter artifacts, peak
memory, and elapsed time next to the high-capacity seed-0 control. Do not select
only favorable tasks. This single seed cannot support a reproduction,
superiority, multi-seed reliability, or task-agnostic claim.
