# Evolving-Core Shared-Frozen-Down + Shared FastKAN ARROW v1 Original Six

## Scope and claim limit

`Evolving-Core-SharedFrozenDown-SharedFastKANAC-StableTargets-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`
is a from-scratch seed-level pilot. It extends the separately named three-task
protocol to the complete original ARROW Atari order without redefining or
overwriting that run. All world-model, shared FastKAN, StableTargets, Replay,
gradient-projection, interface-loss, evaluation-isolation, and checkpoint-state
semantics are inherited unchanged from
`evolving_core_shared_frozen_down_shared_fastkan_arrow_v1_atari.md`.

The scheduler supplies task identity to select task-private world-model routes
and task-conditioned Replay. Task identity is not concatenated to the shared
Actor-Critic input. The method is therefore task-aware, not task-agnostic.

## Fixed curriculum

Each task receives exactly 90 epochs:

1. `ALE/MsPacman-v5`, epochs 0--89;
2. `ALE/Boxing-v5`, epochs 90--179;
3. `ALE/CrazyClimber-v5`, epochs 180--269;
4. `ALE/Frostbite-v5`, epochs 270--359;
5. `ALE/Seaquest-v5`, epochs 360--449;
6. `ALE/Enduro-v5`, epochs 450--539.

This method retains its selected `fixed_v2` Task-0 shared-core learning rate
`3e-4`; later shared-core updates use `1e-4`. The older private-MLP original-six
pilot instead uses `fixed_v1`, so cross-protocol comparisons must disclose that
difference.

## Fixed-budget behavior routing

Exactly 800 shared Actor-Critic optimizer updates occur in every epoch:

| Acquisition task | Current route | Completed-route allocation per epoch |
|---|---:|---|
| Task 0 | 800 | none |
| Task 1 | 600 | Task 0: 200 |
| Task 2 | 600 | Tasks 0--1: 100 each |
| Task 3 | 600 | Tasks 0--1: 67 each; Task 2: 66 |
| Task 4 | 600 | Tasks 0--3: 50 each |
| Task 5 | 600 | Tasks 0--4: 40 each |

The exact route multiset is independently shuffled. Across the full run, route
update totals are Task 0 `113,130`, Task 1 `77,130`, Task 2 `68,040`, Task 3
`62,100`, Task 4 `57,600`, and Task 5 `54,000`, summing to `432,000`.
Every update samples task-homogeneous ARROW context. Evaluation data never
enters Replay.

## Budgets and storage

| Quantity | Fixed budget |
|---|---:|
| Raw Atari frames | 35,389,440 |
| Online world-model updates | 540,000 |
| Extra boundary-consolidation updates | 6,000 |
| Total world-model optimizer steps | 546,000 |
| Actor-Critic optimizer updates | 432,000 |
| Online current sequences | 6,840,000 |
| Online memory sequences | 1,800,000 |
| Consolidation sequences | 96,000 |

ARROW remains 512 FIFO plus 512 LTDM trajectories with 50/50 buffer selection,
`uint8` observations on CPU mmap storage, and unchanged metadata dtypes. Live
Replay allocates `6,486,491,136` tensor bytes, of which `6,442,450,944` are
observations. Rolling `latest_boundary` retention keeps one accepted boundary
asset; atomic replacement can transiently hold two. Live Replay plus peak
boundary observation assets require at least `19,327,352,832` observation
bytes, and the launcher requires at least 48 GiB free for Replay, checkpoints,
atomic-save temporaries, and logs.

## Parameter ledger

| Topology | World model | Behavior | Online total |
|---|---:|---:|---:|
| Shared-Frozen-Down + six private MLP pairs | 71,661,170 | 10,289,766 | 81,950,936 |
| **Shared-Frozen-Down + one shared FastKAN pair** | **71,661,170** | **1,700,670** | **73,361,840** |
| Dense Evolving-Core v2 + six private MLP pairs | 85,414,770 | 10,289,766 | 95,704,536 |

The new topology removes `8,589,096` online behavior parameters relative to
the matched private-MLP topology and is `22,342,696` parameters smaller than
the dense/private reference. It remains `52,148,026` parameters larger than
plain ARROW-50. Its FP32 online parameters occupy `293,447,360` bytes before
optimizer state, gradients, activations, Replay, or boundary teachers.

## Launch and required evidence

Dry-run inspection:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --task0-profile fixed_v2 \
  --behavior-profile shared_fastkan_stable \
  --seed 0 \
  --classification pilot \
  --dry-run
```

The non-dry launch is permitted only from a clean, committed, pushed, and
upstream-synchronized Git state after focused tests and the target-CUDA smoke
pass. A complete run must retain raw per-task evaluations, task-boundary
consolidation outcomes, route-update accounting, byte/parameter ledgers,
continual metrics, and a final run-status record. One seed is still a pilot;
multi-seed matched controls are required for a method claim.
