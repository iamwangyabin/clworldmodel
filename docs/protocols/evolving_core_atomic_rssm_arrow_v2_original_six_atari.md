# Evolving-Core Atomic RSSM ARROW v2 Original-Six Pilot (Atari)

## Scope

`Evolving-Core-Atomic-RSSM-ARROW-v2-OriginalSix-Atari-TaskAware-Pilot` is a
from-scratch, seed-0 complete-horizon extension of the three-task Evolving-Core
v1 pilot. It changes the curriculum length and per-task parameter count, so it
is separately named and does not supersede or retroactively reinterpret v1.

The fixed order is:

1. `ALE/MsPacman-v5`;
2. `ALE/Boxing-v5`;
3. `ALE/CrazyClimber-v5`;
4. `ALE/Frostbite-v5`;
5. `ALE/Seaquest-v5`; and
6. `ALE/Enduro-v5`.

Each task lasts 90 epochs. Training stops at 540 completed epochs, exactly at
the Enduro boundary. Seed index 0 maps to `123456789`. The classification is
fixed to `pilot`; `official` is rejected.

## Inherited Method

All Evolving-Core v1 learning semantics remain fixed:

- one continually plastic CNN and base posterior/recurrent/prior RSSM;
- symmetric zero-effect projectors, four-atom residual mechanisms, private
  decoder/reward/continue heads, and independent Actor-Critics for all tasks;
- 512 FIFO plus 512 LTDM trajectories and 50/50 sub-buffer selection;
- a 16-sequence Task-0 update and 12 current plus four task-homogeneous LTDM
  memory sequences on later tasks;
- posterior, hidden-state, and frozen-old-Actor interface protection;
- component-wise conflicting-current-gradient projection for the shared core;
  and
- 1,000 task-balanced shared-only consolidation updates at every boundary,
  with fixed-cohort return validation and rollback.

Task identity selects private routes and Actor-Critics. Evaluation transitions
never enter Replay or optimization. Future-task evaluation remains isolated
and may be used only for explicitly labeled forward-transfer diagnostics.

## Budgets

| quantity | value |
|---|---:|
| task durations | `90 x 6` epochs |
| raw emulator frames | `35,389,440` |
| online world-model updates | `540,000` |
| boundary-consolidation updates | `6,000` |
| total world-model optimizer steps | `546,000` |
| Actor-Critic updates | `432,000` |
| online current sequences | `6,840,000` |
| online memory sequences | `1,800,000` |
| consolidation sequences | `96,000` |

The additional consolidation compute and task-owned parameters must be
reported. This run is not compute- or capacity-matched to plain ARROW-50.

## Storage Gate

Live uint8 image Replay has 6,442,450,944 logical bytes. This pilot fixes
`evolving_checkpoint_retention=latest_boundary`. The trainer first writes and
checksums the new task's complete pre/post resumable pair. Only after both are
durable does it remove older resumable pairs and their immutable Replay assets.
Peak immutable Replay storage is therefore two boundaries, or
12,884,901,888 bytes; live plus peak Replay observations total
19,327,352,832 bytes. Only the newest pre/post pair supports exact resume.

Raw metrics, TensorBoard events, consolidation records, the retention manifest,
and all task-bank inference snapshots remain preserved. No state is omitted
from the retained checkpoint. Before launch the launcher records capacity and
requires at least 48 GiB free for model/optimizer payloads, checksums, auxiliary
Replay tensors, logs, and one atomic checkpoint temporary in addition to the
observation-byte minimum.

## Execution

Inspect the exact resolved command without interaction or updates:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --seed 0 \
  --classification pilot \
  --dry-run
```

A non-dry run requires a clean, committed, pushed, upstream-synchronized Git
state and a unique target-CUDA smoke. Use unique output and replay-mmap roots;
the launcher refuses to overwrite an existing run.

## Evidence Limits

The primary outputs are raw per-task returns at all evaluation checkpoints,
final average return, forgetting, boundary consolidation decisions, shared
component conflict rates, and exact resource accounting. One seed is pilot
evidence only. It cannot support a reproduction, multi-seed reliability,
task-agnostic, or superiority claim.
