# Dream Rehearsal ArrowMatched v1 Atari protocol

## Status

Implemented; a target-CUDA smoke is required before the six-task pilot. This is
an Atari/DreamerV3 adaptation, not a reproduction of the paper's MiniGrid
numbers.

## Fair comparison contract

Both memory arms use the same original-order DreamerV3 control config and the
same vendored trainer as ARROW. They therefore keep exactly:

| counter | value |
|---|---:|
| epochs | 541 |
| collected agent decisions | 8,863,744 |
| nominal raw frames (repeat 4) | 35,454,976 |
| world-model updates | 541,000 |
| base actor updates | 432,800 |
| base critic updates | 432,800 |
| periodic checkpoints | 55 |
| requested evaluation episodes (6 tasks x 16) | 5,280 |

The historical 541-epoch schedule is preserved exactly: six 90-epoch phases
followed by one task-0 revisit epoch. Evaluation uses the same trainer path and
never enters replay.

Dream Rehearsal's actor-only rehearsal is method-specific extra compute, not an
extra environment-sample allowance. It is logged separately: 554,900 actor
updates for this schedule. Consequently the base training budget is matched to
ARROW, while total optimizer/forward compute is not. Removing or silently
charging these updates to the ordinary actor/critic budget would define a
different algorithm.

## Dream Rehearsal boundary

The integration uses one shared world model, actor, and critic. Task IDs are
replay/scheduler metadata and are never network inputs. Each old-task update:

1. samples 16 retained sequences and observes 64 context steps;
2. imagines horizon-15 trajectories from all 1,024 posterior features;
3. applies the reference realized-first score and top-25-percent selection;
4. bootstraps from the final stored imagined feature, matching the inspected
   `tunnel_update` artifact; and
5. behavior-clones the selected 256 sampled action trajectories into the actor
   only.

The paper cadence of 50 updates per encountered non-current task per 2,000
agent decisions is retained. Because ARROW collects 16,384 decisions per epoch,
due work is performed at the next epoch optimizer boundary without changing
its count.

## Memory pair

The resolved configs differ only in `sac_dv3_data_n_max`:

| arm | trajectory slots | transition capacity | retention |
|---|---:|---:|---|
| `full` | 17,312 | 8,863,744 | every trajectory in the declared run |
| `bounded` | 1,024 | 524,288 | uniform random-key reservoir |

Both use one CPU file-backed uint8 observation store and the same float32
auxiliary tensors. The full arm is finite: its predeclared capacity equals the
entire interaction budget. The bounded arm has no hidden full archive or
frozen phase libraries. Ordinary world-model and actor/critic training samples
the same retained global history used for rehearsal.

Sample capacity is the primary comparison; allocated bytes are still reported
for each arm. The full arm requires about 108.9 GB for observation pixels plus
about 0.74 GB of auxiliary tensors.

## Launch

Inspect either arm without creating a run:

```bash
python scripts/run_dream_rehearsal_arrow_matched_atari.py \
  --history bounded --seed 0 --dry-run
```

Run the two-task target-GPU wiring smoke first:

```bash
python scripts/run_dream_rehearsal_arrow_matched_atari.py \
  --history bounded --seed 0 --smoke --output-dir /persistent/smoke
```

After it passes, launch `full` and `bounded` as separate sequential runs. A
single seed remains a pilot, not a reproduced result.
