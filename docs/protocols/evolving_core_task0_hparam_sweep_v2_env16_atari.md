# Evolving-Core Task-0 Hyperparameter Sweep v2 EnvParallel16 (Atari)

## Question And Scope

This is a seed-0 MsPacman acquisition pilot for the fixed curriculum
**MsPacman, Boxing, CrazyClimber**. It compares the four learning-rate changes
from the v1 Task-0 sweep with a newly trained, matched `fixed_v1` control. Every
candidate stops after 90 MsPacman epochs. No candidate sees Boxing, and no
held-out-final evaluation is run or read during selection.

The runtime collection layout is explicitly named `EnvParallel16`: each epoch
uses 16 parallel Atari environments for 1,024 decisions each instead of four
environments for 4,096 decisions each. Both layouts collect exactly 16,384
agent decisions, or 65,536 raw frames at frame repeat four, per epoch. Changing
the number of workers changes worker seeds and trajectory partitioning, so this
is a new runtime protocol rather than a silent acceleration of the v1 stream.
All five profiles use the same EnvParallel16 layout and are compared only with
one another.

| profile | Task-0 shared-core LR | private projector/atom/head LR | Actor-Critic LR |
|---|---:|---:|---:|
| `fixed_v1` control | `2e-4` | `2e-4` | `1e-4` |
| `task0_shared_lr_1e4` | `1e-4` | `2e-4` | `1e-4` |
| `task0_shared_lr_3e4` | `3e-4` | `2e-4` | `1e-4` |
| `task0_private_lr_3e4` | `2e-4` | `3e-4` | `1e-4` |
| `task0_actor_lr_2e4` | `2e-4` | `2e-4` | `2e-4` |

## Matched Scientific Budget

Each from-scratch candidate uses seed index 0 and exactly:

- 90 epochs, 1,474,560 agent decisions, and 5,898,240 raw emulator frames;
- 90,000 online world-model updates with sequence batch `T=32, N=16`;
- 72,000 Actor-Critic updates;
- the same ARROW-50 FIFO/LTDM capacity and buffer-selection probability;
- uint8 CPU mmap Replay and the same BF16/TF32 numerical settings; and
- 1,000 separately reported boundary-consolidation updates.

Replay paths and all output paths are private to a candidate. Evaluation
transitions never enter Replay. The ranking score is the fixed 16-rollout raw
MsPacman mean written before consolidation, so consolidation cannot improve a
candidate's selection score.

## Preemptible Shared-GPU Operation

This campaign may run on opportunistically idle GPUs. An operator must sample
the accelerator process table before launch and continuously while a candidate
runs. A candidate may start only after its assigned GPU is stably free of
external compute processes. If an external compute process appears on that
GPU, terminate only the campaign's process group, preserve the interrupted
attempt as partial and ineligible, and release the GPU. Do not signal the
external process.

Task-0 has no scientifically equivalent mid-run Replay checkpoint in this
protocol. A preempted profile therefore returns to the queue and later starts
from scratch in a new, monotonically numbered attempt directory. Partial
attempts cannot be selected or described as resumes. An idle GPU may continue
running another candidate when a different GPU becomes occupied.

## Eligibility And Selection

All five profiles must complete under the exact same pushed project commit,
`n_sync=16`, `gen_seq_len=1024`, seed index 0, and fixed validation cohort. The
standalone control is not exempt from completion. An eligible run has a true
`run_status.complete`, both Task-0 resumable boundary checkpoints, a finite
pre-consolidation raw return, no held-out-final artifact, and the required
consolidation success or failure record.

Rank candidates by descending pre-consolidation raw mean. Resolve an exact tie
by the smaller sum of absolute natural-log LR ratios from `fixed_v1`, then the
lexical profile name. Seed 0 is a selection pilot only. The winner must be run
from scratch in a full curriculum, and fresh confirmation seeds are required
before a performance or retention claim. A mathematical winner should not be
promoted if every absolute score is scientifically inadequate.

## Launch And Selection

Launch each profile with a unique output and Replay root:

```bash
python scripts/run_evolving_task0_sweep.py \
  --profile fixed_v1 \
  --collection-envs 16 \
  --seed 0 \
  --output-dir /path/to/attempt/run \
  --replay-mmap-root /path/to/attempt/replay
```

Repeat for the four declared LR profiles. After all five eligible attempts
exist, run `scripts/select_evolving_task0_profile.py` with one
`--candidate-dir` per profile. Both the training launcher and selector reject
protocol mixing. Non-dry training also enforces a clean, committed, pushed,
upstream-synchronized Git state before environment interaction or updates.
