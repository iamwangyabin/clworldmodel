# Never-clear Dream Rehearsal v1 Atari protocol

## Status

Implemented for a target-GPU smoke followed by one explicitly single-seed
pilot.  Neither run is a reproduction claim.  The purpose is to test the
paper's full-history memory semantics before judging Dream Rehearsal from a
storage-bounded Atari port.

## Reference

- Paper: *The World Model Remembers, the Actor Forgets: Dream Rehearsal for
  Continual Model-Based RL*, arXiv:2607.19749.
- Repository: `https://github.com/gurpnijjer/dream-rehearsal`.
- Inspected commit: `7680778f798be3a27a17c320cc875b573c45f0e1`.
- Artifact license: Apache-2.0.

The dream generation, realized-first scoring, top-quarter selection, and
actor-only self-imitation are the same implementation used by
`Bounded-Dream-Rehearsal-v1-Atari`.  This protocol changes replay retention and
ordinary-training data flow rather than silently changing that algorithm.

## Never-clear replay invariant

Every trajectory collected during the configured run receives one unique FIFO
slot.  The store is preallocated because the Atari schedule has a finite known
length, but it has exactly the semantics required here: no collected
trajectory is evicted, overwritten, subsampled, or rejected.

For the 541-epoch original-order seed-0 curriculum:

| quantity | value |
|---|---:|
| trajectories collected per epoch | 32 |
| complete trajectory slots | 17,312 |
| transitions per trajectory | 512 |
| total retained transitions | 8,863,744 |
| uint8 observation bytes | 108,917,686,272 |
| float32 action/reward/continue/reset bytes | 744,554,496 |
| int64 task-metadata bytes | 138,496 |
| total tensor bytes | 109,662,379,264 |

The observations are losslessly stored Atari pixels in one CPU file-backed
`uint8` mmap.  Auxiliary tensors remain float32 CPU tensors.  Filesystem and
Python-container overhead are reported separately when measurable.

One physical mmap is used instead of six Python replay objects.  The task ID
creates exact per-task logical views and is used only by replay sampling and
the scheduler.  It is never passed to the world model, actor, or critic.

## Training data flow

The current scheduled task is the sole source for ordinary DreamerV3 updates:

1. collect the current task and append every complete trajectory;
2. sample current-task sequences for all ordinary world-model updates;
3. sample current-task start states for all ordinary actor-critic updates; and
4. sample old-task sequences only for actor-only Dream Rehearsal.

This separation is load-bearing.  It matches the reference artifact's role of
the current task library versus prior-task libraries and avoids the bounded v1
diagnostic's accidental global mixing of ordinary updates.

`replay_task_id` and model `task_id` are separate interfaces.  This protocol
sets the former to the current replay task and keeps the latter `None`, so task
metadata cannot become a privileged network input.

## Dream Rehearsal schedule

After every 2,000 completed agent decisions, schedule 50 actor-only updates for
each encountered non-current task.  Each update:

- observes four task-filtered sequences for 16 context positions;
- imagines 64 trajectories of horizon 15 with the live shared actor;
- grades them with realized-first reward, continuation, and critic bootstrap;
- retains the top 25% (16 trajectories); and
- minimizes behavior-cloning loss on their sampled actions.

The finite 541-epoch projection is unchanged from the bounded port: 541,000
base world-model updates, 432,800 base actor-critic updates, and 554,900 extra
actor-only rehearsal updates.  Extra compute is reported rather than hidden.

## Declared differences from the paper artifact

1. six Atari games and the repository's ARROW-derived DreamerV3 implementation
   replace MiniGrid and the NM512 stack;
2. uint8 file-backed pixels replace any expanded in-memory observation format,
   without dropping samples;
3. a single physical task-labelled store implements per-task logical
   libraries; and
4. events due every 2,000 decisions are batched after a 16,384-decision Atari
   collection epoch while preserving the exact event and update count.

These differences require the name
`Never-Clear-Dream-Rehearsal-v1-Atari`.  Results must never be presented as the
paper's MiniGrid reproduction.

## Launch

After committing, pushing, and synchronizing the exact revision, run a
non-claimable target-GPU smoke:

```bash
python scripts/run_never_clear_dream_rehearsal_atari.py \
  --seed 0 \
  --smoke-epochs 1 \
  --output-dir /persistent/path/never_clear_dream_rehearsal_smoke_s0
```

Then run the single-seed pilot:

```bash
python scripts/run_never_clear_dream_rehearsal_atari.py \
  --seed 0 \
  --cpu-threads 12 \
  --profile-stages \
  --output-dir /persistent/path/never_clear_dream_rehearsal_original_s0
```

Preserve the full launch manifest, resolved config, replay byte accounting,
rehearsal accounting, raw per-task returns, TensorBoard events, logs, and task
boundary analysis snapshots.
