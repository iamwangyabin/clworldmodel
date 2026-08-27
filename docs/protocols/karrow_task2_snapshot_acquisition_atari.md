# KARROW Task-2 Snapshot Acquisition Atari Protocol

## Status

Diagnostic trainability protocol. It is not a resumable continual-learning
run and cannot by itself support a forgetting or retention claim.

## Question

Can the Task-1 model fit Boxing when initialized from the completed MsPacman
boundary snapshot, if the original shared core is held stable and the KAN
residuals are given enough plasticity?

The source snapshot is:

```text
boundary_01_task_00_epoch_0089.pt
```

The continuation uses the same ARROW-50 per-epoch interaction, replay
capacity, world-model update, actor-critic update, action handling, and
observation protocol as the parent run. It runs only Task 2 for 90 epochs.

## Adaptation arms

### `kan_only`

Load the Task-1 world model and actor-critic. Freeze all original RSSM,
reward/continue, and MLP behavior parameters. Keep every KAN residual module
trainable, including its input projection, normalization, output projection,
and local RBF coefficients. This is the primary test of whether KAN alone can
acquire Boxing from the Task-1 coordinate system.

### `kan_plus_heads`

Use the same KAN plasticity, and additionally open only the final linear
readouts of the posterior, latent prior, reward, continuation, actor, and
critic heads. This is a controlled fallback for testing whether a small amount
of shared readout plasticity is enough to make Task 2 learnable.

No task ID, router, task-specific module, extra grid, or additional replay
capacity is introduced.

## Snapshot semantics

The source artifact is an analysis snapshot. The continuation therefore starts
with an empty ARROW replay buffer, a new optimizer, and RNG state derived from
the configured seed. The loaded actor is used for the first Task-2 collection;
the first collection is not replaced by a random-policy warmup.

This makes the experiment a direct acquisition test. It does not preserve the
old replay stream and must not be presented as equivalent to continuing the
original six-task run.

## Required reporting

Report both raw and scaled Boxing returns, the initial Task-1 snapshot score,
trainable parameter counts by module, world-model feature/reward/continuation
losses, actor entropy, and the exact source snapshot checksum. Compare
`kan_only` and `kan_plus_heads` under the same seed and 90-epoch Task-2 budget.

The next continual experiment should only be designed after `kan_only` or the
small-head fallback demonstrates actual Boxing acquisition; otherwise stronger
consolidation cannot be interpreted as a stability improvement.

## Launch

After committing and pushing a clean revision:

```bash
python scripts/run_karrow_task2_from_snapshot.py \
  --snapshot /path/to/boundary_01_task_00_epoch_0089.pt \
  --adaptation-mode kan_only \
  --cpu-threads 12
```

The launcher writes a resolved config, launch manifest, source checksum, task
adaptation metadata, and new analysis snapshots under a fresh output directory.
