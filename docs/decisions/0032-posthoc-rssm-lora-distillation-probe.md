# 0032: Probe Trained RSSM LoRA with a Frozen Task 1 Encoder

## Status

Accepted as a posthoc task-aware compression probe on 2026-08-26.

## Context

The completed three-task CNN-FullBank pilot stores a separate CNN encoder,
RSSM route, and actor-critic for each task.  A behavior-distilled spatial
projector already showed that the frozen MsPacman encoder can feed the native
Boxing and CrazyClimber routes.  A rank-32 truncated-SVD replacement of the
RSSM weights retained too little CrazyClimber policy performance, but that
diagnostic did not optimize a LoRA route for the task's behavior distribution.

The original mmap Replay no longer exists and the boundary checkpoint is not
resumable.  It omits optimizer, Replay, RNG, environment, and scheduler state,
so it cannot be presented as a continuation of the original 270-epoch run.

## Decision

Run a separately named
`posthoc-shared-task1-encoder-rssm-lora-distillation-v1` probe.  Freeze the
Task 1 CNN encoder and base RSSM.  For each later task, retain one independent
spatial projector, add zero-effect LoRA matrices to the base RSSM, store small
bias and normalization deltas exactly, and initialize one independent full
actor per task from the finished checkpoint.  The actor is adapted jointly to
the LoRA latent interface; its critic remains frozen and is not used for policy
evaluation.

Use rank 128 for the recurrent and representation blocks and rank 32 for the
smaller transition block.  Optimize the projector and LoRA parameters only by
functional distillation from the frozen native task route and actor on a
private deterministic-policy trajectory cohort.  Select weights on a disjoint
validation cohort and evaluate once on the held-out cohort.  No collected
transition enters Replay.  Preserve the already-completed frozen-actor run as
an explicit interface-mismatch ablation rather than overwriting it.

## Consequences

This experiment tests whether a function-aware, trained low-rank route can
compress an already learned task.  It is not evidence that online continual
LoRA learns the task from scratch, and it remains task-aware because the
projector, LoRA route, and actor are selected by task index.  A successful
result justifies a later from-scratch continual ablation; a failed result does
not rule out larger ranks or joint actor training.

The compact inference artifact is deliberately non-resumable and omits the
optimizer, RNG, trajectory frames, environment state, Replay, and task
scheduler position.  Raw task returns are preserved and the single-seed probe
must not be reported as an official comparison.
