# 0040: Fixed-Order Task-0 Hyperparameter Selection

## Status

Accepted as a seed-0 pilot design on 2026-08-29. No general performance claim
exists until fresh confirmation seeds complete.

## Context

The first Evolving-Core campaign initially allocated spare GPUs to task-order
and seed variations. The current research question is narrower: preserve the
original MsPacman, Boxing, CrazyClimber curriculum and test whether Task-0
acquisition is sensitive to shared-core, task-private, or Actor-Critic learning
rate. Changing order would confound that diagnosis.

## Decision

Keep the original GPU-0 `fixed_v1` run as the unchanged control. Reallocate the
four spare GPUs to four one-factor Task-0 LR profiles. Stop each at the common
90-epoch boundary. Select only with the fixed-cohort, 16-rollout,
pre-consolidation raw MsPacman mean. Persist that measurement before any
consolidation update and prohibit held-out-final evaluation in sweep jobs.

Require the complete five-profile set. Break an exact score tie by smallest
log-space distance from the fixed setting and then lexical profile name. Do not
change task order, seed, samples, optimizer-update count, Replay, architecture,
or any loss coefficient.

## Consequences

- The comparison diagnoses one Task-0 optimizer dimension at a time.
- The full control can continue while the four spare GPUs execute selection
  candidates.
- Four candidates each spend 90 epochs of pilot compute; this is explicitly
  extra and does not make their later-task performance comparable.
- A non-default winner requires a fresh full-curriculum launch rather than
  relabeling the sweep checkpoint as an equivalent resume.
- The selection result remains seed-specific and requires fresh confirmation
  seeds before any superiority or reliability claim.
- A sweep reduces risk but cannot guarantee that MsPacman is learned well.

## Rejected Alternatives

- Task-order permutations answer a different question and were stopped without
  being treated as scientific failures.
- Using the held-out-final cohort for selection would contaminate reporting.
- Ranking post-consolidation return would give the selector access to the
  extra 1,000-update phase.
- Simultaneously changing several LRs would make a successful package harder to
  interpret with only one selection seed.
