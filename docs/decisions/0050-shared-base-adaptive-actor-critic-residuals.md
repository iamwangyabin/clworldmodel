# 0050: Share Actor-Critic bases and adaptively compress task residuals

## Status

Accepted for a separately named implementation on 2026-09-04. This decision
authorizes code, tests, and dry-run manifests only. A CUDA smoke or training run
still requires an explicit launch instruction from a clean, pushed commit.

## Context

Decision 0049 makes Q/F/P capacity task adaptive, but method D still stores a
complete MLP Actor and Critic for every task. That behavior bank is a second
linear source of parameter growth. Fixed small behavior heads are risky for the
same reason that fixed Rank-32/128 Q/F/P failed: the required capacity is not
known before acquisition.

The requested hypothesis is to apply the same **acquire wide, then compress by
measured return** rule to behavior. This must remain a new method rather than
silently changing D, because it changes sharing, optimization, rehearsal,
boundary compute, checkpoint topology, and the maximum allocated capacity.

## Decision

Add method
`evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow` and
protocol
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFPAC-SharedDistilledHeads-SharedResidualMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

Actor and Critic each use

```text
task logits = shared MLP logits + routed task residual logits.
```

The two shared MLP bases are stored once. Every task, including Task 0, owns an
exact-zero-initialized nonlinear residual with input width 1,536, acquisition
hidden width 512, residual scale 0.1, and four atoms. Explicit task identity
selects the Actor residual, Critic residual, and learned reuse routes. Old
residuals and old routes are frozen; online optimization updates the shared
bases and only the acquiring task's residual/routes. The unchanged 800
Actor-Critic updates per epoch are allocated 75 percent to the current task and
25 percent uniformly over available old task replay after Task 0. Thus sharing
does not receive hidden extra online optimizer steps.

At every task boundary, after shared-world-model consolidation and Q/F/P
compression, the completed Actor/Critic residual pair is compressed. Four
independent candidates retain hidden-width fractions 0.75, 0.5, 0.25, and
0.125 (`384`, `256`, `128`, and `64`). Each candidate starts from the same
frozen full-width teacher and receives 250 Adam updates at `2e-4`. Initial
latent states come only from that completed task's LTDM replay. Sixteen-step
imagined trajectories provide Actor policy KL and Critic 255-bin categorical
KL targets, each at scale 1.0. Shared bases, older residuals, and all routes are
frozen during this recovery phase.

A separate seed-sequence domain supplies the fixed 16-rollout real-environment
validation cohort. After all candidates consume equal compute, the smallest
candidate whose relative raw-return drop is at most five percent is installed.
If none passes, the original full-width residual pair and its Adam state are
restored. Final held-out evaluation is never used for selection.

## Budget and parameter consequences

Across six tasks, behavior compression adds exactly 6,000 optimizer updates,
1,536,000 imagined states, and 480 real validation rollouts. These are separate
from 432,000 online Actor-Critic updates and from method D's world-model
compression budget.

The shared Actor/Critic MLP bases contain 1,715,985 parameters. A full-width
task residual pair contains 1,720,081 parameters; a width-64 pair contains
220,625. Both behavior banks together add 120 route parameters over six tasks.
Consequently behavior parameters range from 12,036,591 under all-Dense fallback
to 3,039,855 when all tasks accept width 64.

Combining behavior selection with adaptive Q/F/P gives an outcome-dependent
six-task online model between 54,638,216 and 25,679,048 parameters. The upper
bound is 1,740,681 parameters larger than method D because full-width
acquisition stores shared bases in addition to six full residual pairs.
Therefore **parameter reduction is conditional, not guaranteed**. The method
is useful only if post-boundary gates accept enough compact routes without
damaging return or retention.

## Consequences and evidence required

- Method D remains unchanged and is the required attribution control.
- The protocol is task-aware. A separate router may supply task identity, but
  these results cannot be called task-agnostic.
- Candidate-validation return is model-selection data, not final performance.
- Checkpoints persist heterogeneous Actor and Critic width buffers and rebuild
  physical modules before strict state loading.
- Failed compression writes an explicit failure artifact and stops; it never
  silently becomes a successful compact run.
- Seed-0 is only a pilot. Claims require matched multi-seed raw returns, final
  average performance, forgetting, selected widths, and actual parameter bytes.
