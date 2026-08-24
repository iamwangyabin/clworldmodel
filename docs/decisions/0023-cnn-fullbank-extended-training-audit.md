# Decision 0023: Constant-hyperparameter extended training audit

## Status

Accepted for the next CNN-FullBank seed-0 Task 1 diagnostic run. This is an
extended-budget ablation and not a result under the 90-epoch protocol.

## Context

The 90-epoch `x4-full-updates` run used the same CNN-FullBank architecture as
the single-device method, but it was not merely a device-count change. Its
global world-model sequence batch was 64 instead of 16 and Actor-Critic context
batch was 512 instead of 128. With unchanged Adam update counts and learning
rates, each update averaged four times as many samples and followed a different
optimization trajectory.

One periodic evaluation reached 2008.125, but each periodic checkpoint used a
different Atari seed cohort and its standard deviation was 397.794. The point
is not reliable evidence that the policy stably crossed 2000. The broader curve
does show learning, so insufficient task duration remains a live hypothesis.

## Decision

Run a new 180-epoch Task 1 pilot with:

- CNN-FullBank, ARROW-50, BF16, uint8 mmap replay, DP4, and seed 0;
- the unchanged `x4-full-updates` batches, 1,000 WM updates per epoch, 800
  Actor-Critic updates per epoch, and 30,000 initial WM updates;
- constant WM and Actor-Critic learning rates of `1e-4` and constant entropy
  scale `3e-4` for all 180 epochs;
- one fixed periodic validation seed cohort, reused every 10 epochs;
- a disjoint held-out final seed cohort; and
- exact non-resumable task-bank inference snapshots for every evaluated state.

Do not enable the late Actor-Critic cosine schedule in this run. That profile
tests a different hypothesis and would make the source of any improvement
ambiguous.

## Consequences

Relative to the failed 90-epoch run, this audit doubles environment frames,
WM/Actor-Critic updates, optimization samples, and nominal training duration.
Its score cannot be used as a matched-budget reproduction. It can determine
whether the large-batch path continues improving under constant
hyperparameters and whether the apparent epoch-50 peak survives a fixed cohort.

The acquisition criterion remains a predeclared held-out final raw MsPacman
mean of at least 2000. Validation peaks and their retained snapshots are
diagnostic and cannot be substituted for the held-out final result.
