# Decision 0004: use native R2-Dreamer updates with ARROW-50 replay

- Status: accepted for a single-task acquisition sanity run
- Date: 2026-08-17

## Context

`ARROW-R2Rep-50` kept the ARROW model and replaced only its pixel loss. Its
seed-0 pilot completed without a numerical failure, but it did not establish
strong acquisition on several individual tasks. In particular, it coupled a
4,096-dimensional encoder target with only `16 x 32 = 512` flattened Barlow
samples. That is not the official R2-Dreamer geometry and cannot answer a
continual-retention question.

## Decision

Add the separately named `R2Dreamer-ARROW-50` route. It uses the pinned
R2-Dreamer `size12M` model and optimizer profile: decoder-free CNN encoder,
discrete RSSM, `B=16`, `T=64`, 1,024-dimensional embedding, LaProp, AGC,
1,000-update warm-up, and the native model-sample-per-decision train ratio of
128. ARROW provides only trajectory retention and 50/50 FIFO/LTDM whole-
minibatch selection.

The integration keeps the R2 replay contract by adding a float32 CPU sidecar
for posterior RSSM states and a boolean `is_last` label per stored transition.
A sampled ARROW context transition initializes R2's 64 learned steps, and
refreshed posterior states are written back to the same retained slots. The
sidecar is metadata: it does not change which trajectories FIFO or LTDM retain,
nor the 50/50 selection rule. Its byte footprint is saved with every run.

The default `single-task` scope is an acquisition check close to the upstream
Atari-100k 410,000-frame budget. ARROW collects whole 16,384-position blocks,
so it runs seven nominal 458,752-frame blocks. The trainer derives its R2
updates from realized agent decisions, excluding terminal/autoreset positions,
and records the resulting counters. The `continual` scope preserves the
six-task ARROW schedule but is explicitly a native-R2-compute experiment, not
a compute-matched ARROW comparison.

## Consequences

- `R2Dreamer-ARROW-50` and `ARROW-R2Rep-50` are different methods and must not
  share a result table without labels.
- The native-R2 run uses 2,048 `16 x 64` updates per collection block, versus
  ARROW's 1,000 `16 x 32` updates. It has 4.096 times the sampled model
  transitions per block and cannot support a fair ARROW performance claim.
- The first acceptance criterion is single-task learning with finite losses,
  non-degenerate Barlow statistics, and improving deterministic return. Only
  then may the continual scope be considered.
- A later matched-compute continual protocol, if needed, must receive its own
  decision and name; it cannot silently replace this native-R2 route.
