# Decision 0021: CNN-FullBank six-task extra-compute pilot

## Status

Accepted after the seed-0 Task 1 held-out gate passed. This is a pilot protocol,
not an official or matched-budget comparison.

## Context

The 180-epoch CNN-FullBank Task 1 run completed cleanly at commit
`be29e713e67465805f0af5e35085c151fb25e4d2`. Its independent held-out
MsPacman raw return was `2418.75 +/- 227.345` over 16 deterministic rollouts,
strictly greater than 2,000. The early numerical guard passed and all retained
inference snapshots verified by SHA256.

The successful setting uses DDP4, BF16 autocast with FP32 continuation math,
uint8 mmap replay, `x4-full-updates`, and twice the original per-task duration.
It therefore consumes substantially more samples and compute than original
ARROW and cannot be silently promoted into a fair baseline comparison.

## Decision

- Run the six original Atari tasks in their original order for 180 epochs each.
- Name the campaign `six-task-extra-compute-pilot-v1` and require an explicit,
  unique output directory and unique node-local replay backing directory.
- Freeze the original seed-0 ARROW task acquisition references and boundary
  matrix before launch. Copy the exact reference and Task 1 gate evidence into
  the run directory and record both SHA256 values in `launch.json`.
- Use task-specific raw returns. Never apply the MsPacman 2,000 threshold to
  another game and never average raw returns across games.
- Aggregate only the predeclared taskwise normalized ratios. Preserve each
  task's completion, later-boundary, and final raw mean/std for retention,
  forgetting, and backward-transfer analysis.
- Save six immutable complete-bank task-boundary inference snapshots with
  SHA256 sidecars and an atomic index. Evaluate future tasks without allowing
  evaluation transitions to enter replay.
- Describe this run as a single-seed extra-sample/compute pilot. A matched-budget
  control and additional seeds remain required for a method claim.

## Budget and checkpoint consequences

The full run contains 1,080 epochs, 70,778,880 raw environment frames,
1,080,000 WM updates, and 864,000 Actor-Critic updates. Relative to original
ARROW per task, environment interaction and optimizer updates are doubled;
WM sampled-frame and Actor-context use are eightfold.

Replay transition capacity remains ARROW-50, but the pilot stores uint8
observations in file-backed node-local mmap rather than original float32 GPU
observations. The different byte footprint is recorded, not treated as matched.

Boundary and evaluation snapshots are inference artifacts. They omit optimizer
and target state, replay/provenance, RNG, schedule position, and the complete
counter state, so the run is restart-only and must never be described as
resumable.
