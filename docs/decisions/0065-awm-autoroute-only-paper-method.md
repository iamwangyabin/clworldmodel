# 0065 — AWM-AutoRoute is the sole formal paper method

Date: 2026-09-13. Status: accepted by the user.

## Decision

The sole formal project-owned method for the current paper is
**AWM-AutoRoute — Accumulative World Modeling with Automatic Routing**. AWM/D
is no longer a separately claimed method. Its oracle path remains only where
AWM-AutoRoute needs it for training, parity, checkpoint compatibility, an
oracle-routing upper bound, or a controlled diagnostic.

ARROW-50, DreamerV3/FIFO and capacity-organization controls are comparison
methods. They do not become project-owned proposed methods.

## Evidence and compatibility

This scope decision does not rewrite or delete historical results. Old D/AWM,
D-AutoRoute v1/v2, AWM-AutoRoute v3, retired algorithms, failed runs and stopped
runs remain provenance and negative evidence. They cannot fill a current paper
cell unless its exact protocol, benchmark, seed, budget and final evaluation
match the active plan.

Existing serialized keys and launcher paths remain unchanged. The only
maintained router is protocol v4: two-frame probability reconstruction with
unchanged-route policy state preservation.

The active backlog and denominator live in
`docs/experiments/AWM_AUTOROUTE_PAPER_PLAN.md`. No new paper run should launch
unless it occupies a declared cell there or the plan is amended before launch.
