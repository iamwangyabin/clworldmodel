# 0042: Evolving-Core Original-Six-Task Pilot

## Status

Accepted as a separately named seed-0 pilot on 2026-08-30 after explicit
project-level authorization to test the complete original ARROW Atari
curriculum and to use rolling resumable-checkpoint retention on the target
host. It does not change or relabel the existing three-task v1 protocol.

## Context

The first Evolving-Core campaign is deliberately scoped to MsPacman, Boxing,
and CrazyClimber. That truncation is useful for rapid acquisition and
retention diagnosis, but it does not expose the method to the full continual
horizon used by canonical ARROW-50. A complete-horizon pilot is needed before
deciding whether failures after Task 3 are architectural, optimization-related,
or simply absent from the short campaign.

## Decision

Add the separately named
`Evolving-Core-Atomic-RSSM-ARROW-v2-OriginalSix-Atari-TaskAware-Pilot`
protocol. It trains from scratch on the original published order:

1. MsPacman;
2. Boxing;
3. CrazyClimber;
4. Frostbite;
5. Seaquest; and
6. Enduro.

Every task lasts 90 epochs and the run stops exactly at the sixth boundary
after 540 epochs; it does not execute the source config's epoch-540 return to
MsPacman. The Evolving-Core topology, losses, optimizers, 12-current/4-memory
online batch, ARROW-50 replay, fixed evaluation cohorts, and 1,000-update
boundary consolidation remain unchanged. The number of symmetric task-private
routes and Actor-Critics increases from three to six.

The first run uses published seed index 0 (`123456789`) and is classified only
as `pilot`. A favorable single run cannot establish superiority, reliability,
or reproduction.

## Resource And Artifact Consequences

The run spends 35,389,440 raw frames, 540,000 online world-model updates,
432,000 Actor-Critic updates, and 6,000 additional boundary-consolidation
updates. It therefore is not update-matched to canonical ARROW-50.

Uint8 FIFO/LTDM observations occupy 6 GiB for the live replay. This named pilot
uses `latest_boundary` rolling retention: a new task's complete pre/post pair
and checksums must exist before older resumable pairs and their replay assets
are removed. At a transition the peak is two immutable 6-GiB replay assets;
after pruning only the latest remains. All raw metrics, TensorBoard events,
consolidation records, retention manifests, and task-bank inference snapshots
are preserved. The run deliberately gives up exact resume from older
boundaries, but never omits state from the retained latest pair.

The launcher requires at least 48 GiB free before starting, covering live
Replay, peak rolling Replay assets, model/optimizer payloads, task-bank
snapshots, logs, and one atomic-save temporary. Changing the retention mode is
a different artifact protocol.

## Claims

Task identity selects private routes and policies, per-task capacity grows,
and consolidation adds compute. This pilot is a complete-horizon stress test,
not a task-agnostic or budget-matched ARROW replacement. Reports must retain
all six raw-return curves, final average performance, forgetting, boundary
rollback records, component conflict rates, resource use, and failed runs.
