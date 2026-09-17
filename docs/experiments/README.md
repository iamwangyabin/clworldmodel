# Experiment Records

This directory is the repository entry point for experiment evidence. It exists
so that an important result does not have to be rediscovered from a remote
machine, an ignored `runs/` directory, a long protocol document, or a chat.

Start with:

- [AWM_AUTOROUTE_PAPER_PLAN.md](AWM_AUTOROUTE_PAPER_PLAN.md): the sole active
  paper experiment backlog, status matrix and progress denominator;
- [RESULTS.md](RESULTS.md): generated human-readable run index;
- [registry.json](registry.json): generated machine-readable index;
- [`records/<record-id>/record.json`](records/): self-contained record for one
  run;
- [`continual_evaluation_metrics_v1.md`](../protocols/continual_evaluation_metrics_v1.md):
  formulas for normalized continual metrics.

`RESULTS.md` and `registry.json` are generated. The per-run `record.json` files
are the reviewed source of truth in this directory.

The sole formal project-owned paper method is **AWM-AutoRoute** (Decision 0065).
AWM/D is retained only as its oracle/internal reference. Historical names,
record IDs, protocol versions and raw results for retired methods were removed
from the working tree on 2026-09-14 and remain recoverable from Git history;
they never entered the active paper progress count. Renaming never upgrades
old-router evidence to the maintained v4 protocol.

StableTargets and F/D-AutoKAN execution code was retired on 2026-09-07
(Decision 0058); their result records were removed from the working tree on
2026-09-14 and remain recoverable from Git history.

## What the current evidence says

The [CoinRun five-seed baseline archive](coinrun_baselines_5seed_20260906.md)
preserves **ARROW-50 × 5 and DreamerV3/FIFO × 5**, including every scheduled
evaluation's individual raw episode returns, taskwise mean/std/count, resolved
configs, provenance and source hashes. Training/evaluation completed for all
ten; VirtAI's eight stale launcher-finalization records are explicitly retained,
not rewritten as successful exits. Fixed-anchor normalized metrics are labelled
diagnostic rather than exact paper reproduction. The report is recomputable
from Git alone using `python3 scripts/report_coinrun_baselines.py check`.

Retained Atari evidence covers the seed-0 **ARROW-50** and **DreamerV3/FIFO**
original-six records, eighteen single-task normalization anchors, the
AWM-AutoRoute campaign archives
([2-seed](awm_autoroute_atari_2seed_20260908.md),
[5-seed](awm_autoroute_atari_5seed_20260910.md),
[replacement-seed log](awm_autoroute_replacement_seeds_20260909.md)), the
[capacity-organization pilot](capacity_organization_pilot_20260911.md), and the
Dream Rehearsal reference-integration audits
([fidelity](dream_rehearsal_fidelity_audit_20260905.md),
[memory pair](dream_rehearsal_memory_pair_3080_20260906.md)). The completed
Arrow-matched Dream Rehearsal full-history and bounded-history Atari pilots
from 2026-09-10 are indexed as compact pilot records; their large replay and
training artifacts remain outside Git under `runs/`.

There is no universal "best raw return": Atari games have different reward
scales, and records use different task-awareness, evaluation cohorts, or
compute. Raw returns must remain per task.

Within the matched original-six, task-agnostic, advancing-cohort, seed-0 local
comparison group:

| Method | Forgetting (lower) | ACC (higher) | min-ACC (higher) | WC-ACC (higher) |
|---|---:|---:|---:|---:|
| ARROW-50 | 0.849806 | **0.897343** | 0.622579 | 0.548055 |
| DreamerV3/FIFO | 2.386320 | 0.230955 | -0.030802 | 0.012898 |

These are single-seed diagnostic values, not an official multi-seed ranking.
Forward transfer is unavailable because aligned single-task acquisition curves
were not preserved. The exact raw checkpoint matrices and source hashes are in
the corresponding records. Retired-method rows that previously appeared here
(FastKAN StableTargets, Evolving-Core, CNN FullBank) were removed together
with their records on 2026-09-14 and remain recoverable from Git history.

## Repository storage boundary

Git contains only small evidence needed to understand and audit a result:

- clean/pushed Git provenance, seed, task order, protocol and status;
- raw per-task return means and dispersions at important checkpoints;
- explicit evaluator cohort, policy, rollout count, and replay isolation;
- budgets and compact parameter/resource accounting when relevant;
- derived metric summaries with their schema/source;
- SHA256 hashes of the original manifests and logs;
- short, text-only evaluation or failure excerpts when they add evidence.

Git must **not** contain:

- model weights, inference snapshots, or training checkpoints;
- optimizer/scaler/RNG checkpoint state;
- Replay tensors, mmap stores, or downloaded datasets;
- TensorBoard event files, videos, ROMs, or full generated run directories;
- full training logs.

The original large run may remain on a server or offline archive. A record may
name and hash its source artifacts, but understanding its key result must not
depend on that machine still existing. Full logs remain external; only the
small evaluation/failure excerpt belongs here.

Each curated record is limited to 256 KiB per file and 512 KiB total. The
validator rejects nested directories, symlinks, binary content, unapproved
filenames, and heavyweight artifact names.

## Evidence and status labels

- `official`: complete predeclared protocol with the required seed aggregate;
- `pilot`: substantive evidence that is insufficient for an official claim;
- `diagnostic`: matched analysis useful for diagnosis, usually one seed;
- `ablation`: a named controlled experimental difference;
- `smoke`: execution correctness only, never a performance claim.

Run status is recorded separately as `complete`, `partial`, `stopped`, or
`failed`. A completed smoke is still only a smoke. A partial pilot does not
become a final result because its intermediate score is high.

## Adding or updating a record

1. Leave the generated run, weights, checkpoints, Replay, TensorBoard, and full
   log outside Git.
2. Create `docs/experiments/records/<record-id>/record.json` using schema
   version 1. Preserve raw taskwise numbers and claim limitations.
3. If useful, add only `evaluation.log` and/or `notes.md`. An excerpt must state
   the SHA256 of its full source log.
4. List the original small source artifacts and SHA256 values in
   `source_artifacts`. Do not rely on a machine-specific absolute path.
5. Rebuild and check the two indexes:

   ```bash
   python scripts/experiment_registry.py write
   python scripts/experiment_registry.py check
   ```

6. Run `python -m unittest tests.test_experiment_registry` before review.

The validator rejects unknown top-level record fields. A behavior-changing
record-schema update therefore requires a schema-version decision rather than
an ad hoc field silently appearing in one run.
