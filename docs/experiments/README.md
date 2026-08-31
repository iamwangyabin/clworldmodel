# Experiment Records

This directory is the repository entry point for experiment evidence. It exists
so that an important result does not have to be rediscovered from a remote
machine, an ignored `runs/` directory, a long protocol document, or a chat.

Start with:

- [RESULTS.md](RESULTS.md): generated human-readable run index;
- [registry.json](registry.json): generated machine-readable index;
- [`records/<record-id>/record.json`](records/): self-contained record for one
  run;
- [`continual_evaluation_metrics_v1.md`](../protocols/continual_evaluation_metrics_v1.md):
  formulas for normalized continual metrics.

`RESULTS.md` and `registry.json` are generated. The per-run `record.json` files
are the reviewed source of truth in this directory.

## What the current evidence says

There is no universal "best raw return": Atari games have different reward
scales, and several records use different task-awareness, evaluation cohorts,
or compute. Raw returns must remain per task.

Within the matched original-six, task-agnostic, advancing-cohort, seed-0 local
comparison group:

| Method | Forgetting (lower) | ACC (higher) | min-ACC (higher) | WC-ACC (higher) |
|---|---:|---:|---:|---:|
| ARROW-50 | 0.849806 | **0.897343** | 0.622579 | 0.548055 |
| DreamerV3/FIFO | 2.386320 | 0.230955 | -0.030802 | 0.012898 |
| FastKAN StableTargets | **0.114152** | 0.827486 | **0.792280** | **0.675309** |

These are single-seed diagnostic values, not an official multi-seed ranking.
Forward transfer is unavailable because aligned single-task acquisition curves
were not preserved. The exact raw checkpoint matrices and source hashes are in
the corresponding records.

A particularly strong current **early-curriculum task-aware** record is the
partial dense Evolving-Core run. Its accepted fixed-validation vectors were:

- after three tasks: MsPacman `2164.375`, Boxing `86.625`, CrazyClimber
  `96550.001`;
- after four tasks: MsPacman `2074.375`, Boxing `87.5625`, CrazyClimber
  `87457.144`, Frostbite `272.5`.

That run stopped after 404 completed epochs with only four accepted task
boundaries and no final evaluation, so it is not a six-task result. It exposes
task identity, uses a fixed evaluator cohort, and adds boundary-consolidation
updates; it must not be presented as a fair superiority result over ARROW.

For the directly aligned Evolving-Core Task-0 acquisition diagnostic, all three
records use seed 0, the same fixed 16-rollout cohort, and the same 90-epoch
budget:

| Mechanism | Pre-consolidation | Consolidation candidate | Accepted |
|---|---:|---:|---:|
| Dense 512/512/256 | 2025.625 | 1994.375 | 1994.375 |
| SharedDown 512/512/256 | 1588.75 | 1497.5 | 1588.75 (rollback) |
| Compact 128/128/64 | 1100.0 | 1155.0 | 1155.0 |

SharedDown's candidate crossed the five-percent guard and was rolled back. The
compact run later stopped during Boxing, so only its Task-0 row belongs in this
aligned comparison. This is one seed and says nothing yet about continual
retention.

The task-aware CNN FullBank three-task pilot has held-out final raw returns
`[1341.875, 80.0625, 108675.005]` for MsPacman, Boxing, and CrazyClimber. It
uses a different extra-sample protocol, so it is not directly ranked against
the dense run or the matched task-agnostic group. The `2595.625` FullBank
MsPacman number is a single-task acquisition gate, not a continual result.

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
