# Retained-method result archive — 2026-09-06

This is a **dated evidence snapshot**, not a live monitor or a new experiment.
Read-only SSH checks on `4090-1`, `4090-2` and `4090x4` captured D-AutoRoute
at **2026-09-06 04:03:34–04:03:35 UTC** (12:03 China time). The completed D
run was rechecked against the local archive. Existing local StableTargets
artifacts were checked against the curated source hashes. No training,
evaluation, restart, configuration change or seed selection was performed by
this archival task.

## What exists

| Retained method | Evidence available | Interpretation |
|---|---|---|
| [StableTargets](records/fastkan-stable-targets-original-s0/record.json) | Complete six-task seed-0 record; ACC 0.827486, forgetting 0.114152 | Last scheduled stochastic, advancing-cohort evaluation; not a separate held-out final evaluation or a multi-seed result |
| [D](records/evolving-d-adaptive-qfp-original-s0/record.json) | Complete six-task, 540-epoch seed-0 pilot; held-out final evaluation; ACC 2.962900, forgetting 0.290449 | Task-aware, deterministic evaluation with additional consolidation/compression updates; not an evaluator-matched ranking against StableTargets |
| D-AutoRoute | Atari five seeds and CoinRun five seeds, all running at the snapshot; 100–119 completed epochs | Partial acquisition/route diagnostics, not final six-task performance or retention |
| F / D-AutoKAN | No training-result artifact found in the searched local archives or the three servers' experiment manifests | Do not substitute historical SharedFrozenDown + shared FastKAN results for F |

ACC here is normalized RL performance, **not classification accuracy**. Scores
are not clipped and can exceed one. Raw returns from different games are not
averaged. A completed single-seed pilot is not an official reproduction.

The search covered the local run/config archives and the known experiment/run
roots of the three servers above. It does not establish that an unlisted,
offline or inaccessible machine has no additional results.

## D: preserved final raw returns

The final evaluator ran after 540 completed epochs, using the separately
declared held-out cohort. The source's 16-rollout setting uses the historical
trajectory-budget evaluator, not AutoRoute's exact-complete-episode evaluator.

| Task | Raw return mean | Episode-return standard deviation |
|---|---:|---:|
| MsPacman | 4906.250 | 663.239 |
| Boxing | 92.8125 | 3.69491 |
| CrazyClimber | 97766.672 | 25972.018 |
| Frostbite | 2895.000 | 980.829 |
| Seaquest | 907.500 | 158.410 |
| Enduro | 225.600 | 94.1501 |

The record retains all **55 reporting checkpoints**, fixed normalization
references and all six compression boundaries, including all four candidates
at each boundary. Selected width fractions were `1, .75, 1, .5, .5, 1`.
The original project revision is
`2e8a058e5af3be562bb9cc5a46a223e8d98b1b2b`; final evaluation, resolved config,
launch manifest, run status, metrics and full-log hashes still match the local
archive. The historical dense Evolving-Core v2 partial run is a different
method and remains a separate record.

## D-AutoRoute: all ten seeds retained

Completed training epochs and evaluation epochs are deliberately separate.
An evaluation at zero-based epoch 100 precedes that online epoch's updates.
All source records contain six-task raw episode returns, means, population
standard deviations, route eligibility and routing confusion for **every
captured periodic checkpoint**, including untrained future-task evaluations.

| Record | Completed epochs / declared budget | Latest evaluation epoch |
|---|---:|---:|
| [Atari seed 0](records/d-autoroute-atari-s0-20260906/record.json) | 110 / 540 | 100 |
| [Atari seed 1](records/d-autoroute-atari-s1-20260906/record.json) | 105 / 540 | 100 |
| [Atari seed 2](records/d-autoroute-atari-s2-20260906/record.json) | 100 / 540 | 90 |
| [Atari seed 3](records/d-autoroute-atari-s3-20260906/record.json) | 101 / 540 | 100 |
| [Atari seed 4](records/d-autoroute-atari-s4-20260906/record.json) | 100 / 540 | 90 |
| [CoinRun seed 0](records/d-autoroute-coinrun-s0-20260906/record.json) | 119 / 541 | 110 |
| [CoinRun seed 1](records/d-autoroute-coinrun-s1-20260906/record.json) | 111 / 541 | 110 |
| [CoinRun seed 2](records/d-autoroute-coinrun-s2-20260906/record.json) | 103 / 541 | 100 |
| [CoinRun seed 3](records/d-autoroute-coinrun-s3-20260906/record.json) | 102 / 541 | 100 |
| [CoinRun seed 4](records/d-autoroute-coinrun-s4-20260906/record.json) | 103 / 541 | 100 |

Do not aggregate these different latest checkpoints as a matched final result.
Training still uses task identities; only action selection infers routes.
Task 0 is completed, task 1 is acquiring, and future-task routing accuracy
must not be interpreted as recognition of already learned tasks. CoinRun's
541-epoch budget includes its named final task-0 revisit.

### Failure and continuation history

All ten original attempts failed at the first multi-route boundary because
BF16/FP32 route states could not be assigned into one indexed buffer. The
archived lineage retains the failed-parent revision/log hash and the runtime
repair revision `837ec273b05e74dd4c0de08912123b00729ddcdb`, the source checkpoint
hash, restored optimizer/Replay/RNG counters, and separate attempt timing.
It is boundary continuation with fresh environment resets, **not bitwise
mid-episode recovery**.

CoinRun seed 3 discarded one completed online epoch: **1,000 WM and 800 AC
updates**, plus possible partial collection. The other nine discarded no
complete online epoch, but incomplete collection/evaluation costs are unknown,
not zero. Four earlier CoinRun startup failures (seeds 0, 2, 3, 4) caused by a
venv-Python symlink/import problem are also retained. Every retry kept its seed
and protocol; failures are not hidden by relabelling them as new seeds.

## Additional evidence recovered

- [StableTargets Task-0 screen](records/fastkan-stable-targets-task0-s0/record.json):
  the existing 90-epoch single-task result, `2410.625 ± 713.700` MsPacman raw
  return. This is acquisition evidence, not continual retention; its separate
  final call does not declare a held-out seed guarantee.
- [D without atom-output regularization](records/evolving-d-no-atom-reg-original-s0-partial/record.json):
  a separately named zero-regularization ablation with 284 completed epochs and
  29 periodic evaluation vectors through epoch 280. No final/status artifact
  was present; process activity was not inferred. Do not merge this ablation
  into canonical D's result.

## Durable storage and verification

`records/*/record.json` is the self-contained, reviewed source for the generated
[RESULTS.md](RESULTS.md) and [registry.json](registry.json). The archive keeps
raw measurements, declared budgets, evaluator semantics, source SHA256 values
and failure/continuation history. It does **not** require ignored local run
directories or server access to read the results or validate the index.
Weights, Replay, optimizer/checkpoint state, full logs and TensorBoard events
remain outside Git. Absolute machine paths and credentials are not copied into
the curated records.

Rebuild and verify from the repository root:

```bash
python scripts/experiment_registry.py write
python scripts/experiment_registry.py check
python -m unittest discover -s tests -p test_experiment_registry.py -v
```

The focused tests recompute D's normalized metrics from preserved raw vectors,
check every AutoRoute episode mean/std and confusion count, verify all five
seeds per benchmark, and retain the four startup failures and nonzero discarded
work. Source checksums are provenance, not proof of performance replication.
Later updates must preserve earlier raw checkpoints and attempt lineage; the
Git history retains this dated snapshot. Do not replace missing final metrics
with zero or infer them from a successful smoke/restore.
