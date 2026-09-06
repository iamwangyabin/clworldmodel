# CoinRun baseline archive: ARROW-50 and DreamerV3/FIFO, five seeds

Archived 2026-09-06. This is **10 runs / two methods / seeds 0–4**, not the
earlier standalone seed-0 run and not the Atari or task-aware method campaigns.
All **3,300 task/checkpoint records and 844,879 individual episode returns**
are preserved in Git as small text evidence. No seed is selected or discarded.

## Recompute and verify

```sh
python3 scripts/report_coinrun_baselines.py check
python3 scripts/experiment_registry.py check
python3 -m unittest tests.test_coinrun_baseline_archive tests.test_experiment_registry
```

Use `write` instead of `check` to regenerate this report. Only the committed
`record.json` and `evaluation.log` files are needed; no server, TensorBoard,
downloaded run folder, GPU or third-party Python package is required.

## Completion and limitations

All ten logs reach epoch 540 and `training_end`, with all 55 × 6 scheduled
evaluations and finite recorded core losses. **Training/evaluation is complete;
launcher finalization is a separate check.** VirtAI seeds 0–3 retain stale
`state=running`, no exit code and no per-job final checksum manifest after
post-training filesystem I/O errors. These source states were NOT rewritten.
Both 4090-2 seed-4 jobs have `completed`, exit code 0 and verified final
checksum manifests. Record status `complete` refers to training/evaluation
coverage; `strict_finalization_verified` distinguishes the eight incomplete
launcher records. All ten are labelled **diagnostic evidence**, not a verified
reproduction of the paper's numerical results.

The metric is **raw episodic return, not classification accuracy**. Returns
are exactly 0 or 10. Each evaluation has 256 or 257 completed episodes due
to vectorized threshold overshoot; no extra episode was dropped. Within-task
standard deviation uses `ddof=0`; across-seed SD below uses `ddof=1`.

## Final scheduled evaluation (epoch 540)

This evaluation is before epoch-540 optimizer updates: 540,000 world-model
and 432,000 actor-critic updates. Training subsequently ends at 541,000 and
432,800 updates. The last collection revisits task 0 but its extra gradients
do not enter the reported epoch-540 evaluation. There is no separate held-out
post-training evaluation and no saved model/resumable checkpoint in these runs.

Raw columns are six variants of the **same CoinRun game and reward scale**.
Their descriptive mean is not an average across unrelated Atari games.
Taskwise episode counts and mean ± population SD at every checkpoint are
in each linked record; ordered individual returns are in its `evaluation.log`.

| Method / seed | CoinRun | +NB | +NB+RT | +NB+RT+GA | +NB+RT+GA+MA | +NB+RT+GA+MA+CA | Six-variant raw mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| [ARROW-50 / 0](records/coinrun-arrow-ar50-original-s0-20260903/record.json) | 6.054688 | 7.500000 | 7.578125 | 8.085938 | 8.242188 | 4.375000 | 6.972656 |
| [ARROW-50 / 1](records/coinrun-arrow-ar50-original-s1-20260903/record.json) | 6.210938 | 7.382812 | 7.851562 | 8.125000 | 7.890625 | 4.687500 | 7.024740 |
| [ARROW-50 / 2](records/coinrun-arrow-ar50-original-s2-20260903/record.json) | 4.648438 | 6.250000 | 6.328125 | 6.718750 | 7.500000 | 4.179688 | 5.937500 |
| [ARROW-50 / 3](records/coinrun-arrow-ar50-original-s3-20260903/record.json) | 5.078125 | 5.039062 | 6.210938 | 6.289062 | 6.250000 | 4.804688 | 5.611979 |
| [ARROW-50 / 4](records/coinrun-arrow-ar50-original-s4-20260903/record.json) | 5.390625 | 8.125000 | 7.968750 | 8.125000 | 7.773438 | 4.765625 | 7.024740 |
| [DreamerV3/FIFO / 0](records/coinrun-dv3-fifo-original-s0-20260903/record.json) | 6.250000 | 5.703125 | 5.585938 | 4.921875 | 5.859375 | 5.820312 | 5.690104 |
| [DreamerV3/FIFO / 1](records/coinrun-dv3-fifo-original-s1-20260903/record.json) | 6.484375 | 5.039062 | 5.507812 | 5.390625 | 5.898438 | 6.171875 | 5.748698 |
| [DreamerV3/FIFO / 2](records/coinrun-dv3-fifo-original-s2-20260903/record.json) | 6.171875 | 5.976562 | 6.367188 | 5.937500 | 6.757812 | 5.859375 | 6.178385 |
| [DreamerV3/FIFO / 3](records/coinrun-dv3-fifo-original-s3-20260903/record.json) | 5.625000 | 4.414062 | 5.136187 | 5.468750 | 5.742188 | 6.015625 | 5.400302 |
| [DreamerV3/FIFO / 4](records/coinrun-dv3-fifo-original-s4-20260903/record.json) | 2.617188 | 1.875000 | 1.796875 | 2.031250 | 1.953125 | 4.726562 | 2.500000 |

| Method | Across-seed mean ± sample SD | Median [Q25, Q75] |
|---|---:|---:|
| ARROW-50 | 6.514323 ± 0.685213 | 6.972656 [5.937500, 7.024740] |
| DreamerV3/FIFO | 5.103498 ± 1.481728 | 5.690104 [5.400302, 5.748698] |

## Separate fixed-anchor normalization diagnostic

Constants are the rounded median endpoints from [ARROW v3 Table A.16](https://arxiv.org/html/2603.11395v3):
random `[2.78, 2.45, 2.7, 2.62, 2.5, 2.69]`; single-task `[6.09, 7.14, 6.89, 6.85, 7.89, 5.78]`. Apply `(return − random) /
(single-task − random)` per task, then average tasks, then aggregate seeds.
Values are not clipped. These are **not the authors' per-seed/time-aligned
normalization curves**, so they must not be labelled exact paper replication
numbers. Forward transfer is unavailable without aligned single-task curves.
F/min-ACC/WC-ACC use the same fixed anchors for the entire checkpoint matrix.

| Method / seed | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |
|---|---:|---:|---:|---:|
| ARROW-50 / 0 | -0.007612 | 1.022192 | 0.881197 | 0.825216 |
| ARROW-50 / 1 | -0.018536 | 1.044296 | 0.890155 | 0.849536 |
| ARROW-50 / 2 | 0.231075 | 0.769889 | 0.643248 | 0.616390 |
| ARROW-50 / 3 | 0.282563 | 0.721960 | -0.017301 | 0.099643 |
| ARROW-50 / 4 | -0.144514 | 1.034617 | 0.929316 | 0.886384 |
| DreamerV3/FIFO / 0 | 0.375567 | 0.768537 | -0.459342 | -0.213944 |
| DreamerV3/FIFO / 1 | 0.207509 | 0.792272 | 0.256851 | 0.401846 |
| DreamerV3/FIFO / 2 | 0.304751 | 0.875301 | -0.534419 | -0.274402 |
| DreamerV3/FIFO / 3 | 0.431146 | 0.701826 | -0.569553 | -0.295252 |
| DreamerV3/FIFO / 4 | 1.051721 | 0.005184 | -0.198068 | -0.055209 |

| Method | Median F | Median ACC | Median min-ACC | Median WC-ACC |
|---|---:|---:|---:|---:|
| ARROW-50 | -0.007612 | 1.022192 | 0.881197 | 0.825216 |
| DreamerV3/FIFO | 0.375567 | 0.768537 | -0.459342 | -0.213944 |

## Provenance and storage

Launch commit: [`7b3ebec334c54942a40c888d8972aac24761c5ad`](https://github.com/iamwangyabin/clworldmodel/commit/7b3ebec334c54942a40c888d8972aac24761c5ad);
upstream ARROW pin: `cb05e7d97ed83c3cf6e528960db0da6868e29232`.
Seed IDs 0–4 map to `[123456789, 1337, 31337, 42, 987654321]`. Seeds 0–3 ran on VirtAI's four reported
`S2.gpu.xlarge` devices, two jobs per GPU; both seed-4 jobs ran on 4090-2.
Do not infer identical accelerator/runtime behavior from identical seed values.

The [CPU uint8 protocol](../protocols/arrow_coinrun_cpu_uint8_replay.md)
retains 541 epochs, 90/task, 1,024 replay trajectory slots and 6,480,199,680
allocated replay tensor bytes/job. ARROW: FIFO 512 + LTDM 512 with 50/50
whole-minibatch selection; DV3: FIFO 1,024. Only decoded sampled minibatches
enter CUDA. Explicit Python/Procgen seeding is a documented deviation from
released defaults, not a promise of bitwise determinism. Resolved configs,
published-config differences, counter meanings, package/hardware provenance,
loss checks and original artifact SHA256s are embedded in every record.

`evaluation.log` is compact JSONL: one provenance header, followed by epoch,
task index and an ordered integer array of raw returns. Rendering original
0.0/10.0 as 0/10 is numerically exact; no sampling or lossy compression is
used. Means/std/counts are independently recomputed by the command above.
Full training logs, TensorBoard events, runtime dumps and downloaded tarballs
remain outside Git and are identified by hashes. This archive does not claim
that large run packages or nonexistent model weights were uploaded to GitHub.
