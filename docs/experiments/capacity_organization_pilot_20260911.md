# Five-control GPU pilot launch — 2026-09-11

## Scope

User authorized a new protocol and implementation of all five organization controls.
Do not interpret these as restored retired algorithms, official reproductions, or
compute-matched one-switch AWM ablations. AWM main training is not launched here.

Code: `codex/capacity-controls-20260911`, launch commit
`20ba62c9e33ea1ee324528b9f63b7ff9c98aa0e0`. DV3/FIFO CPU-float32 runs use
`f961fa3cc9aec12004660c630e049f50e2771279` on their independent source tree.
Both were clean, pushed to GitHub, and upstream-synced before launch. Controllers
fetched GitHub; targets fetched SHA-verified bundles. Preserve actual launch commits.

## Capacity accounting

All values below exclude private actor-critics. One private AC = 1,715,985 parameters;
six retained private ACs add 10,295,910 parameters for every control.

| Control | Retained/allocated WM | Task 0 trainable WM | Task 5 trainable WM |
| --- | ---: | ---: | ---: |
| shared | 19,498,853 | 19,498,853 | 19,498,853 |
| wide | 34,742,117 | 34,742,117 | 34,742,117 |
| fullbank | 116,993,118 | 19,498,853 | 19,498,853 |
| frozen | 42,601,625 | 23,349,285 | 3,850,492 |
| independent | 42,601,625 | 23,349,285 | 23,349,285 |

The wider shared model uses MLP width1536; it is not exactly parameter-matched
to the dense residual models. All WM modules are allocated upfront, not evidence
of task acquisition. The startup `model_parameter_accounting.json` is written
before task activation, so its trainable count includes future optimizer-owned
tensors. Use the separately preserved post-activation counts above; boundary
accounting is written after activation. No inference snapshot is resumable.

## Validation and preserved failures

- Sixteen focused tests passed on three functioning hosts: all five control
  configurations, actual two-task WM and imagined-AC updates, state roundtrip,
  unchanged shared loss, frozen/storage ownership, baseline parity, and metrics.
- Every control passed a production-width, production-minibatch two-task GPU
  smoke on a functioning target: four WM + four AC updates, both boundary
  inference snapshots and held-out final evaluation. These are execution checks,
  not performance results. Shared/wide were validated on host2 after host1 failed.
- A broader70-case run reported six failures/five errors. Re-running the same
  affected tests on unchanged base `f961fa3` reproduced exactly the same eleven
  failure identities: retired launcher profiles/expectations plus a2.38e-7
  Linux-vs-fixture actor value difference. No assertion was weakened and no
  retired configuration was restored. Baseline/AWM tensor parity and retained
  route/checkpoint tests passed. Do not report the entire old suite as green.
- Host1 shows two free S4 logical GPUs, but platform initialization repeatedly
  exits151. Both logical routes are affected, including an unmocked GPU probe
  on GPU1. The wider smoke launcher exits1 because its child probe exits151.
  Neither shared nor wider full pilot launched there. Replacement requested.
- Explicit CPU-only tests still triggered injected platform CUDA initialization
  during optimizer setup on host1, even without login-shell setup. Their failures
  are preserved; the CUDA-availability mock is not evidence that the card works.

## Allocation

| Host (user-provided order) | Logical GPU 0 | Logical GPU 1 |
| --- | --- | --- |
| 1 | Shared S0 — blocked by platform | Wider S0 — blocked by platform |
| 2 | DV3/FIFO S3 | FullBank S0 |
| 3 | Frozen core/residuals S0 | DV3/FIFO S1 |
| 4 | DV3/FIFO S2 | Independent residuals S0 |

All three DV3 runs and both residual controls have verified completed training
epochs. FullBank also completed its first 1,000 WM and 800 AC updates.
Original seed mapping: S0=123456789,S1=1337,S2=31337,S3=42.

Raw evidence, controller timestamps, bundle digests, PIDs, failed attempts and
phase counts are preserved in the ignored local campaign directory and remotely
under the persistent project mount. Runtime reports S4.gpu.xlarge with24555MiB;
do not claim a physical4090 model or globally distinct physical GPU UUIDs.
The instances have32-CPU/128GiB cgroup quotas, regardless of host-level nproc/free.

Runs do not auto-retry or equivalently resume. No periodic monitor was installed.
Stopping/rebuilding an instance terminates its processes; persisted inference
snapshots cannot restart these pilots equivalently.

## Verified launch snapshot (2026-09-11 09:19 UTC)

| Run | Host | PID | Confirmed WM updates | Confirmed AC updates |
| --- | ---: | ---: | ---: | ---: |
| capacity_fullbank_s0 | 2 | 27093 | 1,000 | 800 |
| dv3_s3 | 2 | 7666 | 17,000 | 13,600 |
| capacity_frozen_s0 | 3 | 26840 | 3,000 | 2,400 |
| dv3_s1 | 3 | 7780 | 18,000 | 14,400 |
| capacity_independent_s0 | 4 | 26768 | 4,000 | 3,200 |
| dv3_s2 | 4 | 7716 | 19,000 | 15,200 |

All six processes were alive at this snapshot. This is not a completion claim.
The two host1 full pilots remain blocked; they have no fake started status.
