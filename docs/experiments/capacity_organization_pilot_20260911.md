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

## Additional allocation — 2026-09-12

After the user requested more experiments, reassign the two not-yet-launched
S0 controls from unavailable host1 to host2, whose DV3 S3 and FullBank S0 had
finished with exit code 0. Do not restart or overwrite their completed runs.

| New run | Host / logical GPU | Seed | Start (UTC) | Launcher PID |
| --- | --- | ---: | --- | ---: |
| Shared S0 | 2 / 0 | 123456789 | 2026-09-12 15:47:13 | 34313 |
| Wider S0 | 2 / 1 | 123456789 | 2026-09-12 15:47:13 | 34314 |
| DV3/FIFO S4 | 3 / 1 | 987654321 | 2026-09-12 15:47:30 | 32941 |

Shared/Wider launch commit: `566b79f171544a167a64a3b0a1467bf27460dcee`
(documentation-only successor to the existing control implementation).
Sixteen focused target tests passed, and both assigned GPUs passed a fresh
two-task production-width smoke including held-out final evaluation before
the full-length pilots started. The target fetched the verified incremental
bundle with SHA-256 `d051e508ea49ac9cbb749ab49754dc1dc48851ea30833e7845f5a1eecf841a94`.

DV3 S4 launch commit: `b783d2fd9395b0434b5ce3cdc489135a08794399`.
The remaining published seed is prospectively recorded in the CPU float32
replay profile; no seed is selected from favorable results. Eleven focused
target tests and a fresh CUDA matrix probe passed; the previous target-GPU
DV3 smoke also exited 0. Bundle SHA-256:
`3c34f77d0d0b067e8338576585a7f6fc651a016e3a7bcfae83502d83f9e8df02`.

All three launch manifests record clean, pushed, upstream-synced code. The
controller fetched GitHub before deployment; targets fetched verified bundles.
The running Frozen and Independent source trees and processes were untouched.
Host4 GPU0 remains unallocated pending the user's scope decision about second
control seeds. No AWM main run, automatic retry, or automatic future queue was
started. These startup records are not completion claims.

## Prospective second-seed allocation — 2026-09-13

Before inspecting any S1 outcome, the user authorized immediately filling every
usable idle card with new experiments. Use original published seed index S1
(1337) for the five capacity controls, rather than inventing or selecting seeds
from S0 results. The fixed queue is FullBank, Independent, Frozen, Shared, then
Wider, assigning a control when a healthy card becomes idle. Never overwrite or
resume an S0 run, never use unavailable host1, and do not launch retired methods
or a new AWM main run as filler. Each S1 launch remains a pilot and keeps the
same `capacity_control_v1` protocol, budgets, task order, held-out evaluation,
capacity accounting and inference-only snapshot limitations as S0.

The first allocation is FullBank S1 on currently idle host4 logical GPU0.
Independent S0 remains active on GPU1 and must not be interrupted. Subsequent
queue entries require a fresh live-idle check, clean/pushed/upstream-synced
source check, existing-output refusal, target CUDA validation, and a recorded
launch manifest. A failed run is reported rather than silently retried.

FullBank S1 actually started at `2026-09-12T16:21:47Z` on host4 logical GPU0,
launcher PID 33683, from clean synchronized commit
`948ff0b23fae0888463b8ab23e4c546cd7227d57`. Sixteen focused tests and a
target-GPU BF16 matrix probe passed immediately before launch. The target used
bundle SHA-256 `be2228505bf6a4fbc83db6601cd2328acdda31a2174782d5a862b7702100f733`.
The manifest resolves seed index1 to 1337 and the full 540-epoch budget. The
process reached epoch0 initialization; this is not a completion claim.

Independent S0 completed successfully at `2026-09-12T16:42:42Z`, releasing
host4 logical GPU1. A live check on 2026-09-13 found that card idle, so the
next fixed queue entry, Independent S1, started there at
`2026-09-13T02:27:57Z` with launcher PID 35351. It uses clean synchronized
commit `ca3bc4c7d0556d25c2f50a77270f73a19edb7652`, seed 1337, bundle SHA-256
`5ded259df282f72aec72b45a2d2db17741aec73a90ee87e3faae2fd73f15cae5`,
and passed the same sixteen focused tests plus target BF16 probe. The process
reached epoch0 data collection; this is not a completion claim.

## Additional four-card allocation — 2026-09-13

The user provided a new four-logical-GPU VirtAI instance and requested using
additional cards to accelerate the paper experiment queue. Before observing
any new result, fix the allocation as Frozen S1, Shared S1, Wider S1, and
FullBank S2 on logical GPUs0–3 respectively. This completes the three remaining
S1 starts in the fixed queue, then begins S2 using the same fixed control order;
it does not select a method or seed based on performance.

The fresh instance reports four idle 24,258 MiB `S2.gpu.xlarge` logical GPUs,
a 256 GiB memory cgroup limit, the pinned PyTorch2.3/CUDA11.8 runtime and a
successful BF16 matrix operation on every device. Use8 CPU threads per run.
All four controls retain the same protocol, budgets, Atari order, evaluation,
capacity accounting and inference-only snapshot limitations. Run the focused
test set and a separate two-task production-width smoke on each assigned GPU
from the clean pushed launch commit before full training. Refuse existing
output paths; do not interrupt other hosts, silently retry failures, or launch
retired methods/AWM as filler.
