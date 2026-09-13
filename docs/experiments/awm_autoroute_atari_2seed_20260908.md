# AWM-AutoRoute Atari two-seed pilot (2026-09-08)

> Historical campaign/provenance record, not the active paper backlog. Use
> [`AWM_AUTOROUTE_PAPER_PLAN.md`](AWM_AUTOROUTE_PAPER_PLAN.md) for current scope
> and progress.

> Current execution uses **v4** after the user-requested state-isolation fix.
> The v3 account below is historical; see the final section for the fresh restarts.

## Original predeclared scope (v3, historical)

The user requests two **new, from-scratch** AWM-AutoRoute Atari pilots:
seed index 0 (123456789) on 4090-1 and seed index 1 (1337) on 4090-2.
These are the first two configured seeds, not a performance-selected subset.
Existing Atari/CoinRun jobs and historical results remain separate and untouched.

Use the sole maintained [two-observation probability routing protocol](
../protocols/awm_autoroute_v3_atari.md), including its explicit episode-local
policy-state initialization and candidate-history switching semantics. Do not
claim identical trajectories or reproduced original AWM returns before testing.
Each run retains six tasks, 90 epochs/task, 540 epochs total, 552,000 world-model
and 432,000 Actor-Critic optimizer updates, BF16 compute, private MLP behavior,
task-aware training and oracle boundary-selection gates. No CoinRun job is added.

## Reproducible entry point

From a clean, pushed, fetched and upstream-synchronized checkout, using the
pinned Atari runtime and one explicitly selected physical GPU per host:

```sh
"$PYTHON" scripts/smoke_evolving_atomic_rssm.py --method-profile d_autoroute
"$PYTHON" -u scripts/run_evolving_atomic_rssm_d_autoroute.py \
  --seed "$SEED_INDEX" --cpu-threads 8 --classification pilot \
  --python "$PYTHON" --output-dir "$NEW_RUN_DIRECTORY"
```

The seed index is 0 or 1 as declared above. Host-specific paths, GPU UUIDs,
commands, Git/dependency fingerprints, smoke outcomes, process identities and
resource preflights belong in each deployment manifest, not hidden defaults.
No launch may overwrite an existing run directory. Smoke results are execution
evidence only and remain separate from pilot metrics.

## Validation and monitoring

Prelaunch fixed/canned-input checks: 45 tests across `test_two_frame_autoroute`,
`test_d_autoroute`, `test_reconstruction_router`, `test_method_retirement`,
`test_retained_method_parity`, and `test_continual_metrics` pass. All 303 vendor
fingerprints verify. The actual trainer CLI is checked through config validation
before CUDA initialization; its stale removed-KAN-field injection and omitted
actor-override handling were corrected without changing the resolved protocol.
The unrelated historical-suite migration limits in Decision 0061 still apply;
this is not a claim that full test discovery is green or that CUDA smoke passed.

The expanded 60-test set passed locally. Its first target-host run on both
Linux/x86 servers found one mismatch against the existing macOS CPU loss trace
(baseline 4353.83740234375 versus 4356.27392578125). Disabling oneDNN in an
isolated diagnostic did not remove the mismatch. Before launching a pilot,
the exact archived pre-retirement source snapshot was checked against all six
hashes in the original fixture and evaluated on the target Linux runtime.
Old and new source produced **identical** initialization hashes, losses and
all gradient norms there (baseline 4353.83740234375, D 4232.0263671875).
The Linux reference fixture records that old-source provenance; the original
fixture and numerical tolerances remain unchanged. The initial failed
preflight is preserved. This is a platform-specific fixed-trace reference,
not a new performance measurement or a change to training precision.

Initial pilot attempts at `929efce7f552686d9972bcbdc3bd5844e47941e2` exited
during startup parameter reporting, before any environment interaction or
optimizer updates: `KeyError: include_task0`. Their outputs and exit records
are preserved, not overwritten. The accounting consumer also retained a call
to deleted prediction adapters. Repair only those reporting contracts and
cover dense/compact reports in unit tests and the target-CUDA smoke before
starting replacement attempts. Do not mislabel these as learned checkpoints
or resumed training.

The second attempts at `18dd7821987edf4831cde5086791d98713ec4a52` collected
the initial batch, evaluated and completed the first 1,000 world-model updates,
then failed before the first AC optimizer step at the removed
`ActorCritic.consolidation_penalty()` call. No complete epoch or resumable task
boundary was reached. Preserve their raw evaluations, replay and failure logs;
replacement attempts start from scratch and must record this extra consumed
work rather than claim equivalent continuation. Repair the zero-valued retired
MLP penalty call in both shared loss paths and add actual private-MLP AC
optimization to CPU regression and target-CUDA smoke coverage. No learned KAN
module, new regularization or changed update budget is introduced.

Monitor both new runs alongside the old campaign, but grant automatic diagnosis
and safe recovery only to these two new runs. Preserve failed attempts and all
raw metrics. Check real process identity, logs, CPU progress, GPU activity,
available RAM/disk and task-boundary artifacts, not just a stale running flag.
Do not treat slow evaluation/compression as a hang or restart based on SSH loss.
Any recovery must validate complete model/optimizer/replay/RNG/schedule state
and record a new attempt and discarded-work accounting. Boundary checkpoints
are required for continuation; restarting from weights alone is not equivalent.
Do not modify old jobs, silently change the protocol, repeatedly retry an
unexplained native crash, or delete prior experiment artifacts to make space.

Normal unchanged progress should remain quiet; report new failures, meaningful
boundaries, completion or required user action. Runtime evidence and current
state are stored in the ignored campaign directory rather than this prelaunch
record. No performance result is asserted here.

## First compact/dense boundary failure and continuation

Seed 0 at `021563eb97ef1fe38e3450e5d76d1b03d7dfdf33` completed task 0,
including consolidation and compression to width fraction 0.5. Its durable
post-boundary checkpoint contains 90 completed epochs, 92,000 world-model
updates, 72,000 AC updates and 5,898,240 protocol-counted raw frames. During
epoch 90 evaluation, a batch containing compact and dense routes failed at
indexed hidden-state assembly: BF16 destination versus FP32 source. This is
a heterogeneous-output assembly error, not evidence of insufficient BF16
precision or a hardware fault. The failed epoch's collection/evaluation work
is retained as overhead; it is not a completed update epoch.

The repair promotes the batch buffer to the common output dtype, retaining
each route's values independently of route ordering. A regression fails on
the original implementation. Restore only the existing retained-method
post-boundary continuation mechanism, adapted to strict v3 configs, not the
retired router or any KAN algorithm. New recovery attempts must validate the
target GPU with the immutable real boundary checkpoint, including full
model/target/optimizer/Replay/RNG/scheduler restoration and mixed-dtype policy
steps, before continuing:

```sh
"$PYTHON" scripts/smoke_d_autoroute_resume.py \
  --checkpoint "$BOUNDARY_CHECKPOINT" --output-dir "$NEW_SMOKE_DIRECTORY"
"$PYTHON" -u scripts/run_evolving_atomic_rssm_d_autoroute.py \
  --seed "$SEED_INDEX" --cpu-threads 8 --classification pilot \
  --python "$PYTHON" --output-dir "$NEW_RUN_DIRECTORY" \
  --resume-from "$BOUNDARY_CHECKPOINT"
```

The source must be a confirmed failed attempt, the config/seed/protocol must
match exactly, and both checkpoint SHA256 and post-compression ownership and
counters must validate. Inherited logs and routing/boundary records retain their
old commit provenance; failed suffixes remain in the parent. This is boundary
continuation with fresh environment resets, not bitwise mid-episode recovery.
Never hot-edit the still-running other seed's source tree. Runtime smoke and
deployment outcomes belong in the ignored campaign manifest, not an advance
claim that a repaired run has completed.


## User-requested v4 state-isolation correction and fresh restarts

On 2026-09-08 the user requested correcting AutoRoute's extra policy-state
reset and restarting these same two seeds from scratch. See Decision 0063 and
[protocol v4](../protocols/awm_autoroute_v4_atari.md). Only actual expert changes
substitute candidate history; a new scoring window on the same expert leaves
AWM's policy state and previous action unchanged. Routing score math and all
training/evaluation budgets remain fixed.

The current v3 attempts were intentionally stopped, not classified as new
crashes: S0 `atari_s0.resume1` had 185 complete epochs and an incomplete epoch
185; S1 `atari_s1.resume2` had 97 complete epochs and an incomplete epoch 97.
Their full logs, metrics, checkpoints and provenance remain historical; no
partial-epoch optimizer count is invented. On the space-constrained S1 host,
the stopped v3 working Replay arrays were losslessly archived and verified
against per-file SHA256, then backed up locally and checked against archive
SHA256 before removing only the redundant uncompressed copies. Restore markers
record the paths/checksums; checkpoint-owned Replay and the old ten runs were
not modified. The 48 GiB launch preflight was not lowered.

Both fresh pilots execute commit
`12d317eaf98eaebeb462894894b6fc83bb5e7041`, pushed and fetched with zero
upstream divergence before launch, in independent clean checkouts. The original
v3 sources were not edited. Current attempt IDs are `atari_s0.v4.pilot1` and
`atari_s1.v4.pilot1`; the same predeclared seeds and hosts are retained.
No resume argument, inherited training prefix, old weights or old Replay is
used. Both hosts passed all 75 targeted tests and the production-width BF16
CUDA smoke, including exact same-route reset state/action/RNG parity, WM/AC
optimizer updates and adaptive-compression topology restoration. Historical
suite limitations remain as recorded in Decision 0063.

Deployment manifests and live acceptance/monitor state record actual PIDs,
progress, GPU UUIDs, source/config hashes and resources. The monitor now follows
only these current v4 attempts for authorized recovery, while the old ten runs
remain read-only. Prior v3 attempts must not be automatically resumed into v4.
Passing the regression and restarting is not evidence of recovered 4000+
returns; new raw evaluations will be assessed at matched training checkpoints.

## User-requested S1 stop and S2 replacement (2026-09-08/09)

After reviewing the partial validation results, the user requested stopping
S1 and starting another seed. S1 (`seed=1337`, `atari_s1.v4.pilot1`) was
intentionally stopped at 188 completed epochs with epoch 188 incomplete.
It is not a new crash, a completed seed, or an experiment that never ran.
Its logs, validation results, model checkpoints and consumed-work provenance
remain part of this pilot campaign; automatic resumption is prohibited.

The replacement is the next configured seed index 2 (`seed=31337`) on
4090-2, from scratch, with no inherited weights, Replay, RNG or counters.
S0 remains untouched. The source commit remains
`12d317eaf98eaebeb462894894b6fc83bb5e7041`, independently pushed/fetched and
tracked through `codex/awm-autoroute-seed2-20260908`. Method, six-task order,
540-epoch budget, evaluation protocol and eight CPU threads are unchanged.
The existing target-host CUDA smoke applies to these unchanged source bytes.
Deployment and acceptance records identify whether startup actually succeeded.

This replacement was requested after observing performance. Consequently,
S0/S2 must not be presented as the original predeclared two-seed cohort or
an unbiased completed-seed estimate. Reports must disclose the stopped S1
and additional work rather than omit an unfavorable partial outcome.

The user explicitly approved lossless archival of the stopped S1 Replay
to satisfy the unchanged 48 GiB storage preflight. Archive content is checked
against each original file's SHA256, and a separately checksum-verified local
copy is required before removing redundant uncompressed arrays. Model
checkpoint files and metrics are not removed. Historical checkpoint reuse
requires restoring the archived Replay to its recorded paths first; archive
manifests and restore markers document this requirement. No other experiment's
data may be removed for this replacement.

S2 actually launched at 2026-09-09 00:05:21 Asia/Shanghai. Initial acceptance
verified its live trainer and GPU process, fresh Replay/counters, zero Git
divergence and an exact resolved-config comparison: only `seed` changed from
1337 to 31337. It entered epoch 0; this is startup evidence, not a performance
result or a completed training run. S1's four Replay arrays total 12 GiB;
the verified archive is 1,576,236,615 bytes, with an identical local backup.

## User-requested additional S3 on 3090 (2026-09-09)

The user requested one additional Atari seed on the idle 3090. Select the
next configured seed index 3 (`seed=42`) before observing this run's results;
do not replace or restart S0/S2, or resume the user-stopped S1. This is an
addition to the existing pilot history, not a new unbiased predeclared cohort.
The original six-task order, 540 epochs, optimizer/evaluation budgets, method
configuration and eight CPU threads remain unchanged. Use the same source
commit `12d317eaf98eaebeb462894894b6fc83bb5e7041`, tracked through the pushed
`codex/awm-autoroute-seed3-20260909` branch, and a new output directory.

The 3090's existing environments have different dependency versions. Deploy
an independent copy of the currently used AWM runtime at the same absolute
prefix, without changing another project's environment. Compare the complete
package/version list and Python version against the source runtime, then run
the retained-method tests and production-shaped CUDA smoke on the 3090 from
the clean synchronized source. Record the host's CPU, RAM, GPU and driver;
matching packages does not imply identical numerical trajectories across
hardware. The unchanged 48 GiB storage preflight must pass. Available disk
space was already sufficient at this request; no experiment data was deleted.

Deployment, target smoke and actual startup outcomes belong in the S3
manifest and acceptance evidence. Do not report a planned or copying
environment as a running experiment, or a smoke result as a performance claim.

S3 actually launched at 2026-09-09 11:32:15 Asia/Shanghai. Its 75 retained
tests and production-shaped CUDA smoke passed on the 3090; the full runtime
package/version list and Python version matched the existing AWM runtime.
Both prelaunch and postlaunch config comparisons found only the seed change
from S2's 31337 to 42. Acceptance verified a real epoch-0 trainer and GPU
process with fresh Replay, and the original S0/S2 processes remained alive.
Direct GitHub access on the target timed out before tests or training, so
source delivery used the existing controller-fetched, SHA256-verified Git
bundle workflow. The target fetched that verified bundle, retained its GitHub
upstream identity, and passed clean/zero-divergence checks before smoke and
training. No failed setup attempt is counted as a trained seed.
