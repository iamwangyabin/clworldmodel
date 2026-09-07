# AWM-AutoRoute Atari two-seed pilot (2026-09-08)

## Predeclared scope

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
