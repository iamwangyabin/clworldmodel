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

Monitor both new runs alongside the old campaign, but grant automatic diagnosis
and safe recovery only to these two new runs. Preserve failed attempts and all
raw metrics. Check real process identity, logs, CPU progress, GPU activity,
available RAM/disk and task-boundary artifacts, not just a stale running flag.
Do not treat slow evaluation/compression as a hang or restart based on SSH loss.
Any recovery must validate complete model/optimizer/replay/RNG/schedule state
and record a new attempt and discarded-work accounting. Boundary checkpoints
exist, but the current standalone launcher has no resume CLI: monitor recovery
must not pretend that restarting from weights is equivalent continuation.
Do not modify old jobs, silently change the protocol, repeatedly retry an
unexplained native crash, or delete prior experiment artifacts to make space.

Normal unchanged progress should remain quiet; report new failures, meaningful
boundaries, completion or required user action. Runtime evidence and current
state are stored in the ignored campaign directory rather than this prelaunch
record. No performance result is asserted here.
