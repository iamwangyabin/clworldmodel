# 0058: Retire StableTargets and F / D-AutoKAN

Date: 2026-09-07

## Decision

The user explicitly requested removal of both methods after reviewing their
evidence. The project-owned retained methods are now **D** and **D-AutoRoute**.
ARROW-50, DV3/FIFO, native R2-Dreamer and Dream Rehearsal references are unchanged.
This updates the retained scope of Decision 0057, not the meaning of any
historical protocol or published result.

## Implementation and ownership

- Remove StableTargets and F launch selectors, presets, typed config choices,
  FastKAN model exports/source, constructor arguments and vendor runtime paths.
  Direct CLI/config requests for retired choices fail rather than falling back
  to MLP. The StableTargets-only task-duration override is removed.
- Remove the shared-FastKAN behavior/checkpoint branches; preserve D's private
  Actor-Critic bank and D-AutoRoute's parameter-free first-frame router,
  episode lock, eligible-route registry and all-seen compression validation.
- Migrate reconstruction routing coverage to `test_reconstruction_router.py`
  and a retained shared-head model fixture. Remove method-only FastKAN tests;
  preserve shared numerical/LaProp tests in `test_actor_critic_numerics.py`.
- Preserve historical protocol bodies, raw metrics, result registry, failed-run
  evidence and the original fixed-input parity fixture. Mark retired protocol
  entry points as historical. Reporting continues to recognize historical
  method identifiers. No checkpoint, remote job or run directory is deleted.
- Keep the ARROW base commit and licenses unchanged; update `UPSTREAM.md` and
  regenerate its file hash manifest. Neutral historical manifest fields remain
  where needed to preserve retained launch contracts; they do not enable a
  retired architecture.

## Invariants and evidence

Checks used Python 3.12 and PyTorch 2.3.0 in an isolated local test environment,
not a target-CUDA experiment. No environment collection, training launch,
commit or push was performed.

- All six default dry-run manifests (ARROW, DV3, D, D-AutoRoute, native R2 and
  bounded Dream Rehearsal) compare equal to the actual pre-change worktree.
  D/D-AutoRoute retain 552,000 world-model and 432,000 Actor-Critic updates,
  with the same 52,897,535-parameter dense acquisition bound.
- **50 focused tests pass**, covering rejected retired selectors/configs,
  retained MLP initialization/outputs, baseline/D losses and gradient norms,
  private-bank compact checkpoint and RNG restoration, routing/evaluation
  isolation, replay, shared numerical helpers and baseline launch contracts.
- The new `retained_mlp_behavior_parity.json` records the actual pre-change
  `ac.py` source hash, seed and PyTorch version. Existing baseline/D fixture
  assertions retain their numerical tolerances. The old StableTargets fixture
  remains historical data but is no longer executed as a retained method.
- Python syntax checks pass; all **303** ARROW manifest entries verify.

Focused check (from the repository root, using the isolated dependency environment):

```sh
PYTHONPATH=src:tests:scripts python -m unittest \
  test_method_retirement test_retained_method_parity \
  test_reconstruction_router test_d_autoroute test_actor_critic_numerics \
  test_task_replay test_arrow_r2_replay_metadata test_uint8_replay \
  test_bounded_dream_rehearsal_config \
  test_training_launchers.TrainingLauncherTests.test_swanlab_mirroring_records_names_but_no_credential \
  test_training_launchers.TrainingLauncherTests.test_arrow_dry_run_records_complete_analysis_snapshot_contract \
  test_training_launchers.TrainingLauncherTests.test_arrow_cpu_replay_profile_keeps_float32_capacity_and_sampling \
  test_training_launchers.TrainingLauncherTests.test_arrow_single_task_cpu_replay_uses_published_task_config \
  test_training_launchers.TrainingLauncherTests.test_native_r2_dry_run_uses_size12m_geometry_and_arrow_replay -q
```

## Remaining validation limits

Full discovery is **not green**: 248 tests ran, with 21 failures, 57 errors and
7 skips. The broader unfinished retirement migration still has obsolete
private-head/old-method fixtures and launch expectations. Discovery also reports
stale CoinRun derived metrics and a component-audit numerical failure; this
change does not regenerate historical results or relax those assertions.

Current D-AutoRoute checkpoint round trips pass, but cross-revision checkpoints
and resolved configs containing retired neutral fields have not been migrated.
They must use their recorded revision until separately validated. This scoped
retirement is not acceptance of the whole worktree for a training campaign.
