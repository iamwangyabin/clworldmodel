# AWM Table 9 Frozen Diagnostics v1

Status: fixed evaluation-only protocol, 2026-09-19.

## Cohort

Run the diagnostic on Atari AWM-AutoRoute seeds **S0, S7, S11, and S12**.
S30 remains in the five-seed main-result cohort, but is excluded from Table 9
because its final inference checkpoint and Replay payload are no longer
available. This exclusion is based on artifact availability, not performance.
Table 9 must therefore be labeled `n=4`; its summaries must not be mixed with
the five-seed main table without an explicit cohort label.

## Frozen-checkpoint return diagnostics

Use each run's final boundary-6 inference snapshot and the exact held-out final
task seeds recorded in `evaluation_seed_manifest.json`.

1. Reuse the saved final auto-route returns, after checking that they are the
   task-ID-free evaluation after 540 completed epochs.
2. Evaluate the same checkpoint and cohort with oracle task routes.
3. Evaluate it again with oracle routes after disabling the three RSSM
   mechanism banks' historical reuse contributions. Do not alter any tensor.

Each newly evaluated condition uses the original 16 rollouts per task. The
reuse comparison is oracle-routed so that changing reconstruction routing does
not confound the causal intervention. Report per-task raw-return differences
for auto minus oracle and reuse-on minus reuse-off.

## Predictive-retention diagnostic

Collect a separate held-out trajectory cohort from the frozen final checkpoint
with the oracle task route and deterministic task policy. The seed domain is a
fixed `SeedSequence` of the training seed, the constant `0x54414239`, and task
index. These transitions never enter Replay and never select model widths.

For every task, collect 4,096 decisions per environment worker, select 64 valid
chunks after excluding open-loop targets that cross an episode boundary, and
teacher-force 16 steps. From the final teacher-forced state, measure open-loop
prediction at horizons 1, 2, 4, 8, and 16:

- pixel MSE on observations normalized to `[0, 1]`;
- reward MAE converted back to raw task reward units.

Report the per-task mean and sample standard deviation. Aggregate across the
four seeds only after all four manifests and checksums pass.

## Isolation and provenance

The command is `scripts/run_awm_table9_diagnostics.py`. It requires a clean,
pushed, upstream-synchronized diagnostic commit. It performs zero optimizer or
training steps, leaves Replay unopened, writes atomic JSON plus SHA-256
sidecars, restores the reuse flags after intervention, and rejects any detected
checkpoint tensor mutation.
