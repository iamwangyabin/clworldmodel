# CoinRun-Frozen-Prefix4-Once-ClosedLoop-Debug-v1

Hypothesis: initialize with the existing first-frame reconstruction route,
execute four real actions, then once replay the five observed frames under each
eligible candidate and select the lowest mean posterior reconstruction MSE.
Thereafter lock the route until the episode ends. This may correct initial
routing while avoiding continuous all-candidate inference. It need not improve
returns; raw closed-loop returns, not classification alone, are the outcome.

## Relationship to the exploratory diagnostic

The earlier `CoinRun-Frozen-Sequence-Routing-Diagnostic-v1` used diagnostic root
seed 20260907 and found equal policy-cohort accuracy for 4/8-transition windows.
Choose 4 as the shorter tied candidate, **not as a pre-exploration optimum**.
Use a new root seed 20260908, NumPy SeedSequence domains 4001 (environment) and
5001 (action-space RNG), episode indices 0–15. Every concrete seed is preserved.
No further window/margin selection occurs on this evaluation cohort.

This is a single-checkpoint **debug**, not a multi-training-seed result. Keep
the previous seed-index-0 270-epoch checkpoint unchanged:
SHA256 `de895252087fe0374d4ba28ed8b11a853cef144766482402bda54eab7d3d375d`,
training seed 123456789, recorded source commit
`837ec273b05e74dd4c0de08912123b00729ddcdb`. Only completed routes 0/1/2,
CoinRun / CoinRun+NB / CoinRun+NB+RT, are eligible. Preserve the later training
failure; do not resume it or claim a final six-task checkpoint. Use the existing
isolated historical-source diagnostic branch, not the dirty current main tree.

## Matched complete episodes

Two arms, 16 exact episodes per task per arm: 48 paired episodes, 96 total.
Environment seeds match within each pair and are paired across visual tasks.
For every task/episode, execute both arms, reversing their order on odd episode
indices to reduce warm-up/shared-GPU ordering bias. No replay writes, parameter
updates, imagined rollouts or controller training. Candidate policies are frozen
and deterministic (latent mode, actor argmax, saved BF16 compute, FP32 MSE).
Zero optimizer updates; the existing constructor's temporary optimizer is discarded.

Reuse the original native CoinRun adapter/options, 15 actions, reset dummy no-op
4, repeat 1, raw rewards and RGB uint8 64x64. Preserve the checkpoint's 32,768
decision safety cap; a cap hit fails rather than accepting a partial return.
The worst-case upper bound is 3,145,728 decisions, but ordinary native terminal
episodes end sooner. Never replace an episode based on its score.

Both arms execute precisely the same first four actions from the first-frame
route. Before action five, the candidate arm uses observed frames o0..o4 and
executed actions a0..a3. Every route reconstructs its own posterior history from
zero/reset, never from the incumbent route's hidden state. The selected route's
**already updated** z4/h4 feed its own actor directly; do not advance o4 twice.
Only subsequent observations use the regular routed collector. Smallest route
ID wins ties. No switching penalty, oracle task ID, reward-based gate, KL,
CUSUM, periodic reprobe or ongoing all-expert state maintenance is introduced.
An episode ending before recheck retains its original route; the native terminal
autoreset image is excluded. Prefix hashes and initial routes must match per pair.

## Evidence and cost

Persist every episode's true label for audit only, seeds, exact action/reward
sequence, raw return/length, initial/final inferred route, recheck score vector,
whether a switch occurred, and per-decision model wall time. Save both arms'
first-five-frame prefixes with checksums. Labels never enter either policy or
prefix-scoring function. Report each seen task and the balanced pooled cohort;
do not invent forgetting from one checkpoint without acquisition evaluations.

Use central `paired_raw_return_summary` for means, unnormalized paired differences
and wins/ties/losses. Report routing corrections and damage separately. CUDA
synchronization brackets each model decision in both arms; recheck timing includes
history transfer/recovery, all candidate decoders and selected actor. Report
median/p95 at the corresponding decision and full model-time/decision totals.
Different episode lengths and shared-GPU interference preclude treating total
wall-time ratios as a clean online speed comparison. Before/after frozen-state
hashes must agree. Preserve preflight/failed-run evidence and exact exit code.

Launch only from a clean pushed commit after upstream fetch/verification. Reuse
the existing checksum-verified complete-history bundle and fresh controller
receipt workflow. Save typed resolved configs, hardware/dependencies, seeds,
budgets, source/checkpoint/vendor provenance and full launch command. New entry:
`scripts/eval_coinrun_prefix_refinement.py` with required `--checkpoint`,
`--output-dir`, `--upstream-verification`; paths resolve against repository root.
No existing trainer, vendor source, baseline protocol or active job is modified.

Validation: `PYTHONPATH=src:tests:scripts python -m unittest test_coinrun_prefix_refinement test_coinrun_sequence_routing_probe test_d_autoroute_coinrun test_d_autoroute test_fastkan_autoroute`.
The launcher writes `SUMMARY.md` via versioned `render_report(results)`; tables
are reproducible from the preserved `results.json` without further interaction.
