# Frozen CoinRun periodic task-ID routing — closed-loop pilot v1

Protocol: `CoinRun-Frozen-Periodic-TaskID-ClosedLoop-Pilot-v1`.
Entrypoint: `scripts/eval_coinrun_periodic_task_id.py`.

## Fixed hypothesis and arms

Keep the existing reconstruction scorer and frozen world model/private actors.
An initial observation supplies the first route; execute it without further
task classification between sparse rechecks. The experiment asks whether this
improves **true task-ID accuracy**, not game return. Do not train a new router,
maintain all experts continuously, or use labels/rewards to select routes.

Three arms: original episode lock (interval 0), every 16 actual decisions, and
every 32 actual decisions. Recheck before actions at t=16,32,... or t=32,64,...,
only if the episode remains active. The current route stays fixed otherwise.
Every recheck scores the most recent nine actual observations (eight intervening
executed actions). Each candidate independently filters this window from zero
using a dummy action/reset on its first frame and deterministic posterior mode;
select minimum mean posterior reconstruction MSE, ties to smallest eligible ID.
No prior, KL/CUSUM, confidence gate, hysteresis, adaptive interval or early lock
is introduced in this pilot. These choices are fixed before examining this
cohort; do not retune after seeing its accuracy and call that held-out evidence.

**Window initialization is approximate context, not a real environment reset.**
The nine-frame window is different from the earlier nine-frame *episode-prefix*
diagnostic; its accuracy cannot be inherited from that report. If the selected
route matches the incumbent, preserve the incumbent's full-history state and
process the current frame normally. If it changes, restore **only the selected
route** from the complete actual episode observation/action history, without a
decoder, and apply its actor to that already-current state. Never inherit the
old expert's hidden state, double-process the current observation, or replay
full histories for all candidates at each recheck. Inactive states are not cached.

## Cohort and unchanged environment contract

Use the existing boundary-03/epoch-270 checkpoint, SHA256
`de895252087fe0374d4ba28ed8b11a853cef144766482402bda54eab7d3d375d`,
training seed 123456789, source `837ec273b05e74dd4c0de08912123b00729ddcdb`.
Only its completed routes 0/1/2 are eligible: CoinRun, CoinRun+NB (no backgrounds),
CoinRun+NB+RT (restrict themes). These are visual variants, not a new dynamics
benchmark. No Atari evaluation or resumed training. Preserve historical failures.

128 full episodes per task per arm: 384 paired task/seed groups, 1,152 episodes.
New root seed 20260910; `seed_for` domains 7401 for environments and 7501 for action
spaces, episode indices 0..127. Pair the same seeds across arms and visual tasks.
Loop tasks then episode indices; rotate the three-arm order by episode index.
Do not discard/replace inconvenient seeds or incomplete episodes. All arms must
exactly match baseline observations/actions/rewards before their first recheck;
an arm that never switches must reproduce the complete baseline trajectory.

Native adapter/options, RGB uint8 64x64, /255 inference, 15 actions, dummy no-op 4,
repeat 1, raw rewards, termination handling and the 32,768-decision safety cap
remain unchanged. Native terminal autoreset images are excluded from history.
A cap hit fails the run, never creates an accepted partial episode. The maximum
budget is 37,748,736 decisions; report the actual count separately. WM/Actor/
Router updates, imagined rollouts, replay writes, replay capacity and bytes are
all zero. All data are evaluation-only. Latent mode and actor argmax, batch 1,
checkpoint BF16 inference with FP32 MSE, TF32 off; no batch-size equivalence claim.

## Metrics and costs

Central `task_id_trace_accuracy` scores the held route at **every actual action**.
Primary: mean over episodes of the within-episode correct-route fraction. Also
report decision-weighted (micro) accuracy, initial/final accuracy, final confusion
matrix, each task, recheck coverage, total checks/switches, wrong-to-correct and
correct-to-wrong events, and initial-wrong/final-correct versus initial-correct/
final-wrong counts. Do not use only late surviving episodes as the denominator.
Preserve raw returns but never reinterpret a misclassified task as correct merely
because it earns reward. Different arm policies can induce different later
trajectories; this is a closed-loop evaluation, not a fixed-trajectory classifier.

Paired differences in the primary metric use 4,000 bootstrap draws over the 128
environment-seed groups (three matched visual tasks per group), seed 20260910.
Intervals condition on this one checkpoint, with no multiple-comparison adjustment.
One checkpoint/seed is a pilot, not proof of high accuracy or a reproduced result.

Save per-episode actions, raw rewards, held routes, complete uint8 observation
history, initial scores, recheck scores/window bounds, executed state hashes,
per-decision timings, episode timings and seeds. Hash raw NPZ files and final JSON.
Before/after hashes of all frozen parameters and buffers must match. The report
is reproducible without environment interaction or updates from saved raw rows.

Synchronize CUDA around each model decision, including scoring and selected-route
restoration; audit-state hashing occurs outside inference latency but inside
episode wall time. Report ordinary versus recheck median/p95 and mean model time
per decision. Shared GPU timing is descriptive, not a dedicated throughput test.
Count RSSM calls: K initial probes; L minus switches ordinary filters;
K times window length for each recheck; t+1 selected-history filters per switch.
Full history retention and occasional long restore latency are real costs; no
claim that sparse decisions make this approach free or always cheap is allowed.

## Reproducibility and validation

Work in the isolated historical diagnostic branch, not the unrelated dirty main
worktree. No vendored behavior, existing baseline, training config or active job
is changed. Launch only from a clean committed/pushed source after upstream fetch
and zero ahead/behind verification; preserve the fresh controller receipt and
checksum-verified bundle, full command/env, hardware/dependencies and provenance.

Fixed-tensor tests (no simulator or parameter updates):

```sh
PYTHONPATH=src:scripts:tests python -m unittest test_coinrun_periodic_task_id test_coinrun_prefix_refinement test_coinrun_sequence_routing_probe test_d_autoroute_coinrun test_d_autoroute test_fastkan_autoroute
```

Launch: `python scripts/eval_coinrun_periodic_task_id.py evaluate --checkpoint SNAPSHOT --output-dir OUTPUT --upstream-verification RECEIPT`.
Recompute: `PYTHONPATH=src:scripts python scripts/eval_coinrun_periodic_task_id.py report OUTPUT`.
All relative input/output paths resolve against the repository root. Save failures
with context and traceback; no later commit may retroactively label a launch.
