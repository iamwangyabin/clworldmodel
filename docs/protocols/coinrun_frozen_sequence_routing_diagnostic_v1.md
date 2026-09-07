# CoinRun-Frozen-Sequence-Routing-Diagnostic-v1

Classification: **debug diagnostic**, not training, a reproduction, or a
closed-loop return experiment. Hypothesis: a short observation-action history
can correct errors of the existing first-frame reconstruction router without
changing the trained D-AutoRoute world model or private actors.

## Fixed initial diagnostic

Select seed index 0 by seed order, using its 270-completed-epoch inference
snapshot (training seed 123456789; recorded source commit
`837ec273b05e74dd4c0de08912123b00729ddcdb`). The checkpoint digest is
`de895252087fe0374d4ba28ed8b11a853cef144766482402bda54eab7d3d375d`.
Eligible routes are only the three completed tasks: CoinRun, CoinRun+NB,
CoinRun+NB+RT. The later training failure does not invalidate these earlier
weights and is not hidden or restarted by this diagnostic.

Use that historical source in an isolated worktree for checkpoint compatibility;
do not restore retired implementations into the current main line. Only this
script, its test and this named protocol are new. No vendor behavior is changed.

Two predeclared collection cohorts: uniform random native actions, and the
unchanged deterministic first-frame-routed policy. Each cohort gets 16 independently
seeded episode prefixes per task, at most 32 decisions per prefix: **3,072
maximum native frames/decisions**, no gradient updates, no replay allocation or
writes, no imagined rollouts, no controller training. Root diagnostic seed is
20260907. NumPy SeedSequence domains 1001/2001/3001 separate environment seeds,
random action seeds and shuffled-action permutations. Episode environment/action
seeds are paired across tasks and cohorts. Every concrete seed is recorded.

Retain the existing CoinRun adapter, native 15 actions, reset no-op index 4,
repeat 1, uint8 HWC RGB 64x64 and float32 CHW [0,1] model inputs. Stop at a
terminal step and exclude its native autoreset image from prediction targets;
preserve its raw reward separately. Prefix reward sums are not episode returns.
Environment labels enter construction and reporting only, not routing/policy.

## Matched comparisons

All candidate routes independently replay **the same saved prefix**, starting
from their own zero state and reset no-op, and use posterior-mode filtering.
No candidate inherits another candidate's hidden state. At W = 1,4,8,16,32
valid transitions, compare:

- **A:** first-frame posterior reconstruction pixel MSE.
- **B:** mean filtered posterior reconstruction MSE on frames 0 through W.
- **C:** mean one-step prior-mode decoded pixel MSE on frames 1 through W.
- **C-shuffled:** repeat C with a seeded permutation of each prefix's actions;
  retain observations, action multiset and valid-prefix length. Record how many
  actions actually changed; nearly constant policy actions weaken this control.

The prior is computed before seeing the target frame, then the observation
updates the posterior for the next transition. Reset has no predictive score.
Also save per-frame posterior-to-prior categorical KL (summed latent factors,
no free bits) for a later monitoring-signal analysis; it does not choose routes
in this diagnostic. C is a mode-based proxy, not exact predictive likelihood.
Scores reduce in float32, ties choose the smallest eligible ID. Model compute
uses its saved precision setting. First-frame offline winners must match the
production collector on the policy cohort, or the run fails.

For each cohort/window use the same complete-prefix subset for every method;
report denominators, exclusions, confusion matrices, per-task accuracy, initial
errors corrected and initial correct routes broken. Longer windows may have
survivor selection; do not attribute cross-window changes solely to evidence.
No threshold fitting, winning-window selection or gate implementation is done.
Single-seed results establish only whether further work is worth testing.

NB removes backgrounds and RT restricts themes; these are visual variants,
**not evidence of different transition laws**. Better routing here cannot by
itself establish identification of different physical dynamics. Additional
frames, rather than actions, might explain any gain; report B and shuffled C.
This test spends extra inference compute offline and is not a low-cost online
router or evidence of improved game returns.

## Reproducibility and execution

Entry point: `scripts/probe_coinrun_sequence_routing.py`. Required arguments are
`--checkpoint`, `--output-dir` (new directory), and `--upstream-verification`.
Relative paths resolve against the repository, never the shell working directory.
CLI overrides appear in the resolved typed diagnostic config; a changed budget
is a separately recorded debug run, not a change to any baseline budget.

Before environment interaction, commit and push the exact diagnostic branch,
fetch upstream and verify clean/zero ahead/behind. On hosts without GitHub access,
use the campaign's controller fetch plus full-history Git bundle: verify SHA256
on both ends, preserve the GitHub origin/tracking ref and record the controller
receipt (commit, upstream, ahead, behind, origin, fetched_at_utc, bundle_sha256).
The script checks clean/synced state and a receipt no older than one hour.

Persist checkpoint/config/provenance, dependency/hardware/determinism settings,
raw uint8 trajectories, executed and shuffled actions, masks/seeds, score arrays,
raw rewards and result tables. Check frozen model/actor state hashes before and
after. Record timings, GPU peak allocated bytes and shared-GPU caveat; combined
batched scoring time is not online router FPS. Preserve failures and exit codes.
The external launcher records its exact command and environment overrides.

Focused check (fixed tensors/mocks only, no simulator or parameter updates):

```bash
PYTHONPATH=src:tests:scripts python -m unittest test_coinrun_sequence_routing_probe test_d_autoroute_coinrun test_d_autoroute test_fastkan_autoroute
```

After completion, render all predeclared windows from the preserved metric
records without a model, GPU or simulator (stdlib-only reporting command):

```bash
python scripts/report_coinrun_sequence_routing_probe.py /absolute/path/to/run-directory
```
