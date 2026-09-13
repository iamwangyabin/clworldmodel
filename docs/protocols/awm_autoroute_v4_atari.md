# AWM-AutoRoute v4: preserve AWM policy state on unchanged routes

## Identity and scope

- Method: **AWM-AutoRoute**, same method key and independent launcher.
- Protocol: `Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-TwoFrameProbabilityRouter-ARROWParity-v4-OriginalSix-Atari-TaskAwareTraining-TaskIDFreeInference-Pilot`.
- Resolved config: `task_route_inference=two_frame_probability_reconstruction`,
  `task_route_inference_version=4`. Only this version is maintained.
- User approved this correction and fresh Atari S0/S1 restarts on 2026-09-08.
  Historical v3 logs/checkpoints stay separate; they cannot resume as v4.

## Invariant and correction

The [v3](awm_autoroute_v3_atari.md) two-observation score, probability-only
routing decoder, deterministic candidate histories, eligibility, ties,
diagnostics and bounded computation remain unchanged. A route window restarts
on the actual reset observation, independently for each worker. Its scoring
histories still start from zero with dummy action 0.

The **selected policy** is not a scoring history. With no previous route, use
the collector's initial state and previous action. If the selected ID remains
unchanged, preserve the collector's z, h and previous action exactly, even at
the start of a new episode. In particular, do not impose an additional reset
on top of AWM/ARROW's existing NextStep/RSSM mask timing. Preserve AWM's exact
uint8-to-float observation conversion path as well.

Only a genuine change from an already selected route substitutes the selected
candidate's own prior history. On observation 1 this is its zero state and
dummy action; on observation 2 it is its independent deterministic history
from observation 1 with the actual executed action. Never carry another
expert's hidden state into a new expert. Subsequent observations do not probe.

A single eligible expert must match oracle AWM's state/action/RNG trace for
identical weights, observations and sampling RNG, including staggered NextStep
resets. This is a fixed-input execution contract, not a promise to reproduce a
particular historical return. With multiple experts, route changes intentionally
change the policy path; final per-episode routing accuracy alone is not proof
that every action used the oracle route.

## Unchanged protocol

Training and Replay remain task-aware; interaction and reported evaluation
infer IDs. Private MLP Actor-Critics, original six Atari tasks, order, 90 epochs
per task (540 total), ARROW legacy collection and evaluation accounting, fixed
validation/held-out cohorts, oracle consolidation/compression, 552,000 WM and
432,000 AC updates, 480 nominal selection rollouts, BF16 compute with FP32
master weights, and uint8 replay budgets are unchanged. Evaluation never enters
Replay. No new parameters, evaluations or optimizer steps are added.

The config schema rejects v3/missing AutoRoute versions before interaction.
Both inference and resumable checkpoints record the version; full restoration
rejects cross-version configs. Historical oracle AWM checkpoints can still
receive the default-off version 0 without changing their behavior.

## Evidence and launch requirements

`tests/test_two_frame_autoroute.py` checks a real retained RSSM over staggered
NextStep terminal/reset observations, deterministic and stochastic actions, and
exact state/action/RNG equality to the oracle path. A two-worker regression
separates a same-ID reset from an actual first-frame expert switch; the
existing second-frame switch and mixed-precision tests remain in place.
The new regressions fail on the pre-fix adapter. The target CUDA smoke also
checks production-width BF16 state/action/RNG parity alongside WM/AC updates
and adaptive-compression checkpoint topology.

```bash
PYTHONPATH=src:tests:scripts python -m unittest test_two_frame_autoroute test_d_autoroute
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 0 --dry-run
```

Every smoke/update or training launch requires a clean pushed synchronized
commit, recorded provenance and resource preflight. Fresh v4 seeds do not
inherit v3 weights, Replay, RNG, metrics or progress. v3 stopped/failed records
are retained as negative/incomplete evidence, not discarded seeds. A v4
performance improvement is unproven until the new evaluations exist.
