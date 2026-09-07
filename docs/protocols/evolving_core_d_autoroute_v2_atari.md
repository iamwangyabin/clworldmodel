# AWM-AutoRoute v2: AWM-compatible collection and evaluation

Historical only; retired from current code on 2026-09-08. The sole maintained
method is [AWM-AutoRoute](awm_autoroute_v3_atari.md). The definition below is
unchanged; current code rejects its route mode. Historical reproduction requires
the recorded revision, not a runtime version switch.

Formerly D-AutoRoute v2. AWM stands for **Accumulative World Modeling**.
Decision 0060 changes the formal name only; entry-point paths, method keys,
protocol IDs and the v1/v2 distinction remain unchanged.

## Identity and hypothesis

- Entry point: `scripts/run_evolving_atomic_rssm_d_autoroute.py`.
- Method key:
  `evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`.
- Protocol:
  `Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-FirstFrameRouter-ARROWParity-v2-OriginalSix-Atari-TaskAwareTraining-TaskIDFreeInference-Pilot`.

This is the controlled comparison requested by Decision 0059: retain AWM's
training, collection, evaluation and selection protocol, changing only how an
acquired world-model/private-Actor route is selected during interaction and
reported inference. Training and Replay remain task-labelled, so this is not
fully task-agnostic continual learning.

## AWM parity contract

AWM and AWM-AutoRoute v2 retain identical task order, 90-epoch durations, BF16
profile, Replay, model/behavior topology, learning rates, update counts,
environment seeds, deterministic evaluation policy and Q/F/P candidate grid.
The resolved method differs only by its method identity and
`task_route_inference=first_frame_reconstruction`.

The Atari collector deliberately preserves pinned ARROW behavior:

- construct `AsyncVectorEnv` without an `autoreset_mode` override, yielding
  Gymnasium 1.1.1's default `NextStep` behavior;
- preserve AWM's reset mask, reward/continuation shifting and stored action;
- preserve AWM's stochastic collection and deterministic evaluation choices;
- preserve AWM's trajectory-budget evaluator and S/E return extraction.

ARROW's comment describes same-step observations even though its pinned runtime
uses next-step autoreset. This is recorded as an upstream inconsistency, not
silently repaired here. A corrected collector requires a separately named,
matched rerun of all compared methods.

## Route timing under NextStep

At the initial `env.reset`, the first observation selects and locks one eligible
route. After a terminal observation, Gymnasium ignores the following action and
returns the new reset observation. The old route is retained for that ignored
action; the returned reset frame selects the next episode's route before its
first effective action. RSSM reset handling and stored transitions remain AWM's.

For every route candidate `k`, selection uses a deterministic zero-state
posterior and shared-decoder pixel MSE. The lowest score wins, ties choose the
lowest eligible ID, and the matching private Actor supplies the action. Future
unacquired routes and true environment task IDs are unavailable to selection.
Collected transitions keep the scheduler's task label for Replay and updates.

## Evaluation and boundary selection

Periodic and final evaluation use automatic routing but otherwise call AWM's
seeded deterministic legacy evaluator with the same nominal 16-rollout request.
Routing labels are attached only afterward for accuracy/confusion diagnostics.

Training-time model selection remains task-aware exactly as in AWM:

- shared-core consolidation validates every seen task with its oracle route;
- adaptive Q/F/P compression validates only the completed task with its oracle
  route and fixed pruning seed;
- each Dense teacher and four candidates receives 16 nominal rollouts, totaling
  `6 * 5 * 16 = 480` selector rollouts across six boundaries.

The optimizer budgets remain 552,000 world-model and 432,000 Actor-Critic
updates. The six private MLP Actor-Critic pairs remain 10,295,910 parameters;
the router adds no learned parameters.

## Verification and claim boundary

The focused contract requires:

- no `autoreset_mode` override for either AWM or AWM-AutoRoute;
- the same legacy evaluator budget and return extraction;
- oracle consolidation/compression gates;
- identical deterministic trajectories when only one route is eligible;
- inferred private-Actor selection when multiple routes are eligible.

Inspect without environment interaction or gradient updates:

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 0 --dry-run
PYTHONPATH=src:tests python -m unittest test_d_autoroute test_reconstruction_router
```

Historical v1 SameStep/exact runs remain separate evidence and cannot support a
v2 performance claim. A real v2 pilot still requires a clean pushed commit and
target-CUDA smoke under the repository reproducibility contract.
