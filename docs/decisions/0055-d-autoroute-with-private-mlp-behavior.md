# 0055: D-AutoRoute retains independent MLP Actor-Critics

## Status

Accepted for implementation on 2026-09-05. This authorizes code, deterministic
tests and dry runs, not training from the current worktree. Framework cleanup
is explicitly deferred.

## Decision

Add a standalone entry point, `scripts/run_evolving_atomic_rssm_d_autoroute.py`,
for method
`evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`.
Compose the existing D launcher and trainer rather than duplicating them.
Keep original D and D-AutoKAN independently selectable and unchanged in meaning.

This method changes how acquired world-model routes and private policies are
selected during interaction/evaluation. It does **not** share, shrink, or
distill Actor-Critics. All D MLP/optimizer settings, task-private ownership,
world-model acquisition, and Q/F/P compression candidates are preserved.
Training and Replay remain task-labelled; the new method is not fully
task-agnostic continual learning.

At every episode start, reuse the parameter-free shared-decoder reconstruction
router introduced for D-AutoKAN. Group RSSM inference by inferred route, then
apply the corresponding private Actor to each group. A temporary read-only
`RoutedActorBank` references existing Actors, without new weights, optimizer
state, or copies. The acquired registry comes from the training schedule, not
the evaluation task. Never reuse the currently acquiring Actor for all workers
or retrieve an Actor using the true evaluation label.

Retain D-AutoKAN's exact seeded complete-episode evaluator, explicit same-step
autoreset, and all-seen auto-routed compression gate. These are declared
differences from legacy D: matching update counts alone does not match total
compute or evaluation conditions. Validate every seen task after each candidate
because changing one route can change other tasks' route rankings. A relative
return gate is not proof of correct recognition or zero forgetting.

## Accounting and evidence

- D's six independent MLP pairs remain `10,295,910` parameters; routing adds no
  learned parameters. Total online parameter bounds remain D's `32,935,103`
  through `52,897,535`, depending on actual Q/F/P widths.
- World-model optimizer steps remain `552,000`; online AC steps remain
  `432,000`. AC compression adds zero steps.
- Compression selection uses `1,680` complete episodes across six boundaries,
  versus D's `480` nominal rollouts, plus per-eligible-route reset probes.
- Final held-out evaluation is never used for model selection. Evaluation
  transitions never enter Replay or any update.
- Existing private-bank checkpoint schema persists Actors, Critics, optimizers,
  RNG and Replay; routing metadata additionally records acquired eligibility.
- Tensor, mock collection/evaluation, config parity, parameter identity,
  compact checkpoint reload and dry-run tests establish execution contracts
  only. Closed-loop Atari performance needs a separately authorized clean,
  pushed-commit pilot and ultimately matched multi-seed evidence.

See `docs/protocols/evolving_core_d_autoroute_v1_atari.md` for the full protocol.
