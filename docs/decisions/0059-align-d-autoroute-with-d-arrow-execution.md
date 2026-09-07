# 0059: Align D-AutoRoute with D/ARROW execution

## Status

Accepted by project-level review on 2026-09-07. This supersedes Decision 0055
for the maintained D-AutoRoute entry point while preserving v1 results and
provenance as historical evidence.

## Context

D-AutoRoute v1 reused the former D-AutoKAN inference harness. That changed
Gymnasium autoreset from ARROW's implicit `NextStep` behavior to `SameStep`,
replaced D's trajectory-budget evaluator with an exact-complete-episode
evaluator, and changed D's current-task oracle compression gate to an all-seen
auto-routed gate. These changes were declared, but they confounded the intended
comparison of D against D plus observation-based route selection.

The pinned ARROW Atari collector itself is internally inconsistent: with
Gymnasium 1.1.1 it actually uses the default `NextStep` mode, while its comments
and boundary packing describe same-step reset observations. That upstream issue
does not authorize a silent correction inside only one new method.

## Decision

The maintained D-AutoRoute entry point becomes the separately named v2
protocol and preserves D's observable execution contract:

- implicit Gymnasium `NextStep` autoreset and unchanged trajectory packing;
- unchanged action values at reset boundaries;
- D's seeded deterministic trajectory-budget evaluation and return extraction;
- all-seen oracle validation for shared-core consolidation;
- current-task oracle validation for adaptive Q/F/P selection, retaining the
  same 480 nominal selector-rollout budget as D;
- unchanged task-labelled Replay and all optimizer/update budgets.

Only online interaction plus periodic/final inference replace the scheduler's
route with first-frame reconstruction routing. Under `NextStep`, a terminating
worker keeps its old route for the ignored autoreset action and selects its new
route from the reset observation returned by that autoreset step. This preserves
D's RSSM reset mask and stored transition layout.

A one-eligible-route deterministic collection must match D's actions and
trajectory tensors. Any corrected autoreset or exact-episode evaluator remains
a separate future protocol and must be applied to D and D-AutoRoute together.

## Consequences

Existing v1 checkpoints and results remain valid only for the recorded
SameStep/exact/all-seen protocol and are not resume-compatible evidence for v2.
Their performance cannot estimate the isolated cost of automatic routing.
See `docs/protocols/evolving_core_d_autoroute_v2_atari.md`.
