# 0052: Add D-AutoKAN without redefining D

Date: 2026-09-05

## Decision

Add a distinct original-six profile combining D's Dense-acquire adaptive Q/F/P
and shared prediction heads with one shared FastKAN StableTargets Actor-Critic
and first-frame reconstruction routing. Keep D and the adaptive-behavior
residual scheme E unchanged as separately named controls.

The user selected **task-aware training, task-ID-free interaction/evaluation**,
not label-free Replay grouping or automatic task discovery. Implement a
parameter-free per-worker first-frame pixel-MSE argmin and episode lock.
Eligible routes come from acquired model slots, never evaluation task identity.
Keep true task labels strictly on the training and posthoc diagnostic sides.

Use the existing width-53 shared FastKAN and 75/25 current/old AC rehearsal
without adding online optimizer steps. Do not import SharedFrozenDown, private
prediction heads, or the earlier three-task `fixed_v2` optimizer profile.

Require every seen task's **auto-routed** return to pass the compression gate.
A compact candidate may alter first-frame routing even when oracle-routed
current-task return is unchanged. Budget the extra all-seen validation rather
than presenting it as compute-matched to D.

## Consequences

- Fixed online AC size 1,700,670 replaces the six-pair 10,295,910 bank; WM
  private state still grows and compression remains outcome-dependent.
- No restored private decoder bank or learned classifier is needed, but first
  observations require one posterior/decoder probe per acquired route.
- The new collector explicitly uses same-step autoreset and reset no-op actions.
  The new evaluator consumes exactly 16 seeded episodes rather than inheriting
  legacy partial/undersized return estimates. Both are named deviations; old D
  results are not silently relabeled as evaluator-matched.
- Historical high recognition accuracy came from a private FullBank diagnostic,
  not this topology or closed-loop policy. New performance claims require runs.
- Restore validates the recorded eligible prefix. Old-method checkpoints can
  omit the newly added default-off inference fields without changing behavior.

Details, resource accounting, claim limits and validation are in
[the protocol](../protocols/evolving_core_fastkan_autoroute_v1_atari.md).
