# 0045: Extend Shared-Frozen-Down + Shared FastKAN to the original six tasks

## Status

Accepted for a separately named seed-0 pilot on 2026-08-31. This decision does
not change, stop, or relabel the already launched three-task pilot, and it does
not make a performance claim.

## Context

Decision 0044 authorized a three-task combination of the Shared-Frozen-Down
world model and one persistent StableTargets FastKAN Actor-Critic. The original
ARROW curriculum contains six tasks. Extending the run duration and task set is
a protocol change because it doubles interaction and optimizer budgets, adds
three task-private world-model routes, changes the rehearsal allocation, and
requires rolling checkpoint storage.

The existing private-MLP original-six Evolving-Core pilot intentionally keeps
the older `fixed_v1` Task-0 learning rate. The Shared FastKAN method was defined
with `fixed_v2`. Silently switching it back to `fixed_v1` for the longer run
would confound curriculum length with the selected Task-0 optimizer profile.

## Decision

Add
`Evolving-Core-SharedFrozenDown-SharedFastKANAC-StableTargets-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.
It runs the unchanged method from decision 0044 for 90 epochs on each task in
the canonical order: MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and
Enduro. It retains `fixed_v2` (`3e-4`) for Task 0 and `1e-4` for later shared-core
training.

Keep exactly 800 Actor-Critic optimizer steps per epoch. Task 0 assigns all 800
to the current route; each later task assigns 600 to the current route and
uniformly divides the other 200 over completed routes. Integer remainders go to
lower task indices before the exact multiset is shuffled by the independent
behavior-route RNG. No behavior rehearsal step is added beyond the fixed 800.

Use rolling `latest_boundary` checkpoint retention and the existing 48 GiB free
space preflight for the original-six run. Preserve every raw evaluation vector
and consolidation decision even though only the latest pre/post boundary
checkpoint and Replay snapshot remain live.

## Consequences

- Interaction doubles to `35,389,440` raw frames.
- Online world-model updates double to `540,000`; six boundary consolidations
  add `6,000` explicitly extra updates.
- Actor-Critic updates double to `432,000` without changing per-epoch compute.
- The six-task online model has `71,661,170` world-model parameters and one
  `1,700,670`-parameter FastKAN pair, totaling `73,361,840`.
- The run is directly comparable to the three-task method only on their shared
  prefix and matched checkpoints. Comparisons with older original-six
  Evolving-Core pilots must disclose the `fixed_v2` versus `fixed_v1` Task-0
  difference.
- A single seed remains pilot evidence. It cannot establish superiority or
  support task-agnostic claims.
