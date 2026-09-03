# 0050: Port Dream Rehearsal with a fixed sample-capacity budget

## Status

Accepted and implemented on 2026-09-04. This decision authorizes code, tests,
and a dry-run launcher. It does not authorize describing an unrun experiment as
validated or reproduced.

## Context

The Dream Rehearsal preprint and its official artifact propose a useful direct
baseline for continual Dreamer: start imagined trajectories from old real
states, grade them with the live world model and critic, and behavior-clone the
shared actor only on the best dreams. The artifact's primary rehearsal library,
however, retains each earlier phase's data without one global fixed capacity.
That storage rule is not comparable to ARROW-50's fixed 524,288-transition
budget.

Counting only bytes would not fix the comparison. A method must not receive
more stored experience merely because its observations use a smaller dtype.
The primary constraint must therefore be the number of stored trajectory slots
and transitions; actual bytes remain a separate mandatory resource metric.

## Decision

Add `Bounded-Dream-Rehearsal-v1-Atari` with:

- the existing DreamerV3 world model and one persistent shared MLP
  Actor-Critic;
- one global random-key `LongTermReplay` reservoir;
- default capacity 1,024 complete trajectories of 512 transitions, exactly
  524,288 transitions total;
- uint8 observation storage in a CPU-backed mmap, without using the smaller
  dtype to enlarge the sample capacity;
- an integer task ID per replay trajectory used only by the scheduler and
  replay filter, never passed to the world model, actor, or critic;
- the reference realized-first score, top-25% selection, horizon 15, and
  actor-only behavior-cloning loss;
- 50 extra actor-only updates for every previously encountered non-current
  task whenever another 2,000 agent decisions become due; and
- explicit separate counters for base Actor-Critic updates, extra rehearsal
  actor updates, dreamed trajectories, selected trajectories, transition
  capacity, and allocated tensor bytes.

The vendored trainer collects 16,384 decisions before an optimizer phase. It
therefore executes all 2,000-decision events crossed during collection at the
next epoch optimizer boundary. It preserves the exact number of due updates
but not their within-collection timing. This is a declared protocol deviation,
not an implementation detail to hide.

The 541st published Atari epoch revisits task 0 after the six-task schedule has
cycled. "Prior" consequently means all encountered tasks other than the task
providing current real data, rather than numerically smaller task IDs.

## Consequences

- This is a bounded adaptation, not a faithful reproduction of the preprint's
  unbounded-data result.
- The method receives substantial extra actor compute. The default original
  541-epoch schedule projects 554,900 extra actor-only optimizer steps in
  addition to 432,800 base Actor-Critic steps. The manifest reports both.
- Each rehearsal step preserves the artifact's 4-sequence by 16-step batch
  layout, imagines from all 64 posterior states, and keeps 16 trajectories.
  Even though this is smaller than one 128-start base actor update, the very
  large number of extra optimizer steps must not be called compute matched.
- A storage ablation changes `--replay-capacity-transitions` and receives an
  ablation role. Non-integral trajectory capacities are rejected.
- If the finite global reservoir loses every trajectory from an encountered
  old task, the run fails instead of silently reducing that task's rehearsal
  budget.
- A clean causal study should additionally run a bounded DreamerV3/LTDM
  buffer-only control. Until that control exists, distinguish the combined
  retention-plus-rehearsal comparison from a rehearsal-only claim.
