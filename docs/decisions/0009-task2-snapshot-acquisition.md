# Decision 0009: Task-2 Snapshot Acquisition Diagnostic

## Decision

Add a separate Task-2 acquisition launcher that initializes from the Task-1
analysis snapshot and runs only the second Atari task. The primary arm,
`kan_only`, freezes the original shared core and leaves the complete KAN
residual modules plastic. A matched `kan_plus_heads` arm additionally opens
only the final latent, reward/continuation, actor, and critic readouts.

## Rationale

The six-task replay-consolidated run does not establish whether Boxing is
unlearnable or merely over-constrained by the frozen-core rule. The existing
analysis snapshots contain model weights but intentionally omit replay,
optimizer, RNG, and schedule state. A fresh Task-2 replay and optimizer make
the diagnostic explicit instead of pretending to resume the original
continual run.

KAN-only is tested first because the research claim requires KAN to provide
the new-task plasticity. The small-head arm is a diagnosis and fallback, not a
replacement for the primary method.

## Consequences

- The result is a trainability result, not a forgetting result.
- Task-2 environment interaction and update budgets remain 90 epochs and are
  recorded separately from the parent six-task campaign.
- A new resumable checkpoint format is not introduced by this decision.
- The source snapshot checksum and all reset-state semantics are recorded in
  the launch and resume manifests.
