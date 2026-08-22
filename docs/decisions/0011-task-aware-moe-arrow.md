# Decision 0011: Replace fixed residual capacity with task-routed experts

## Status

Accepted for the separately named `MoE-ARROW-v1-Atari-TaskAware` experimental
protocol. The completed seed-0 two-task pilot was negative and V1 is preserved
as a failed partial-expert acquisition result. It does not replace ARROW-50 as
the repository baseline. Decision 0012 defines the separately named corrected
full-bank protocol.

## Context

KARROW v1-v4 gives one fixed residual parameter set responsibility for every
future game. Freezing the base protects old behavior, but it also forces all
new dynamics through a small external correction. Keeping the base trainable
improves plasticity but reintroduces interference. A separate RSSM per task is
the natural storage-unconstrained upper bound, and separate policies remove an
independent source of actor forgetting.

The project already uses ARROW replay and a sequential scheduler that knows
task boundaries. Using that scheduler identity inside the agent is useful, but
it changes the comparison class and must be explicit.

## Decision

Implement `moe_arrow` as a task-aware method with hard one-task-one-expert
routing. DINOv3, the posterior representation, and the feature predictor are
shared. The RSSM recurrent dynamics, latent prior, reward head, and continuation
head have one full expert per scheduled game. Each game also owns an independent
DreamerV3 MLP Actor-Critic and optimizer.

The scheduler's integer task ID selects components but is not concatenated to
the latent state. A new task copies the previous task's world-model expert and
Actor-Critic weights once. Its optimizer is fresh, and subsequent parameters are
independent. ARROW stores one task ID per trajectory slot. Every sampled
minibatch is task-homogeneous.

The frozen visual target is a pooled `4 x 4` DINOv3 patch grid projected from
384 to 64 channels by a seeded, task-independent orthogonal matrix. It avoids a
pixel VAE and avoids fitting a projection only on Task 1.

World-model and Actor-Critic update counts remain equal to the source ARROW-50
config. Once old tasks exist, half of each update budget goes to the current
task and half is divided uniformly among replay-available old tasks. This is
budget reallocation, not extra rehearsal.

## Consequences

This method spends substantially more parameter and optimizer memory than
ARROW or KARROW. It also receives privileged task identity. Results must be
reported as a task-aware upper-bound method and never as a like-for-like
task-agnostic ARROW improvement.

The v1 router assumes known boundaries and a fixed task inventory. Learned
routing, expert sharing, task inference, and dynamic expert growth are future
protocols, not silent extensions of v1.

The vendored trainer still cannot create a fully resumable checkpoint because
it does not serialize replay and every optimizer/RNG state. MoE-ARROW therefore
saves final inference weights and records `resumable: false`; an interrupted
run cannot be presented as equivalent to an uninterrupted run.
