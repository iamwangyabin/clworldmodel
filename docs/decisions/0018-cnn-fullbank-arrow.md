# Decision 0018: Bank the complete CNN world model by task

## Status

Accepted as the separately named `CNN-FullBank-ARROW-v1` task-aware method.
This does not replace published ARROW-50 or reinterpret the negative DINO
pilots.

## Context

The DINO task-bank variants changed the visual representation while also
testing task routing. Their first-task acquisition was too weak to justify a
continual campaign. The original Dreamer CNN path has already demonstrated
stronger MsPacman acquisition in this repository, but sharing that trainable
encoder across tasks would leave a direct forgetting path into every old RSSM.

The intended question is therefore narrower: retain the original CNN and pixel
world-model objective, while giving the known scheduler task ID a completely
isolated model and policy route.

## Decision

Add `cnn_fullbank_arrow`. Each scheduled task owns its CNN encoder, posterior,
recurrent dynamics, latent prior, pixel decoder, reward head, continuation
head, and Actor-Critic. The scheduler hard-routes a homogeneous minibatch; task
identity is not concatenated to observations or latent state.

Task 0 starts from the normal initialization. At each later boundary, the prior
task's complete world-model route, including the CNN, is copied exactly once.
The new Actor-Critic is initialized independently. Only the current route is
plastic, and all fixed world-model and Actor-Critic updates are assigned to the
current task. Old routes are frozen and used unchanged for old-task evaluation.

The execution profile requires BF16 autocast and uint8 file-backed observation
replay. It supports fixed-global-batch DDP with 1, 2, or 4 devices using the
same sample and update budgets as the existing DINO-ConvBank DDP path.

## Consequences

- This is a privileged task-aware upper bound, not a task-agnostic continual
  learner or a published ARROW reproduction.
- Six tasks allocate six complete world models and six Actor-Critics. The CNN
  alone has 691,104 parameters per task, or 4,146,624 across six routes.
- Old-task functional isolation is exact at the parameter-routing level; there
  is no shared trainable visual adapter.
- Positive retention would not show that replay alone prevented forgetting,
  because old routes receive no later updates.
- The first gate is a completed 90-epoch MsPacman pilot with final raw mean at
  strictly greater than 2,000. A smoke run or favorable intermediate peak
  cannot pass it.
