# Decision 0003: test R2 latent prediction as a named ARROW ablation

- Status: completed pilot; acquisition criterion not met
- Date: 2026-08-16

## Context

The component audits motivate testing whether pixel reconstruction contributes
to encoder or latent-interface drift. A decoder-only interpretation is not
supported by the existing evidence, so the experiment must change the
observation objective without changing replay retention, interaction, update,
or evaluation budgets.

## Decision

Add `ARROW-R2Rep-50` as an opt-in representation-objective ablation. Keep the
full ARROW-50 replay and training protocol, remove the image decoder, and use a
bias-free projector with the stop-gradient R2-Dreamer Barlow Twins objective.
The reusable objective belongs under `src/clworldmodel/`; the vendored trainer
contains only the integration needed to preserve the existing runnable stack.

Keep reconstruction as the default. Do not call this method a faithful
R2-Dreamer baseline because it retains ARROW's world model, KL scales,
actor-critic, and continual protocol.

## Consequences

- Any improvement can be attributed to the observation-objective/head change
  only within the frozen matched protocol.
- Decoder and visual-reconstruction audit metrics are not applicable; encoder,
  RSSM, actor, critic, return, and forgetting metrics remain required.
- Removing the decoder is not assumed to reduce wall time or peak memory: the
  `4096 x 4096` cross-correlation matrix must be measured on the target GPU.
- A smoke run and one seed are not evidence of better retention. The five
  frozen seeds are required for an official comparison.
- The completed seed-0 pilot is retained as a negative result: it was stable
  numerically but did not establish reliable single-task acquisition. The
  native-R2 follow-up is documented separately in Decision 0004.
