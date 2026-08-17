# ARROW-KANActorBounded-50 Atari trainability protocol

## Status

Implementation-ready, unrun trainability pilot. This is a corrected successor
to `ARROW-KANActor-50`, not a continuation of that method name. It changes
only the interface between the two fixed-grid KAN layers in the actor; ARROW
replay, the world model, critic, curriculum, environment interaction, and
per-epoch update budgets remain unchanged.

## Why a new variant is necessary

The completed seed-0 `ARROW-KANActor-50-T2Pilot` used two directly composed
fixed-grid KAN layers. Although the external Dreamer state was mapped to
`[0, 1]`, the output of the first KAN layer was unbounded before entering the
second layer's fixed `[0, 1]` grid.

An offline diagnostic on valid-shaped categorical latents and bounded recurrent
states found that, at both epoch 89 and epoch 179, no second-layer basis was
active in the fixed diagnostic batch. The first-layer output had standard
deviation above `3.2`, while the union of the second-layer compact supports was
only `[-0.6, 1.6]`. On those inputs, the second layer reduced to its output
biases. The run's raw returns (`MsPacman 403.125` at epoch 90 and `Boxing
-1.9375` at final evaluation) therefore do not distinguish a weak KAN
hypothesis from a broken fixed-grid interface.

More training steps cannot reliably repair a compact-support layer receiving
out-of-support values: its basis coefficients receive no gradient on those
examples. The next experiment fixes that trainability issue before changing
the interaction or update budget.

## Method definition

The actor is:

```text
Dreamer state (z, h)
  -> z unchanged; h mapped from [-1, 1] to [0, 1]
  -> fixed-grid ReLU-KAN: 1536 -> 64
  -> LayerNorm(64, eps=1e-3) -> Sigmoid
  -> fixed-grid ReLU-KAN: 64 -> 18
  -> LogSoftmax
```

The sigmoid guarantees every input to the second KAN layer is strictly inside
`(0, 1)`, where its fixed local basis functions are active. LayerNorm keeps the
sigmoid from starting saturated and mirrors the normalization role in the
published MLP actor. The grid locations remain fixed; only KAN coefficients,
KAN biases, and the 64-dimensional LayerNorm affine parameters are trainable.

| Actor | Trainable parameters | Difference from MLP actor |
| --- | ---: | ---: |
| ARROW-50 MLP | 797,202 | 0 |
| direct fixed-grid KAN, historical `relu_kan` | 795,730 | -1,472 (-0.185%) |
| bounded fixed-grid KAN, `relu_kan_bounded` | 795,858 | -1,344 (-0.169%) |

The code contract tests that every second-layer input stays in `(0, 1)`, at
least one compact basis is active per hidden feature, and gradients reach both
KAN coefficient tensors. The old `relu_kan` implementation remains selectable
only to reproduce its checkpoint and failure diagnosis.

## T1 trainability pilot

The first run is deliberately one task, not a continual-learning claim:

| Stage | Environment | Epochs | Purpose |
| ---: | --- | ---: | --- |
| 1 | `ALE/MsPacman-v5` | 0-89 | Establish that the corrected actor can acquire a policy under the unchanged ARROW-50 per-epoch budget. |

The T1 run still uses the frozen ARROW-50 replay allocation: 512 FIFO and 512
LTDM trajectories, sampled with probability 0.5 each. It retains the original
90-epoch task duration, environment frames, world-model updates, actor-critic
updates, optimizer, reward scaling, and 16-rollout stochastic final evaluation.
It does not claim that a single-task run tests forgetting.

At the end of T1, inspect the raw MsPacman curve and final return before
changing a training budget. The historical MLP value of `2109.375` is useful as
a screening reference only because it predates the current seed-isolation
fixes; it is not a publication-grade paired control. If the corrected actor
has a healthy acquisition curve, the next run is the named two-task T2 pilot.
If it remains weak despite active bases, tune actor optimization in a separate
named pilot rather than silently lengthening all budgets.

## Launch

After the implementation commit is clean, pushed, and synchronized on the GPU
host, launch the one-task pilot into a fresh persistent directory:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan_bounded \
  --task-prefix-length 1 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_kan_actor_bounded_ar50_t1_s0
```

The run manifest records `kan_hidden_adapter=layer_norm_sigmoid`, the 795,858
actor parameter count, the frozen replay accounting, environment seed streams,
and final evaluation semantics. This result is a trainability pilot, not a
reproduction or a KAN continual-learning conclusion.
