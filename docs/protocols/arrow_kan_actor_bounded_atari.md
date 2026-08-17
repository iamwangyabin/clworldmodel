# ARROW-KANActorBounded-50 Atari trainability protocol

## Status

Completed T1 pilot. A user-directed T1 duration extension was started fresh,
then deliberately stopped before its final evaluation. This is a corrected
successor to `ARROW-KANActor-50`, not a continuation of that method name. It
changes only the interface between the two fixed-grid KAN layers in the actor;
ARROW replay, the world model, critic, curriculum, environment interaction,
and per-epoch update budgets remain unchanged.

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
examples. This bounded actor fixes that interface failure. The next requested
experiment instead makes the support anchors trainable; it has its own named
protocol in [`arrow_kan_actor_adaptive_atari.md`](arrow_kan_actor_adaptive_atari.md).

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

The completed seed-0 T1 pilot reached a final MsPacman raw return of `1597.5`
with a raw standard deviation of `397.327` over 16 stochastic evaluation
rollouts. This is a trainability result only: it has no Boxing result and makes
no forgetting claim. The historical MLP value of `2109.375` remains a screening
reference only because it predates the current seed-isolation fixes; it is not
a publication-grade paired control.

## T1 180-epoch trainability extension (stopped)

The user requested exactly double the T1 duration before making a continual
learning claim. The named `ARROW-KANActorBounded-50-T1-180EpochTrainabilityPilot`
started fresh on MsPacman and changed only the following schedule budget:

| Item | T1 | T1 180-epoch extension |
| --- | ---: | ---: |
| MsPacman epochs | 90 | 180 |
| Sequential task boundary | 90 | 180 |
| Tasks trained | 1 | 1 |
| Per-epoch interaction and update budgets | unchanged | unchanged |
| Replay capacities and sampling | unchanged | unchanged |

The launcher wrote a resolved run-local config with both `epochs=180` and
`esc.kwargs.swap_sched=180`. Changing both values is required: changing only
the total epochs would silently switch to Boxing after epoch 90. The run was
deliberately stopped by the user after it reached epoch 128, before final
evaluation. Its last observed regular MsPacman value was `1753.125` raw at
epoch 120; this is a partial training observation, not a final experiment
result. The new adaptive-anchor run starts from scratch and does not resume it.

This was an explicit 2x training-budget trainability extension, not a matched
ARROW-50 comparison and not a retention experiment.

## Launch

This historical command describes the stopped fixed-grid extension:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan_bounded \
  --task-prefix-length 1 \
  --task-duration-epochs 180 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_kan_actor_bounded_ar50_t1_e180_s0
```

The manifest records `kan_hidden_adapter=layer_norm_sigmoid`, the 795,858 actor
parameter count, frozen replay accounting, environment seed streams, and the
resolved 180-epoch task boundary. Its missing final evaluation confirms that
the stopped process is not a completed result. The active follow-up protocol is
the trainable-anchor variant in
[`arrow_kan_actor_adaptive_atari.md`](arrow_kan_actor_adaptive_atari.md).
