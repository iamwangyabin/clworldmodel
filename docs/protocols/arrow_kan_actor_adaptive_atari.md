# ARROW-KANActorAdaptive-50 Atari trainability protocol

## Status

This is a new, fresh one-task trainability experiment requested after the
fixed-grid bounded actor learned MsPacman but its longer run was stopped before
final evaluation. It is a new named actor method, not a resume or a relabeling
of `ARROW-KANActorBounded-50`.

The first launch is
`ARROW-KANActorAdaptive-50-T1-180EpochTrainabilityPilot`: a seed-0,
180-epoch MsPacman run. It tests acquisition only. It has no Boxing stage and
must not be used to make a forgetting claim.

## Method definition

The actor is:

```text
Dreamer state (z, h)
  -> z unchanged; h mapped from [-1, 1] to [0, 1]
  -> trainable-anchor ReLU-KAN: 1536 -> 64
  -> LayerNorm(64, eps=1e-3) -> Sigmoid
  -> trainable-anchor ReLU-KAN: 64 -> 18
  -> LogSoftmax
```

It keeps the bounded interface from `relu_kan_bounded`: the sigmoid makes every
second-layer input lie in `(0, 1)`, avoiding the inactive-basis failure of the
historical directly composed KAN actor. ARROW-50 replay, world model, critic,
curriculum, environment interaction, per-epoch world-model updates, and
actor-critic updates are unchanged.

For grid size `g = 5` and spline order `k = 3`, each input feature has eight
compact bases. The ReLU-KAN basis is

```text
R_i(x) = [ReLU(e_i - x) ReLU(x - s_i) 4 / (e_i - s_i)^2]^2.
```

The original ReLU-KAN source calls its trainable-anchor configuration
`train_ab=True` and makes every per-input, per-basis start and end point
trainable. This actor implements the same ordered support family with
`s_i = anchor_start_i` and
`e_i = anchor_start_i + softplus(anchor_raw_width_i)`. The softplus
parameterization is an intentional project stabilization: it prevents a
non-positive support width and the resulting singular normalization while still
allowing both support position and width to change. It has the fixed-grid
supports at initialization, to floating-point precision, and uses the same
initial coefficient and bias draws as the bounded fixed-grid actor under the
same seed.

The method is independently implemented under
`src/clworldmodel/models/relu_kan.py`, based on the ReLU-KAN formula and its
official reference implementation: [Qiu et al., 2024](https://arxiv.org/abs/2406.02075),
[official source](https://github.com/quiqi/relu_kan). No third-party KAN source
code is vendored.

## Parameter accounting

| Actor | Trainable parameters | Difference from MLP actor |
| --- | ---: | ---: |
| ARROW-50 MLP | 797,202 | 0 |
| bounded fixed-grid KAN | 795,858 | -1,344 (-0.169%) |
| adaptive-anchor KAN | 821,458 | +24,256 (+3.04%) |

The adaptive actor adds `25,600` anchor parameters: two values for each of the
`1536 x 8` first-layer and `64 x 8` second-layer supports. It is deliberately
not parameter matched to the MLP; the run manifest records this increase. This
pilot therefore tests whether trainable anchors can be optimized in this actor,
not whether an exactly parameter-matched architecture wins.

Focused tests verify fixed-grid initialization, positive ordered supports,
nonzero gradients to starts and widths, the bounded second-layer interface,
and the intentional parameter accounting.

## T1 180-epoch trainability pilot

| Item | Value |
| --- | --- |
| Environment | `ALE/MsPacman-v5` only |
| Epochs | 180 |
| Sequential task boundary | epoch 180 |
| Replay | 512 FIFO + 512 LTDM trajectories; 0.5 / 0.5 buffer selection |
| Evaluation | final 16-rollout stochastic evaluation; raw and scaled returns |
| Scope | trainability only; no retention or forgetting conclusion |

The launcher writes a run-local resolved config with
`actor_network=relu_kan_adaptive`, `actor_kan_trainable_grid=true`,
`epochs=180`, and `esc.kwargs.swap_sched=180`. Moving both schedule values is
necessary so the one-task run does not transition to Boxing after epoch 90.
It is an explicit 2x trainability budget, not a matched ARROW-50 comparison.

The preceding fixed-grid 180-epoch extension was deliberately stopped by the
user before final evaluation after it had reached epoch 128. Its last observed
regular MsPacman value was `1753.125` raw at epoch 120. That partial run is not
a final result and does not transfer weights, replay, or optimizer state into
this fresh adaptive-anchor run.

## Launch

After the implementation commit is clean, pushed, and synchronized on the GPU
host, launch into a new persistent directory:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network relu_kan_adaptive \
  --task-prefix-length 1 \
  --task-duration-epochs 180 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_kan_actor_adaptive_ar50_t1_e180_s0
```

The result should be compared first with the completed 90-epoch bounded KAN T1
result (`1597.5` raw) and the historical MLP screening value (`2109.375` raw).
Neither reference is a publication-grade paired control, and neither turns a
single-task result into evidence that KAN solves continual learning.
