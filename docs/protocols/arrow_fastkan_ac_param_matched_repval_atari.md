# ARROW-FastKANAC-ParamMatchedRepVal-50 Atari protocol

## Status and hypothesis

This is a seed-0, one-task MsPacman trainability extension. It tests whether
the first FastKAN Actor/Critic pilot was limited by three controlled factors:
its 37.67% smaller behavior model, omission of the DreamerV3 replay-value
objective, and its 68-epoch acquisition budget. It is not a continual run and
cannot support a forgetting or retention claim.

The ARROW world model, environment interaction, Atari preprocessing, FIFO/LTDM
capacity split, 0.5/0.5 buffer-selection probability, world-model updates, and
actor-critic update count per epoch remain unchanged.

## Parameter-matched behavior heads

Both actor and critic use the same independent PyTorch FastKAN implementation
as `ARROW-FastKANAC-KDAligned-50`: three hidden layers, eight fixed Gaussian
centers over `[-2, 2]`, RMSNorm, a SiLU linear base branch, 1% actor unimix, and
a zero-initialized 255-bin symlog critic output. Only the hidden width changes
from 34 to 53.

| Behavior head | MLP reference | Width-53 FastKAN | Difference |
| --- | ---: | ---: | ---: |
| Actor | 797,202 | 793,692 | -3,510 |
| Critic | 917,759 | 906,978 | -10,781 |
| Combined | 1,714,961 | 1,700,670 | -14,291 (-0.83%) |

The slow critic remains frozen training state and is not included in online
model parameter matching.

## Replay critic objective

The critic receives the existing imagination loss plus a replay loss with
scale `0.3`. No extra replay minibatch is drawn. Each actor-critic update
already samples four real context frames to initialize imagination; their
posterior RSSM states, rewards, and continuation flags are reused.

For context index `t`, the detached target is

```text
R_t = r_t + discount * continue_t *
      ((1 - lambda) * V_boot(t + 1) + lambda * R_(t + 1))
```

and the first three context states receive the unchanged categorical symlog
critic loss plus slow-critic regularization. This follows the purpose and scale
of DreamerV3's `repval` loss but adapts reward timing to ARROW's same-index
storage convention. It is not line-for-line parity with the current JAX
DreamerV3 joint loss pass. Replay sampling counts and ARROW-50 sampling
semantics are unchanged.

This historical extension retains ARROW's online-critic imagination targets
and its pre-transition horizon bootstrap. Those semantics were identified as
likely contributors to unstable FastKAN value learning after this protocol was
defined, so they are not changed retroactively. The separately named
`ARROW-FastKANAC-StableTargets-50` route uses EMA targets and the actual final
imagined state.

## Optimization and duration

The KAN-Dreamer-aligned behavior settings remain LaProp at `4e-5`, epsilon
`1e-20`, betas `0.9/0.999`, 1,000-update warmup, AGC `0.3`, imagination horizon
15, discount horizon 333, lambda `0.95`, entropy scale `3e-4`, persistent p95-p5
return normalization, and slow-critic regularizer/decay `1.0/0.98`.

The run lasts 136 epochs, exactly twice the first FastKAN pilot:

```text
136 * 16,384 = 2,228,224 agent decisions
2,228,224 * frame_repeat(4) = 8,912,896 raw Atari frames
```

After 68 completed epochs, evaluation and a non-resumable analysis snapshot are
added as a declared midpoint. Evaluation RNG state is restored, so this extra
evaluation does not change subsequent training draws. The only task boundary
and final evaluation occur after 136 completed epochs.

## Diagnostics and online logging

TensorBoard records actor reinforce loss, entropy, imagination critic loss,
replay critic loss, total behavior loss, return mean and scale, and behavior
gradient norm against the explicit actor-critic update counter. Non-finite
losses or metrics terminate the run instead of being silently logged.

SwanLab mirroring is optional through `--swanlab-project` and
`--swanlab-experiment-name`. Authentication must come from the host's SwanLab
configuration or environment; the launcher accepts and persists no API key.
TensorBoard remains the canonical local record when SwanLab is unavailable.

## Launch

After a clean pushed commit is synchronized on the GPU host:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac_param_matched \
  --task-prefix-length 1 \
  --task-duration-epochs 136 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_fastkan_ac_param_matched_repval_ar50_t1_e136_s0
```

The predeclared comparisons are the same run's 68-epoch midpoint, the completed
width-34 FastKAN result (`1001.875` raw at epoch 68), fixed-grid bounded KAN
(`1597.5` raw at epoch 90), and historical ARROW-50 MLP screen (`2109.375` raw
at epoch 90). Only the midpoint comparison has the same environment-step
budget, and architecture/loss settings still differ.

## References

- KAN-Dreamer: <https://arxiv.org/abs/2512.07437>
- Official DreamerV3 replay-value reference, commit
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`:
  <https://github.com/danijar/dreamerv3>
- FastKAN: <https://github.com/ZiyaoLi/fast-kan>
