# ARROW-FastKANAC-KDAligned-50 Atari trainability protocol

## Status

This is a fresh one-task architecture and optimization pilot. It replaces both
behavior heads with FastKAN while retaining ARROW-50 as the replay and world
model base. Its first run is
`ARROW-FastKANAC-KDAligned-50-T1-68EpochTrainabilityPilot`, a seed-0 MsPacman
acquisition screen. It has no second task and cannot support a forgetting claim.

The seed-0 pilot completed successfully. Its final 16-rollout MsPacman raw
return was `1001.875 +/- 450.017`; the highest periodic observation was
`1458.125` at epoch index 50. This is a negative acquisition result relative to
the historical MLP screen, not evidence about continual retention.

KAN-Dreamer reports a JAX implementation but does not link author-released
source code. This protocol is therefore an independent PyTorch reconstruction
from the paper, the official DreamerV3 configuration, and the original FastKAN
reference implementation. It is not source-code parity with KAN-Dreamer.

## Behavior architecture

Actor and critic consume the unchanged ARROW Dreamer state `(z, h)`, with 1,536
features in the published Atari configuration. Each head uses three hidden
FastKAN layers of width 34 followed by a FastKAN output layer. A layer computes

```text
RBF(x)  = exp(-((RMSNorm(x) - center) / bandwidth)^2)
output  = tensor_contract(RBF(x), rbf_weights) + Linear(SiLU(x))
```

There are eight fixed uniform centers over `[-2, 2]`; the derived bandwidth is
`4/7`. Centers are persistent buffers and receive no gradients. Both branches
use Dreamer-style fan-in truncated-normal initialization and fixed branch
scales of one. The actor output weights use scale `0.01`, followed by a 1%
uniform mixture and log probabilities. The critic output weights start at zero
and produce the unchanged 255-bin symlog distribution.

The paper explicitly specifies FastKAN width 34, eight centers, the center
range, fixed grids, RMSNorm plus SiLU, and joint Actor/Critic replacement. It
does not state layer count, RMSNorm epsilon, output scales, or warmup in the
architecture table. The three hidden layers, epsilon `1e-4`, actor output scale
`0.01`, critic output scale zero, and 1,000-update optimizer warmup follow the
official DreamerV3 behavior-head and optimizer configuration used to interpret
the paper's stated direct replacement.

## Parameter accounting

| Behavior head | Trainable parameters |
| --- | ---: |
| ARROW MLP actor | 797,202 |
| ARROW MLP critic | 917,759 |
| FastKAN actor | 498,090 |
| FastKAN critic | 570,849 |

The online FastKAN pair has 1,068,939 trainable parameters, 646,022 fewer
(`37.67%`) than ARROW's 1,714,961-parameter MLP pair. Width 34 is copied from
KAN-Dreamer's approximately 10.5M whole-model comparison; it does not
parameter-match ARROW's shallower width-512 behavior heads. The critic EMA adds
another 570,849 frozen training-state parameters and is reported separately in
the run accounting artifact.

## Training settings

The directly portable KAN-Dreamer settings are enabled only for this named
protocol:

| Setting | Value |
| --- | ---: |
| Actor/Critic optimizer | LaProp |
| Learning rate | `4e-5` |
| Optimizer epsilon | `1e-20` |
| Betas | `0.9`, `0.999` |
| Gradient clipping | per-tensor AGC `0.3` |
| Imagination horizon | 15 |
| Discount horizon | 333 |
| Lambda | `0.95` |
| Entropy regularizer | `3e-4` |
| Actor unimix | 1% |
| Return normalization | p95-p5, minimum 1, EMA decay `0.99` |
| Critic EMA regularizer / decay | `1.0` / `0.98` |

LaProp applies AGC, bias-corrected RMS gradient normalization, then
bias-corrected momentum, matching the ordering in official DreamerV3. Return
normalization state persists across ARROW epochs for this protocol.

## Preserved ARROW behavior and deviations

ARROW-50's equal FIFO/LTDM capacities and 0.5/0.5 buffer-selection probability,
Atari preprocessing, RSSM, world-model optimizer and update count, behavior
update count, and evaluation remain unchanged. Consequently, this is a
KAN-Dreamer-aligned behavior pilot, not a reproduction of its DMC experiment.

In particular, KAN-Dreamer's replay capacity of five million and batch length
64 are not copied because they would replace the requested ARROW sample
protocol and change the world-model update budget. ARROW continues to use
16-example, length-32 world-model minibatches and 128 imagined starts per
behavior update. KAN-Dreamer's critic replay-loss scale `0.3` is also not
ported: its implementation is coupled to DreamerV3's joint replay/world-model
training pass, whereas ARROW trains behavior in a separate imagination loop.
The manifest records the applied scale as zero and the paper value separately.

## Duration mapping

KAN-Dreamer reports its final result at 1.1M DMC environment steps rather than
epochs. In the frozen seed-0 ARROW configuration, pretraining collection is
disabled and each epoch collects `4 x 4096 = 16,384` agent decisions. The
smallest whole-epoch run reaching the paper target is therefore 68 epochs:

```text
68 x 16,384 = 1,114,112 agent decisions (+1.283%)
1,114,112 x frame_repeat(4) = 4,456,448 raw Atari frames
```

Agent decisions are used only as the declared mapping counter. Raw Atari frames
are not equated to DMC steps. Epoch 68 is also the only task boundary, so no
Boxing samples can enter replay.

## Launch

After a clean, pushed commit is synchronized on the GPU host:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac \
  --task-prefix-length 1 \
  --task-duration-epochs 68 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_fastkan_ac_kd_aligned_ar50_t1_e68_s0
```

The first question is acquisition trainability relative to the completed
bounded ReLU-KAN result (`1597.5` raw at epoch 90) and historical MLP screening
result (`2109.375` raw at epoch 90). The budgets differ, so these are diagnostic
references rather than matched claims.

## References

- KAN-Dreamer: <https://arxiv.org/abs/2512.07437>
- Official DreamerV3: <https://github.com/danijar/dreamerv3>
- FastKAN: <https://github.com/ZiyaoLi/fast-kan>
