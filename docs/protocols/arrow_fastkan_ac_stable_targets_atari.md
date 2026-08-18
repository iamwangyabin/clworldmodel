# ARROW-FastKANAC-StableTargets-50 Atari protocol

## Status and hypothesis

This is a seed-0, one-task MsPacman trainability pilot for the existing
parameter-matched FastKAN actor and critic. It tests a training-target
correction, not a new replay method and not a continual-learning claim.

The preceding FastKAN routes use the online critic for imagination targets and
the actor advantage baseline. Their terminal lambda-return bootstrap also
re-evaluates the last pre-transition state instead of the final imagined
state. This creates a rapidly moving critic target and duplicates the last
online value at the horizon boundary. The effect is shared by the historical
ARROW behavior loop, but it is especially suspect for FastKAN because its
critic loss is already higher and noisier than the MLP critic in the
KAN-Dreamer behavior experiment.

The hypothesis is that a correct terminal state and the existing EMA critic as
the value target will reduce target drift enough for the FastKAN critic to
provide a more useful actor advantage. The completed historical protocols are
left unchanged so their recorded results remain reproducible.

## Controlled changes

The online actor and critic remain width-53 FastKAN heads with three hidden
layers, eight fixed Gaussian centers over `[-2, 2]`, RMSNorm, a SiLU base
branch, 1% actor unimix, and a zero-initialized 255-bin symlog critic output.
Their combined online parameter count remains 1,700,670, which is 14,291
parameters below the ARROW MLP pair.

Only the value-target path changes:

1. The final lambda-return bootstrap evaluates the post-transition imagined
   state `(z_H, h_H)`, not the preceding state `(z_(H-1), h_(H-1))`.
2. Imagination lambda returns use the existing EMA slow critic at all states,
   including the terminal bootstrap.
3. The actor advantage subtracts that same detached EMA value rather than the
   online critic value.
4. Replay-value TD-lambda bootstraps use the EMA critic. The online FastKAN
   critic still receives the categorical imagination loss, the `0.3` replay
   loss, and the unchanged EMA consistency regularizer.

Items 2-4 align the target source with DreamerV3's default `slowtar=True`
behavior. No additional model, replay minibatch, environment interaction,
world-model update, or actor-critic update is introduced.

## Preserved behavior

ARROW-50 FIFO/LTDM capacity and sampling, Atari preprocessing, task identity
isolation, RSSM and world-model training, actor sampling, behavior-update count
per epoch, optimizer, return normalization, evaluation, and parameter budgets
remain unchanged. The historical `fast_kan_ac` and
`fast_kan_ac_param_matched` names preserve their original online-target and
legacy-bootstrap semantics.

## Budget and comparison

The pilot runs for the frozen 90-epoch MsPacman task duration:

```text
90 * 16,384 = 1,474,560 agent decisions
1,474,560 * frame_repeat(4) = 5,898,240 raw Atari frames
```

This matches the historical 90-epoch MLP and bounded ReLU-KAN acquisition
screens. It does not match the 68-epoch first FastKAN run or the 136-epoch
parameter-matched extension, so those comparisons remain diagnostic.

Success requires more than finite losses: report final and periodic raw return,
critic imagination and replay losses, actor loss and entropy, return scale, and
gradient norm. A single seed remains a trainability screen only.

## Launch

After a clean pushed commit is synchronized on the GPU host:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac_stable \
  --task-prefix-length 1 \
  --task-duration-epochs 90 \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_fastkan_ac_stable_targets_ar50_t1_e90_s0
```

## References

- KAN-Dreamer: <https://arxiv.org/abs/2512.07437>
- Official DreamerV3: <https://github.com/danijar/dreamerv3>
- FastKAN: <https://github.com/ZiyaoLi/fast-kan>

## Continual follow-up

The completed seed-0 screen reached raw MsPacman return `2410.625 +/- 713.700`
after 90 epochs. The next registered experiment keeps this exact behavior
package and restores the full six-task, 541-epoch ARROW curriculum. Its
retention definitions, attribution limits, and launch command are in
`arrow_fastkan_ac_stable_targets_continual_atari.md`.
