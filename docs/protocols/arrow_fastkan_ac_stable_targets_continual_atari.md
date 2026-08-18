# ARROW-FastKANAC-StableTargets-50 continual Atari protocol

## Status and question

This is a seed-0, six-task continual-learning pilot. The preceding 90-epoch
MsPacman screen established that the stable-target FastKAN actor and critic can
learn the first task; this run asks whether that behavior is retained while the
same agent continues through the frozen ARROW Atari curriculum.

The direct comparison is the ARROW-50 MLP behavior pair. A lower forgetting
score in one seed is evidence for a follow-up, not a claim that KAN generally
prevents forgetting. This protocol changes the behavior architecture together
with its KAN-Dreamer-aligned optimizer and target semantics, so it measures the
complete stable FastKAN behavior package. An architecture-only attribution
would additionally require an MLP control with the same optimizer and target
semantics.

## Frozen continual curriculum

The run uses the original-order ARROW-50 seed-0 configuration without an epoch
or schedule override:

1. `ALE/MsPacman-v5`, epochs 0-89
2. `ALE/Boxing-v5`, epochs 90-179
3. `ALE/CrazyClimber-v5`, epochs 180-269
4. `ALE/Frostbite-v5`, epochs 270-359
5. `ALE/Seaquest-v5`, epochs 360-449
6. `ALE/Enduro-v5`, epochs 450-539

The source configuration has 541 loop epochs. Epoch 540 preserves the upstream
ARROW comparison point after the six 90-epoch acquisition blocks. The budget
is 8,863,744 agent decisions and 35,454,976 raw Atari frames. ARROW-50 retains
512 FIFO and 512 LTDM sequence slots and selects each replay buffer with
probability 0.5. Task identity is available only to orchestration and
evaluation; it is not an actor, critic, or world-model input.

World-model architecture and losses, replay capacity and sampling, collection,
world-model updates, actor-critic updates, preprocessing, reward scaling, and
evaluation cadence remain unchanged. The only method-level change is the
stable width-53 FastKAN actor-critic package defined by
`arrow_fastkan_ac_stable_targets_atari.md`.

## Retention readout

Evaluation remains isolated from training and replay and covers all six tasks
every ten epochs. Future-task evaluations are retained for forward-transfer
analysis but are excluded from forgetting references.

For task `i`, the acquisition evaluation is the first periodic evaluation
after its 90-epoch block: epochs 90, 180, 270, 360, 450, and 540. The final
matched comparison is epoch 540. Preserve the complete raw-return vector and
compute, for tasks 1-5:

```text
forgetting_i = max raw_return_i(e), acquisition_i <= e <= 540
               - raw_return_i(540)

backward_transfer_i = raw_return_i(540)
                      - raw_return_i(acquisition_i)
```

Task 6 has no post-acquisition retention interval, so its current-task return
is reported but excluded from mean forgetting and mean backward transfer.
Report per-task raw values first. Any cross-game aggregate is a separate
normalized metric with the same fixed constants for KAN and MLP.

The two completed historical seed-0 MLP runs are useful screening references,
but they predate explicit environment-worker seeding and evaluation-RNG
isolation. A strict matched conclusion therefore requires a fresh MLP control
from the same commit and multiple paired seeds. No seed may be selected or
discarded based on the result.

## Launch

After a clean pushed commit is synchronized on the GPU host:

```bash
python scripts/run_arrow_ar50_atari.py \
  --actor-network fast_kan_ac_stable \
  --seed 0 \
  --profile-stages \
  --cpu-threads 12 \
  --output-dir /persistent/path/arrow_fastkan_ac_stable_targets_ar50_continual_original_s0
```

Omitting both task-prefix flags is intentional: it selects the complete frozen
curriculum rather than another one-task trainability screen.

