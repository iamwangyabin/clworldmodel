# Bounded Dream Rehearsal v1 Atari protocol

## Status

Implemented; target-CUDA smoke and multi-seed evaluation have not yet run.
Results must be labeled `smoke`, `pilot`, or `official` according to the
repository contract. The implementation is not evidence that the paper result
has been reproduced.

## Reference method and provenance

- Preprint: *The World Model Remembers, the Actor Forgets: Dream Rehearsal for
  Continual Model-Based RL*, arXiv:2607.19749.
- Official repository: `https://github.com/gurpnijjer/dream-rehearsal`
- Inspected artifact commit:
  `7680778f798be3a27a17c320cc875b573c45f0e1`
- Artifact license: Apache-2.0.

The project reimplements the small algorithmic boundary against its existing
DreamerV3 interfaces; it does not vendor the reference source tree. The thin
integration lives in `third_party/arrow/Code/ARROW_and_DV3/Atari/ac.py` and
`train.py`. Framework-independent scoring and selection primitives live in
`src/clworldmodel/continual/dream_rehearsal.py`.

## Algorithm

For each rehearsal update for old replay task `j`:

1. preserve the artifact's batch layout by sampling four real replay sequences
   labelled `j` and observing 16 context frames with the shared world model;
2. flatten all `16 x 4 = 64` posterior states into imagined starts;
3. sample 15-step trajectories from the live shared actor;
4. predict reward `r_t`, continuation probability `c_t`, and final critic
   value `V(s_H)` without passing `j` to any network;
5. compute

   \[
   S_t=\prod_{k=0}^{t} c_k,\qquad
   R=\sum_{t=0}^{H-1}\gamma^t
      \left(\prod_{k=0}^{t-1}c_k\right)r_t,
   \]

   \[
   q=10\,\mathbf{1}[R>0.3]+R+\gamma^H S_{H-1}V(s_H);
   \]

6. retain the top `floor(0.25 x 64) = 16` trajectories by `q`; and
7. minimize the negative log probability of their already sampled actions.

Only actor parameters receive gradients. The world model and critic grade the
dreams but are unchanged by the rehearsal loss. The standard DreamerV3 update
phase remains unchanged.

## Bounded replay contract

The default library has one global `LongTermReplay` random-key reservoir:

| quantity | value |
|---|---:|
| complete trajectory slots | 1,024 |
| transitions per trajectory | 512 |
| total transition capacity | 524,288 |
| Atari observation | 64 x 64 x 3, uint8 |
| observation device | CPU file-backed mmap |
| action/reward/continue/reset | float32 CPU tensors |
| task metadata | one int64 per trajectory |

The top fixed random keys are an exchangeable uniform reservoir over eligible
complete trajectories. The buffer never grows after allocation. The task ID is
storage metadata used to request homogeneous old-task starts; it is neither an
observation nor a network input.

**Fairness rule:** 524,288 stored transitions is the primary match to ARROW-50
and DreamerV3/FIFO. uint8 changes actual bytes but does not buy additional
samples. Launch and runtime artifacts report observation bytes, auxiliary
tensor bytes, task-ID bytes, and the unmeasured Python sorted-key index
separately.

The default allocation is 6,486,499,328 tensor bytes: 6,442,450,944 mmap
observation bytes, 44,040,192 auxiliary bytes, and 8,192 task-metadata bytes.
Filesystem and Python-container overhead are not included.

## Update and compute budget

The reference cadence is retained: after every 2,000 newly completed agent
decisions, schedule 50 actor-only updates for every encountered non-current
task. The Atari collector emits 16,384 decisions per epoch, so due events are
batched after collection. Integer division of the cumulative decision counter
preserves remainders and the exact event count.

For the original-order 541-epoch config, the launch projection is:

| counter | value |
|---|---:|
| base world-model updates | 541,000 |
| base Actor-Critic updates | 432,800 |
| extra rehearsal actor updates | 554,900 |
| total actor-bearing optimizer steps | 987,700 |
| starts per rehearsal update | 64 |
| imagined steps per rehearsal update | 960 |

Thus this baseline is storage matched but intentionally **not compute matched**
to plain DreamerV3 or ARROW-50. Reports must show base and extra compute rather
than folding them into one ambiguous step counter.

## Declared differences from the reference artifact

1. one globally bounded uniform reservoir replaces never-clear per-phase data;
2. Atari and this repository's DreamerV3 replace the artifact's MiniGrid NM512
   stack;
3. optimizer work due within collection is delayed to the next epoch boundary;
4. both implementations imagine from every posterior state in the artifact's
   4-sequence by 16-step batch layout, but the Atari observation/action and
   RSSM interfaces necessarily differ; and
5. the fixed 541-epoch Atari schedule, evaluation, preprocessing, and full
   action-space semantics remain those of the existing matched DV3 protocol.

These differences require the name `Bounded-Dream-Rehearsal-v1-Atari`. Do not
label its result as the paper's MiniGrid result.

## Launch

Inspect the resolved contract without creating files:

```bash
python scripts/run_bounded_dream_rehearsal_atari.py --seed 0 --dry-run
```

After committing and pushing a clean synchronized revision, run:

```bash
python scripts/run_bounded_dream_rehearsal_atari.py \
  --seed 0 \
  --cpu-threads 12 \
  --profile-stages \
  --output-dir /persistent/path/bounded_dream_rehearsal_original_s0
```

For a named smaller-memory ablation, specify a multiple of 512 transitions:

```bash
python scripts/run_bounded_dream_rehearsal_atari.py \
  --seed 0 \
  --replay-capacity-transitions 131072 \
  --output-dir /persistent/path/bounded_dream_rehearsal_m131072_s0
```

The launcher refuses actual training from a dirty, local-only, ahead, or behind
Git state. A real run writes the resolved config, launch manifest, runtime
replay accounting, rehearsal accounting, TensorBoard events, logs, and analysis
snapshots. Analysis snapshots are not resumable checkpoints.

## Evaluation matrix

At minimum compare against the same seeds, curriculum, environment decisions,
task durations, and evaluation checkpoints for:

1. DreamerV3/FIFO at 524,288 transitions;
2. ARROW-50 at 524,288 transitions; and
3. Bounded Dream Rehearsal at 524,288 transitions.

A bounded DV3/LTDM control is required before attributing a difference purely
to dream rehearsal rather than long-term retention. Preserve raw per-task
returns and compute final average performance and forgetting using
`docs/protocols/continual_evaluation_metrics_v1.md`.
