# ARROW-50 versus DV3-RS MiniGrid five-seed campaign

## Status and scope

This is a direct formal comparison on the Continual-Dreamer three-task
MiniGrid curriculum. It schedules **five ARROW-50 seeds and five DV3-RS
seeds**, for ten result-bearing runs. There is no additional smoke or pilot
stage. The two completed 384-interaction jobs remain engineering smokes only;
they are excluded from the formal aggregate.

The repository's existing ARROW-50 and DreamerV3/FIFO results are retained and
reused rather than rerun, but they use a six-task Atari protocol and therefore
are not direct MiniGrid controls. The ten new runs are needed because both
methods must share this MiniGrid environment, interaction, update, evaluation,
seed, replay-capacity, and replay-byte protocol.

Unit, config, replay, provenance, and target-host preflight checks must pass
before launch. Those checks are not extra experiments.

## Methods

### ARROW-50

- vendored DreamerV3 world model and actor-critic;
- 50% of replay capacity assigned to FIFO and 50% to LTDM;
- each update chooses a whole minibatch from FIFO or LTDM with probability 0.5;
- LTDM uses exact random-key top-k uniform retention.

### DV3-RS

- the identical vendored DreamerV3 world model and actor-critic;
- 100% of replay capacity assigned to exact random-key top-k reservoir replay;
- uniform sampling over retained fixed-length trajectories;
- no Plan2Explore.

DV3-RS is a DreamerV3 mechanism port of Continual-Dreamer, not a reproduction
of its DreamerV2 results. The inspected public source is
`https://github.com/skezle/continual-dreamer` at commit
`77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6`.

Both methods are task-agnostic: task boundaries are known only to orchestration
and evaluation and are never model, policy, or replay-sampler inputs.

## Formal run count

The predeclared seeds are:

1. `123456789`
2. `1337`
3. `31337`
4. `42`
5. `987654321`

| Method | Seeds | Formal runs |
| --- | ---: | ---: |
| ARROW-50 | 5 | 5 |
| DV3-RS | 5 | 5 |
| **Total** | | **10** |

No DV3-FIFO jobs are part of this campaign.

## Environment and interaction budget

The fixed task order is the order in the released Continual-Dreamer MiniGrid
code:

1. `MiniGrid-DoorKey-9x9-v0`
2. `MiniGrid-LavaCrossingS9N1-v0`
3. `MiniGrid-SimpleCrossingS9N1-v0`

Every task receives exactly 750,000 training interactions. A run therefore
contains 2,250,000 interactions and the ten-run campaign contains 22,500,000.
Evaluation interactions are isolated and never enter replay or training
counters.

The environment exposes agent-centred partial RGB without mission text, a
100-step episode limit, seven discrete actions, and no action repeat. The
legacy 56 x 56 observation is resized to 64 x 64 with OpenCV `INTER_AREA` for
the vendored DreamerV3 encoder. The modern MiniGrid package no longer registers
the legacy DoorKey-9x9 identifier, so the adapter constructs an actual
`DoorKeyEnv(size=9)` rather than substituting 8 x 8.

## Collection, prefill, and updates

The formal configs use:

- `n_sync = 4` and `gen_seq_len = 250`, giving 1,000 normal interactions per
  epoch;
- `data_t = 50` and `data_n = 20`, so every retained unit remains inside one
  worker stream and matches the released DreamerV2 dataset sequence length;
- one 10,000-interaction random prefill at the start of the run, matching the
  released Continual-Dreamer MiniGrid configuration;
- task-duration epochs `[741, 750, 750]`: the first epoch collects ten normal
  batches, so each task still receives exactly 750,000 interactions;
- 610 world-model updates after the prefill epoch and 61 after every later
  epoch;
- 49 actor-critic updates after every epoch.

This gives each run exactly 137,250 world-model updates and 109,809
actor-critic updates. ARROW-50 and DV3-RS receive identical collection and
update budgets. The 61/49 normal update schedule is the integer translation of
the vendored ARROW DreamerV3 update-to-interaction ratios; the 610-update first
epoch preserves that world-model ratio across the 10x prefill.

## Matched CPU replay

Persistent replay resides on CPU. Images are stored as uint8 and converted to
float32 only after sampled minibatches are copied to the selected GPU.

A run has 40,000 fixed-length slots of 50 transitions, for exactly 2,000,000
transitions:

| Method | FIFO slots | Reservoir slots | Total transitions |
| --- | ---: | ---: | ---: |
| ARROW-50 | 20,000 | 20,000 | 2,000,000 |
| DV3-RS | 0 | 40,000 | 2,000,000 |

Per run, tensor storage excluding allocator overhead is:

- uint8 observations: 24,576,000,000 bytes (22.89 GiB);
- float32 actions, rewards, continuation, and reset: 80,000,000 bytes
  (0.07 GiB);
- total: **24,656,000,000 bytes (22.96 GiB)**.

Four concurrent jobs therefore reserve approximately 91.85 GiB for replay.
The launcher performs a Linux RAM preflight and requires at least 8 GiB to
remain beyond the fixed replay tensor budget. It refuses rather than reducing
capacity. Replay is in-memory and is not checkpointed, so analysis snapshots
are explicitly non-resumable.

## Evaluation

Periodic evaluation occurs after the 10,000-interaction prefill and every
10,000 normal training interactions, with explicit task-boundary checkpoints.
Every checkpoint evaluates all three tasks, including unseen tasks needed for
forward-transfer reference points. Each task uses 16 rollouts from a fixed
validation seed cohort. Final evaluation uses a separate fixed held-out seed
cohort.

Evaluation uses deterministic action argmax and latent mode. Its transitions
never enter replay and do not affect training RNG streams. Preserve unscaled
per-task return means and standard deviations at every checkpoint. The final
aggregate reports final average performance, forgetting, forward transfer,
backward transfer where defined, and paired seed-wise ARROW-50 minus DV3-RS
uncertainty.

## `4090x4` schedule

Each process owns one GPU and CPU replay. Subject to a fresh resource check,
ten jobs require three waves:

| Wave | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | --- | --- | --- | --- |
| 1 | ARROW S0 | ARROW S1 | DV3-RS S0 | DV3-RS S1 |
| 2 | ARROW S2 | ARROW S3 | DV3-RS S2 | DV3-RS S3 |
| 3 | ARROW S4 | DV3-RS S4 | idle | idle |

Pairing methods within each wave reduces time-dependent host confounding. If a
GPU or sufficient RAM is unavailable, reduce concurrency instead of
colocating or shrinking replay.

## Failure and claim rules

A result becomes an official comparison only after all ten predeclared runs
complete and the aggregate is generated from preserved raw metrics. Failed
runs remain in the experiment record and may be repeated only under a declared
infrastructure-failure rule. Seeds are never selected or discarded based on
performance.
