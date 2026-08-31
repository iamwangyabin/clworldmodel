# ARROW CoinRun with CPU uint8 replay

## Status

This is a storage-optimized execution profile for the published CoinRun
ARROW-50 protocol. It is not a new retention method and it is not evidence of a
baseline reproduction until the full published schedule has completed with the
required seed set.

The behavioral base is the published original-order CoinRun configuration:

`third_party/arrow/Configs/CoinRun configs/CL-task configs/Original Order/CoinRun,CoinRun+NB,CoinRun+NB+RT,CoinRun+NB+RT+GA,CoinRun+NB+RT+GA+MA,CoinRun+NB+RT+GA+MA+CA-s0-arrow.json`

## Motivation

The released configuration places both replay buffers on CUDA and stores
normalized observations as float32. With `data_t=512` and 512 trajectory slots
in each of FIFO and LTDM, replay alone allocates approximately 24.04 GiB. That
does not leave room for the model, optimizer, sampled minibatches, or training
activations on a 24-GiB RTX 4090.

Replay placement is not part of ARROW retention or sampling. This profile keeps
the full published sample capacity while storing source pixels as uint8 in CPU
memory. Only the sampled minibatch is decoded to the original float32 `[0, 1]`
interface and transferred to the training device.

## Fixed protocol

Unless a separately named smoke or pilot says otherwise, retain all published
settings:

- six CoinRun variants in original order;
- 90 epochs per task and 541 epochs total;
- `data_t=512`, `data_n_max=512`, and 1,024 total trajectory slots;
- 512 FIFO slots and 512 LTDM slots for ARROW-50;
- 0.5/0.5 whole-minibatch FIFO/LTDM selection;
- the published collection, world-model update, Actor-Critic update, evaluation,
  action, reward, and preprocessing settings.

The storage-only overrides are:

```json
{
  "replay_observation_dtype": "uint8",
  "replay_buffers": [
    {"rb_type": "FifoReplay", "rb_device": "cpu"},
    {"rb_type": "LongTermReplay", "rb_device": "cpu"}
  ]
}
```

The published default remains float32 and existing JSON files remain unchanged.

## Formal-run reproducibility

Project formal runs also set:

```json
{"deterministic_runtime_seeding": true}
```

This seeds Python's replay-buffer selector with the configured run seed and
passes explicit seeds to every Procgen constructor. Training and evaluation use
separate deterministic streams; task `i` starts at `seed + i * 1,000,000` for
training and adds `1,000,000,000` for evaluation, modulo `2**31`. Repeated
constructors increment that task/stream seed by one. This changes neither the
Procgen distribution settings nor task identity available to the agent.

The released JSON files retain their original unseeded Procgen/Python-random
behavior by default. Therefore a project formal run records the deterministic
execution option as an explicit compatibility correction rather than silently
presenting it as byte-identical released-code execution.

At every evaluation checkpoint, `evaluation_returns.jsonl` stores all complete
raw episode returns with the epoch, gradient-update counter, task index, and
task name. TensorBoard means and standard deviations remain derived outputs;
they are not the only retained evaluation record.

## Byte accounting

For each sub-buffer, observations have shape `[512, 512, 3, 64, 64]`.

| Storage | FIFO + LTDM observations | Other replay tensors | Total replay |
| --- | ---: | ---: | ---: |
| released float32 CUDA | 24.000 GiB | 0.035 GiB | 24.035 GiB |
| this uint8 CPU profile | 6.000 GiB | 0.035 GiB | 6.035 GiB |

The regular world-model minibatch contains `32 * 16 = 512` observations, or
24 MiB after decoding to float32, before model activations.

## Parity requirement

CoinRun frames originate as uint8 pixels. Encoding normalized values with
`round(x * 255)` and decoding with `float(pixel) / 255` must reproduce the
float32 replay sample exactly for these source values. Focused tests cover:

- unchanged default float32 storage;
- exact FIFO values and overwrite order through wraparound;
- exact LTDM random-key retention and sampled values under matched NumPy RNG;
- one byte per stored observation coordinate; and
- rejection of invalid normalized observations;
- deterministic, disjoint training/evaluation Procgen seed streams; and
- preservation of hand-computed raw episode returns.

Every run must record storage dtype, storage device, sample capacity, allocated
bytes, and the fact that only sampled data moves to CUDA.
