# ARROW CPU-resident float32 replay campaign

## Decision

The remaining ARROW-50 Atari baseline seeds may run with both replay buffers on
CPU when the target accelerators have 24 GiB of VRAM. The execution profile is
selected explicitly with `--replay-device cpu` and is recorded as
`vendored-optimized-cpu-float32-replay`.

## Preserved invariants

- FIFO and LTDM each retain 512 trajectories of length 512.
- Observation storage remains float32.
- Action, reward, continuation, and reset storage remains float32.
- FIFO overwrite, LTDM retention, and their random-number streams are unchanged.
- Whole-minibatch FIFO/LTDM selection remains 0.5/0.5.
- Sample shapes, sampled values, world-model updates, actor-critic updates,
  interaction budgets, task schedule, and evaluation schedule are unchanged.
- Sampled CPU tensors are copied to CUDA before model computation through the
  existing replay interface.

## Declared deviation and resources

The published JSON stores replay on CUDA. This profile changes only that
persistent storage device and must not be described as bitwise identical to the
published execution environment. Each run allocates 25,813,843,968 bytes of
replay tensors on CPU. A four-process campaign therefore requires
103,255,375,872 bytes for replay before Python, environments, models, and cache.
The campaign host must expose at least 128 GiB of memory and record the actual
peak usage and runtime.

## Validation

Launcher tests require a generated resolved config with both replay devices set
to CPU, explicit float32 observation storage, unchanged byte/capacity accounting,
and a manifest stating that retention and sampling semantics are unchanged.
The first target-GPU run remains a monitored runtime validation; any OOM,
throughput regression, or value mismatch is preserved as experiment evidence.
The canonical launcher also prepends the project `src/` directory to
`PYTHONPATH` for every ARROW execution because the vendored trainer imports
project-owned runtime helpers on the baseline path; it does not rely on an
ambient editable installation.
