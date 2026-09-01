# Atari shared random-policy reference v1

## Purpose

This protocol measures the uniform-random lower reference for the frozen
six-task ARROW Atari environment. It is one environment/protocol reference
shared by ARROW, DreamerV3 controls, and project methods; it is not rerun or
renormalized separately for each method.

The reference performs no model inference, replay insertion, gradient update,
or GPU computation. Evaluation transitions remain isolated and are never used
for training.

## Frozen environment and metric

- tasks: MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, Enduro;
- ALE IDs: the corresponding `ALE/<game>-v5` environments;
- base ALE settings: `frameskip=1`, `repeat_action_probability=0`, and the
  full 18-action space;
- preprocessing: `AtariPreprocessing`, frame repeat 4, RGB 64 x 64;
- action policy: independent uniform samples from the 18 valid actions;
- output metric: raw, unscaled episode return;
- target evaluation count: 16 completed episodes per task, cohort, and seed;
- synchronization: four CPU environment workers.

The evaluator preserves the vendored ARROW episode-boundary extraction
semantics. Because vector workers can terminate on the same step, the result
records both the target and the actual number of complete episodes used.

## Seeds and reporting

Published ARROW seed indices map to `123456789`, `1337`, `31337`, `42`, and
`987654321`. Environment reset seeds use the same fixed validation and held-out
task streams as project continual runs. Random-policy action draws use separate
validation and held-out `SeedSequence` children so they cannot perturb or reuse
environment seed streams.

Keep every episode return. Report mean and population standard deviation for
each task/seed, followed by the median and interquartile range of the five seed
means. Fewer than five seeds are a pilot and not an official local reference.

## Launch

The exact code commit must be clean, pushed, and synchronized before any Atari
interaction:

```bash
CUDA_VISIBLE_DEVICES="" nice -n 10 \
  python scripts/evaluate_atari_random_policy.py \
  --output-dir /persistent/path/atari_random_reference_v1 \
  --seed-indices 0 1 2 3 4 \
  --cohorts validation \
  --rollouts 16 \
  --n-sync 4 \
  --cpu-threads 4
```

The output contains `launch.json`, all raw returns, aggregate summaries, an
atomic completion status, and a result checksum. Do not substitute this local
reference for published ARROW normalization constants without also declaring a
separate local metric schema and obtaining a compatible local single-task
upper reference.
