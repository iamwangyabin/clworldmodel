# ARROW-50 MiniGrid three-task execution smoke

`ARROW-50-MiniGrid-3Task-Smoke-v1` is an execution-correctness smoke test. It
does **not** reproduce the Continual-Dreamer paper and must not be reported as
evidence of learning, retention, forgetting, or transfer.

## Invariant under test

The existing task-agnostic ARROW-50 FIFO/LTDM composition can collect, retain,
sample, update its world model and Actor-Critic, cross two MiniGrid task
boundaries, and evaluate without assuming Atari's 18-action interface.

## Environments

The fixed order follows the authors' released Continual-Dreamer code:

1. `MiniGrid-DoorKey-9x9-v0`
2. `MiniGrid-LavaCrossingS9N1-v0`
3. `MiniGrid-SimpleCrossingS9N1-v0`

The model receives only the agent-centred RGB partial observation. Mission text
is removed. The native `56 x 56 x 3` `uint8` rendering is resized with OpenCV
`INTER_AREA` to the vendored ARROW model's `64 x 64 x 3` visual boundary.
Episodes are capped at 100 agent decisions, actions are the seven native
MiniGrid actions, and action repeat is one. No task identity is exposed to the
world model or policy. MiniGrid 3.x no longer registers the paper's legacy
`DoorKey-9x9` identifier, so the adapter constructs `DoorKeyEnv(size=9)`
explicitly rather than substituting the registered 8x8 environment.

## Smoke budgets

- one epoch and 128 collected transitions per task;
- two synchronized collection environments;
- six world-model updates total;
- six Actor-Critic updates total;
- ARROW-50 capacity and selection: eight FIFO plus eight LTDM trajectory slots,
  with equal whole-minibatch selection probability;
- 16 steps per trajectory and ARROW's default `float32` replay observations on
  CUDA;
- final evaluation is isolated from replay and runs 16 episodes per seen task.

The matched reservoir-only control is
`DV3-RS-MiniGrid-3Task-Smoke-v1`, documented in
`docs/protocols/dv3_rs_minigrid_smoke.md`. It keeps the same DreamerV3
backbone and all non-replay budgets while assigning all 16 trajectory slots to
the uniform long-term reservoir.

These deliberately tiny budgets establish execution only. A later pilot or
paper-aligned protocol must freeze its own interaction counts, model/update
budgets, reward/termination semantics, replay byte budget, evaluation cohorts,
and seed set before launch.

## Launch

After installing the `minigrid` optional dependencies in the pinned CUDA
environment:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_arrow_minigrid_smoke.py \
  --python /path/to/pinned/python \
  --seed 0 \
  --profile-stages
```

The launcher refuses real environment interaction unless the exact commit is
clean, pushed, and synchronized with its configured upstream.
