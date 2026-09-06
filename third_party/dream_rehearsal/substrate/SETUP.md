# Substrate setup (dreamerv3-torch)

Base: https://github.com/NM512/dreamerv3-torch (+ its requirements, plus `pip install minigrid`).

**1.** Copy `minigrid_env.py` → `envs/minigrid.py`.

**2.** In `dreamer.py`'s `make_env`, add a suite branch:

```python
    elif suite == "minigrid":
        import envs.minigrid as minigrid

        env = minigrid.MiniGrid(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
```

**3.** In `configs.yaml`, add:

```yaml
minigrid:
  task: minigrid_SimpleCrossingS9N1
  steps: 5e5
  action_repeat: 1
  envs: 1
  time_limit: 256
  train_ratio: 512
  video_pred_log: true
  encoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32, mlp_layers: 5, mlp_units: 512}
  decoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32, mlp_layers: 5, mlp_units: 512}
  actor: {layers: 5, dist: 'onehot', std: 'none'}
  value: {layers: 5}
  reward_head: {layers: 5}
  cont_head: {layers: 5}
  imag_gradient: 'reinforce'
```

**4.** Drop `../src/*` into the repo root. Orchestrators import `dreamer`, `tools`, and
`parallel` from there.
