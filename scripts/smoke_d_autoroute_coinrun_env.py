#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit real-environment CoinRun adapter parity and collection smoke.

Run only from a clean, pushed, freshly verified commit. This is not a result
seed or shortened campaign; it checks simulator adapter, seeding and IPC.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Launcher-only vendor compatibility bridge, not a library import convention.
for path in (ROOT / "src", ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"):
    sys.path.insert(0, str(path))


def main():
    from git_provenance import require_synced_training_git_state
    provenance = require_synced_training_git_state(ROOT)
    import numpy as np
    import torch
    from procgen.gym_registration import make_env
    from clworldmodel.environments.coinrun import COINRUN_TASKS, CoinRunFactory
    from generate_trajectory import generate_trajectories
    records = []
    for index, name in enumerate(COINRUN_TASKS):
        factory = CoinRunFactory(name)
        env = factory.prepare(1, 100 + index)
        reference = make_env(env_name="coinrun", rand_seed=20260906 + index, **factory.options)
        try:
            obs, _ = env.reset(seed=20260906 + index)
            np.testing.assert_array_equal(obs, reference.reset())
            digest = hashlib.sha256(obs.tobytes())
            for decision in range(128):
                action = decision % factory.action_count
                obs, rew, done, trunc, _ = env.step(action)
                expected, expected_rew, expected_done, _ = reference.step(action)
                np.testing.assert_array_equal(obs, expected)
                if (rew, done, trunc) != (float(expected_rew), bool(expected_done), False):
                    raise RuntimeError("Adapter changed native reward/termination")
                digest.update(obs.tobytes())
                if done:
                    np.testing.assert_array_equal(env.reset()[0], reference.reset())
            records.append({"task": name, "matched_decisions": 128, "trace_sha256": digest.hexdigest()})
        finally:
            env.close()
            reference.close()
    np.random.seed(20260906)
    diagnostics = {}
    factory = CoinRunFactory(COINRUN_TASKS[0])
    acts, obs, _, _, resets = generate_trajectories(
        32, 2, env_fns=[factory]*2, env_repeat=1, seed=20260906,
        eligible_route_ids=(0,), routing_diagnostics=diagnostics,
    )
    if acts.shape != (32,15) or obs.shape != (32,3,64,64):
        raise RuntimeError("Production collector action/image shapes changed")
    if not torch.all(acts.argmax(-1)[resets[:,0].bool()] == 4):
        raise RuntimeError("Collector did not use the adapter's dummy reset action")
    print(json.dumps({"classification": "smoke", "project_git": provenance,
                      "seed": 20260906, "native_parity": records,
                      "async_collection": diagnostics,
                      "performance_claimed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
