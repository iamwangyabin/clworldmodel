# SPDX-License-Identifier: Apache-2.0
"""Seeded Procgen CoinRun adapter for the shared D-AutoRoute collector.

Native Procgen automatically returns the *next* episode's first observation on
termination. Gymnasium SameStep reset must consume that observation without
advancing/skipping an episode. Explicitly seeded resets (exact evaluation)
construct a new native environment, so episode seeds are order-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np

COINRUN_TASKS = (
    "CoinRun", "CoinRun+NB", "CoinRun+NB+RT", "CoinRun+NB+RT+GA",
    "CoinRun+NB+RT+GA+MA", "CoinRun+NB+RT+GA+MA+CA",
)
PROCGEN_COMMIT = "5e1dbf341d291eff40d1f9e0c0a0d5003643aebf"


def _native_environment(**kwargs):
    # Optional simulator dependency; not imported or constructed by config/dry-run.
    from procgen import ProcgenGym3Env
    return ProcgenGym3Env(**kwargs)


@dataclass(frozen=True)
class CoinRunFactory:
    name: str
    action_count: ClassVar[int] = 15
    dummy_previous_action: ClassVar[int] = 4  # Native Procgen combo () / no-op.

    def __post_init__(self):
        if self.name not in COINRUN_TASKS:
            raise ValueError(f"Unknown original-six CoinRun variant: {self.name!r}")

    @property
    def options(self) -> dict:
        parts = self.name.split("+")[1:]
        return {
            "use_backgrounds": "NB" not in parts,
            "restrict_themes": "RT" in parts,
            "use_generated_assets": "GA" in parts,
            "use_monochrome_assets": "MA" in parts,
            "center_agent": "CA" not in parts,
            "paint_vel_info": False,
            "distribution_mode": "hard",
            "num_levels": 0, "start_level": 0,
            "use_sequential_levels": False,
        }

    def __call__(self) -> CoinRunEnv:
        return CoinRunEnv(self)

    def prepare(self, env_repeat: int, action_seed: int | None) -> CoinRunEnv:
        if env_repeat != 1:
            raise ValueError("The CoinRun protocol requires env_repeat=1")
        env = self()
        if action_seed is not None:
            env.action_space.seed(action_seed)
        return env


class CoinRunEnv(gym.Env):
    """Gymnasium adapter: RGB [64,64,3] uint8; 15 native discrete actions.

    Rewards remain raw, done maps to terminated (native Procgen does not expose
    a separate truncation signal), and no terminal image is fabricated. As in
    the published collector, the terminal reward is attached to the preceding
    valid transition while the native autoreset image begins the next episode.
    """
    metadata = {"render_modes": []}

    def __init__(self, factory: CoinRunFactory):
        self.factory = factory
        self.observation_space = gym.spaces.Box(0, 255, (64,64,3), np.uint8)
        self.action_space = gym.spaces.Discrete(factory.action_count)
        self._native = None
        self._pending_reset = None

    def _observation(self, observations):
        rgb = np.asarray(observations["rgb"])
        if rgb.shape != (1,64,64,3) or rgb.dtype != np.uint8:
            raise ValueError(f"Unexpected Procgen RGB boundary: {rgb.shape}, {rgb.dtype}")
        return rgb[0].copy()  # C-owned storage may be reused by the next act().

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("CoinRun reset options are not part of this fixed protocol")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError("CoinRun reset seed must be a non-negative integer")
            super().reset(seed=int(seed))
            self.close()
            self._native = _native_environment(
                num=1, env_name="coinrun", num_threads=0,
                rand_seed=int(seed) % (2**31), **self.factory.options,
            )
            _, observations, first = self._native.observe()
            if not bool(first[0]):
                self.close()
                raise RuntimeError("New Procgen constructor did not begin an episode")
            return self._observation(observations), {}
        if self._pending_reset is not None:
            observation, self._pending_reset = self._pending_reset, None
            return observation, {}
        raise ValueError("CoinRun needs an explicit initial seed or a pending native autoreset")

    def step(self, action):
        if self._native is None:
            raise RuntimeError("Seeded reset is required before CoinRun step")
        if self._pending_reset is not None:
            raise RuntimeError("Consume the native autoreset with reset() before step")
        if not self.action_space.contains(action):
            raise ValueError(f"CoinRun action must lie in [0, {self.action_space.n}): {action!r}")
        self._native.act(np.array([action], dtype=np.int32))
        rewards, observations, first = self._native.observe()
        observation = self._observation(observations)
        reward = float(rewards[0])
        if not np.isfinite(reward):
            raise FloatingPointError("Procgen returned a non-finite raw reward")
        done = bool(first[0])
        if done:
            self._pending_reset = observation.copy()
        return observation, reward, done, False, {}

    def close(self):
        if self._native is not None:
            self._native.close()
            self._native = None
        self._pending_reset = None
