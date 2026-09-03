"""MiniGrid adapter for the Continual-Dreamer three-task curriculum.

The paper implementation supplies only the agent-centred RGB view to the
learner and caps episodes at 100 decisions.  This adapter deliberately removes
the symbolic mission string and resizes the 56x56 partial rendering to the
64x64 image boundary used by the vendored ARROW world model.
"""

from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np


MINIGRID_PAPER_TASKS = (
    "MiniGrid-DoorKey-9x9-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
)


class _ResizeRgbObservation(gym.ObservationWrapper):
    """Expose a fixed 64x64 uint8 RGB observation."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim != 3 or observation.shape[-1] != 3:
            raise ValueError(
                "MiniGrid RGB observation must have shape [height, width, 3], "
                f"got {observation.shape}"
            )
        resized = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_AREA)
        return np.asarray(resized, dtype=np.uint8)


def make_minigrid_environment(
    name: str,
    *,
    max_episode_steps: int = 100,
    tile_size: int = 8,
    **kwargs: Any,
) -> gym.Env[Any, Any]:
    """Construct one paper-aligned, task-agnostic MiniGrid environment.

    ``kwargs`` are passed only to the underlying registered MiniGrid task.  In
    particular, no Atari-only construction options are accepted or discarded.
    """

    if name not in MINIGRID_PAPER_TASKS:
        raise ValueError(f"Unsupported Continual-Dreamer MiniGrid task: {name!r}")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")

    # Importing minigrid registers its environments with Gymnasium.
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    if name == "MiniGrid-DoorKey-9x9-v0":
        # MiniGrid 3.x removed this legacy registration while retaining the
        # parameterized environment class.  Construct size 9 explicitly; using
        # the still-registered 8x8 task would silently change the benchmark.
        from minigrid.envs import DoorKeyEnv

        env = DoorKeyEnv(size=9, **kwargs)
    else:
        env = gym.make(name, **kwargs)
    env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
    env = ImgObsWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = _ResizeRgbObservation(env)

    # The vendored collector normally applies AtariPreprocessing.  Mark this
    # environment as already normalized at the shared 64x64 visual boundary.
    setattr(env.unwrapped, "_clworldmodel_visual_preprocessed", True)
    return env
