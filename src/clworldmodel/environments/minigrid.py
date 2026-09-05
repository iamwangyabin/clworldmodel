"""MiniGrid adapter for the Continual-Dreamer three-task curriculum.

The released implementation supplies only the agent-centred RGB view to the
learner and caps episodes at 100 decisions.  This adapter deliberately removes
the symbolic mission string and resizes the 56x56 partial rendering to the
64x64 image boundary used by the vendored ARROW world model.

The public Continual-Dreamer source registers ``DoorKey-9x9`` without passing a
size to ``DoorKeyEnv``.  Its vendored ``DoorKeyEnv`` defaults to size 8.  The
two geometries are therefore named explicitly below instead of treating the
paper label and released executable behavior as if they were identical.
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

DOORKEY_GEOMETRY_SIZES = {
    "paper_label_9x9": 9,
    "released_source_8x8": 8,
}


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
    doorkey_geometry: str = "paper_label_9x9",
    **kwargs: Any,
) -> gym.Env[Any, Any]:
    """Construct one named, task-agnostic MiniGrid environment.

    ``kwargs`` are passed only to the underlying registered MiniGrid task.  In
    particular, no Atari-only construction options are accepted or discarded.
    ``doorkey_geometry`` has an effect only for the legacy DoorKey task name.
    """

    if name not in MINIGRID_PAPER_TASKS:
        raise ValueError(f"Unsupported Continual-Dreamer MiniGrid task: {name!r}")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if doorkey_geometry not in DOORKEY_GEOMETRY_SIZES:
        raise ValueError(
            "Unsupported DoorKey geometry: "
            f"{doorkey_geometry!r}; expected one of "
            f"{tuple(DOORKEY_GEOMETRY_SIZES)}"
        )

    # Importing minigrid registers its environments with Gymnasium.
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    if name == "MiniGrid-DoorKey-9x9-v0":
        # MiniGrid 3.x removed this legacy registration while retaining the
        # parameterized environment class.  The paper label says 9x9, whereas
        # the released source's registration actually instantiated the class's
        # 8x8 default.  Keep both interpretations explicitly named.
        from minigrid.envs import DoorKeyEnv

        env = DoorKeyEnv(size=DOORKEY_GEOMETRY_SIZES[doorkey_geometry], **kwargs)
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
