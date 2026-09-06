"""Gymnasium/ALE -> NM512's old-Gym dict-observation boundary.

No model, policy, reward shaping, or replay selection is implemented here.
Agent decisions exclude reset no-ops; actual emulator frames include them.
"""

from __future__ import annotations

import gym
import gymnasium
import numpy as np


class FrameCounter(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_frames = 0

    def step(self, action):
        result = self.env.step(action)
        self.raw_frames += 1  # The underlying ALE is explicitly frameskip=1.
        return result


class AtariNM512(gym.Env):
    """HWC uint8 image, scalar reset/terminal flags, raw scalar rewards."""

    metadata = {}

    def __init__(self, env, frame_counter: FrameCounter, seed: int):
        self._env = env
        self._frame_counter = frame_counter
        self._seed = seed
        self._first_reset = True
        self.agent_decisions = 0
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._return = 0.0
        self._length = 0
        if env.action_space.n != 18:
            raise ValueError("The named Atari protocol requires all 18 actions")
        self.action_space = gym.spaces.Discrete(18)
        self.action_space.seed(seed)
        self.action_space.discrete = True
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, env.observation_space.shape, dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.uint8),
        })

    @property
    def raw_frames(self) -> int:
        return self._frame_counter.raw_frames

    @staticmethod
    def _observation(image, first=False, last=False, terminal=False):
        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("Expected unscaled HWC uint8 Atari pixels")
        return {"image": image, "is_first": first, "is_last": last, "is_terminal": terminal}

    def reset(self):
        seed = self._seed if self._first_reset else None
        image, _ = self._env.reset(seed=seed)
        self._first_reset = False
        self._return, self._length = 0.0, 0
        return self._observation(image, first=True)

    def step(self, action):
        image, reward, terminated, truncated, info = self._env.step(int(action))
        self.agent_decisions += 1
        self._return += float(reward)
        self._length += 1
        done = bool(terminated or truncated)
        if done:
            self.episode_returns.append(self._return)
            self.episode_lengths.append(self._length)
        # A time limit ends an episode, but is NOT an MDP terminal.
        info = dict(info)
        info["discount"] = np.float32(0.0 if terminated else 1.0)
        return (self._observation(image, last=done, terminal=bool(terminated)),
                np.float32(reward), done, info)

    def close(self):
        self._env.close()


def make_atari(task: str, config, seed: int) -> AtariNM512:
    import ale_py

    gymnasium.register_envs(ale_py)
    native = gymnasium.make(task, frameskip=1, full_action_space=True,
                           repeat_action_probability=config.sticky_probability,
                           max_num_frames_per_episode=config.episode_limit_raw_frames)
    native.action_space.seed(seed)
    counter = FrameCounter(native)
    visual = gymnasium.wrappers.AtariPreprocessing(
        counter, noop_max=config.noop_max, frame_skip=config.action_repeat,
        screen_size=config.image_size, terminal_on_life_loss=False,
        grayscale_obs=False, scale_obs=False,
    )
    limited = gymnasium.wrappers.TimeLimit(
        visual, max_episode_steps=config.episode_limit_raw_frames // config.action_repeat,
    )
    return AtariNM512(limited, counter, seed)
