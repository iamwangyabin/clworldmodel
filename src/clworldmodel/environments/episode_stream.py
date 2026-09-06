"""Single-environment collection preserving incoming-action/terminal alignment."""

from __future__ import annotations

import numpy as np


class EpisodeStream:
    """Gymnasium stream without autoreset, reward clipping, or task metadata."""

    def __init__(self, env, seed: int):
        self.env = env
        env.action_space.seed(seed)
        self.seed = seed
        self.observation = None
        self.rows: list[dict] = []
        self.complete = True
        self.resets = 0

    def reset(self) -> dict:
        image, _ = self.env.reset(seed=self.seed if self.resets == 0 else None)
        self.resets += 1
        self.complete = False
        self.observation = dict(image=np.asarray(image).copy(), is_first=np.bool_(True), is_terminal=np.bool_(False))
        self.rows = [{**self.observation, "action": np.zeros(self.env.action_space.n, np.float32), "reward": np.float32(0), "is_last": np.bool_(False)}]
        return self.observation

    def step(self, action: int):
        if self.complete:
            raise RuntimeError("Reset the completed stream before taking another action")
        image, reward, terminated, truncated, _ = self.env.step(action)
        self.complete = bool(terminated or truncated)
        # The final observation is the env.step result, never the next reset.
        self.observation = dict(image=np.asarray(image).copy(), is_first=np.bool_(False), is_terminal=np.bool_(terminated))
        onehot = np.zeros(self.env.action_space.n, np.float32)
        onehot[action] = 1
        self.rows.append({**self.observation, "action": onehot, "reward": np.float32(reward), "is_last": np.bool_(self.complete)})
        episode = None
        if self.complete:
            episode = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
            self.rows = []
        return self.observation, episode, float(reward)

    def discard_partial(self) -> int:
        dropped = max(0, len(self.rows) - 1)
        self.rows = []
        self.complete = True
        return dropped
