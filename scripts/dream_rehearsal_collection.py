"""Single-env storage adapter shared by BOTH Dream Rehearsal memory arms.

Reset -> policy -> environment -> replay insertion semantics follow NM512
tools.simulate (MIT, copyright 2023 NM512; see its retained LICENSE). The
project-owned difference is streaming rows into the common retention store,
instead of keeping/saving a second unbounded episode cache. No learning,
reward transformation, exploration, or rehearsal math is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CollectionState:
    done: bool = True
    observation: dict | None = None
    agent_state: Any = None
    episode_length: int = 0
    episode_return: float = 0.0


class RetainedHistoryCollector:
    def __init__(self, env, history, tools, *, phase: int, num_actions: int):
        self.env, self.history, self.tools = env, history, tools
        self.phase, self.num_actions = phase, num_actions
        self.online = False

    def simulate(self, agent, envs, cache, directory, logger, *, limit=0, steps=0,
                 state=None, is_eval=False, episodes=0):
        """Compatible boundary for author random_prefill and chunk collection."""
        if is_eval or episodes:
            raise ValueError("Evaluation must never use the training history collector")
        if len(envs) != 1 or envs[0] is not self.env or cache is not self.history.episodes:
            raise ValueError("Collector must own exactly one environment and the shared history")
        if limit or type(steps) is not int or steps < 1:
            raise ValueError("History capacity belongs to the store; collection needs positive decisions")
        state = CollectionState() if state is None else state
        for _ in range(steps):
            reset = state.done
            if reset:
                state.observation = self.env.reset()()
                state.episode_length, state.episode_return = 0, 0.0
                initial = {k: self.tools.convert(v) for k, v in state.observation.items()}
                initial.update(
                    reward=np.array(0, np.float32), discount=np.array(1, np.float32),
                    action=np.zeros(self.num_actions, np.float32), logprob=np.array(0, np.float32),
                )
                self.history.begin_episode(self.env.id, initial, phase=self.phase, online=self.online)
            obs = {k: np.stack([v]) for k, v in state.observation.items() if "log_" not in k}
            action, state.agent_state = agent(obs, np.array([reset]), state.agent_state)
            if set(action) != {"action", "logprob"}:
                raise ValueError("Reference policy must return only action and logprob")
            action = {k: np.array(v[0].detach().cpu()) for k, v in action.items()}
            observation, reward, done, info = self.env.step(action)()
            transition = {k: self.tools.convert(v) for k, v in observation.items()}
            transition.update({k: self.tools.convert(v) for k, v in action.items()})
            transition["reward"] = self.tools.convert(reward)
            transition["discount"] = self.tools.convert(info.get("discount", 1 - float(done)))
            self.history.add_transition(self.env.id, transition)
            state.observation, state.done = observation, bool(done)
            state.episode_length += 1
            state.episode_return += float(reward)
            if done:
                accounting = self.history.accounting()
                logger.scalar("dataset_size", accounting["collected_transitions_retained"])
                logger.scalar("train_return", state.episode_return)
                logger.scalar("train_length", state.episode_length)
                logger.scalar("train_episodes", accounting["episodes_or_retained_segments"])
                logger.write(step=logger.step)
        return state
