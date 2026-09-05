from bisect import bisect_right
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.distributions as td
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm

from ac import ActorCritic, zh_to_ac_state
from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT, WorldModel


def _autocast_context(device: torch.device, compute_dtype: str):
    if compute_dtype == "float32":
        return nullcontext()
    from clworldmodel.precision import autocast_context

    return autocast_context(device, compute_dtype)


def _environment_worker_seeds(seed: int, n_sync: int) -> tuple[list[int], list[int]]:
    """Derive disjoint reset and action-space seeds for vector workers."""
    if seed < 0:
        raise ValueError("environment seed must be non-negative")
    if n_sync < 1:
        raise ValueError("n_sync must be positive")
    reset_root, action_root = np.random.SeedSequence(seed).spawn(2)

    def expand(root: np.random.SeedSequence) -> list[int]:
        return [
            int(worker.generate_state(1, dtype=np.uint32)[0])
            for worker in root.spawn(n_sync)
        ]

    return expand(reset_root), expand(action_root)


def _make_visual_env(
    env_fn: Callable[[], Any], env_repeat: int, action_seed: Optional[int]
) -> Any:
    env = env_fn()
    already_preprocessed = bool(
        getattr(env.unwrapped, "_clworldmodel_visual_preprocessed", False)
    )
    if already_preprocessed:
        if env_repeat != 1:
            raise ValueError("Preprocessed visual environments require env_repeat=1")
        if env.observation_space.shape != (64, 64, 3):
            raise ValueError(
                "Preprocessed visual environments must expose 64x64 RGB observations"
            )
    else:
        env = AtariPreprocessing(
            env, frame_skip=env_repeat, screen_size=64, grayscale_obs=False
        )
    if action_seed is not None:
        env.action_space.seed(action_seed)
    return env


class EnvironmentSchedule:
    def __init__(self, n_sync: int, templates: list[Callable[[], Any]]) -> None:
        self._step = 0
        self.templates = templates
        self.n_sync = n_sync

    def step(self) -> None:
        self._step += 1

    def funcs(self) -> list[Callable[[], Any]]:
        raise NotImplementedError

    def eval_funcs(self) -> list[list[Callable[[], Any]]]:
        return [[t for _ in range(self.n_sync)] for t in self.templates]

    def is_new_env(self) -> bool:
        raise NotImplementedError

    def current_task_index(self) -> int:
        raise NotImplementedError


class AllEnvironments(EnvironmentSchedule):
    def __init__(self, n_sync: int, templates: list[Callable[[], Any]]) -> None:
        super().__init__(n_sync, templates)

    def funcs(self) -> list[Callable[[], Any]]:
        res = []
        for _ in range(self.n_sync):
            res.append(np.random.choice(self.templates))
        return res

    def is_new_env(self) -> bool:
        return self._step == 0

    def current_task_index(self) -> int:
        raise ValueError("AllEnvironments does not expose one homogeneous task id")


class SequentialEnvironments(EnvironmentSchedule):
    def __init__(
        self,
        n_sync: int,
        templates: list[Callable[[], Any]],
        swap_sched: Optional[int] = None,
        task_durations: Optional[list[int]] = None,
    ) -> None:
        super().__init__(n_sync, templates)
        if task_durations is not None and swap_sched is not None:
            raise ValueError(
                "SequentialEnvironments accepts swap_sched or task_durations, not both"
            )
        if task_durations is None:
            if not isinstance(swap_sched, int) or swap_sched < 1:
                raise ValueError("swap_sched must be a positive integer")
            task_durations = [swap_sched] * len(templates)
        if len(task_durations) != len(templates):
            raise ValueError("task_durations must match the environment count")
        if any(
            not isinstance(duration, int) or duration < 1
            for duration in task_durations
        ):
            raise ValueError("task_durations must contain positive integers")

        self.swap_sched = swap_sched
        self.task_durations = tuple(task_durations)
        total = 0
        boundaries = []
        for duration in self.task_durations:
            total += duration
            boundaries.append(total)
        self._task_boundaries = tuple(boundaries)
        self._task_starts = frozenset((0, *self._task_boundaries[:-1]))
        self._schedule_duration = total

    def funcs(self) -> list[Callable[[], Any]]:
        i = self.current_task_index()
        return [self.templates[i] for _ in range(self.n_sync)]

    def is_new_env(self) -> bool:
        return self._step in self._task_starts

    def current_task_index(self) -> int:
        schedule_step = self._step % self._schedule_duration
        return bisect_right(self._task_boundaries, schedule_step)


class SyncVectorEnvAtHome:
    @staticmethod
    def resize64(obs):
        if len(obs) == 2:
            obs = obs[0]
        assert len(obs.shape) == 3, obs.shape
        return cv2.resize(obs, (64, 64))

    def __init__(self, create_fns, repeat: int = 1) -> None:
        self.create_fns = create_fns
        self.repeat = repeat
        self.envs = [f() for f in self.create_fns]

    def reset(self) -> np.ndarray:
        return np.stack([self.resize64(e.reset()) for e in self.envs])

    def step(self, act):
        w, x, y, z = [], [], [], []
        for a, e in zip(act, self.envs):
            # obs rew reset
            # _w, _x, _y, _z = e.step(a)
            _x_acc = 0  # reward accumulator
            for _ in range(self.repeat):
                try:
                    dat = e.step(a)
                except IndexError:
                    dat = e.step(a % e.action_space.n)
                if len(dat) == 5:
                    _w, _x, _y1, _y2, _z = dat
                    _y = _y1 or _y2
                else:
                    _w, _x, _y, _z = dat
                _x_acc += _x
                if _y:
                    _w = e.reset()
                    break
            w.append(SyncVectorEnvAtHome.resize64(_w))
            x.append(_x)
            y.append(_y)
            z.append(_z)
        return np.stack(w), np.stack(x), np.stack(y), np.stack(z)


@torch.no_grad()
def evaluate(
    n_sync: int,
    wm: Optional[WorldModel] = None,
    ac: Optional[ActorCritic] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    n_rollouts: int = 10,
    seed: Optional[int] = None,
    task_id: Optional[int] = None,
    deterministic_policy: bool = False,
    action_space: int = 18,
    autoreset_mode: str = "legacy_next_step",
) -> tuple[float, float]:
    _, _, rews, conts, resets = generate_trajectories(
        n_rollouts * 2**13 // n_sync,
        n_sync,
        wm,
        ac,
        env_fns,
        env_repeat,
        n_rollouts,
        no_images=True,
        seed=seed,
        task_id=task_id,
        deterministic_policy=deterministic_policy,
        action_space=action_space,
        autoreset_mode=autoreset_mode,
    )
    terms = torch.where(conts == 0)[0]
    starts = torch.where(resets == 1)[0]
    collection = [(t.item(), "E") for t in terms] + [(s.item(), "S") for s in starts]
    collection.sort()
    where_se = []
    for i in range(len(collection) - 1):
        (si, s), (ei, e) = collection[i : i + 2]
        if s == "S" and e == "E":
            where_se.append((si, ei))
    eps_rews = [rews[s : e + 1].sum().item() for s, e in where_se]
    if not eps_rews:
        n_eps = resets.sum().item()
        var = rews.var().item() * rews.numel() / n_eps**2
        return rews.sum().item() / n_eps, np.sqrt(var)
    return np.mean(eps_rews), np.std(eps_rews)


@torch.no_grad()
def generate_trajectories(
    n: int,
    n_sync: int,
    wm: Optional[WorldModel] = None,
    ac: Optional[ActorCritic] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    target_terminals: Optional[int] = None,
    no_images: bool = False,
    seed: Optional[int] = None,
    task_id: Optional[int] = None,
    deterministic_policy: bool = False,
    action_space: int = 18,
    autoreset_mode: str = "legacy_next_step",
    diagnostics: Optional[dict[str, Any]] = None,
) -> tuple[ActionT, Optional[ImageT], RewardT, ContT, ResetT]:
    # Returns [ X ... ] packed as [ N*T ... ] (sort of)
    # To change to [ T N ... ], do reshape and swapaxes
    # `target_terminals` if not None, forces at least some number of environment resets
    # (not including initial resets)

    class DummyList(list):
        def append(self, __object: Any) -> None:
            return

    acts = [[] for _ in range(n_sync)]  # [ N T ] int
    obss = [DummyList() if no_images else [] for _ in range(n_sync)]  # [ N T 64 64 3 ] uint8
    rews = [[] for _ in range(n_sync)]  # [ N T ] float
    conts = [[] for _ in range(n_sync)]  # [ N T ] bool
    # Important: should be 1 for new sequence
    resets = [[] for _ in range(n_sync)]  # [ N T ] bool
    n_samples = 0
    n_terminals = 0

    reset_seeds: Optional[list[int]] = None
    action_seeds: list[Optional[int]] = [None] * n_sync
    if seed is not None:
        reset_seeds, seeded_actions = _environment_worker_seeds(seed, n_sync)
        action_seeds = seeded_actions
    default_env_fn = partial(
        gym.make,
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=0,
    )
    source_env_fns = [
        env_fns[i] if env_fns is not None else default_env_fn
        for i in range(n_sync)
    ]
    if action_space < 1:
        raise ValueError("action_space must be positive")
    modes = {
        "legacy_next_step": AutoresetMode.NEXT_STEP,
        "same_step": AutoresetMode.SAME_STEP,
    }
    if autoreset_mode not in modes:
        raise ValueError(f"Unsupported collector autoreset mode: {autoreset_mode!r}")
    env = AsyncVectorEnv(
        [
            partial(_make_visual_env, env_fn, env_repeat, action_seed)
            for env_fn, action_seed in zip(source_env_fns, action_seeds)
        ],
        autoreset_mode=modes[autoreset_mode],
    )
    # Diagnostics do not sample RNGs or feed information back to the policy.
    episode_returns = np.zeros(n_sync, dtype=np.float64)
    completed_returns: list[float] = []
    actual_actions = 0
    positive_reward_events = 0
    raw_reward_sum = 0.0
    ignored_reset_actions = 0
    action_counts = np.zeros(action_space, dtype=np.int64)
    if (
        not isinstance(env.single_action_space, gym.spaces.Discrete)
        or env.single_action_space.n != action_space
    ):
        observed_action_space = env.single_action_space
        env.close()
        raise ValueError(
            "Configured categorical action dimension does not match the "
            f"environment: config={action_space}, environment={observed_action_space}"
        )
    z = None

    if wm is not None and ac is not None:
        post = " (+WM/AC)"
    else:
        post = ""
    with tqdm(total=n, desc=f"Generating trajectories{post}",disable = True) as progbar:
        while n_samples < n:
            _n_samples = n_samples
            if target_terminals is not None and n_terminals >= target_terminals:
                break
            if n_samples == 0:  # First step
                n_samples += n_sync
                obs, _ = env.reset(seed=reset_seeds)
                for i in range(n_sync):
                    acts[i].append(0)
                    obss[i].append(obs[i])
                    rews[i].append(0)
                    conts[i].append(True)
                    resets[i].append(True)
                reset = np.zeros(n_sync, dtype=bool)
                continue

            n_samples += n_sync
            if wm is None or ac is None:
                act = np.random.randint(0, action_space, size=n_sync)
            else:
                if z is None:
                    z, h = wm.rssm.initial_state(n_sync)
                    act_t = torch.zeros(n_sync, action_space, device=z.device)
                    act_t[:, 0] = 1  # Previous move would have been all 0s
                # Follow a stochastic policy
                rssm_kwargs = {}
                if task_id is not None:
                    rssm_kwargs["task_id"] = task_id
                if deterministic_policy:
                    rssm_kwargs["stochastic"] = False
                with _autocast_context(
                    z.device, getattr(wm, "compute_dtype", "float32")
                ):
                    _, z, h = wm.rssm(
                        z,
                        act_t,
                        h,
                        torch.from_numpy(obs / 255)
                        .float()
                        .permute(0, 3, 1, 2)
                        .to(z.device),
                        torch.from_numpy(reset).float().unsqueeze(-1).to(z.device),
                        **rssm_kwargs,
                    )
                    ac_state = zh_to_ac_state(z, h)
                    act_prob = ac.actor(ac_state).float()
                if deterministic_policy:
                    act = act_prob.argmax(dim=-1)
                else:
                    act_prob_dist = td.Categorical(logits=act_prob)
                    act = act_prob_dist.sample()
                act_t = torch.nn.functional.one_hot(act, action_space)
                act = act.cpu().numpy()

            executed = (
                ~reset
                if autoreset_mode == "legacy_next_step"
                else np.ones(n_sync, dtype=bool)
            )
            obs, rew, term, trunc, _ = env.step(act)
            reset = term | trunc
            if diagnostics is not None:
                actual_actions += int(executed.sum())
                ignored_reset_actions += int((~executed).sum())
                action_counts += np.bincount(act[executed], minlength=action_space)
                positive_reward_events += int((rew > 0).sum())
                raw_reward_sum += float(rew.sum())
                episode_returns += rew
                for worker in np.flatnonzero(reset):
                    completed_returns.append(float(episode_returns[worker]))
                    episode_returns[worker] = 0.0
            # When there is an episode termination (`reset`):
            # `obs` is of the new episode
            # `rew` is of the previous episode
            # Final observation information not used due to inconsistency with procgen
            # This loses a frame but this change is insignificant
            for i in range(n_sync):
                acts[i].append(act[i])
                obss[i].append(obs[i])
                rews[i].append(rew[i])
                conts[i].append(True)
                resets[i].append(reset[i])
                if reset[i]:
                    conts[i][-2] = False
                    rews[i][-2] = rew[i]
                    rews[i][-1] = 0
                    n_terminals += 1

            progbar.update(n_samples - _n_samples)

    env.close()
    if diagnostics is not None:
        diagnostics.update(
            autoreset_mode=autoreset_mode,
            stored_rows=n_samples,
            initial_reset_rows=n_sync,
            actual_environment_actions=actual_actions,
            ignored_reset_actions=ignored_reset_actions,
            positive_reward_events=positive_reward_events,
            observed_reward_sum=raw_reward_sum,
            completed_episodes=len(completed_returns),
            completed_positive_return_episodes=sum(value > 0 for value in completed_returns),
            completed_episode_return_sum=sum(completed_returns),
            completed_episode_returns=completed_returns,
            executed_action_counts=action_counts.tolist(),
        )

    acts = [np.stack(e) for e in acts]
    obss = [np.stack(e) for e in obss] if not no_images else None
    rews = [np.stack(e) for e in rews]
    conts = [np.stack(e) for e in conts]
    resets = [np.stack(e) for e in resets]

    return (
        torch.nn.functional.one_hot(
            torch.from_numpy(np.concatenate(acts)[:n]).long(), action_space
        ).float(),
        torch.from_numpy(np.concatenate(obss)[:n] / 255).float().permute(0, 3, 1, 2)
        if not no_images
        else None,
        torch.from_numpy(np.concatenate(rews)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(conts)[:n]).float().unsqueeze(-1),
        torch.from_numpy(np.concatenate(resets)[:n]).float().unsqueeze(-1),
    )


def reinterpret_nt_to_t_n(
    acts: ActionT, obss: ImageT, rews: RewardT, conts: ContT, resets: ResetT, t: int, n: int
) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
    if t * n != acts.shape[0]:
        raise ValueError(f"Illegal reinterpret (acts.shape={acts.shape}[0] != {t * n})")
    if acts.ndim != 2 or acts.shape[-1] < 1:
        raise ValueError(f"Actions must have shape [samples, categories], got {acts.shape}")
    return (
        acts.reshape(n, t, acts.shape[-1]).swapaxes(0, 1),
        obss.reshape(n, t, 3, 64, 64).swapaxes(0, 1),
        rews.reshape(n, t, 1).swapaxes(0, 1),
        conts.reshape(n, t, 1).swapaxes(0, 1),
        resets.reshape(n, t, 1).swapaxes(0, 1),
    )
