from bisect import bisect_right
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.distributions as td
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing
from tqdm import tqdm

from ac import ActorCritic, zh_to_ac_state
from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT, WorldModel

if TYPE_CHECKING:
    from clworldmodel.routing import RoutedActorBank


@torch.no_grad()
def _routed_policy_step(wm, ac, router, obs, z, h, previous_action, reset, *, stochastic, dummy_previous_action=0):
    """Route each worker's RSSM and, when private, Actor by the same inferred ID.

    Candidate probes always start from zero state and a dummy no-op action. They
    use posterior mode and cannot advance the executed trajectory's state/RNG.
    No environment/task label is accepted by this boundary.
    """
    from clworldmodel.routing import EpisodeReconstructionRouter, RoutedActorBank

    if not isinstance(router, EpisodeReconstructionRouter):
        raise TypeError("Expected an episode reconstruction router")
    if isinstance(ac, RoutedActorBank) and ac.route_ids != router.eligible_route_ids:
        raise ValueError("Private actor and world-model eligibility registries must match")

    def reconstruct(route_id, frames):
        probe_z, probe_h = wm.rssm.initial_state(len(frames))
        probe_action = torch.zeros(len(frames), previous_action.shape[-1], device=z.device)
        probe_action[:, dummy_previous_action] = 1
        _, probe_z, probe_h = wm.rssm(
            probe_z, probe_action, probe_h, frames,
            torch.ones(len(frames), 1, device=z.device),
            task_id=route_id, stochastic=False,
        )
        return wm.decoder_for(route_id)(zh_to_ac_state(probe_z, probe_h))

    with _autocast_context(z.device, getattr(wm, "compute_dtype", "float32")):
        route_ids = router.route(obs, reset.bool().reshape(-1), reconstruct)
        previous_action = previous_action.clone()
        restarting = reset.bool().reshape(-1)
        previous_action[restarting] = 0
        previous_action[restarting, dummy_previous_action] = 1
        next_z, next_h = None, None
        for route_id in route_ids.unique(sorted=True).tolist():
            rows = torch.where(route_ids == route_id)[0]
            _, route_z, route_h = wm.rssm(
                z[rows], previous_action[rows], h[rows], obs[rows], reset[rows],
                task_id=route_id, stochastic=stochastic,
            )
            if next_z is None:
                next_z = route_z.new_empty((len(obs), *route_z.shape[1:]))
                next_h = route_h.new_empty((len(obs), *route_h.shape[1:]))
            next_z[rows] = route_z
            next_h[rows] = route_h
        features = zh_to_ac_state(next_z, next_h)
        logits = (ac(features, route_ids) if isinstance(ac, RoutedActorBank)
                  else ac.actor(features)).float()
    action = td.Categorical(logits=logits).sample() if stochastic else logits.argmax(-1)
    return next_z, next_h, action


@torch.no_grad()
def _evaluate_routed_exact(
    n_sync, wm, ac, env_fns, env_repeat, n_rollouts, seed,
    eligible_route_ids, max_agent_decisions_per_episode, diagnostics,
):
    """Evaluate exactly N independently seeded episodes, without Replay.

    This is opt-in for the new named protocol. The legacy trajectory-derived
    evaluator remains unchanged. Homogeneous per-task factories are required.
    A safety-cap hit fails instead of substituting an unfinished episode return.
    """
    from clworldmodel.routing import EpisodeReconstructionRouter

    if wm is None or ac is None or seed is None:
        raise ValueError("Exact routed evaluation requires a model, routed policy and fixed seed")
    if n_sync < 1 or n_rollouts < 1 or max_agent_decisions_per_episode < 1:
        raise ValueError("Evaluation worker, episode and safety-cap counts must be positive")
    if env_fns is None or len(env_fns) != n_sync:
        raise ValueError("Exact evaluation requires one homogeneous factory per worker")
    router = EpisodeReconstructionRouter(eligible_route_ids)
    action_count, dummy_action = _collection_action_spec(env_fns, wm)
    reset_seeds, action_seeds = _environment_worker_seeds(seed, n_rollouts)
    workers = min(n_sync, n_rollouts)
    envs, observations = [], []
    episode_ids = list(range(workers))
    lengths = [0] * workers
    returns = [0.] * workers
    records = []
    next_episode = workers
    model_modes = [(module, module.training) for root in (wm, ac) for module in root.modules()]
    try:
        wm.eval()
        ac.eval()
        for worker in range(workers):
            env = _make_collection_env(env_fns[worker], env_repeat, action_seeds[worker])
            envs.append(env)
            observations.append(env.reset(seed=reset_seeds[worker])[0])
        z, h = wm.rssm.initial_state(workers)
        action_count = wm.a_dim
        previous_action = torch.zeros(workers, action_count, device=z.device)
        previous_action[:, dummy_action] = 1
        reset = torch.ones(workers, 1, device=z.device)
        while len(records) < n_rollouts:
            obs = torch.from_numpy(np.stack(observations)).to(z.device).float().permute(0, 3, 1, 2) / 255
            z, h, action = _routed_policy_step(
                wm, ac, router, obs, z, h, previous_action, reset, stochastic=False,
                dummy_previous_action=dummy_action,
            )
            previous_action = torch.nn.functional.one_hot(action, action_count)
            reset.zero_()
            for worker, act in enumerate(action.cpu().tolist()):
                episode = episode_ids[worker]
                if episode is None:
                    continue  # Never step an extra episode to fill a vector batch.
                obs, reward, terminated, truncated, _ = envs[worker].step(act)
                observations[worker] = obs
                if not np.isfinite(float(reward)):
                    raise FloatingPointError("Exact evaluation received a non-finite reward")
                returns[worker] += float(reward)
                lengths[worker] += 1
                if terminated or truncated:
                    event = next(e for e in reversed(router.events) if e["worker_index"] == worker)
                    records.append({
                        "episode_index": episode, "reset_seed": reset_seeds[episode],
                        "action_seed": action_seeds[episode], "scaled_return": returns[worker],
                        "agent_decisions": lengths[worker], "terminated": bool(terminated),
                        "truncated": bool(truncated), "routing": dict(event),
                    })
                    if next_episode < n_rollouts:
                        episode_ids[worker] = next_episode
                        envs[worker].action_space.seed(action_seeds[next_episode])
                        observations[worker] = envs[worker].reset(seed=reset_seeds[next_episode])[0]
                        returns[worker], lengths[worker] = 0., 0
                        reset[worker] = 1
                        next_episode += 1
                    else:
                        episode_ids[worker] = None
                elif lengths[worker] >= max_agent_decisions_per_episode:
                    raise RuntimeError(
                        f"Exact evaluation episode {episode} exceeded the decision safety cap; "
                        "no partial return is accepted"
                    )
        records.sort(key=lambda row: row["episode_index"])
        if diagnostics is not None:
            diagnostics.update({
                "episode_count_mode": "exact", "completed_episodes": len(records),
                "eligible_route_ids": list(eligible_route_ids), "episodes": records,
                "routing_events": [row["routing"] for row in records],
                "agent_decisions": sum(row["agent_decisions"] for row in records),
            })
        values = [row["scaled_return"] for row in records]
        return float(np.mean(values)), float(np.std(values))
    finally:
        for module, training in model_modes:
            module.training = training
        for env in envs:
            env.close()


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


def _make_atari_env(
    env_fn: Callable[[], Any], env_repeat: int, action_seed: Optional[int]
) -> Any:
    env = AtariPreprocessing(
        env_fn(), frame_skip=env_repeat, screen_size=64, grayscale_obs=False
    )
    if action_seed is not None:
        env.action_space.seed(action_seed)
    return env


def _make_collection_env(env_fn, env_repeat, action_seed):
    from clworldmodel.environments import PreparedEnvironmentFactory
    if isinstance(env_fn, PreparedEnvironmentFactory):
        return env_fn.prepare(env_repeat, action_seed)
    return _make_atari_env(env_fn, env_repeat, action_seed)


def _collection_action_spec(env_fns, wm=None):
    from clworldmodel.environments import PreparedEnvironmentFactory
    specs = [(f.action_count, f.dummy_previous_action)
             if isinstance(f, PreparedEnvironmentFactory) else (18, 0)
             for f in env_fns]
    if not specs or any(spec != specs[0] for spec in specs):
        raise ValueError("Collection workers must have matching action semantics")
    count, dummy = specs[0]
    if count < 1 or not 0 <= dummy < count:
        raise ValueError("Invalid action count or dummy previous action")
    if wm is not None and hasattr(wm, "a_dim") and wm.a_dim != count:
        raise ValueError("World-model action size does not match environment adapter")
    return count, dummy


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
    ac: Optional[Union[ActorCritic, "RoutedActorBank"]] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    n_rollouts: int = 10,
    seed: Optional[int] = None,
    task_id: Optional[int] = None,
    deterministic_policy: bool = False,
    eligible_route_ids: Optional[tuple[int, ...]] = None,
    max_agent_decisions_per_episode: int = 32768,
    diagnostics: Optional[dict[str, Any]] = None,
) -> tuple[float, float]:
    if eligible_route_ids is not None:
        if task_id is not None or not deterministic_policy:
            raise ValueError("Auto-routed evaluation forbids oracle task IDs and requires deterministic policy")
        return _evaluate_routed_exact(
            n_sync, wm, ac, env_fns, env_repeat, n_rollouts, seed,
            eligible_route_ids, max_agent_decisions_per_episode, diagnostics,
        )
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
    ac: Optional[Union[ActorCritic, "RoutedActorBank"]] = None,
    env_fns: Optional[list[Callable[[], Any]]] = None,
    env_repeat: int = 4,
    target_terminals: Optional[int] = None,
    no_images: bool = False,
    seed: Optional[int] = None,
    task_id: Optional[int] = None,
    deterministic_policy: bool = False,
    eligible_route_ids: Optional[tuple[int, ...]] = None,
    routing_diagnostics: Optional[dict[str, Any]] = None,
) -> tuple[ActionT, Optional[ImageT], RewardT, ContT, ResetT]:
    # Returns [ X ... ] packed as [ N*T ... ] (sort of)
    # To change to [ T N ... ], do reshape and swapaxes
    # `target_terminals` if not None, forces at least some number of environment resets
    # (not including initial resets)

    router = None
    if eligible_route_ids is not None:
        from clworldmodel.routing import EpisodeReconstructionRouter

        if task_id is not None:
            raise ValueError("Auto-routed collection forbids oracle task IDs")
        router = EpisodeReconstructionRouter(eligible_route_ids)
    if ac is not None and task_id is not None:
        ac.set_task_route(task_id)

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
    action_count, dummy_action = _collection_action_spec(source_env_fns, wm)
    # Gymnasium 1.1 defaults to NEXT_STEP. Only the new protocol explicitly
    # requests SAME_STEP, so the router sees a reset frame, not a terminal frame.
    vector_kwargs = {"autoreset_mode": "SameStep"} if router is not None else {}
    env = AsyncVectorEnv(
        [
            partial(_make_collection_env, env_fn, env_repeat, action_seed)
            for env_fn, action_seed in zip(source_env_fns, action_seeds)
        ],
        **vector_kwargs,
    )
    z = None

    if wm is not None and ac is not None:
        post = " (+WM/AC)"
    else:
        post = ""
    try:
        with tqdm(total=n, desc=f"Generating trajectories{post}",disable = True) as progbar:
            while n_samples < n:
                _n_samples = n_samples
                if target_terminals is not None and n_terminals >= target_terminals:
                    break
                if n_samples == 0:  # First step
                    n_samples += n_sync
                    obs, _ = env.reset(seed=reset_seeds)
                    for i in range(n_sync):
                        acts[i].append(dummy_action)
                        obss[i].append(obs[i])
                        rews[i].append(0)
                        conts[i].append(True)
                        resets[i].append(True)
                    reset = np.zeros(n_sync, dtype=bool)
                    continue

                n_samples += n_sync
                if wm is None or ac is None:
                    act = np.random.randint(0, action_count, size=n_sync)
                elif router is not None:
                    if z is None:
                        z, h = wm.rssm.initial_state(n_sync)
                        act_t = torch.zeros(n_sync, action_count, device=z.device)
                        act_t[:, dummy_action] = 1
                    z, h, act = _routed_policy_step(
                        wm, ac, router,
                        torch.from_numpy(obs).float().permute(0, 3, 1, 2).to(z.device) / 255,
                        z, h, act_t,
                        torch.from_numpy(reset).float().unsqueeze(-1).to(z.device),
                        stochastic=not deterministic_policy, dummy_previous_action=dummy_action,
                    )
                    act_t = torch.nn.functional.one_hot(act, action_count)
                    act = act.cpu().numpy()
                else:
                    if z is None:
                        z, h = wm.rssm.initial_state(n_sync)
                        act_t = torch.zeros(n_sync, action_count, device=z.device)
                        act_t[:, dummy_action] = 1  # Previous move would have been all 0s
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
                    act_t = torch.nn.functional.one_hot(act, action_count)
                    act = act.cpu().numpy()

                obs, rew, term, trunc, _ = env.step(act)
                reset = term | trunc
                # When there is an episode termination (`reset`):
                # `obs` is of the new episode
                # `rew` is of the previous episode
                # Final observation information not used due to inconsistency with procgen
                # This loses a frame but this change is insignificant
                for i in range(n_sync):
                    # Reset observations have a dummy no-op previous action,
                    # matching the new router's zero-context posterior probes.
                    acts[i].append(dummy_action if router is not None and reset[i] else act[i])
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

    finally:
        env.close()
    if routing_diagnostics is not None and router is not None:
        routing_diagnostics.update({
            "environment_agent_decisions": max(0, n_samples - n_sync),
            "random_policy": wm is None or ac is None,
            "eligible_route_ids": list(eligible_route_ids),
            "routing_events": router.events,
        })
    acts = [np.stack(e) for e in acts]
    obss = [np.stack(e) for e in obss] if not no_images else None
    rews = [np.stack(e) for e in rews]
    conts = [np.stack(e) for e in conts]
    resets = [np.stack(e) for e in resets]

    return (
        torch.nn.functional.one_hot(torch.from_numpy(np.concatenate(acts)[:n]).long(), action_count).float(),
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
    return (
        acts.reshape(n, t, acts.shape[-1]).swapaxes(0, 1),
        obss.reshape(n, t, *obss.shape[1:]).swapaxes(0, 1),
        rews.reshape(n, t, 1).swapaxes(0, 1),
        conts.reshape(n, t, 1).swapaxes(0, 1),
        resets.reshape(n, t, 1).swapaxes(0, 1),
    )
