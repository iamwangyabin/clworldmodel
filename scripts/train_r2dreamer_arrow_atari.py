#!/usr/bin/env python3
"""Train native R2-Dreamer with ARROW trajectory retention on continual Atari.

This is intentionally separate from the vendored ARROW trainer. The agent,
optimizer, `B=16,T=64` geometry, and decoder-free R2 objective are native to
R2-Dreamer; only the FIFO/LTDM storage and mixed sampling remain from ARROW.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
for path in (PROJECT_SRC, ROOT, VENDORED_ATARI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import ale_py
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.wrappers import AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter

from clworldmodel.r2dreamer import R2DreamerAgent, R2DreamerConfig
from clworldmodel.replay import ArrowR2ReplayAdapter
from config import Config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train R2-Dreamer size12M with ARROW FIFO/LTDM replay"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=_positive_int, required=True)
    parser.add_argument("--epochs", type=_positive_int, required=True)
    parser.add_argument("--world-model-updates-per-epoch", type=_positive_int)
    parser.add_argument("--native-train-ratio", type=_positive_int)
    parser.add_argument(
        "--require-optimizer-step",
        action="store_true",
        help="Fail unless the final update is finite and at least one optimizer step succeeds.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--analysis-snapshot-dir", type=Path)
    parser.add_argument(
        "--launcher-created-log-dir",
        action="store_true",
        help="Allow the fresh directory pre-created by the provenance launcher only.",
    )
    return parser


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_arrow_config(path: Path, *, task_count: int, epochs: int, device: str) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = raw["esc"]["env_configs"]
    if task_count > len(tasks):
        raise ValueError(f"task_count={task_count} exceeds the configured {len(tasks)} tasks")
    raw["esc"]["env_configs"] = tasks[:task_count]
    raw["epochs"] = epochs
    for replay_config in raw["replay_buffers"]:
        replay_config["rb_device"] = device
    return Config.from_dict(raw)


def _make_vector_environment(
    env_fns: list[Any], *, action_repeat: int
) -> AsyncVectorEnv:
    def wrap(factory: Any) -> Any:
        def build() -> Any:
            # Each async worker needs the ALE namespace before `gym.make`.
            gym.register_envs(ale_py)
            return AtariPreprocessing(
                factory(),
                frame_skip=action_repeat,
                screen_size=64,
                grayscale_obs=False,
            )

        return build

    return AsyncVectorEnv(
        [wrap(factory) for factory in env_fns],
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )


@contextmanager
def _vector_environment(
    env_fns: list[Any], *, action_repeat: int
) -> Iterator[AsyncVectorEnv]:
    environment = _make_vector_environment(env_fns, action_repeat=action_repeat)
    try:
        yield environment
    finally:
        environment.close()


@dataclass(frozen=True)
class CollectedTrajectories:
    """One fixed ARROW trajectory block with native R2 transition labels."""

    actions: torch.Tensor
    observations: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    resets: torch.Tensor
    is_last: torch.Tensor
    stoch_states: torch.Tensor
    deter_states: torch.Tensor
    environment_decisions: int


def _pack_worker_streams_for_arrow(
    value: torch.Tensor,
    *,
    sequence_length: int,
    sequence_count: int,
) -> torch.Tensor:
    """Split each worker stream into ARROW's fixed `[T, N, ...]` trajectories."""
    source_steps, worker_count = value.shape[:2]
    if source_steps % sequence_length:
        raise ValueError(
            "worker trajectory length must be divisible by the ARROW replay sequence length"
        )
    chunks_per_worker = source_steps // sequence_length
    if worker_count * chunks_per_worker != sequence_count:
        raise ValueError(
            "collected worker streams do not match the configured ARROW replay sequence count"
        )
    packed = value.swapaxes(0, 1).reshape(
        worker_count, chunks_per_worker, sequence_length, *value.shape[2:]
    )
    return packed.reshape(sequence_count, sequence_length, *value.shape[2:]).swapaxes(0, 1)


@torch.no_grad()
def _collect_trajectories(
    agent: R2DreamerAgent,
    env_fns: list[Any],
    *,
    action_repeat: int,
    trajectory_steps: int,
    replay_sequence_length: int,
    replay_sequence_count: int,
    seed: int,
) -> CollectedTrajectories:
    """Collect ARROW-format transition blocks with R2's action alignment.

    The current action is stored with the current observation. The adapter uses
    `action[t] -> observation[t+1]`, exactly as native R2-Dreamer does after
    its one-step replay context shift. Gymnasium's explicit ``NEXT_STEP``
    autoreset is preserved: terminal observations are stored as `is_last`, and
    the following reset observation is stored as `is_first`.
    """
    n_sync = len(env_fns)
    if trajectory_steps % n_sync:
        raise ValueError("trajectory_steps must be divisible by the environment count")
    if trajectory_steps != replay_sequence_length * replay_sequence_count:
        raise ValueError("trajectory budget does not match the ARROW replay shape")
    steps_per_env = trajectory_steps // n_sync
    rng = np.random.default_rng(seed)
    with _vector_environment(env_fns, action_repeat=action_repeat) as environment:
        observations, _ = environment.reset(seed=rng.integers(0, 2**31, size=n_sync).tolist())
        is_first = np.ones(n_sync, dtype=bool)
        is_last = np.zeros(n_sync, dtype=bool)
        is_terminal = np.zeros(n_sync, dtype=bool)
        previous_rewards = np.zeros(n_sync, dtype=np.float32)
        policy_state = agent.initial_policy_state(n_sync)

        actions_history = []
        observations_history = []
        rewards_history = []
        continues_history = []
        resets_history = []
        last_history = []

        stoch_history = []
        deter_history = []
        environment_decisions = 0
        for _ in range(steps_per_env):
            actions, policy_state = agent.act(
                torch.from_numpy(observations).float().permute(0, 3, 1, 2) / 255.0,
                torch.from_numpy(is_first),
                policy_state,
                deterministic=False,
            )
            live = torch.from_numpy(~is_last).to(actions.device, dtype=actions.dtype)
            actions = actions * live.unsqueeze(-1)
            action_indices = actions.argmax(-1).cpu().numpy()

            actions_history.append(actions.cpu())
            observations_history.append(torch.from_numpy(observations).float().permute(0, 3, 1, 2) / 255.0)
            rewards_history.append(torch.from_numpy(previous_rewards).unsqueeze(-1))
            continues_history.append(torch.from_numpy((~is_terminal).astype(np.float32)).unsqueeze(-1))
            resets_history.append(torch.from_numpy(is_first.astype(np.float32)).unsqueeze(-1))
            last_history.append(torch.from_numpy(is_last).unsqueeze(-1))
            stoch_history.append(policy_state.stoch.cpu())
            deter_history.append(policy_state.deter.cpu())
            environment_decisions += int((~is_last).sum())

            observations, rewards, terminated, truncated, _ = environment.step(action_indices)
            next_is_first = is_last
            is_last = np.logical_or(terminated, truncated)
            is_terminal = terminated
            is_first = next_is_first
            previous_rewards = rewards.astype(np.float32)
    return CollectedTrajectories(
        actions=_pack_worker_streams_for_arrow(
            torch.stack(actions_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        observations=_pack_worker_streams_for_arrow(
            torch.stack(observations_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        rewards=_pack_worker_streams_for_arrow(
            torch.stack(rewards_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        continues=_pack_worker_streams_for_arrow(
            torch.stack(continues_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        resets=_pack_worker_streams_for_arrow(
            torch.stack(resets_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        is_last=_pack_worker_streams_for_arrow(
            torch.stack(last_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        stoch_states=_pack_worker_streams_for_arrow(
            torch.stack(stoch_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        deter_states=_pack_worker_streams_for_arrow(
            torch.stack(deter_history),
            sequence_length=replay_sequence_length,
            sequence_count=replay_sequence_count,
        ),
        environment_decisions=environment_decisions,
    )


@torch.no_grad()
def _evaluate_policy(
    agent: R2DreamerAgent,
    env_fns: list[Any],
    *,
    action_repeat: int,
    episode_count: int,
    seed: int,
    reward_scale: float,
) -> tuple[float, float, float, float]:
    """Evaluate frozen parameters with deterministic R2 policy modes only."""
    n_sync = len(env_fns)
    rng = np.random.default_rng(seed)
    with _vector_environment(env_fns, action_repeat=action_repeat) as environment:
        observations, _ = environment.reset(seed=rng.integers(0, 2**31, size=n_sync).tolist())
        is_first = np.ones(n_sync, dtype=bool)
        is_last = np.zeros(n_sync, dtype=bool)
        policy_state = agent.initial_policy_state(n_sync)
        returns = np.zeros(n_sync, dtype=np.float64)
        completed = []
        while len(completed) < episode_count:
            actions, policy_state = agent.act(
                torch.from_numpy(observations).float().permute(0, 3, 1, 2) / 255.0,
                torch.from_numpy(is_first),
                policy_state,
                deterministic=True,
            )
            actions = actions * torch.from_numpy(~is_last).to(actions.device, dtype=actions.dtype).unsqueeze(-1)
            observations, rewards, terminated, truncated, _ = environment.step(
                actions.argmax(-1).cpu().numpy()
            )
            returns += rewards
            next_is_last = np.logical_or(terminated, truncated)
            for index, done in enumerate(next_is_last):
                if done and len(completed) < episode_count:
                    completed.append(float(returns[index]))
                    returns[index] = 0.0
            is_first = is_last
            is_last = next_is_last
    scaled = np.asarray(completed, dtype=np.float64)
    raw = scaled / reward_scale
    return (
        float(np.mean(raw)),
        float(np.std(raw)),
        float(np.mean(scaled)),
        float(np.std(scaled)),
    )


def _snapshot(
    directory: Path,
    agent: R2DreamerAgent,
    *,
    epoch: int,
    update_count: int,
    reason: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"r2dreamer_{reason}_e{epoch:04d}.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(
        {
            "format": "r2dreamer-arrow-analysis-snapshot-v1",
            "epoch": epoch,
            "world_model_updates": update_count,
            "reason": reason,
            "state_dict": {key: value.detach().cpu() for key, value in agent.state_dict().items()},
        },
        temporary,
    )
    os.replace(temporary, target)


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    args = _parser().parse_args()
    if (args.world_model_updates_per_epoch is None) == (args.native_train_ratio is None):
        raise ValueError(
            "Provide exactly one of --world-model-updates-per-epoch or --native-train-ratio"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("R2-Dreamer ARROW runs require a CUDA accelerator")
    arrow_config = _load_arrow_config(
        args.config.resolve(),
        task_count=args.task_count,
        epochs=args.epochs,
        device=args.device,
    )
    r2_config = R2DreamerConfig(device=args.device)
    _seed_everything(arrow_config.seed)
    torch.set_float32_matmul_precision("high")

    log_dir = args.log_dir.resolve()
    if log_dir.exists():
        if not args.launcher_created_log_dir:
            raise FileExistsError(f"Refusing to overwrite existing run directory: {log_dir}")
        allowed_entries = {"launch.json", "train.log"}
        unexpected = sorted(
            path.name for path in log_dir.iterdir() if path.name not in allowed_entries
        )
        if unexpected:
            raise FileExistsError(
                "Refusing a launcher-created directory with existing run artifacts: "
                f"{log_dir} contains {unexpected}"
            )
    else:
        log_dir.mkdir(parents=True)
    _write_json(log_dir / "config.json", arrow_config.to_dict())
    _write_json(log_dir / "r2dreamer_config.json", r2_config.to_dict())

    agent = R2DreamerAgent(r2_config)
    _write_json(log_dir / "model_parameter_accounting.json", agent.parameter_accounting())
    replay = arrow_config.get_replay_buffer()
    replay_adapter = ArrowR2ReplayAdapter(replay, r2_config)
    _write_json(log_dir / "replay_storage_accounting.json", replay_adapter.storage_accounting())
    writer = SummaryWriter(log_dir=str(log_dir))
    analysis_directory = (
        args.analysis_snapshot_dir.resolve()
        if args.analysis_snapshot_dir is not None
        else None
    )
    schedule = arrow_config.get_env_schedule()
    update_count = 0
    successful_optimizer_steps = 0
    model_sample_remainder = 0
    total_env_decisions = 0
    task_boundary = arrow_config.esc.kwargs.get("swap_sched")
    latest_metrics: dict[str, float] = {}

    try:
        for epoch in range(arrow_config.epochs):
            epoch_started = time.perf_counter()
            current_env_fns = schedule.funcs()
            collect_started = time.perf_counter()
            trajectories = _collect_trajectories(
                agent,
                current_env_fns,
                action_repeat=arrow_config.env_repeat,
                trajectory_steps=arrow_config.n_sync * arrow_config.gen_seq_len,
                replay_sequence_length=arrow_config.data_t,
                replay_sequence_count=arrow_config.data_n,
                seed=arrow_config.seed + epoch,
            )
            replay_adapter.add(
                trajectories.actions,
                trajectories.observations,
                trajectories.rewards,
                trajectories.continues,
                trajectories.resets,
                trajectories.is_last,
                trajectories.stoch_states,
                trajectories.deter_states,
            )
            total_env_decisions += trajectories.environment_decisions
            _synchronize(args.device)
            collect_seconds = time.perf_counter() - collect_started

            update_started = time.perf_counter()
            metrics: dict[str, float] = {}
            if args.world_model_updates_per_epoch is not None:
                updates_this_epoch = args.world_model_updates_per_epoch
            else:
                model_sample_remainder += (
                    trajectories.environment_decisions * args.native_train_ratio
                )
                updates_this_epoch, model_sample_remainder = divmod(
                    model_sample_remainder,
                    r2_config.sample_count,
                )
            for _ in range(updates_this_epoch):
                sample = replay_adapter.sample()
                update = agent.update_batch(sample.batch)
                replay_adapter.update_latent_states(sample.reference, update)
                metrics = update.metrics
                successful_optimizer_steps += int(metrics["opt/optimizer_step"])
                update_count += 1
            latest_metrics = metrics
            _synchronize(args.device)
            update_seconds = time.perf_counter() - update_started
            for name, value in metrics.items():
                writer.add_scalar(name, value, update_count)
            writer.add_scalar("counter/environment_decisions", total_env_decisions, epoch)
            writer.add_scalar(
                "counter/raw_environment_frames",
                total_env_decisions * arrow_config.env_repeat,
                epoch,
            )

            evaluation_started = time.perf_counter()
            seen_tasks = min(args.task_count, epoch // task_boundary + 1)
            evaluations = []
            for task_index, task_env_fns in enumerate(schedule.eval_funcs()[:seen_tasks]):
                reward_scale = arrow_config.esc.env_configs[task_index].rew_scale
                mean_raw, std_raw, mean_scaled, std_scaled = _evaluate_policy(
                    agent,
                    task_env_fns,
                    action_repeat=arrow_config.env_repeat,
                    episode_count=10,
                    seed=arrow_config.seed + 10_000 + epoch * args.task_count + task_index,
                    reward_scale=reward_scale,
                )
                evaluation = {
                    "task_index": task_index,
                    "environment": arrow_config.esc.env_configs[task_index].name,
                    "reward_scale": reward_scale,
                    "mean_raw_return": mean_raw,
                    "std_raw_return": std_raw,
                    "mean_optimization_return": mean_scaled,
                    "std_optimization_return": std_scaled,
                }
                evaluations.append(evaluation)
                writer.add_scalar(f"eval/task_{task_index}/mean_raw_return", mean_raw, epoch)
                writer.add_scalar(f"eval/task_{task_index}/std_raw_return", std_raw, epoch)
                writer.add_scalar(
                    f"eval/task_{task_index}/mean_optimization_return", mean_scaled, epoch
                )
                writer.add_scalar(
                    f"eval/task_{task_index}/std_optimization_return", std_scaled, epoch
                )
                print(
                    f"Eval task={task_index} raw_mean={mean_raw:.6f} raw_std={std_raw:.6f} "
                    f"scaled_mean={mean_scaled:.6f} scaled_std={std_scaled:.6f}"
                )
            _synchronize(args.device)
            evaluation_seconds = time.perf_counter() - evaluation_started

            _append_jsonl(
                log_dir / "metrics.jsonl",
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "world_model_updates": update_count,
                    "successful_optimizer_steps": successful_optimizer_steps,
                    "world_model_updates_this_epoch": updates_this_epoch,
                    "environment_decisions": total_env_decisions,
                    "raw_environment_frames": total_env_decisions * arrow_config.env_repeat,
                    "native_train_ratio_sample_remainder": model_sample_remainder,
                    "latest_train_metrics": metrics,
                    "evaluations": evaluations,
                },
            )

            is_boundary = task_boundary is not None and (epoch + 1) % task_boundary == 0
            if analysis_directory is not None and (is_boundary or epoch == arrow_config.epochs - 1):
                _snapshot(
                    analysis_directory,
                    agent,
                    epoch=epoch,
                    update_count=update_count,
                    reason="task_boundary" if is_boundary else "final",
                )
            schedule.step()
            writer.flush()
            if args.profile_stages:
                total_seconds = time.perf_counter() - epoch_started
                print(
                    "[stage-time] "
                    f"epoch={epoch} collect={collect_seconds:.3f}s "
                    f"r2_update={update_seconds:.3f}s eval={evaluation_seconds:.3f}s "
                    f"total={total_seconds:.3f}s"
                )
    finally:
        writer.close()

    if args.require_optimizer_step:
        if successful_optimizer_steps < 1:
            raise RuntimeError("R2 smoke finished without a successful optimizer step")
        nonfinite_metrics = sorted(
            name for name, value in latest_metrics.items() if not math.isfinite(value)
        )
        if nonfinite_metrics:
            raise FloatingPointError(
                "R2 smoke ended with non-finite metrics: " + ", ".join(nonfinite_metrics)
            )

    print(
        "training_end "
        f"epochs={arrow_config.epochs} world_model_updates={update_count} "
        f"environment_decisions={total_env_decisions} "
        f"native_train_ratio_sample_remainder={model_sample_remainder}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
