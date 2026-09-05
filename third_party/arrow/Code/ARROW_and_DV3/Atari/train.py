import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import socket
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange
import ale_py  # noqa: F401  # Registers ALE environments with Gymnasium.
import replay
from replay import MultiTypeReplay
from ac import (
    ActorCriticOpt,
    actor_policy_kl,
    build_actor_critic_opt,
    dream_rollout,
    train_bounded_dream_rehearsal,
    train_ac_from_wm,
    zh_to_ac_state,
)
from config import (
    Config,
    _arrow_fifo_ltdm_capacity_ns,
    _arrow_fifo_ltdm_sampling_weights,
)
from generate_trajectory import (
    SequentialEnvironments,
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)
from wm import WorldModel


class _NoOpWriter:
    """Keep non-primary ranks out of TensorBoard and filesystem side effects."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_scalars(self, *args, **kwargs) -> None:
        return None

    def add_images(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _autocast_context(device: torch.device, compute_dtype: str):
    if compute_dtype == "float32":
        return nullcontext()
    from clworldmodel.precision import autocast_context

    return autocast_context(device, compute_dtype)


def _require_cuda_compute_support(compute_dtype: str) -> None:
    if compute_dtype == "float32":
        return
    from clworldmodel.precision import require_cuda_compute_support

    require_cuda_compute_support(compute_dtype)


def _environment_seed_streams(
    seed: int,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    collection_seed, validation_seed, final_seed = np.random.SeedSequence(seed).spawn(3)
    return (
        np.random.default_rng(collection_seed),
        np.random.default_rng(validation_seed),
        np.random.default_rng(final_seed),
    )


def _next_environment_seed(seed_rng: np.random.Generator) -> int:
    return int(seed_rng.integers(0, 2**32, dtype=np.uint64))


def _adaptive_compression_task_seeds(
    seed: int, task_count: int
) -> tuple[int, ...]:
    """Derive the fixed pruning cohort without advancing training RNGs.

    Spawn indices 0--2 are already assigned to collection, periodic
    validation, and final held-out evaluation.  Constructing spawn index 3
    directly preserves those existing streams byte-for-byte for every older
    protocol while giving compression a separately declared seed domain.
    """

    if task_count < 1:
        raise ValueError("Adaptive compression requires at least one task seed")
    compression_rng = np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(3,))
    )
    return tuple(
        _next_environment_seed(compression_rng) for _ in range(task_count)
    )


def _adaptive_behavior_compression_task_seeds(
    seed: int, task_count: int
) -> tuple[int, ...]:
    """Derive fixed behavior-pruning seeds in an isolated RNG domain.

    Spawn index 4 is distinct from collection (0), periodic validation (1),
    held-out final evaluation (2), and Q/F/P pruning validation (3).
    """

    if task_count < 1:
        raise ValueError(
            "Adaptive behavior compression requires at least one task seed"
        )
    compression_rng = np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(4,))
    )
    return tuple(
        _next_environment_seed(compression_rng) for _ in range(task_count)
    )


@contextmanager
def _preserve_training_rng_state():
    """Keep stochastic evaluation from changing subsequent training draws."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _print_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[cuda-mem] {tag}: CUDA is not available.")
        return
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    peak_alloc = torch.cuda.max_memory_allocated(dev)
    peak_reserved = torch.cuda.max_memory_reserved(dev)
    # Reserved peak is usually the safer number for scheduler sizing.
    suggested = peak_reserved * 1.15
    print(
        f"[cuda-mem] {tag} | device={dev} ({torch.cuda.get_device_name(dev)}) "
        f"allocated={_bytes_to_gib(allocated):.2f} GiB "
        f"reserved={_bytes_to_gib(reserved):.2f} GiB "
        f"peak_allocated={_bytes_to_gib(peak_alloc):.2f} GiB "
        f"peak_reserved={_bytes_to_gib(peak_reserved):.2f} GiB "
        f"suggested_slurm_gpu_mem={_bytes_to_gib(suggested):.2f} GiB"
    )


def _print_replay_buffer_debug(config: Config, buf) -> None:
    """Log replay capacities and ARROW FIFO vs LTDM minibatch sampling weights."""
    print(
        f"[replay] algorithm={config.algorithm} data_t={config.data_t} "
        f"data_n_max={config.data_n_max}"
    )
    if isinstance(buf, MultiTypeReplay):
        total_slots = 2 * config.data_n_max
        n_fifo, n_ltdm = _arrow_fifo_ltdm_capacity_ns(
            total_slots, config.arrow_replay_capacity_ratio
        )
        w_fifo, w_ltdm = _arrow_fifo_ltdm_sampling_weights(
            config.arrow_replay_capacity_ratio
        )
        print(
            f"[replay] ARROW total_trajectory_slots={total_slots} "
            f"(2 * data_n_max), arrow_replay_capacity_ratio={config.arrow_replay_capacity_ratio}"
        )
        print(
            f"[replay] capacity split: FifoReplay n={n_fifo} ({n_fifo / total_slots:.4f}), "
            f"LongTermReplay n={n_ltdm} ({n_ltdm / total_slots:.4f})"
        )
        print(
            f"[replay] minibatch sampling weights (random.choices): "
            f"Fifo={w_fifo}, LTDM={w_ltdm} (sum={w_fifo + w_ltdm})"
        )
        for i, sub in enumerate(buf.replays):
            sw = buf.sampling_weights[i]
            nv = getattr(sub, "n_valid", None)
            print(
                f"[replay]   [{i}] {type(sub).__name__}: t={sub.t} n={sub.n} "
                f"n_valid={nv} sampling_weight={sw} "
                f"observation_storage={getattr(sub, 'observation_storage_path', None)}"
            )
    else:
        dv3_max = getattr(config, "sac_dv3_data_n_max", None)
        print(
            f"[replay] single buffer: {type(buf).__name__} t={buf.t} n={buf.n} "
            f"n_valid={buf.n_valid} (config.sac_dv3_data_n_max={dv3_max})"
        )


def _mapped_replay_storage_accounting(
    buf: replay.Replay,
) -> dict[str, object]:
    buffers = []
    observation_dtypes = set()
    sub_replays = buf.replays if isinstance(buf, MultiTypeReplay) else (buf,)
    for index, sub_replay in enumerate(sub_replays):
        storage_path = getattr(sub_replay, "observation_storage_path", None)
        if storage_path is None:
            continue
        stat = storage_path.stat()
        dtype = str(sub_replay.obss.dtype).removeprefix("torch.")
        observation_dtypes.add(dtype)
        buffers.append(
            {
                "index": index,
                "type": type(sub_replay).__name__,
                "path": str(storage_path),
                "dtype": dtype,
                "shape": list(sub_replay.obss.shape),
                "logical_storage_bytes": (
                    sub_replay.obss.numel() * sub_replay.obss.element_size()
                ),
                "allocated_file_bytes_at_accounting": stat.st_blocks * 512,
            }
        )
    if len(observation_dtypes) != 1:
        raise RuntimeError(
            "Mapped replay sub-buffers must use one observation dtype"
        )
    observation_dtype = next(iter(observation_dtypes))
    return {
        "schema_version": 2,
        "storage_backend": "file_mmap",
        "observation_dtype": observation_dtype,
        "sampled_dtype": "float32",
        "numeric_semantics": (
            "uint8 pixels are transferred before exact float32 division by 255"
            if observation_dtype == "uint8"
            else "float32 observations unchanged"
        ),
        "buffers": buffers,
    }


def _stage_clock(enabled: bool) -> float:
    if not enabled:
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _stage_elapsed(start: float, enabled: bool) -> float:
    if not enabled:
        return 0.0
    return _stage_clock(True) - start


def _autorouted_behavior(
    config: Config,
    aco: Optional[ActorCriticOpt],
    actor_critic_bank,
    eligible_task_count: Optional[int],
) -> torch.nn.Module:
    """Expose acquired policies only; never choose one from an environment label."""
    if eligible_task_count is None or not 1 <= eligible_task_count <= config.rssm_num_experts:
        raise ValueError("Declare acquired route count explicitly; never infer it from evaluation labels")
    if config.task_private_actor_critic:
        from clworldmodel.routing import RoutedActorBank

        if actor_critic_bank is None:
            raise ValueError("Private auto-routed behavior requires the acquired Actor-Critic bank")
        return RoutedActorBank({
            task: actor_critic_bank.get(task).ac.actor
            for task in range(eligible_task_count)
        })
    if aco is None or actor_critic_bank is not None:
        raise ValueError("Shared auto-routed behavior requires exactly one shared Actor-Critic")
    return aco.ac


def _evaluate_policy_tasks(
    config: Config,
    wm: WorldModel,
    aco: Optional[ActorCriticOpt],
    eval_funcs,
    task_seeds: Sequence[int],
    actor_critic_bank=None,
    distributed_context=None,
    eligible_task_count: Optional[int] = None,
    routing_diagnostics: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[float], list[float]]:
    if len(task_seeds) != len(eval_funcs):
        raise ValueError(
            "Evaluation task functions and fixed task seeds must have equal length"
        )
    if getattr(config, "uses_reconstruction_task_inference", False):
        from clworldmodel.routing import routing_audit

        if distributed_context is not None and distributed_context.enabled:
            raise ValueError("The named autoroute pilot is single-device")
        behavior = _autorouted_behavior(config, aco, actor_critic_bank, eligible_task_count)
        means, stds = [], []
        with _preserve_training_rng_state():
            for audit_task_id, (env_fns, task_seed) in enumerate(zip(eval_funcs, task_seeds)):
                diagnostic = {}
                mean, std = evaluate(
                    config.n_sync, wm=wm, ac=behavior, env_fns=env_fns,
                    env_repeat=config.env_repeat, n_rollouts=16, seed=task_seed,
                    deterministic_policy=True,
                    eligible_route_ids=tuple(range(eligible_task_count)),
                    max_agent_decisions_per_episode=config.evaluation_max_agent_decisions_per_episode,
                    diagnostics=diagnostic,
                )
                diagnostic["audit"] = routing_audit(
                    diagnostic["routing_events"], true_task_id=audit_task_id,
                    task_count=config.rssm_num_experts,
                )
                diagnostic["true_task_is_eligible"] = audit_task_id < eligible_task_count
                if routing_diagnostics is not None:
                    routing_diagnostics.append(diagnostic)
                means.append(mean)
                stds.append(std)
        return means, stds
    if distributed_context is not None and distributed_context.enabled:
        local_values = torch.zeros(
            len(eval_funcs), 2, dtype=torch.float64, device=distributed_context.device
        )
        local_present = torch.zeros_like(local_values)
        with _preserve_training_rng_state():
            for task_id, (env_fns, task_seed) in enumerate(
                zip(eval_funcs, task_seeds)
            ):
                if task_id % distributed_context.world_size != distributed_context.rank:
                    continue
                task_aco = (
                    actor_critic_bank.get_optional(task_id)
                    if actor_critic_bank is not None
                    else aco
                )
                evaluation_kwargs = {}
                if config.uses_task_experts:
                    evaluation_kwargs = {
                        "task_id": task_id,
                        "deterministic_policy": True,
                    }
                mean, std = evaluate(
                    config.n_sync,
                    wm=wm,
                    ac=task_aco.ac if task_aco is not None else None,
                    env_fns=env_fns,
                    env_repeat=config.env_repeat,
                    n_rollouts=16,
                    seed=task_seed,
                    **evaluation_kwargs,
                )
                local_values[task_id] = torch.tensor(
                    (mean, std), dtype=torch.float64, device=local_values.device
                )
                local_present[task_id] = 1
        combined = distributed_context.combine_sparse_task_values(
            local_values, local_present
        ).cpu()
        return combined[:, 0].tolist(), combined[:, 1].tolist()

    means = []
    stds = []
    with _preserve_training_rng_state():
        for task_id, (env_fns, task_seed) in enumerate(zip(eval_funcs, task_seeds)):
            task_aco = (
                actor_critic_bank.get_optional(task_id)
                if actor_critic_bank is not None
                else aco
            )
            evaluation_kwargs = {}
            if config.uses_task_experts:
                evaluation_kwargs = {
                    "task_id": task_id,
                    "deterministic_policy": True,
                }
            mean, std = evaluate(
                config.n_sync,
                wm=wm,
                ac=task_aco.ac if task_aco is not None else None,
                env_fns=env_fns,
                env_repeat=config.env_repeat,
                n_rollouts=16,
                seed=task_seed,
                **evaluation_kwargs,
            )
            means.append(mean)
            stds.append(std)
    return means, stds



def _write_routing_diagnostic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _actor_critic_schedule_values(
    config: Config, epoch: int
) -> tuple[float, float, int]:
    """Resolve actor hyperparameters from task age without using evaluation data."""
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    if config.ac_schedule == "constant":
        return config.ac_lr, config.ac_entropy_scale, epoch + 1
    _, task_epoch = _sequential_task_position(config, epoch)
    if config.ac_schedule != "task_cosine_decay":
        raise ValueError(f"Unknown actor-critic schedule: {config.ac_schedule!r}")

    start = config.ac_decay_start_task_epoch
    end = config.ac_decay_end_task_epoch
    progress = min(1.0, max(0.0, (task_epoch - start) / (end - start)))
    remaining = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = config.ac_final_lr + (config.ac_lr - config.ac_final_lr) * remaining
    entropy_scale = config.ac_final_entropy_scale + (
        config.ac_entropy_scale - config.ac_final_entropy_scale
    ) * remaining
    return learning_rate, entropy_scale, task_epoch


def _sequential_task_durations(config: Config) -> tuple[int, ...]:
    """Return validated per-task durations for a sequential schedule."""
    kwargs = config.esc.kwargs
    task_durations = kwargs.get("task_durations")
    swap_sched = kwargs.get("swap_sched")
    if task_durations is not None and swap_sched is not None:
        raise ValueError(
            "Sequential scheduling accepts swap_sched or task_durations, not both"
        )
    task_count = len(getattr(config.esc, "env_configs", ()))
    if task_durations is None:
        if not isinstance(swap_sched, int) or swap_sched < 1:
            raise ValueError("Sequential scheduling requires positive task durations")
        return (swap_sched,) * max(1, task_count)
    if not isinstance(task_durations, (list, tuple)):
        raise ValueError("task_durations must be a list of positive integers")
    durations = tuple(task_durations)
    if task_count and len(durations) != task_count:
        raise ValueError("task_durations must match the environment count")
    if not durations or any(
        not isinstance(duration, int) or duration < 1 for duration in durations
    ):
        raise ValueError("task_durations must contain positive integers")
    return durations


def _sequential_task_position(config: Config, epoch: int) -> tuple[int, int]:
    """Return the task index and one-based local epoch for a global epoch."""
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    durations = _sequential_task_durations(config)
    schedule_epoch = epoch % sum(durations)
    task_start = 0
    for task_index, duration in enumerate(durations):
        task_end = task_start + duration
        if schedule_epoch < task_end:
            return task_index, schedule_epoch - task_start + 1
        task_start = task_end
    raise AssertionError("Validated sequential schedule did not contain the epoch")


def _sequential_seen_task_count(config: Config, completed_epochs: int) -> int:
    if completed_epochs < 0:
        raise ValueError("Completed epochs must be non-negative")
    durations = _sequential_task_durations(config)
    if completed_epochs >= sum(durations):
        return len(durations)
    task_index, _ = _sequential_task_position(config, completed_epochs)
    return task_index + 1


def _raw_return_statistics(
    task_configs, scaled_means: list[float], scaled_stds: list[float]
) -> tuple[list[float], list[float]]:
    if not (len(task_configs) == len(scaled_means) == len(scaled_stds)):
        raise ValueError("Evaluation tasks and statistics must have matching lengths")
    raw_means = []
    raw_stds = []
    for task, scaled_mean, scaled_std in zip(
        task_configs, scaled_means, scaled_stds
    ):
        if task.rew_scale == 0:
            raise ValueError(f"Task {task.name!r} has zero reward scale")
        raw_means.append(float(scaled_mean / task.rew_scale))
        raw_stds.append(float(scaled_std / abs(task.rew_scale)))
    return raw_means, raw_stds


def _task_boundary_metadata(config: Config, epoch: int) -> Optional[dict]:
    if config.esc.env_schedule_type is not SequentialEnvironments:
        return None
    durations = _sequential_task_durations(config)
    completed_epochs = epoch + 1
    boundaries = np.cumsum(durations).tolist()
    if completed_epochs not in boundaries:
        return None

    task_index = boundaries.index(completed_epochs)
    boundary_index = task_index + 1
    task = config.esc.env_configs[task_index]
    return {
        "boundary_index": boundary_index,
        "task_index": task_index,
        "task_name": task.name,
        "task_reward_scale": task.rew_scale,
    }


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _optimizer_bank_state_dict(
    optimizers: Mapping[int, torch.optim.Optimizer],
) -> dict[str, dict[str, object]]:
    return {
        str(task_id): optimizer.state_dict()
        for task_id, optimizer in sorted(optimizers.items())
    }


def _snapshot_checkpoint_replay_mmaps(
    replay_state: dict[str, object],
    *,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Copy mutable mmap replay files into immutable checkpoint-owned assets."""

    snapshotted = copy.deepcopy(replay_state)
    replay_states = snapshotted.get("replays")
    if not isinstance(replay_states, list):
        replay_states = [snapshotted]
    shared_stem = checkpoint_path.stem.replace("_pre_consolidation", "").replace(
        "_post_consolidation", ""
    )
    asset_dir = checkpoint_path.parent / f"{shared_stem}_replay_assets"
    for index, sub_state in enumerate(replay_states):
        if not isinstance(sub_state, dict):
            raise ValueError("Replay checkpoint state must contain mappings")
        observations = sub_state.get("observations")
        if not isinstance(observations, dict) or observations.get("kind") != "mmap":
            continue
        source = Path(str(observations["path"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Replay mmap is missing: {source}")
        asset_dir.mkdir(parents=True, exist_ok=True)
        destination = asset_dir / f"{index}_{source.name}"
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        expected_bytes = int(observations["byte_size"])
        if destination.stat().st_size != expected_bytes:
            raise ValueError(
                "Checkpoint replay mmap byte size changed: "
                f"{destination.stat().st_size} != {expected_bytes}"
            )
        observations["path"] = str(destination)
        observations["sha256"] = _sha256(destination)
        observations["immutable_checkpoint_asset"] = True
    return snapshotted


def _actor_critic_opt_resumable_state_dict(aco: ActorCriticOpt) -> dict[str, Any]:
    """Serialize one persistent actor-critic including optimizer/target state."""

    return {
        "schema_version": 1,
        "actor_critic": aco.ac.state_dict(),
        "optimizer": aco.opt.state_dict(),
        "slow_critic": (
            None if aco.slow_critic is None else aco.slow_critic.state_dict()
        ),
        "return_scale_ema": aco.return_scale_ema,
        "return_mean_ema": aco.return_mean_ema,
    }


def _refresh_actor_critic_optimizer_parameters(aco: ActorCriticOpt) -> None:
    """Rebind one optimizer after a structured module replacement/load.

    Adaptive residual loading can rebuild compact modules from serialized width
    buffers.  The optimizer must therefore own the new ``Parameter`` objects,
    not the dense constructor objects that were replaced during ``load_state_dict``.
    """

    if len(aco.opt.param_groups) != 1:
        raise RuntimeError("Actor-Critic optimizer must own one parameter group")
    parameters = list(aco.ac.parameters())
    parameter_ids = {id(parameter) for parameter in parameters}
    for parameter in list(aco.opt.state):
        if id(parameter) not in parameter_ids:
            del aco.opt.state[parameter]
    aco.opt.param_groups[0]["params"] = parameters


def _load_actor_critic_opt_resumable_state_dict(
    aco: ActorCriticOpt, state: Mapping[str, Any]
) -> None:
    """Restore a shared actor-critic without replacing owned Parameter objects."""

    if state.get("schema_version") != 1:
        raise ValueError("Shared Actor-Critic state is not resumable schema v1")
    aco.ac.load_state_dict(state["actor_critic"], strict=True)
    _refresh_actor_critic_optimizer_parameters(aco)
    aco.opt.load_state_dict(state["optimizer"])
    slow_state = state.get("slow_critic")
    if (aco.slow_critic is None) != (slow_state is None):
        raise ValueError("Shared Actor-Critic slow-target topology changed on resume")
    if aco.slow_critic is not None:
        aco.slow_critic.load_state_dict(slow_state, strict=True)
    for name in ("return_scale_ema", "return_mean_ema"):
        value = state.get(name)
        setattr(aco, name, None if value is None else value.detach().clone())


def _save_evolving_resumable_checkpoint(
    path: Path,
    *,
    config: Config,
    wm: WorldModel,
    boundary_teacher: WorldModel,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizers: Mapping[int, torch.optim.Optimizer],
    route_optimizers: Mapping[int, torch.optim.Optimizer],
    actor_critic_bank,
    aco: Optional[ActorCriticOpt] = None,
    shared_behavior_update_rng: Optional[np.random.Generator] = None,
    shared_behavior_replay_updates: Optional[Mapping[int, int]] = None,
    replay_buffer,
    environment_schedule,
    epoch: int,
    current_task_id: int,
    world_model_updates: int,
    actor_critic_updates: int,
    adaptive_behavior_compression_updates: int = 0,
    total_env_steps: int,
    task_update_rng: np.random.Generator,
    collection_environment_seed_rng: np.random.Generator,
    validation_environment_seed_rng: np.random.Generator,
    final_environment_seed_rng: np.random.Generator,
) -> Path:
    """Atomically persist every state required for an equivalent resume."""

    if not config.uses_evolving_atomic_rssm:
        raise ValueError("Resumable evolving checkpoints require the named protocol")
    if current_task_id < 0:
        raise ValueError("current_task_id must be non-negative")
    path = path.expanduser().resolve()
    replay_state = _snapshot_checkpoint_replay_mmaps(
        replay_buffer.state_dict(), checkpoint_path=path
    )
    uses_shared_behavior = config.uses_replay_rehearsed_shared_behavior
    if uses_shared_behavior:
        if (
            actor_critic_bank is not None
            or aco is None
            or shared_behavior_update_rng is None
            or shared_behavior_replay_updates is None
        ):
            raise ValueError(
                "Shared replay-rehearsed behavior checkpoints require exactly one "
                "actor-critic and "
                "its independent route-schedule RNG plus routed update counters"
            )
    elif (
        actor_critic_bank is None
        or aco is not None
        or shared_behavior_update_rng is not None
        or shared_behavior_replay_updates is not None
    ):
        raise ValueError(
            "Private-behavior Evolving-Core checkpoints require only an actor bank"
        )
    behavior_optimizer_state = (
        {
            "shared_actor_critic": _actor_critic_opt_resumable_state_dict(aco),
        }
        if uses_shared_behavior
        else {"actor_critic_bank": actor_critic_bank.resumable_state_dict()}
    )
    rng_state = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "task_update": copy.deepcopy(task_update_rng.bit_generator.state),
        "collection_environment": copy.deepcopy(
            collection_environment_seed_rng.bit_generator.state
        ),
        "validation_environment": copy.deepcopy(
            validation_environment_seed_rng.bit_generator.state
        ),
        "final_environment": copy.deepcopy(
            final_environment_seed_rng.bit_generator.state
        ),
    }
    if uses_shared_behavior:
        rng_state["shared_behavior_update"] = copy.deepcopy(
            shared_behavior_update_rng.bit_generator.state
        )
    routed_behavior_updates: dict[str, int] | None = None
    if uses_shared_behavior:
        routed_behavior_updates = {
            str(int(task_id)): int(update_count)
            for task_id, update_count in sorted(
                shared_behavior_replay_updates.items()
            )
        }
        if any(
            int(task_id) < 0 or update_count < 0
            for task_id, update_count in routed_behavior_updates.items()
        ):
            raise ValueError(
                "Shared behavior routed update counters must be non-negative"
            )
        if sum(routed_behavior_updates.values()) != actor_critic_updates:
            raise ValueError(
                "Shared behavior routed update counters must sum to the total "
                "Actor-Critic optimizer updates"
            )
    payload = {
        "schema_version": 2 if uses_shared_behavior else 1,
        "artifact_kind": "evolving_core_atomic_rssm_resumable_checkpoint",
        "resumable": True,
        "config": config.to_dict(),
        "world_model": wm.state_dict(),
        "boundary_teacher": boundary_teacher.state_dict(),
        "optimizers": {
            "shared": shared_optimizer.state_dict(),
            "private_by_task": _optimizer_bank_state_dict(private_optimizers),
            "route_by_task": _optimizer_bank_state_dict(route_optimizers),
            **behavior_optimizer_state,
        },
        "replay": replay_state,
        "rng": rng_state,
        "schedule": {
            "environment_step": int(environment_schedule._step) + 1,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "current_task_id": current_task_id,
        },
        "counters": {
            "raw_environment_frames": total_env_steps,
            "world_model_updates": world_model_updates,
            "actor_critic_updates": actor_critic_updates,
            "adaptive_behavior_compression_updates": int(
                adaptive_behavior_compression_updates
            ),
            **(
                {
                    "actor_critic_updates_by_task_route": (
                        routed_behavior_updates
                    )
                }
                if uses_shared_behavior
                else {}
            ),
        },
        "replay_checkpoint_semantics": (
            "mapped observations are copied into immutable checkpoint-owned assets; "
            "all other replay tensors and retention indices are embedded"
        ),
    }
    if getattr(config, "uses_reconstruction_task_inference", False):
        payload["inference_routing"] = {
            "mode": config.task_route_inference,
            "eligible_route_ids": list(range(current_task_id + 1)),
            "episode_state_checkpointed": False,
            "resume_semantics": "boundary checkpoint; collection starts with fresh environment resets",
        }
    if getattr(config, "uses_adaptive_qfp_compression", False):
        world_model_layout = wm.rssm.adaptive_compression_layout()
        teacher_layout = boundary_teacher.rssm.adaptive_compression_layout()
        if world_model_layout != teacher_layout:
            raise ValueError(
                "Adaptive checkpoint teacher and student compression layouts differ"
            )
        payload["adaptive_compression"] = {
            "schema_version": 1,
            "world_model_hidden_widths": world_model_layout,
            "boundary_teacher_hidden_widths": teacher_layout,
            "private_optimizer_task_ids": sorted(
                int(task_id) for task_id in private_optimizers
            ),
            "route_optimizer_task_ids": sorted(
                int(task_id) for task_id in route_optimizers
            ),
            "topology_rebuild_source": (
                "persistent mechanism_hidden_features buffers in each Q/F/P bank"
            ),
            "full_dense_teacher_persistent": False,
        }
    if uses_shared_behavior:
        payload["shared_behavior"] = {
            "topology": (
                "single_shared_mlp_plus_task_adaptive_residuals"
                if getattr(config, "uses_adaptive_behavior_compression", False)
                else "single_shared_fastkan_actor_critic"
            ),
            "future_task_teacher_actor": aco.ac.actor.state_dict(),
            "teacher_seen_tasks": current_task_id + 1,
            "teacher_semantics": (
                "the just-completed shared actor becomes the single frozen "
                "cumulative policy-interface teacher for the next task"
            ),
        }
        if getattr(config, "uses_adaptive_behavior_compression", False):
            payload["shared_behavior"]["adaptive_hidden_widths"] = (
                aco.ac.adaptive_behavior_layout()
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    temporary_checksum = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    temporary_checksum.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    os.replace(temporary_checksum, checksum_path)
    return path


def _apply_evolving_checkpoint_retention(
    checkpoint_dir: Path,
    *,
    completed_task_id: int,
    retention: str,
) -> dict[str, object]:
    """Prune only older resumable boundaries after the new pair is durable."""

    if retention not in {"all_boundaries", "latest_boundary"}:
        raise ValueError(f"Unknown Evolving-Core checkpoint retention: {retention!r}")
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    current_paths = [
        checkpoint_dir / f"task_{completed_task_id:02d}_pre_consolidation.pt",
        checkpoint_dir / f"task_{completed_task_id:02d}_post_consolidation.pt",
    ]
    for path in current_paths:
        checksum = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not checksum.is_file():
            raise FileNotFoundError(
                "Checkpoint retention requires the complete current pre/post pair: "
                f"{path}"
            )

    removed: list[str] = []
    if retention == "latest_boundary":
        for old_task_id in range(completed_task_id):
            for phase in ("pre_consolidation", "post_consolidation"):
                path = checkpoint_dir / f"task_{old_task_id:02d}_{phase}.pt"
                checksum = path.with_suffix(path.suffix + ".sha256")
                for candidate in (path, checksum):
                    if candidate.exists():
                        candidate.unlink()
                        removed.append(str(candidate))
            asset_dir = checkpoint_dir / f"task_{old_task_id:02d}_replay_assets"
            if asset_dir.exists():
                shutil.rmtree(asset_dir)
                removed.append(str(asset_dir))

    artifact = {
        "schema_version": 1,
        "artifact_kind": "evolving_core_checkpoint_retention",
        "retention": retention,
        "completed_task_id": completed_task_id,
        "retained_checkpoint_pair": [str(path) for path in current_paths],
        "removed_older_artifacts": removed,
        "task_boundary_inference_snapshots_retained": True,
        "consolidation_records_retained": True,
    }
    artifact_path = checkpoint_dir / "retention.json"
    temporary = artifact_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, artifact_path)
    return artifact


def _restore_evolving_resumable_checkpoint(
    path: Path,
    *,
    config: Config,
    wm: WorldModel,
    boundary_teacher: WorldModel,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizers: Mapping[int, torch.optim.Optimizer],
    route_optimizers: Mapping[int, torch.optim.Optimizer],
    actor_critic_bank,
    actor_critic_factory,
    aco: Optional[ActorCriticOpt] = None,
    shared_behavior_update_rng: Optional[np.random.Generator] = None,
    replay_buffer,
    environment_schedule,
    task_update_rng: np.random.Generator,
    collection_environment_seed_rng: np.random.Generator,
    validation_environment_seed_rng: np.random.Generator,
    final_environment_seed_rng: np.random.Generator,
) -> dict[str, Any]:
    """Restore a preconstructed Evolving-Core training topology exactly."""

    path = path.expanduser().resolve()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Evolving-Core checkpoint checksum is missing: {checksum_path}"
        )
    checksum_fields = checksum_path.read_text(encoding="ascii").split()
    if not checksum_fields or checksum_fields[0] != _sha256(path):
        raise ValueError("Evolving-Core checkpoint checksum does not match")
    payload = torch.load(
        path,
        map_location=next(wm.parameters()).device,
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Evolving-Core checkpoint must contain a mapping")
    expected_schema = 2 if config.uses_replay_rehearsed_shared_behavior else 1
    if (
        payload.get("schema_version") != expected_schema
        or payload.get("artifact_kind")
        != "evolving_core_atomic_rssm_resumable_checkpoint"
        or payload.get("resumable") is not True
    ):
        raise ValueError(
            f"Checkpoint is not resumable Evolving-Core schema v{expected_schema}"
        )
    checkpoint_config = payload.get("config")
    if isinstance(checkpoint_config, Mapping) and not getattr(config, "uses_reconstruction_task_inference", False):
        # Historical D/other checkpoints predate these opt-in, default-off fields.
        checkpoint_config = dict(checkpoint_config)
        for name, default in (
            ("task_route_inference", "oracle"),
            ("evaluation_episode_count_mode", "legacy"),
            ("evaluation_max_agent_decisions_per_episode", 32768),
        ):
            if name in config.to_dict():
                checkpoint_config.setdefault(name, default)
    if checkpoint_config != config.to_dict():
        raise ValueError("Resolved config changed across Evolving-Core resume")
    if getattr(config, "uses_reconstruction_task_inference", False):
        routing = payload.get("inference_routing", {})
        completed_id = int(payload["schedule"]["current_task_id"])
        if (not 0 <= completed_id < config.rssm_num_experts
                or routing.get("eligible_route_ids") != list(range(completed_id + 1))
                or routing.get("mode") != config.task_route_inference):
            raise ValueError("Checkpoint inference eligibility does not match acquisition state")

    wm.load_state_dict(payload["world_model"], strict=True)
    boundary_teacher.load_state_dict(payload["boundary_teacher"], strict=True)
    if getattr(config, "uses_adaptive_qfp_compression", False):
        adaptive_state = payload.get("adaptive_compression")
        if not isinstance(adaptive_state, Mapping):
            raise ValueError(
                "Adaptive checkpoint is missing compression topology metadata"
            )
        expected_layout = adaptive_state.get("world_model_hidden_widths")
        if (
            wm.rssm.adaptive_compression_layout() != expected_layout
            or boundary_teacher.rssm.adaptive_compression_layout()
            != adaptive_state.get("boundary_teacher_hidden_widths")
        ):
            raise ValueError(
                "Adaptive checkpoint topology did not rebuild to recorded widths"
            )
    optimizers = payload["optimizers"]
    if getattr(config, "uses_adaptive_qfp_compression", False):
        for metadata_name, optimizer_name in (
            ("private_optimizer_task_ids", "private_by_task"),
            ("route_optimizer_task_ids", "route_by_task"),
        ):
            recorded_ids = adaptive_state.get(metadata_name)
            payload_ids = sorted(
                int(task_id) for task_id in optimizers[optimizer_name]
            )
            if recorded_ids != payload_ids:
                raise ValueError(
                    "Adaptive checkpoint optimizer ownership metadata is inconsistent"
                )
    shared_optimizer.load_state_dict(optimizers["shared"])
    for name, targets in (
        ("private_by_task", private_optimizers),
        ("route_by_task", route_optimizers),
    ):
        states = optimizers[name]
        if {int(task_id) for task_id in states} != set(targets):
            raise ValueError(f"Checkpoint {name} ownership does not match target")
        for task_id, optimizer in targets.items():
            optimizer.load_state_dict(states[str(task_id)])
    restored_teacher = None
    restored_teacher_seen_tasks = 0
    if config.uses_replay_rehearsed_shared_behavior:
        if (
            actor_critic_bank is not None
            or aco is None
            or shared_behavior_update_rng is None
        ):
            raise ValueError(
                "Shared replay-rehearsed behavior restore requires exactly one "
                "actor-critic and "
                "its independent route-schedule RNG"
            )
        _load_actor_critic_opt_resumable_state_dict(
            aco, optimizers["shared_actor_critic"]
        )
        shared_behavior = payload.get("shared_behavior")
        if not isinstance(shared_behavior, Mapping):
            raise ValueError("Shared behavior checkpoint is missing behavior state")
        if getattr(config, "uses_adaptive_behavior_compression", False) and (
            aco.ac.adaptive_behavior_layout()
            != shared_behavior.get("adaptive_hidden_widths")
        ):
            raise ValueError(
                "Adaptive Actor-Critic checkpoint did not rebuild recorded widths"
            )
        restored_teacher = copy.deepcopy(aco.ac.actor).eval()
        restored_teacher.requires_grad_(False)
        restored_teacher.load_state_dict(
            shared_behavior["future_task_teacher_actor"], strict=True
        )
        restored_teacher_seen_tasks = int(shared_behavior["teacher_seen_tasks"])
    else:
        if (
            actor_critic_bank is None
            or aco is not None
            or shared_behavior_update_rng is not None
        ):
            raise ValueError(
                "Private-behavior restore requires only an actor-critic bank"
            )
        actor_critic_bank.load_resumable_state_dict(
            optimizers["actor_critic_bank"], actor_critic_factory
        )
    replay_buffer.load_state_dict(payload["replay"])

    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy_legacy"])
    torch.random.set_rng_state(rng["torch_cpu"].cpu())
    if rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    for generator, name in (
        (task_update_rng, "task_update"),
        (collection_environment_seed_rng, "collection_environment"),
        (validation_environment_seed_rng, "validation_environment"),
        (final_environment_seed_rng, "final_environment"),
    ):
        generator.bit_generator.state = copy.deepcopy(rng[name])
    if config.uses_replay_rehearsed_shared_behavior:
        shared_behavior_update_rng.bit_generator.state = copy.deepcopy(
            rng["shared_behavior_update"]
        )
    schedule = payload["schedule"]
    environment_schedule._step = int(schedule["environment_step"])
    if (
        config.uses_adaptive_behavior_compression
        and aco is not None
    ):
        # The task route is scheduler state, deliberately not a CUDA buffer in
        # the Actor-Critic. Restore it before any resumed collection can run.
        aco.ac.set_task_route(int(schedule["current_task_id"]))
    counters = payload["counters"]
    restored = {
        "completed_epochs": int(schedule["completed_epochs"]),
        "current_task_id": int(schedule["current_task_id"]),
        "raw_environment_frames": int(counters["raw_environment_frames"]),
        "world_model_updates": int(counters["world_model_updates"]),
        "actor_critic_updates": int(counters["actor_critic_updates"]),
        "adaptive_behavior_compression_updates": int(
            counters.get("adaptive_behavior_compression_updates", 0)
        ),
    }
    if config.uses_replay_rehearsed_shared_behavior:
        restored["shared_actor_teacher"] = restored_teacher
        restored["shared_actor_teacher_seen_tasks"] = restored_teacher_seen_tasks
        routed_updates = counters.get("actor_critic_updates_by_task_route")
        if not isinstance(routed_updates, Mapping):
            raise ValueError(
                "Shared behavior checkpoint is missing routed update counters"
            )
        restored_updates = {
            int(task_id): int(update_count)
            for task_id, update_count in routed_updates.items()
        }
        if (
            any(
                task_id < 0 or update_count < 0
                for task_id, update_count in restored_updates.items()
            )
            or sum(restored_updates.values())
            != restored["actor_critic_updates"]
        ):
            raise ValueError(
                "Shared behavior checkpoint has inconsistent routed update counters"
            )
        restored["shared_behavior_replay_updates"] = restored_updates
    return restored


_ATOMIC_LORA_SHARED_HEADS_METHOD = (
    "evolving_atomic_rssm_atomic_lora_shared_heads_arrow"
)
_TASK0_TRANSITION_SOURCE_METHOD = (
    "evolving_atomic_rssm_learned_base_adapters_arrow"
)
_TASK0_TRANSITION_ALLOWED_CONFIG_CHANGES = frozenset(
    {
        "continual_method",
        "task_mechanism_reuse",
        "task_mechanism_parameterization",
        "task_mechanism_low_rank",
        "task_private_prediction_adapters",
        "prediction_adapter_rank",
        "freeze_shared_prediction_heads_after_task0",
    }
)


def _load_evolving_task0_transition_checkpoint(
    path: Path,
    *,
    config: Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one named Task-0 boundary transition used by method C.

    This is deliberately not the normal equivalent-resume path.  It preserves
    Task-0 data, weights, behavior state, counters, and RNG while changing the
    *future-task* Q/F/P and prediction-head ownership topology.  Optimizers for
    the changed world-model ownership are therefore rebuilt explicitly.
    """

    path = path.expanduser().resolve()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            "Task-0 transition requires a checkpoint and checksum sidecar: "
            f"{path}"
        )
    fields = checksum_path.read_text(encoding="ascii").split()
    actual_sha256 = _sha256(path)
    if not fields or fields[0] != actual_sha256:
        raise ValueError("Task-0 transition checkpoint checksum does not match")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Task-0 transition checkpoint must contain a dictionary")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "evolving_core_atomic_rssm_resumable_checkpoint"
        or payload.get("resumable") is not True
    ):
        raise ValueError(
            "Task-0 transition source must be private-behavior Evolving-Core "
            "resumable schema v1"
        )
    if config.continual_method != _ATOMIC_LORA_SHARED_HEADS_METHOD:
        raise ValueError(
            "Task-0 boundary transition is restricted to the named atomic-LoRA "
            "shared-head method"
        )
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("Task-0 transition source is missing its resolved config")
    if source_config.get("continual_method") != _TASK0_TRANSITION_SOURCE_METHOD:
        raise ValueError(
            "Task-0 transition must come from the named learned-base adapter pilot"
        )
    target_config = config.to_dict()
    changed = {
        key
        for key in set(source_config) | set(target_config)
        if source_config.get(key) != target_config.get(key)
    }
    if changed != _TASK0_TRANSITION_ALLOWED_CONFIG_CHANGES:
        raise ValueError(
            "Task-0 transition changed fields outside the declared topology "
            f"boundary: {sorted(changed)}"
        )
    expected_values = {
        "source": {
            "task_mechanism_reuse": False,
            "task_mechanism_parameterization": "learned_task0_low_rank",
            "task_mechanism_low_rank": 32,
            "task_private_prediction_adapters": True,
            "prediction_adapter_rank": 32,
            "freeze_shared_prediction_heads_after_task0": True,
        },
        "target": {
            "task_mechanism_reuse": True,
            "task_mechanism_parameterization": "dense_task0_low_rank_atoms",
            "task_mechanism_low_rank": 128,
            "task_private_prediction_adapters": False,
            "prediction_adapter_rank": 0,
            "freeze_shared_prediction_heads_after_task0": False,
        },
    }
    for label, values in expected_values.items():
        actual = source_config if label == "source" else target_config
        mismatches = {
            name: (actual.get(name), expected)
            for name, expected in values.items()
            if actual.get(name) != expected
        }
        if mismatches:
            raise ValueError(
                f"Task-0 transition {label} topology is not the declared v1: "
                f"{mismatches}"
            )

    durations = _sequential_task_durations(config)
    first_task_epochs = int(durations[0])
    schedule = payload.get("schedule")
    counters = payload.get("counters")
    if not isinstance(schedule, Mapping) or not isinstance(counters, Mapping):
        raise ValueError("Task-0 transition source is missing schedule/counters")
    expected_schedule = {
        "environment_step": first_task_epochs,
        "epoch": first_task_epochs - 1,
        "completed_epochs": first_task_epochs,
        "current_task_id": 0,
    }
    schedule_mismatches = {
        name: (int(schedule.get(name, -1)), expected)
        for name, expected in expected_schedule.items()
        if int(schedule.get(name, -1)) != expected
    }
    if schedule_mismatches:
        raise ValueError(
            "Task-0 transition source is not exactly the post-Task-0 boundary: "
            f"{schedule_mismatches}"
        )
    expected_counters = {
        "raw_environment_frames": (
            first_task_epochs
            * config.n_sync
            * config.gen_seq_len
            * config.env_repeat
        ),
        "world_model_updates": (
            first_task_epochs * config.steps_per_batch
            + config.boundary_consolidation_steps
        ),
        "actor_critic_updates": first_task_epochs * config.ac_train_steps,
    }
    counter_mismatches = {
        name: (int(counters.get(name, -1)), expected)
        for name, expected in expected_counters.items()
        if int(counters.get(name, -1)) != expected
    }
    if counter_mismatches:
        raise ValueError(
            "Task-0 transition source counters do not match the fixed budget: "
            f"{counter_mismatches}"
        )
    for name in ("world_model", "boundary_teacher", "optimizers", "replay", "rng"):
        if name not in payload:
            raise ValueError(f"Task-0 transition source is missing {name!r}")

    metadata = {
        "schema_version": 1,
        "artifact_kind": "evolving_task0_cross_topology_transition",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": actual_sha256,
        "source_method": _TASK0_TRANSITION_SOURCE_METHOD,
        "target_method": _ATOMIC_LORA_SHARED_HEADS_METHOD,
        "completed_epochs": first_task_epochs,
        "source_counters": expected_counters,
        "allowed_config_changes": sorted(changed),
        "replay_state": "exact_task0_checkpoint_state_copied_to_new_working_mmaps",
        "world_model_optimizer_state": "reset_due_to_ownership_transition",
        "actor_critic_task0_optimizer_state": "restored_exactly_but_frozen",
        "rng_state": "restored_after_target_topology_construction",
        "environment_schedule_restart_task": 1,
        "scientific_scope": (
            "post-Task-0 boundary bootstrap across a declared topology change; "
            "not an equivalent resume and not a from-scratch C run"
        ),
    }
    return payload, metadata


def _seed_atomic_lora_task0_world_model(
    wm: WorldModel,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Copy only shared and Task-0-compatible state into method C."""

    exact_names = {"task_expert_initialized"}
    prefixes = (
        "rssm.image_embedder.",
        "rssm.observation_adapter.",
        "rssm.recurrent.",
        "rssm.representation.",
        "rssm.transition.",
        "rssm.image_projectors.0.",
        "rssm.recurrent_mechanism_bank.mechanisms.0.",
        "rssm.representation_mechanism_bank.mechanisms.0.",
        "rssm.transition_mechanism_bank.mechanisms.0.",
        "zh_transform.",
        "decoder.",
        "reward_fc.",
        "continue_fc.",
    )
    # The published Atari shape uses identity observation and latent-feature
    # adapters, which have no state-dict entries. Transfer shaped adapters when
    # present, but do not require empty identity modules to manufacture state.
    optional_stateless_prefixes = {
        "rssm.observation_adapter.",
        "zh_transform.",
    }
    required_prefixes = tuple(
        prefix
        for prefix in prefixes
        if prefix not in optional_stateless_prefixes
    )
    target_state = wm.state_dict()
    selected = {
        name: value
        for name, value in source_state.items()
        if name in exact_names or name.startswith(prefixes)
    }
    missing_required_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in selected)
    ]
    if missing_required_prefixes:
        raise ValueError(
            "Task-0 transition source lacks required world-model modules: "
            f"{missing_required_prefixes}"
        )
    if exact_names - selected.keys():
        raise ValueError("Task-0 transition source lacks task initialization state")
    unknown = sorted(set(selected) - set(target_state))
    if unknown:
        raise ValueError(
            f"Task-0 transition selected unknown target state: {unknown[:5]}"
        )
    mismatched = {
        name: (tuple(value.shape), tuple(target_state[name].shape))
        for name, value in selected.items()
        if value.shape != target_state[name].shape or value.dtype != target_state[name].dtype
    }
    if mismatched:
        raise ValueError(
            "Task-0 transition shared/Task-0 tensors changed shape or dtype: "
            f"{mismatched}"
        )
    merged = dict(target_state)
    merged.update(selected)
    wm.load_state_dict(merged, strict=True)
    parameter_names = {name for name, _parameter in wm.named_parameters()}
    transferred_parameters = sum(
        value.numel()
        for name, value in selected.items()
        if name in parameter_names
    )
    return {
        "selected_tensor_count": len(selected),
        "selected_parameter_count": transferred_parameters,
        "selected_prefixes": list(prefixes),
        "future_task_modules": "target initialization preserved",
        "future_task_routes": "target zero-gate initialization preserved",
        "prediction_adapters": "source-only modules omitted",
    }


def _restore_task0_transition_rng(
    rng: Mapping[str, Any],
    *,
    task_update_rng: np.random.Generator,
    collection_environment_seed_rng: np.random.Generator,
    validation_environment_seed_rng: np.random.Generator,
    final_environment_seed_rng: np.random.Generator,
) -> None:
    """Restore the source boundary RNG after all method-C modules exist."""

    random.setstate(rng["python"])
    np.random.set_state(rng["numpy_legacy"])
    torch.random.set_rng_state(rng["torch_cpu"].cpu())
    cuda_state = rng.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Task-0 transition contains CUDA RNG but CUDA is absent")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("Task-0 transition CUDA device count changed")
        torch.cuda.set_rng_state_all(cuda_state)
    for generator, name in (
        (task_update_rng, "task_update"),
        (collection_environment_seed_rng, "collection_environment"),
        (validation_environment_seed_rng, "validation_environment"),
        (final_environment_seed_rng, "final_environment"),
    ):
        generator.bit_generator.state = copy.deepcopy(rng[name])


def _parameter_accounting(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def _module_state_accounting(module: torch.nn.Module) -> dict[str, int]:
    accounting = _parameter_accounting(module)
    buffers = list(module.buffers())
    buffer_values = sum(buffer.numel() for buffer in buffers)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    return {
        **accounting,
        "buffers": buffer_values,
        "buffer_bytes": buffer_bytes,
        "parameter_and_buffer_bytes": accounting["parameter_bytes"] + buffer_bytes,
    }


def _actor_critic_parameter_accounting(aco: ActorCriticOpt) -> dict:
    actor = aco.ac.actor
    critic = aco.ac.critic
    if not isinstance(actor, torch.nn.Module) or not isinstance(critic, torch.nn.Module):
        raise TypeError("Actor and critic must be torch modules for resource accounting")
    accounting = {
        "schema_version": 1,
        "actor_class": type(actor).__name__,
        "actor": _module_state_accounting(actor),
        "critic_class": type(critic).__name__,
        "critic": _module_state_accounting(critic),
        "actor_critic": _module_state_accounting(aco.ac),
        "accounting_scope": (
            "parameters and registered buffers; excludes gradients, optimizer state, "
            "and activations; non-persistent buffers are included when allocated"
        ),
    }
    if aco.slow_critic is not None:
        accounting["slow_critic_class"] = type(aco.slow_critic).__name__
        accounting["slow_critic"] = _module_state_accounting(aco.slow_critic)
        accounting["accounting_scope"] += "; slow critic is training state only"
    return accounting


def _actor_critic_bank_parameter_accounting(bank) -> dict:
    per_task = {
        str(task_id): _actor_critic_parameter_accounting(bank.get(task_id))
        for task_id in bank.task_ids()
    }
    return {
        "schema_version": 1,
        "topology": "per_task_actor_critic_bank",
        "task_ids": list(bank.task_ids()),
        "per_task": per_task,
        "aggregate_actor_critic_parameters": sum(
            task["actor_critic"]["parameters"] for task in per_task.values()
        ),
        "optimizer_state_excluded": True,
    }


def _shared_actor_parameter_accounting(
    aco: ActorCriticOpt,
    teacher_actor: Optional[torch.nn.Module],
) -> dict:
    accounting = _actor_critic_parameter_accounting(aco)
    adaptive_behavior = bool(aco.ac.adaptive_behavior_residuals)
    accounting["topology"] = (
        "single_shared_mlp_plus_task_adaptive_residuals"
        if adaptive_behavior
        else "single_shared_actor_critic"
    )
    accounting["persistent_actor_copies"] = 1
    accounting["per_task_actor_growth"] = (
        "outcome-dependent private actor/critic residual width"
        if adaptive_behavior
        else 0
    )
    if adaptive_behavior:
        accounting["adaptive_behavior"] = {
            "layout": aco.ac.adaptive_behavior_layout(),
            "actor": aco.ac.actor.parameter_report(),
            "critic": aco.ac.critic.parameter_report(),
            "shared_bases_trainable": True,
            "task_routes_explicit": True,
        }
    accounting["transient_teacher"] = (
        {
            "persistent": False,
            "lifetime": "current task only",
            "actor": _module_state_accounting(teacher_actor),
        }
        if teacher_actor is not None
        else None
    )
    return accounting


def _active_actor_critic_parameter_accounting(
    *,
    config: Config,
    aco: ActorCriticOpt,
    actor_critic_bank,
    shared_actor_teacher: Optional[torch.nn.Module],
) -> dict:
    """Account for the persistent behavior topology selected by the protocol."""

    if actor_critic_bank is not None:
        return _actor_critic_bank_parameter_accounting(actor_critic_bank)
    if config.uses_shared_actor:
        return _shared_actor_parameter_accounting(aco, shared_actor_teacher)
    return _actor_critic_parameter_accounting(aco)


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _world_model_parameter_accounting(wm: WorldModel) -> dict:
    if wm.observation_objective == "reconstruction":
        observation_head_name = "decoder"
        observation_head = wm.decoder
    elif wm.observation_objective == "r2":
        observation_head_name = "r2_projector"
        observation_head = wm.r2_projector
    else:
        observation_head_name = "feature_predictor"
        observation_head = wm.feature_predictor
    observation_encoders_per_task = {
        str(task_id): _parameter_accounting(
            wm.rssm.image_embedder_for(task_id)
        )
        for task_id in range(wm.rssm.num_task_experts)
        if wm.rssm.task_banked_image_encoder
    }
    projectors_per_task = {
        str(task_id): _parameter_accounting(
            wm.rssm.image_projector_for(task_id)
        )
        for task_id in range(
            0 if wm.rssm.task_symmetric_image_projectors else 1,
            wm.rssm.num_task_experts,
        )
        if wm.rssm.task_projected_image_encoder
    }
    mechanism_banks = {}
    mechanism_parameters_per_later_task = {}
    if wm.rssm.task_mechanism_bank_enabled:
        banks = {
            "recurrent": wm.rssm.recurrent_mechanism_bank,
            "representation": wm.rssm.representation_mechanism_bank,
            "transition": wm.rssm.transition_mechanism_bank,
        }
        mechanism_banks = {
            name: {
                **bank.parameter_report(),
                "route_values": {
                    str(task_id): bank.route_values(task_id)
                    for task_id in range(bank.num_tasks)
                },
            }
            for name, bank in banks.items()
        }
        mechanism_parameters_per_later_task = {
            str(task_id): sum(
                report["mechanism_parameters_per_task"][
                    task_id if report["include_task0"] else task_id - 1
                ]
                + report["route_parameters_per_later_task"][
                    task_id if report["include_task0"] else task_id - 1
                ]
                for report in mechanism_banks.values()
            )
            for task_id in range(
                0 if wm.rssm.task_symmetric_mechanisms else 1,
                wm.rssm.num_task_experts,
            )
        }
    return {
        "schema_version": 1,
        "observation_objective": wm.observation_objective,
        "world_model": _parameter_accounting(wm),
        "world_model_parameter_and_buffer_state": _module_state_accounting(wm),
        "observation_encoder": _parameter_accounting(wm.rssm.image_embedder),
        "observation_encoder_topology": (
            "per_task_bank"
            if wm.rssm.task_banked_image_encoder
            else "shared"
        ),
        "observation_encoders_per_task": observation_encoders_per_task,
        "task_projected_image_encoder": wm.rssm.task_projected_image_encoder,
        "task_symmetric_image_projectors": (
            wm.rssm.task_symmetric_image_projectors
        ),
        "observation_projectors_per_task": projectors_per_task,
        "rssm_task_lora_enabled": wm.rssm.task_lora_enabled,
        "rssm_task_lora_reports": wm.rssm.task_lora_reports,
        "rssm_recurrent_output_adapter_enabled": (
            wm.rssm.task_recurrent_output_adapter_enabled
        ),
        "rssm_recurrent_output_adapter_features": (
            wm.rssm.task_recurrent_output_adapter_features
        ),
        "rssm_task_mechanism_bank_enabled": (
            wm.rssm.task_mechanism_bank_enabled
        ),
        "rssm_task_mechanism_reuse": wm.rssm.task_mechanism_reuse,
        "rssm_task_mechanism_parameterization": (
            wm.rssm.task_mechanism_parameterization
        ),
        "rssm_task_mechanism_low_rank": wm.rssm.task_mechanism_low_rank,
        "rssm_task_symmetric_mechanisms": wm.rssm.task_symmetric_mechanisms,
        "rssm_task_mechanism_banks": mechanism_banks,
        "rssm_task_mechanism_parameters_per_later_task": (
            mechanism_parameters_per_later_task
        ),
        "aggregate_observation_encoder_parameters": (
            sum(
                accounting["parameters"]
                for accounting in observation_encoders_per_task.values()
            )
            if observation_encoders_per_task
            else _parameter_accounting(wm.rssm.image_embedder)["parameters"]
        ),
        "observation_adapter_kind": wm.rssm.observation_adapter_kind,
        "observation_adapter": _parameter_accounting(
            wm.rssm.observation_adapter
        ),
        "posterior_embedding_size": wm.rssm.observation_embedding_size,
        "observation_head_name": observation_head_name,
        "observation_head": _parameter_accounting(observation_head),
        "prediction_head_topology": (
            "frozen_task0_base_plus_private_feature_adapters"
            if wm.task_private_prediction_adapters
            else "single_shared"
            if wm.task_shared_prediction_heads
            else "task_routed"
            if wm.rssm.num_task_experts > 1
            else "single_task"
        ),
        "task_shared_prediction_heads": wm.task_shared_prediction_heads,
        "task_private_prediction_adapters": (
            wm.task_private_prediction_adapters
        ),
        "freeze_shared_prediction_heads_after_task0": (
            wm.freeze_shared_prediction_heads_after_task0
        ),
        "prediction_adapter_rank": wm.prediction_adapter_rank,
        "prediction_adapter_residual_scale": (
            wm.prediction_adapter_residual_scale
        ),
        "prediction_adapter_parameters_per_task": {
            str(task_id): sum(
                parameter.numel()
                for head_name in ("observation", "reward", "continue")
                for adapter in [wm.prediction_adapter_for(head_name, task_id)]
                if adapter is not None
                for parameter in adapter.parameters()
            )
            for task_id in range(wm.rssm.num_task_experts)
        },
        "reward_head": _parameter_accounting(wm.reward_fc),
        "continue_head": _parameter_accounting(wm.continue_fc),
        "prediction_head_expert_parameters": sum(
            parameter.numel()
            for modules in (
                wm.decoder_experts,
                wm.feature_predictor_experts,
                wm.reward_experts,
                wm.continue_experts,
            )
            for module in modules
            for parameter in module.parameters()
        ),
        "accounting_scope": (
            "legacy component entries count parameters only; "
            "world_model_parameter_and_buffer_state also counts registered buffers; "
            "all entries exclude gradients, optimizer state, and activations"
        ),
    }


@torch.no_grad()
def _encode_frozen_observation_features(
    wm: WorldModel,
    observations: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Encode collected CPU observations once before writing the replay sidecar."""
    if observations.ndim != 5:
        raise ValueError("Collected observations must have [time, batch, C, H, W] axes")
    if batch_size < 1:
        raise ValueError("DINOv3 encoding batch size must be positive")
    time, sequences = observations.shape[:2]
    flat = observations.reshape(-1, *observations.shape[-3:])
    try:
        encoder_device = next(wm.rssm.image_embedder.parameters()).device
    except StopIteration:
        encoder_device = next(wm.parameters()).device
    encoded = []
    for start in range(0, flat.shape[0], batch_size):
        images = flat[start : start + batch_size].to(encoder_device)
        encoded.append(wm.rssm.image_embedder(images).detach().cpu())
    features = torch.cat(encoded, dim=0)
    return features.view(time, sequences, -1)


@torch.no_grad()
def _fit_dinov3_patch_projection(
    wm: WorldModel,
    observations: torch.Tensor,
    *,
    calibration_frames: int,
) -> dict[str, object]:
    """Learn one Task-1 PCA bottleneck before any world-model update."""
    encoder = wm.rssm.image_embedder
    if not getattr(encoder, "requires_projection_fit", False):
        raise RuntimeError("The DINOv3 patch projection does not require fitting")
    if observations.ndim != 5:
        raise ValueError("Collected observations must have [time, batch, C, H, W] axes")
    flat = observations.reshape(-1, *observations.shape[-3:])
    if calibration_frames < 1 or calibration_frames > flat.shape[0]:
        raise ValueError(
            "Patch projection calibration frames must fit in the first collection"
        )
    indices = torch.linspace(
        0,
        flat.shape[0] - 1,
        steps=calibration_frames,
        dtype=torch.float64,
    ).round().long()
    try:
        encoder_device = next(encoder.parameters()).device
    except StopIteration:
        encoder_device = next(wm.parameters()).device
    calibration_images = flat.index_select(0, indices).to(encoder_device)
    raw_patch_features = encoder.extract_patch_features(calibration_images)
    metadata = encoder.fit_patch_projection(raw_patch_features)
    return {
        **metadata,
        "calibration_frames": calibration_frames,
        "frame_selection": "uniform over first Task-1 random collection",
        "fit_timing": "before first world-model update",
        "frozen_after_fit": True,
    }


def _restrict_optimizer_to_trainable(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
) -> list[torch.nn.Parameter]:
    """Drop frozen parameters without resetting optimizer state for adapters."""
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Freezing the shared core left no trainable parameters")
    trainable_ids = {id(parameter) for parameter in trainable}
    for parameter in list(optimizer.state):
        if id(parameter) not in trainable_ids:
            del optimizer.state[parameter]
    for parameter_group in optimizer.param_groups:
        parameter_group["params"] = trainable
    return trainable


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    """Return the unique parameters currently owned by an optimizer."""
    parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]
    if not parameters:
        raise RuntimeError("World-model optimizer contains no parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("World-model optimizer contains duplicate parameters")
    return parameters


RESUME_ADAPTATION_MODES = frozenset({"kan_only", "kan_plus_heads"})


def _load_analysis_snapshot(path: Path) -> Mapping[str, object]:
    """Load a portable boundary snapshot for a task-2 acquisition run."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Analysis snapshot must contain a mapping: {path}")
    required = {
        "artifact_kind",
        "resumable",
        "world_model_state_dict",
        "actor_critic_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Analysis snapshot is missing {missing}: {path}")
    if payload["artifact_kind"] != "analysis_snapshot" or payload["resumable"]:
        raise ValueError(f"Snapshot is not a non-resumable analysis snapshot: {path}")
    if not isinstance(payload["world_model_state_dict"], Mapping):
        raise ValueError(f"World-model state is not a mapping: {path}")
    if not isinstance(payload["actor_critic_state_dict"], Mapping):
        raise ValueError(f"Actor-critic state is not a mapping: {path}")
    return payload


def _load_task1_boundary_snapshot(path: Path) -> Mapping[str, object]:
    """Load a finished Task-1 inference bank as a new incremental-run seed."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Task boundary snapshot must contain a mapping: {path}")
    required = {
        "artifact_kind",
        "resumable",
        "completed_epochs",
        "completed_task",
        "world_model_state_dict",
        "actor_critic_bank_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Task boundary snapshot is missing {missing}: {path}")
    if (
        payload["artifact_kind"] != "task_bank_boundary_inference_snapshot"
        or payload["resumable"]
    ):
        raise ValueError(f"Snapshot is not a non-resumable task boundary: {path}")
    completed_task = payload["completed_task"]
    if not isinstance(completed_task, Mapping) or int(
        completed_task.get("task_index", -1)
    ) != 0:
        raise ValueError("The incremental seed must be the completed first task")
    actor_bank = payload["actor_critic_bank_state_dict"]
    if not isinstance(actor_bank, Mapping):
        raise ValueError("Task boundary actor bank is not a mapping")
    tasks = actor_bank.get("tasks")
    if not isinstance(tasks, Mapping) or "0" not in tasks:
        raise ValueError("Task boundary snapshot does not contain the Task-1 actor")
    if not isinstance(payload["world_model_state_dict"], Mapping):
        raise ValueError("Task boundary world-model state is not a mapping")
    return payload


def _load_prefixed_module_state(
    module: torch.nn.Module,
    state: Mapping[str, object],
    *,
    prefix: str,
    label: str,
) -> int:
    prefix_with_dot = f"{prefix}."
    selected = {
        key[len(prefix_with_dot) :]: value
        for key, value in state.items()
        if key.startswith(prefix_with_dot)
    }
    if not selected:
        raise ValueError(f"Task-1 snapshot has no {label} state")
    result = module.load_state_dict(selected, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"Task-1 {label} mismatch: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )
    return len(selected)


def _seed_task1_world_model_from_fullbank(
    wm: WorldModel, payload: Mapping[str, object]
) -> dict[str, int]:
    """Import only Task-1 core/head tensors into the new frozen-core topology."""
    state = payload["world_model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Task-1 world-model state must be a mapping")
    modules = {
        "rssm.image_embedder": (wm.rssm.image_embedder, "CNN encoder"),
        "rssm.recurrent": (wm.rssm.recurrent, "recurrent RSSM"),
        "rssm.representation": (wm.rssm.representation, "posterior RSSM"),
        "rssm.transition": (wm.rssm.transition, "prior RSSM"),
        "decoder": (wm.decoder, "pixel decoder"),
        "reward_fc": (wm.reward_fc, "reward head"),
        "continue_fc": (wm.continue_fc, "continuation head"),
    }
    report = {
        prefix: _load_prefixed_module_state(
            module, state, prefix=prefix, label=label
        )
        for prefix, (module, label) in modules.items()
    }
    if wm.task_expert_initialized is None:
        raise ValueError("Task-1 seed requires a task-routed world model")
    wm.task_expert_initialized.zero_()
    wm.task_expert_initialized[0] = True
    return report


def _load_snapshot_state(
    module: torch.nn.Module,
    state: Mapping[str, object],
    *,
    label: str,
) -> dict[str, list[str]]:
    """Load weights while allowing only stale consolidation buffers."""
    target_state = module.state_dict()
    filtered_state = {}
    ignored_stale_buffers: list[str] = []
    for key, value in state.items():
        target = target_state.get(key)
        if (
            "consolidation_" in key
            and target is not None
            and hasattr(value, "shape")
            and target.shape != value.shape
        ):
            ignored_stale_buffers.append(key)
            continue
        filtered_state[key] = value
    result = module.load_state_dict(filtered_state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    disallowed_missing = [
        key for key in missing if "consolidation_" not in key
    ]
    disallowed_unexpected = [
        key for key in unexpected if "consolidation_" not in key
    ]
    if disallowed_missing or disallowed_unexpected:
        raise ValueError(
            f"{label} snapshot incompatibility: missing={disallowed_missing} "
            f"unexpected={disallowed_unexpected}"
        )
    return {
        "missing": missing,
        "unexpected": [*unexpected, *ignored_stale_buffers],
    }


def _last_linear(module: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    candidates = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear)
    ]
    if not candidates:
        raise ValueError(f"No linear readout found in {type(module).__name__}")
    return candidates[-1]


def _configure_resume_world_model(
    wm: WorldModel,
    mode: str,
) -> list[str]:
    """Freeze the shared model and optionally open a few task-2 readouts."""
    if mode not in RESUME_ADAPTATION_MODES:
        raise ValueError(f"Unknown resume adaptation mode: {mode!r}")
    wm.freeze_shared_core()
    opened: list[str] = []
    if mode == "kan_plus_heads":
        candidates = {
            "world_model.rssm.representation.eh_to_inter": (
                wm.rssm.representation.eh_to_inter
            ),
            "world_model.rssm.transition.h_to_z_prior": (
                wm.rssm.transition.h_to_z_prior
            ),
            "world_model.reward_fc": wm.reward_fc,
            "world_model.continue_fc": wm.continue_fc,
        }
        for name, module in candidates.items():
            child_name, linear = _last_linear(module)
            linear.requires_grad_(True)
            opened.append(f"{name}.{child_name}")
    return opened


def _configure_resume_actor_critic(
    aco: ActorCriticOpt,
    mode: str,
) -> list[str]:
    """Freeze the MLP behavior core and optionally open its output heads."""
    if mode not in RESUME_ADAPTATION_MODES:
        raise ValueError(f"Unknown resume adaptation mode: {mode!r}")
    aco.ac.freeze_shared_core()
    if mode == "kan_only":
        return []
    opened: list[str] = []
    for name in ("actor", "critic"):
        head = getattr(aco.ac, name)
        if not hasattr(head, "base_head"):
            raise ValueError(
                "Task-2 checkpoint adaptation requires residual MLP behavior heads"
            )
        head.base_head.requires_grad_(True)
        opened.append(f"actor_critic.{name}.base_head")
    return opened


def _actor_critic_kwargs(
    config: Config,
    *,
    feature_cache,
    protect_residual_updates: bool,
) -> dict[str, object]:
    return {
        "dream_steps": config.ac_dream_steps,
        "actor_network": config.actor_network,
        "actor_kan_hidden_features": config.actor_kan_hidden_features,
        "actor_kan_grid_size": config.actor_kan_grid_size,
        "actor_kan_spline_order": config.actor_kan_spline_order,
        "actor_kan_input_min": config.actor_kan_input_min,
        "actor_kan_input_max": config.actor_kan_input_max,
        "actor_kan_normalize_recurrent_state": (
            config.actor_kan_normalize_recurrent_state
        ),
        "fastkan_hidden_features": config.fastkan_hidden_features,
        "fastkan_hidden_layers": config.fastkan_hidden_layers,
        "fastkan_grid_size": config.fastkan_grid_size,
        "fastkan_input_min": config.fastkan_input_min,
        "fastkan_input_max": config.fastkan_input_max,
        "fastkan_rms_norm_epsilon": config.fastkan_rms_norm_epsilon,
        "fastkan_actor_output_scale": config.fastkan_actor_output_scale,
        "fastkan_actor_unimix": config.fastkan_actor_unimix,
        "optimizer_name": config.ac_optimizer,
        "optimizer_eps": config.ac_optimizer_eps,
        "optimizer_beta1": config.ac_optimizer_beta1,
        "optimizer_beta2": config.ac_optimizer_beta2,
        "optimizer_warmup_steps": config.ac_optimizer_warmup_steps,
        "agc_clip": config.ac_agc_clip,
        "grad_clip": config.ac_grad_clip,
        "discount": config.ac_discount,
        "lam": config.ac_lambda,
        "entropy_scale": config.ac_entropy_scale,
        "return_norm_decay": config.ac_return_norm_decay,
        "persistent_return_norm": config.ac_persistent_return_norm,
        "slow_critic_regularizer": config.ac_slow_critic_regularizer,
        "slow_critic_decay": config.ac_slow_critic_decay,
        "replay_critic_loss_scale": config.ac_replay_critic_loss_scale,
        "use_slow_critic_targets": config.ac_use_slow_critic_targets,
        "corrected_imagination_bootstrap": config.ac_corrected_imagination_bootstrap,
        "residual_correction": config.residual_correction,
        "residual_bottleneck_features": config.residual_bottleneck_features,
        "residual_grid_size": config.residual_grid_size,
        "residual_input_min": config.residual_input_min,
        "residual_input_max": config.residual_input_max,
        "residual_rms_norm_epsilon": config.residual_rms_norm_epsilon,
        "residual_alpha": config.residual_alpha,
        "residual_input_mode": config.residual_input_mode,
        "residual_consolidation": config.residual_consolidation,
        "adaptive_behavior_residuals": config.adaptive_behavior_residuals,
        "adaptive_behavior_num_tasks": config.rssm_num_experts,
        "adaptive_behavior_hidden_features": (
            config.adaptive_behavior_hidden_features
        ),
        "adaptive_behavior_residual_scale": (
            config.adaptive_behavior_residual_scale
        ),
        "adaptive_behavior_num_atoms": config.adaptive_behavior_num_atoms,
        "adaptive_behavior_reuse": config.adaptive_behavior_reuse,
        "protect_residual_updates": protect_residual_updates,
        "feature_cache": feature_cache,
    }


def _actor_critic_constructor_kwargs(
    config: Config,
) -> dict[str, object]:
    kwargs = _actor_critic_kwargs(
        config,
        feature_cache=None,
        protect_residual_updates=False,
    )
    for key in (
        "dream_steps",
        "grad_clip",
        "discount",
        "lam",
        "entropy_scale",
        "return_norm_decay",
        "persistent_return_norm",
        "replay_critic_loss_scale",
        "use_slow_critic_targets",
        "corrected_imagination_bootstrap",
        "protect_residual_updates",
        "feature_cache",
    ):
        kwargs.pop(key)
    return kwargs


@torch.no_grad()
def _exercise_world_model_residual_heads(
    wm: WorldModel,
    z: torch.Tensor,
    h: torch.Tensor,
    *,
    prior_log_probs: Optional[torch.Tensor] = None,
) -> None:
    """Visit every non-RSSM residual at deterministic replay/imagination states."""
    zhs = wm.zh_transform(z, h)
    for residual in (wm.reward_residual, wm.continue_residual):
        if residual is not None:
            residual(zhs)
    if wm.feature_predictor_residual is not None:
        if wm.observation_objective == "dinov3_next_feature":
            if prior_log_probs is None:
                raise ValueError("Prior features require deterministic prior logits")
            feature_state = wm.zh_transform(prior_log_probs.exp(), h)
        else:
            feature_state = zhs
        wm.feature_predictor_residual(feature_state)


@torch.no_grad()
def _observe_replay_for_kan_importance(
    wm: WorldModel,
    aco: ActorCriticOpt,
    feature_cache,
    *,
    batches: int,
    sequence_length: int,
    sequences: int,
    imagination_horizon: int,
) -> None:
    """Exercise KAN adapters on replay posteriors and deterministic imagination."""
    device = next(wm.parameters()).device
    for _ in range(batches):
        actions, _, features, _, _, resets = feature_cache.minibatch(
            sequence_length,
            sequences,
            mb_device=device,
        )
        initial_z, initial_h = wm.rssm.initial_state(actions.shape[1])
        _, posterior_z, hiddens = wm.rssm.observe_embeddings(
            initial_z,
            actions,
            initial_h,
            wm.rssm.adapt_observation_embeddings(features),
            resets,
            stochastic=False,
        )
        prior_log_probs = wm.rssm.transition(hiddens)
        _exercise_world_model_residual_heads(
            wm,
            posterior_z,
            hiddens,
            prior_log_probs=prior_log_probs,
        )
        posterior_states = zh_to_ac_state(posterior_z, hiddens)
        aco.ac.actor(posterior_states)
        aco.ac.critic(posterior_states)

        z = posterior_z[-1]
        h = hiddens[-1]
        no_reset = torch.zeros(actions.shape[1], 1, device=device)
        for _ in range(imagination_horizon):
            state = zh_to_ac_state(z, h)
            action_log_probs = aco.ac.actor(state)
            aco.ac.critic(state)
            action = torch.nn.functional.one_hot(
                action_log_probs.argmax(dim=-1),
                num_classes=wm.a_dim,
            ).to(dtype=z.dtype)
            imagined_prior, z, h = wm.rssm(
                z,
                action,
                h,
                None,
                no_reset,
                stochastic=False,
            )
            _exercise_world_model_residual_heads(
                wm,
                z,
                h,
                prior_log_probs=imagined_prior,
            )


def _consolidate_kan_from_replay(
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    feature_cache,
    epoch: int,
    global_step: int,
    log_dir: Path,
    writer,
) -> dict[str, dict[str, float | int]]:
    """Estimate, persist, and log task-boundary KAN coefficient importance."""
    if feature_cache is None:
        raise RuntimeError("Replay KAN consolidation requires frozen feature replay")
    from clworldmodel.continual import (
        begin_kan_importance_estimation,
        cancel_kan_importance_estimation,
        finish_kan_importance_estimation,
    )

    roots = {"world_model": wm, "actor_critic": aco.ac}
    residuals = begin_kan_importance_estimation(roots)
    wm_was_training = wm.training
    ac_was_training = aco.ac.training
    try:
        with _preserve_training_rng_state():
            wm.eval()
            aco.ac.eval()
            _observe_replay_for_kan_importance(
                wm,
                aco,
                feature_cache,
                batches=config.residual_consolidation_batches,
                sequence_length=config.mb_t_size,
                sequences=config.mb_n_size,
                imagination_horizon=(
                    config.residual_consolidation_imagination_horizon
                ),
            )
    except Exception:
        cancel_kan_importance_estimation(residuals)
        raise
    finally:
        wm.train(wm_was_training)
        aco.ac.train(ac_was_training)

    diagnostics = finish_kan_importance_estimation(
        residuals,
        gradient_power=config.residual_consolidation_gradient_power,
        min_plasticity=config.residual_consolidation_min_plasticity,
        anchor_loss_scale=config.residual_consolidation_anchor_loss_scale,
    )
    completed_task_index, _ = _sequential_task_position(config, epoch - 1)
    upcoming_task_index, _ = _sequential_task_position(config, epoch)
    boundary_index = completed_task_index + 1
    artifact = {
        "schema_version": 1,
        "artifact_kind": "replay_functional_kan_consolidation",
        "epoch": epoch,
        "world_model_updates": global_step,
        "boundary_index": boundary_index,
        "completed_task": {
            "index": completed_task_index,
            "name": config.esc.env_configs[completed_task_index].name,
        },
        "upcoming_task": {
            "index": upcoming_task_index,
            "name": config.esc.env_configs[upcoming_task_index].name,
        },
        "estimator": {
            "quantity": "squared local output Jacobian per Gaussian RBF coefficient",
            "replay_batches": config.residual_consolidation_batches,
            "sequence_length": config.mb_t_size,
            "sequences_per_batch": config.mb_n_size,
            "deterministic_imagination_horizon": (
                config.residual_consolidation_imagination_horizon
            ),
            "replay_capacity_and_sampling": "unchanged ARROW mixture",
            "training_rng_state_restored": True,
            "gradient_updates": 0,
            "environment_interactions": 0,
            "task_identity_exposed_to_agent": False,
        },
        "protection": {
            "cumulative_rule": "coefficient-wise maximum across boundaries",
            "anchor_rule": "replace only when the new normalized importance is larger",
            "gradient_power": config.residual_consolidation_gradient_power,
            "minimum_plasticity": config.residual_consolidation_min_plasticity,
            "anchor_loss_scale": config.residual_consolidation_anchor_loss_scale,
        },
        "modules": diagnostics,
    }
    output_dir = log_dir / "kan_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"boundary_{boundary_index:02d}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    for module_name, values in diagnostics.items():
        tag_name = module_name.replace(".", "/")
        writer.add_scalar(
            f"KANConsolidation/{tag_name}/importance_mean",
            values["importance_mean"],
            global_step,
        )
        writer.add_scalar(
            f"KANConsolidation/{tag_name}/protected_fraction_ge_0_9",
            values["protected_fraction_ge_0_9"],
            global_step,
        )
        writer.add_scalar(
            f"KANConsolidation/{tag_name}/gradient_scale_mean",
            values["gradient_scale_mean"],
            global_step,
        )
    return diagnostics


def _rec_mechanism_banks(wm: WorldModel) -> dict[str, Any]:
    if not wm.rssm.task_mechanism_bank_enabled:
        raise ValueError("REC-RSSM consolidation requires mechanism banks")
    return {
        "recurrent": wm.rssm.recurrent_mechanism_bank,
        "posterior": wm.rssm.representation_mechanism_bank,
        "prior": wm.rssm.transition_mechanism_bank,
    }


def _rec_optimizer_parameter_groups(
    wm: WorldModel, *, wm_lr: float, route_lr_scale: float
) -> list[dict[str, Any]]:
    """Keep every future mechanism in Adam while assigning routes their own LR."""
    route_parameters = [
        parameter
        for bank in _rec_mechanism_banks(wm).values()
        for route in bank.routes
        for parameter in route.parameters()
    ]
    route_parameter_ids = {id(parameter) for parameter in route_parameters}
    if len(route_parameter_ids) != len(route_parameters):
        raise RuntimeError("REC-RSSM route parameters must not be shared")
    normal_parameters = [
        parameter
        for parameter in wm.parameters()
        if id(parameter) not in route_parameter_ids
    ]
    if not normal_parameters or not route_parameters:
        raise RuntimeError("REC-RSSM optimizer requires normal and route parameters")
    return [
        {"params": normal_parameters, "lr": wm_lr},
        {"params": route_parameters, "lr": wm_lr * route_lr_scale},
    ]


def _flatten_parameter_groups(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> list[torch.nn.Parameter]:
    parameters = [parameter for values in groups.values() for parameter in values]
    if not parameters:
        raise RuntimeError("Parameter groups contain no parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("Parameter groups contain duplicate parameters")
    return parameters


_SHARED_PREDICTION_GROUP_NAMES = frozenset(
    {"observation_head", "reward_head", "continue_head"}
)


def _evolving_shared_optimizer_parameter_groups(
    wm: WorldModel,
    *,
    core_lr: float,
    prediction_head_lr: float,
) -> list[dict[str, Any]]:
    """Build persistent Adam groups without changing Dense Task-0 head LR."""

    if min(core_lr, prediction_head_lr) <= 0:
        raise ValueError("Evolving shared optimizer learning rates must be positive")
    named = wm.shared_parameter_groups()
    core = [
        parameter
        for name, parameters in named.items()
        if name not in _SHARED_PREDICTION_GROUP_NAMES
        for parameter in parameters
    ]
    prediction_heads = [
        parameter
        for name, parameters in named.items()
        if name in _SHARED_PREDICTION_GROUP_NAMES
        for parameter in parameters
    ]
    all_parameters = (*core, *prediction_heads)
    if not core or len({id(parameter) for parameter in all_parameters}) != len(
        all_parameters
    ):
        raise RuntimeError(
            "Evolving shared optimizer groups are empty or overlapping"
        )
    groups: list[dict[str, Any]] = [
        {"params": core, "lr": core_lr, "ownership": "core"}
    ]
    if (
        wm.task_shared_prediction_heads
        and not wm.freeze_shared_prediction_heads_after_task0
    ):
        if not prediction_heads:
            raise RuntimeError("Shared prediction heads have no optimizer parameters")
        groups.append(
            {
                "params": prediction_heads,
                "lr": prediction_head_lr,
                "ownership": "prediction_heads",
            }
        )
    elif prediction_heads:
        raise RuntimeError(
            "Private prediction-head topology leaked into shared optimizer groups"
        )
    return groups


def _set_evolving_shared_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    core_lr: float,
    prediction_head_lr: float,
) -> None:
    """Change core LR by task while retaining the Dense private-head LR."""

    if min(core_lr, prediction_head_lr) <= 0:
        raise ValueError("Evolving shared optimizer learning rates must be positive")
    observed = []
    for group in optimizer.param_groups:
        ownership = group.get("ownership")
        observed.append(ownership)
        if ownership == "core":
            group["lr"] = core_lr
        elif ownership == "prediction_heads":
            group["lr"] = prediction_head_lr
        else:
            raise ValueError(
                "Evolving shared optimizer lost its ownership metadata"
            )
    expected = ["core", "prediction_heads"] if len(observed) == 2 else ["core"]
    if observed != expected:
        raise ValueError(
            f"Unexpected Evolving shared optimizer groups: {observed}"
        )


def _ensure_evolving_private_optimizers(
    *,
    wm: WorldModel,
    task_id: int,
    private_optimizers: dict[int, torch.optim.Optimizer],
    route_optimizers: dict[int, torch.optim.Optimizer],
    private_lr: float,
    route_lr: float,
    fused: bool,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
    """Create each task optimizer once and retain its Adam state thereafter."""

    if task_id not in private_optimizers:
        private_parameters = wm.private_parameters(task_id)
        if not private_parameters:
            raise RuntimeError(f"Task {task_id} has no private world-model parameters")
        private_optimizers[task_id] = Adam(
            private_parameters,
            lr=private_lr,
            fused=fused,
        )
    route_parameters = wm.route_parameters(task_id)
    if route_parameters and task_id not in route_optimizers:
        route_optimizers[task_id] = Adam(
            route_parameters,
            lr=route_lr,
            fused=fused,
        )
    route_optimizer = route_optimizers.get(task_id)
    owned_ids = {
        id(parameter)
        for optimizer in (
            private_optimizers[task_id],
            route_optimizer,
        )
        if optimizer is not None
        for parameter in _optimizer_parameters(optimizer)
    }
    expected_ids = {
        id(parameter) for parameter in (*wm.private_parameters(task_id), *route_parameters)
    }
    if owned_ids != expected_ids:
        raise RuntimeError("Task-private optimizer ownership is incomplete or overlapping")
    return private_optimizers[task_id], route_optimizer


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    if learning_rate <= 0:
        raise ValueError("Optimizer learning rate must be positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _restore_sampling_rng(
    cpu_state: torch.Tensor,
    cuda_states: Optional[list[torch.Tensor]],
) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _evolving_memory_loss(
    *,
    config: Config,
    wm: WorldModel,
    teacher: WorldModel,
    frozen_actor: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    task_id: int,
    mechanism_output_scale: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute old-task Dreamer and Q/H/policy interface protection losses."""

    from clworldmodel.continual import (
        interface_distillation_losses,
        mechanism_output_distillation_losses,
        prediction_head_distillation_losses,
    )

    if mechanism_output_scale < 0:
        raise ValueError("Mechanism-output distillation scale must be non-negative")
    if hasattr(frozen_actor, "set_task_route"):
        frozen_actor.set_task_route(task_id)

    actions, observations, rewards, continues, resets = batch
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    with torch.no_grad(), _autocast_context(actions.device, config.compute_dtype):
        _teacher_loss, _teacher_metrics, teacher_trace = (
            teacher.compute_loss_and_trace(
                actions,
                observations,
                rewards,
                continues,
                resets,
                task_id=task_id,
            )
        )
    _restore_sampling_rng(cpu_state, cuda_states)
    with _autocast_context(actions.device, config.compute_dtype):
        dreamer_loss, dreamer_metrics, student_trace = wm.compute_loss_and_trace(
            actions,
            observations,
            rewards,
            continues,
            resets,
            task_id=task_id,
        )
        interface = interface_distillation_losses(
            student_trace=student_trace,
            teacher_trace=teacher_trace,
            frozen_actor=frozen_actor,
        )
        prediction_distillation = None
        mechanism_distillation = (
            mechanism_output_distillation_losses(student_trace, teacher_trace)
            if mechanism_output_scale
            else None
        )
        if config.uses_shared_prediction_heads:
            student_outputs = student_trace.get("prediction_head_outputs")
            teacher_outputs = teacher_trace.get("prediction_head_outputs")
            if not isinstance(student_outputs, Mapping) or not isinstance(
                teacher_outputs, Mapping
            ):
                raise RuntimeError(
                    "Shared prediction-head distillation requires output traces"
                )
            prediction_distillation = prediction_head_distillation_losses(
                student_outputs,
                teacher_outputs,
            )
        total = (
            dreamer_loss
            + config.interface_q_scale * interface["posterior"]
            + config.interface_h_scale * interface["hidden"]
            + config.interface_actor_scale * interface["actor"]
            + (
                config.shared_prediction_distill_scale
                * prediction_distillation["total"]
                if prediction_distillation is not None
                else 0.0
            )
            + (
                mechanism_output_scale * mechanism_distillation["total"]
                if mechanism_distillation is not None
                else 0.0
            )
        )
    metrics = {
        **dreamer_metrics,
        "Loss/evolving_interface_q": interface["posterior"].detach(),
        "Loss/evolving_interface_h": interface["hidden"].detach(),
        "Loss/evolving_interface_actor": interface["actor"].detach(),
        "Loss/evolving_memory_total": total.detach(),
    }
    if prediction_distillation is not None:
        metrics.update(
            {
                "Loss/evolving_prediction_observation_distill": (
                    prediction_distillation["observation"].detach()
                ),
                "Loss/evolving_prediction_reward_distill": (
                    prediction_distillation["reward"].detach()
                ),
                "Loss/evolving_prediction_continue_distill": (
                    prediction_distillation["continue"].detach()
                ),
                "Loss/evolving_prediction_distill_scaled": (
                    config.shared_prediction_distill_scale
                    * prediction_distillation["total"].detach()
                ),
            }
        )
    if mechanism_distillation is not None:
        metrics.update(
            {
                "Loss/evolving_qfp_recurrent_distill": (
                    mechanism_distillation["recurrent"].detach()
                ),
                "Loss/evolving_qfp_posterior_distill": (
                    mechanism_distillation["posterior"].detach()
                ),
                "Loss/evolving_qfp_prior_distill": (
                    mechanism_distillation["prior"].detach()
                ),
                "Loss/evolving_qfp_distill_scaled": (
                    mechanism_output_scale
                    * mechanism_distillation["total"].detach()
                ),
            }
        )
    return total, metrics


def _evolving_world_model_update(
    *,
    config: Config,
    wm: WorldModel,
    boundary_teacher: Optional[WorldModel],
    actor_critic_bank,
    frozen_actor: Optional[torch.nn.Module] = None,
    replay_buffer,
    current_task_id: int,
    memory_task_id: Optional[int],
    sequence_length: int,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizer: torch.optim.Optimizer,
    route_optimizer: Optional[torch.optim.Optimizer],
    materialize_diagnostics: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
    """Run one fixed-budget Evolving-Core world-model optimizer update."""

    from clworldmodel.continual import (
        assign_component_projected_gradients,
        assign_unprojected_current_gradients,
        atom_output_penalty,
    )

    current_sequences = (
        config.mb_n_size if current_task_id == 0 else config.current_batch_n
    )
    current_batch = replay_buffer.minibatch_for_task(
        current_task_id,
        sequence_length,
        current_sequences,
        source="mixed",
    )
    with _autocast_context(current_batch[0].device, config.compute_dtype):
        current_dreamer, current_metrics, current_trace = (
            wm.compute_loss_and_trace(*current_batch, task_id=current_task_id)
        )
        atom_penalty = atom_output_penalty(current_trace)
        current_loss = (
            current_dreamer
            + config.task_atom_output_regularization * atom_penalty
        )

    for optimizer in (shared_optimizer, private_optimizer, route_optimizer):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    shared_groups = wm.shared_parameter_groups()
    shared_parameters = _flatten_parameter_groups(shared_groups)
    private_parameters = list(wm.private_parameters(current_task_id))
    private_parameters.extend(wm.route_parameters(current_task_id))
    diagnostics: dict[str, Any] = {}
    memory_metrics: dict[str, torch.Tensor] = {}
    if current_task_id == 0:
        assign_unprojected_current_gradients(
            current_loss,
            (*shared_parameters, *private_parameters),
        )
    else:
        if boundary_teacher is None:
            raise RuntimeError("Old-task protection requires a boundary teacher")
        if memory_task_id is None or not 0 <= memory_task_id < current_task_id:
            raise ValueError("memory_task_id must select a previously completed task")
        memory_batch = replay_buffer.minibatch_for_task(
            memory_task_id,
            sequence_length,
            config.memory_batch_n,
            source="ltdm",
        )
        memory_actor = frozen_actor
        if memory_actor is None:
            if actor_critic_bank is None:
                raise RuntimeError(
                    "Old-task interface protection requires an actor teacher"
                )
            memory_actor = actor_critic_bank.get(memory_task_id).ac.actor
        memory_loss, memory_metrics = _evolving_memory_loss(
            config=config,
            wm=wm,
            teacher=boundary_teacher,
            frozen_actor=memory_actor,
            batch=memory_batch,
            task_id=memory_task_id,
        )
        diagnostics = assign_component_projected_gradients(
            current_loss=current_loss,
            memory_loss=memory_loss,
            shared_parameter_groups=shared_groups,
            private_parameters=private_parameters,
            memory_scale=config.memory_loss_scale,
            project_conflicts=config.component_gradient_projection,
            materialize_diagnostics=materialize_diagnostics,
        )

    active_parameters = (*shared_parameters, *private_parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1000)
    shared_optimizer.step()
    private_optimizer.step()
    if route_optimizer is not None:
        route_optimizer.step()

    metrics = {
        **current_metrics,
        **{f"Memory/{name}": value for name, value in memory_metrics.items()},
        "Loss/evolving_atom_output_penalty": atom_penalty.detach(),
        "Loss/evolving_current_total": current_loss.detach(),
    }
    return metrics, diagnostics, grad_norm


def _restore_torch_rng_state(
    cpu_state: torch.Tensor, cuda_states: Optional[list[torch.Tensor]]
) -> None:
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def _rec_loss_over_batches(
    *,
    config: Config,
    wm: WorldModel,
    task_id: int,
    batches: list[tuple[torch.Tensor, ...]],
    cpu_rng_state: torch.Tensor,
    cuda_rng_states: Optional[list[torch.Tensor]],
) -> float:
    _restore_torch_rng_state(cpu_rng_state, cuda_rng_states)
    losses = []
    for actions, observations, rewards, continues, resets in batches:
        with _autocast_context(actions.device, config.compute_dtype):
            loss, _metrics = wm.compute_loss(
                actions,
                observations,
                rewards,
                continues,
                resets,
                task_id=task_id,
            )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("REC-RSSM consolidation observed a non-finite loss")
        losses.append(float(loss.detach().float().cpu()))
    return float(np.mean(losses))


def _evaluate_rec_route(
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    task_id: int,
    env_fns,
    seed: int,
) -> tuple[float, float]:
    with _preserve_training_rng_state():
        return evaluate(
            config.n_sync,
            wm=wm,
            ac=aco.ac,
            env_fns=env_fns,
            env_repeat=config.env_repeat,
            n_rollouts=16,
            seed=seed,
            task_id=task_id,
            deterministic_policy=True,
        )


def _consolidate_rec_routes(
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    replay_buffer,
    completed_task_id: int,
    eval_env_fns,
    validation_seed: int,
    epoch: int,
    global_step: int,
    log_dir: Path,
    writer,
) -> dict[str, Any]:
    """Ablate old atoms, hard-prune weak reuse, and validate the route."""
    if config.continual_method != "rec_rssm_arrow":
        raise ValueError("REC-RSSM consolidation requires rec_rssm_arrow")
    if completed_task_id < 1:
        raise ValueError("REC-RSSM consolidates only post-Task-1 routes")

    banks = _rec_mechanism_banks(wm)
    route_index = completed_task_id - 1
    original_masks = {
        name: bank.routes[route_index].hard_mask.detach().clone()
        for name, bank in banks.items()
    }
    original_shared_masks = {
        name: bank.routes[route_index].validated_shared_mask.detach().clone()
        for name, bank in banks.items()
    }
    candidate_coordinates = [
        (name, old_index, atom_index)
        for name, mask in original_masks.items()
        for old_index in range(mask.shape[0])
        for atom_index in range(mask.shape[1])
        if bool(mask[old_index, atom_index].item())
    ]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "rec_rssm_atom_route_consolidation",
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "world_model_updates": global_step,
        "completed_task": {
            "index": completed_task_id,
            "name": config.esc.env_configs[completed_task_id].name,
        },
        "settings": {
            "num_atoms": config.task_mechanism_num_atoms,
            "replay_batches": config.task_mechanism_consolidation_batches,
            "minimum_contribution": config.task_mechanism_min_contribution,
            "maximum_validation_drop": (
                config.task_mechanism_max_validation_drop
            ),
        },
        "candidate_count": len(candidate_coordinates),
        "gradient_updates": 0,
        "training_replay_writes": 0,
        "evaluation_transitions_enter_replay": False,
    }
    if not candidate_coordinates:
        artifact.update(
            {
                "reason": "completed route has no old mechanisms to reuse",
                "candidates": [],
                "validation": None,
                "rollback": False,
                "accepted_masks": {
                    name: bank.routes[route_index]
                    .hard_mask.detach()
                    .cpu()
                    .tolist()
                    for name, bank in banks.items()
                },
                "accepted_shared_masks": {
                    name: bank.routes[route_index]
                    .validated_shared_mask.detach()
                    .cpu()
                    .tolist()
                    for name, bank in banks.items()
                },
                "route_manifest": {
                    name: bank.route_manifest(completed_task_id)
                    for name, bank in banks.items()
                },
            }
        )
    else:
        wm_was_training = wm.training
        batches: list[tuple[torch.Tensor, ...]] = []
        try:
            with _preserve_training_rng_state():
                wm.eval()
                for _ in range(config.task_mechanism_consolidation_batches):
                    batches.append(
                        tuple(
                            tensor.detach()
                            for tensor in replay_buffer.minibatch(
                                config.mb_t_size,
                                config.mb_n_size,
                                task_id=completed_task_id,
                            )
                        )
                    )
                condition_cpu_rng = torch.random.get_rng_state()
                condition_cuda_rngs = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )
                for bank in banks.values():
                    bank.begin_contribution_recording(completed_task_id)
                full_loss = _rec_loss_over_batches(
                    config=config,
                    wm=wm,
                    task_id=completed_task_id,
                    batches=batches,
                    cpu_rng_state=condition_cpu_rng,
                    cuda_rng_states=condition_cuda_rngs,
                )
                contributions = {
                    name: bank.finish_contribution_recording()
                    for name, bank in banks.items()
                }

                candidates = []
                proposed_masks = {
                    name: mask.detach().clone()
                    for name, mask in original_masks.items()
                }
                proposed_shared_masks = {
                    name: mask.detach().clone()
                    for name, mask in original_shared_masks.items()
                }
                for bank_name, old_index, atom_index in candidate_coordinates:
                    bank = banks[bank_name]
                    temporary_mask = original_masks[bank_name].detach().clone()
                    temporary_mask[old_index, atom_index] = 0
                    bank.apply_consolidated_mask(completed_task_id, temporary_mask)
                    ablated_loss = _rec_loss_over_batches(
                        config=config,
                        wm=wm,
                        task_id=completed_task_id,
                        batches=batches,
                        cpu_rng_state=condition_cpu_rng,
                        cuda_rng_states=condition_cuda_rngs,
                    )
                    bank.apply_consolidated_mask(
                        completed_task_id, original_masks[bank_name]
                    )
                    delta_loss = ablated_loss - full_loss
                    contribution = float(
                        contributions[bank_name]["contribution_ratio"][old_index][
                            atom_index
                        ]
                    )
                    should_prune = (
                        delta_loss <= 0
                        or contribution < config.task_mechanism_min_contribution
                    )
                    if should_prune:
                        proposed_masks[bank_name][old_index, atom_index] = 0
                        proposed_shared_masks[bank_name][old_index, atom_index] = 0
                    else:
                        proposed_shared_masks[bank_name][old_index, atom_index] = 1
                    candidates.append(
                        {
                            "component": bank_name,
                            "owner_task": old_index + 1,
                            "atom_index": atom_index,
                            "full_loss": full_loss,
                            "ablated_loss": ablated_loss,
                            "delta_loss": delta_loss,
                            "functional_contribution": contribution,
                            "proposed_prune": should_prune,
                        }
                    )

                full_mean, full_std = _evaluate_rec_route(
                    config=config,
                    wm=wm,
                    aco=aco,
                    task_id=completed_task_id,
                    env_fns=eval_env_fns,
                    seed=validation_seed,
                )
                for name, bank in banks.items():
                    bank.apply_consolidated_mask(
                        completed_task_id, proposed_masks[name]
                    )
                pruned_mean, pruned_std = _evaluate_rec_route(
                    config=config,
                    wm=wm,
                    aco=aco,
                    task_id=completed_task_id,
                    env_fns=eval_env_fns,
                    seed=validation_seed,
                )
                rollback = pruned_mean < (
                    (1.0 - config.task_mechanism_max_validation_drop) * full_mean
                )
                if rollback:
                    for name, bank in banks.items():
                        bank.apply_consolidated_mask(
                            completed_task_id, original_masks[name]
                        )
                        bank.apply_validated_shared_mask(
                            completed_task_id, original_shared_masks[name]
                        )
                else:
                    for name, bank in banks.items():
                        bank.apply_validated_shared_mask(
                            completed_task_id, proposed_shared_masks[name]
                        )
                reward_scale = config.esc.env_configs[completed_task_id].rew_scale
                artifact.update(
                    {
                        "full_world_model_loss": full_loss,
                        "functional_contributions": contributions,
                        "candidates": candidates,
                        "validation": {
                            "cohort": "fixed_periodic_validation",
                            "seed": validation_seed,
                            "rollouts_per_condition": 16,
                            "full_scaled_mean": full_mean,
                            "full_scaled_std": full_std,
                            "full_raw_mean": full_mean / reward_scale,
                            "full_raw_std": full_std / abs(reward_scale),
                            "pruned_scaled_mean": pruned_mean,
                            "pruned_scaled_std": pruned_std,
                            "pruned_raw_mean": pruned_mean / reward_scale,
                            "pruned_raw_std": pruned_std / abs(reward_scale),
                            "acceptance_threshold_scaled": (
                                (1.0 - config.task_mechanism_max_validation_drop)
                                * full_mean
                            ),
                        },
                        "rollback": rollback,
                        "accepted_masks": {
                            name: bank.routes[route_index]
                            .hard_mask.detach()
                            .cpu()
                            .tolist()
                            for name, bank in banks.items()
                        },
                        "accepted_shared_masks": {
                            name: bank.routes[route_index]
                            .validated_shared_mask.detach()
                            .cpu()
                            .tolist()
                            for name, bank in banks.items()
                        },
                        "route_manifest": {
                            name: bank.route_manifest(completed_task_id)
                            for name, bank in banks.items()
                        },
                    }
                )
        except Exception:
            for name, bank in banks.items():
                bank.cancel_contribution_recording()
                bank.apply_consolidated_mask(
                    completed_task_id, original_masks[name]
                )
                bank.apply_validated_shared_mask(
                    completed_task_id, original_shared_masks[name]
                )
            raise
        finally:
            wm.train(wm_was_training)

    output_dir = log_dir / "rec_rssm_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"task_{completed_task_id:02d}_boundary.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    writer.add_scalar(
        "RECRSSM/consolidation_candidate_count",
        len(candidate_coordinates),
        global_step,
    )
    writer.add_scalar(
        "RECRSSM/consolidation_rollback",
        int(bool(artifact["rollback"])),
        global_step,
    )
    return artifact


def _consolidate_evolving_shared_core(
    *,
    config: Config,
    wm: WorldModel,
    shared_optimizer: torch.optim.Optimizer,
    replay_buffer,
    actor_critic_bank,
    aco: Optional[ActorCriticOpt] = None,
    completed_task_id: int,
    eval_funcs,
    validation_task_seeds: Sequence[int],
    epoch: int,
    global_step: int,
    log_dir: Path,
    writer,
) -> dict[str, Any]:
    """Run task-balanced shared-only consolidation with whole-core rollback."""

    from clworldmodel.continual import recursive_python_scalars

    if not config.uses_evolving_atomic_rssm:
        raise ValueError("Shared consolidation requires Evolving-Core Atomic RSSM")
    if not 0 <= completed_task_id < len(eval_funcs):
        raise ValueError("completed_task_id is outside the evaluation task set")
    seen_count = completed_task_id + 1
    seen_eval_funcs = eval_funcs[:seen_count]
    seen_validation_seeds = validation_task_seeds[:seen_count]
    shared_before = wm.shared_core_state_dict()
    optimizer_before = copy.deepcopy(shared_optimizer.state_dict())
    learning_rates_before = [group["lr"] for group in shared_optimizer.param_groups]
    was_training = wm.training

    if bool(getattr(config, "uses_shared_actor", False)):
        if actor_critic_bank is not None or aco is None:
            raise ValueError(
                "Shared-behavior consolidation requires exactly one actor-critic"
            )
        evaluation_aco = aco
        evaluation_bank = None
    else:
        if actor_critic_bank is None or aco is not None:
            raise ValueError(
                "Private-behavior consolidation requires only an actor bank"
            )
        evaluation_aco = actor_critic_bank.get(completed_task_id)
        evaluation_bank = actor_critic_bank

    pre_routing, post_routing = [], []
    pre_scaled_mean, pre_scaled_std = _evaluate_policy_tasks(
        config,
        wm,
        evaluation_aco,
        seen_eval_funcs,
        seen_validation_seeds,
        actor_critic_bank=evaluation_bank,
        eligible_task_count=seen_count, routing_diagnostics=pre_routing,
    )
    pre_raw_mean, pre_raw_std = _raw_return_statistics(
        config.esc.env_configs[:seen_count], pre_scaled_mean, pre_scaled_std
    )
    # Persist the selection observation before any consolidation gradient is
    # taken.  A failed consolidation must not erase the completed online
    # acquisition result or tempt a sweep to fall back to held-out-final data.
    pre_validation = recursive_python_scalars(
        {
            "schema_version": 1,
            "artifact_kind": "evolving_core_pre_consolidation_validation",
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "completed_task_id": completed_task_id,
            "world_model_updates": global_step,
            "seed_cohort": "fixed_validation",
            "rollouts_per_task": 16,
            "validation": {
                "task_seeds": list(seen_validation_seeds),
                "scaled_mean": pre_scaled_mean,
                "scaled_std": pre_scaled_std,
                "raw_mean": pre_raw_mean,
                "raw_std": pre_raw_std,
                "routing": pre_routing,
            },
            "selection_metric": (
                "validation.raw_mean[completed_task_id]"
                if completed_task_id == 0
                else None
            ),
            "evaluation_transitions_enter_replay": False,
            "consolidation_updates_completed": 0,
            "heldout_final_data_used": False,
        }
    )
    output_dir = log_dir / "evolving_core_consolidation"
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_validation_path = output_dir / (
        f"task_{completed_task_id:02d}_pre_validation.json"
    )
    pre_validation_temporary = pre_validation_path.with_suffix(".json.tmp")
    pre_validation_temporary.write_text(
        json.dumps(pre_validation, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(pre_validation_temporary, pre_validation_path)

    losses: list[float] = []
    rollback = False
    rollback_reasons: list[dict[str, float | int]] = []
    try:
        wm.train()
        wm.activate_shared_only()
        _set_optimizer_learning_rate(
            shared_optimizer, config.boundary_consolidation_lr
        )
        shared_parameters = _flatten_parameter_groups(wm.shared_parameter_groups())
        for consolidation_index in range(config.boundary_consolidation_steps):
            task_id = consolidation_index % seen_count
            batch = replay_buffer.minibatch_for_task(
                task_id,
                config.mb_t_size,
                config.mb_n_size,
                source="ltdm",
            )
            shared_optimizer.zero_grad(set_to_none=True)
            with _autocast_context(batch[0].device, config.compute_dtype):
                loss, _metrics = wm.compute_loss(*batch, task_id=task_id)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Evolving-Core consolidation produced a non-finite loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_parameters, 1000)
            shared_optimizer.step()
            losses.append(float(loss.detach().float().cpu()))

        wm.eval()
        post_scaled_mean, post_scaled_std = _evaluate_policy_tasks(
            config,
            wm,
            evaluation_aco,
            seen_eval_funcs,
            seen_validation_seeds,
            actor_critic_bank=evaluation_bank,
            eligible_task_count=seen_count, routing_diagnostics=post_routing,
        )
        post_raw_mean, post_raw_std = _raw_return_statistics(
            config.esc.env_configs[:seen_count],
            post_scaled_mean,
            post_scaled_std,
        )
        for task_id, (before, after) in enumerate(
            zip(pre_raw_mean, post_raw_mean)
        ):
            relative_drop = (before - after) / max(abs(before), 1.0)
            if relative_drop > config.boundary_max_return_drop:
                rollback = True
                rollback_reasons.append(
                    {
                        "task_id": task_id,
                        "pre_return": before,
                        "post_return": after,
                        "relative_drop": relative_drop,
                    }
                )
        if rollback:
            wm.load_shared_core_state_dict(shared_before)
            shared_optimizer.load_state_dict(optimizer_before)
            post_scaled_mean = pre_scaled_mean
            post_scaled_std = pre_scaled_std
            post_raw_mean = pre_raw_mean
            post_raw_std = pre_raw_std
    except Exception:
        # Restore exact pre-consolidation training state before propagating the
        # failure to the boundary safety wrapper in the trainer.
        wm.load_shared_core_state_dict(shared_before)
        shared_optimizer.load_state_dict(optimizer_before)
        raise
    finally:
        for group, learning_rate in zip(
            shared_optimizer.param_groups, learning_rates_before
        ):
            group["lr"] = learning_rate
        wm.train(was_training)
        wm.activate_task_expert(completed_task_id)

    artifact = recursive_python_scalars(
        {
            "schema_version": 1,
            "artifact_kind": "evolving_core_boundary_consolidation",
            "epoch": epoch,
            "completed_task_id": completed_task_id,
            "world_model_update_start": global_step,
            "world_model_update_stop": (
                global_step + config.boundary_consolidation_steps
            ),
            "steps": config.boundary_consolidation_steps,
            "learning_rate": config.boundary_consolidation_lr,
            "sampling": "round-robin task-balanced LTDM",
            "private_modules_frozen": True,
            "shared_prediction_heads_consolidated": bool(
                getattr(config, "uses_shared_prediction_heads", False)
            ),
            "loss_mean": float(np.mean(losses)),
            "validation": {
                "task_seeds": list(seen_validation_seeds),
                "pre_scaled_mean": pre_scaled_mean,
                "pre_scaled_std": pre_scaled_std,
                "post_scaled_mean": post_scaled_mean,
                "post_scaled_std": post_scaled_std,
                "pre_raw_mean": pre_raw_mean,
                "pre_raw_std": pre_raw_std,
                "post_raw_mean": post_raw_mean,
                "post_raw_std": post_raw_std,
                "maximum_relative_drop": config.boundary_max_return_drop,
                "pre_routing": pre_routing,
                "attempted_post_routing": post_routing,
                "selected_routing": pre_routing if rollback else post_routing,
            },
            "rollback": rollback,
            "rollback_reasons": rollback_reasons,
            "evaluation_transitions_enter_replay": False,
        }
    )
    path = output_dir / f"task_{completed_task_id:02d}_boundary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    writer.add_scalar(
        "EvolvingCoreConsolidation/rollback", int(rollback), global_step
    )
    writer.add_scalar(
        "EvolvingCoreConsolidation/loss_mean", artifact["loss_mean"], global_step
    )
    return artifact


_ADAPTIVE_QFP_COMPRESSION_METHOD = (
    "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow"
)
_ADAPTIVE_QFP_AC_COMPRESSION_METHOD = (
    "evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow"
)


def _adaptive_compression_candidate_passes(
    *,
    teacher_return: float,
    candidate_return: float,
    maximum_relative_drop: float,
) -> bool:
    """Apply the fixed raw-return gate, including negative-return games."""

    if not all(
        math.isfinite(value)
        for value in (teacher_return, candidate_return, maximum_relative_drop)
    ):
        raise ValueError("Adaptive compression return gates require finite values")
    if not 0 <= maximum_relative_drop < 1:
        raise ValueError("Maximum adaptive-compression return drop must lie in [0, 1)")
    relative_drop = (teacher_return - candidate_return) / max(
        abs(teacher_return), 1.0
    )
    return relative_drop <= maximum_relative_drop + 1e-12


def _adaptive_compression_target_width(
    full_width: int, fraction: float, num_atoms: int
) -> int:
    if full_width < 1 or num_atoms < 1 or full_width % num_atoms:
        raise ValueError("Dense acquisition width must be positive and atom-divisible")
    if not 0 < fraction < 1:
        raise ValueError("Adaptive compression fraction must lie in (0, 1)")
    target = int(round(full_width * fraction))
    if target < num_atoms or target % num_atoms:
        raise ValueError(
            "Adaptive compression fraction produced a non atom-divisible width"
        )
    return target


def _adaptive_qfp_modules(
    wm: WorldModel, task_id: int
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, bank in wm.rssm.mechanism_banks().items():
        mechanism = bank.mechanism_for(task_id)
        if mechanism is None:
            raise RuntimeError(f"Task {task_id} is missing its {name} mechanism")
        modules[name] = mechanism
    return modules


def _install_adaptive_qfp_modules(
    wm: WorldModel,
    task_id: int,
    modules: Mapping[str, torch.nn.Module],
) -> list[dict[str, Any]]:
    banks = wm.rssm.mechanism_banks()
    if set(modules) != set(banks):
        raise ValueError("Adaptive Q/F/P replacement must contain all three banks")
    reference = next(wm.parameters())
    reports = []
    for name, bank in banks.items():
        module = modules[name].to(device=reference.device, dtype=reference.dtype)
        report = bank.install_task_mechanism(task_id, module)
        report["component"] = name
        reports.append(report)
    return reports


def _structured_adaptive_qfp_candidate(
    *,
    wm: WorldModel,
    dense_teacher: WorldModel,
    task_id: int,
    fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.nn.Module]]:
    from clworldmodel.models.mechanism_bank import ResidualMechanism

    student_banks = wm.rssm.mechanism_banks()
    teacher_banks = dense_teacher.rssm.mechanism_banks()
    candidates: dict[str, torch.nn.Module] = {}
    selection_reports: list[dict[str, Any]] = []
    for name, student_bank in student_banks.items():
        source = teacher_banks[name].mechanism_for(task_id)
        if not isinstance(source, ResidualMechanism):
            raise TypeError("Adaptive compression teacher must own Dense Q/F/P")
        if source.hidden_features != student_bank.hidden_features:
            raise ValueError(
                "Adaptive compression must begin from the declared full Dense width"
            )
        target_width = _adaptive_compression_target_width(
            student_bank.hidden_features,
            fraction,
            student_bank.num_atoms,
        )
        candidate, selected = ResidualMechanism.structured_pruned_copy(
            source, hidden_features=target_width
        )
        candidates[name] = candidate
        selection_bytes = ",".join(str(index) for index in selected).encode("ascii")
        selection_reports.append(
            {
                "component": name,
                "dense_hidden_features": source.hidden_features,
                "candidate_hidden_features": target_width,
                "selected_channel_count": len(selected),
                "selected_channel_sha256": hashlib.sha256(
                    selection_bytes
                ).hexdigest(),
            }
        )
    install_reports = _install_adaptive_qfp_modules(wm, task_id, candidates)
    installed_by_name = {report["component"]: report for report in install_reports}
    for report in selection_reports:
        report.update(installed_by_name[report["component"]])
    return selection_reports, _adaptive_qfp_modules(wm, task_id)


def _evaluate_adaptive_compression_task(
    *,
    config: Config,
    wm: WorldModel,
    actor_critic_bank=None,
    aco: Optional[ActorCriticOpt] = None,
    task_id: int,
    eval_env_fns,
    validation_seed: int,
    eligible_task_count: Optional[int] = None,
) -> dict[str, Any]:
    if (actor_critic_bank is None) == (aco is None):
        raise ValueError(
            "Adaptive Q/F/P evaluation requires exactly one behavior topology"
        )
    diagnostic = {}
    inference_kwargs = {"task_id": task_id}
    if getattr(config, "uses_reconstruction_task_inference", False):
        behavior = _autorouted_behavior(config, aco, actor_critic_bank, eligible_task_count)
        inference_kwargs = {
            "eligible_route_ids": tuple(range(eligible_task_count)),
            "max_agent_decisions_per_episode": config.evaluation_max_agent_decisions_per_episode,
            "diagnostics": diagnostic,
        }
    else:
        task_aco = actor_critic_bank.get(task_id) if actor_critic_bank is not None else aco
        task_aco.ac.set_task_route(task_id)
        behavior = task_aco.ac
    with _preserve_training_rng_state():
        scaled_mean, scaled_std = evaluate(
            config.n_sync,
            wm=wm,
            ac=behavior,
            env_fns=eval_env_fns,
            env_repeat=config.env_repeat,
            n_rollouts=config.adaptive_compression_rollouts,
            seed=validation_seed,
            deterministic_policy=True,
            **inference_kwargs,
        )
    raw_mean, raw_std = _raw_return_statistics(
        [config.esc.env_configs[task_id]],
        [scaled_mean],
        [scaled_std],
    )
    result = {
        "scaled_mean": float(scaled_mean),
        "scaled_std": float(scaled_std),
        "raw_mean": raw_mean[0],
        "raw_std": raw_std[0],
    }
    if diagnostic:
        from clworldmodel.routing import routing_audit

        diagnostic["audit"] = routing_audit(
            diagnostic["routing_events"], true_task_id=task_id,
            task_count=config.rssm_num_experts,
        )
        result["inference"] = diagnostic
    return result


def _adaptive_qfp_validation_gate(teacher, candidate, maximum_relative_drop):
    """All-seen auto-routed gates cannot be replaced by a current-task average."""
    teacher_rows = teacher.get("seen_task_validation", [teacher])
    candidate_rows = candidate.get("seen_task_validation", [candidate])
    if not teacher_rows or len(teacher_rows) != len(candidate_rows):
        raise ValueError("Teacher and candidate validation task sets must match")
    drops = []
    passed = True
    for before, after in zip(teacher_rows, candidate_rows):
        if before.get("task_id") != after.get("task_id"):
            raise ValueError("Validation task order must match")
        accepted = _adaptive_compression_candidate_passes(
            teacher_return=before["raw_mean"], candidate_return=after["raw_mean"],
            maximum_relative_drop=maximum_relative_drop,
        )
        passed = passed and accepted
        drops.append((before["raw_mean"] - after["raw_mean"]) / max(abs(before["raw_mean"]), 1.0))
    return passed, drops


def _compress_evolving_task_qfp(
    *,
    config: Config,
    wm: WorldModel,
    replay_buffer,
    actor_critic_bank,
    aco: Optional[ActorCriticOpt] = None,
    completed_task_id: int,
    eval_env_fns,
    validation_seed: int,
    epoch: int,
    global_step: int,
    log_dir: Path,
    writer,
    fused_adam: bool,
    seen_eval_env_fns=None,
    seen_validation_seeds: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Train/evaluate fixed compact candidates and install the smallest pass.

    Every candidate starts from the same post-consolidation Dense teacher, gets
    the same LTDM recovery budget, and is evaluated on a dedicated pruning
    cohort.  All candidates are attempted even after a failure so optimizer
    compute is fixed rather than performance-dependent.  Held-out final seeds
    never enter this selection procedure.
    """

    from clworldmodel.continual import recursive_python_scalars

    if config.continual_method not in {
        _ADAPTIVE_QFP_COMPRESSION_METHOD,
        _ADAPTIVE_QFP_AC_COMPRESSION_METHOD,
        "evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow",
    }:
        raise ValueError("Adaptive Q/F/P compression requires its named method")
    if (actor_critic_bank is None) == (aco is None):
        raise ValueError(
            "Adaptive Q/F/P compression requires exactly one behavior topology"
        )
    if not 0 <= completed_task_id < len(config.esc.env_configs):
        raise ValueError("Completed task is outside the adaptive curriculum")
    if wm.rssm.task_mechanism_parameterization != "adaptive_dense_width":
        raise ValueError("World model is not an adaptive dense-width topology")

    autoroute = getattr(config, "uses_reconstruction_task_inference", False)
    if autoroute and (
        seen_eval_env_fns is None or seen_validation_seeds is None
        or len(seen_eval_env_fns) != completed_task_id + 1
        or len(seen_validation_seeds) != completed_task_id + 1
    ):
        raise ValueError("Auto-routed compression must validate every seen task")

    def evaluate_condition(model):
        if not autoroute:
            return _evaluate_adaptive_compression_task(
                config=config, wm=model, actor_critic_bank=actor_critic_bank, aco=aco,
                task_id=completed_task_id, eval_env_fns=eval_env_fns,
                validation_seed=validation_seed,
            )
        rows = []
        for audit_id, (functions, seed) in enumerate(zip(seen_eval_env_fns, seen_validation_seeds)):
            row = _evaluate_adaptive_compression_task(
                config=config, wm=model, actor_critic_bank=actor_critic_bank,
                aco=aco, task_id=audit_id,
                eval_env_fns=functions, validation_seed=seed,
                eligible_task_count=completed_task_id + 1,
            )
            rows.append({"task_id": audit_id, **row})
        result = dict(rows[completed_task_id])
        result["seen_task_validation"] = rows
        return result

    was_training = wm.training
    dense_teacher = copy.deepcopy(wm).eval()
    dense_teacher.requires_grad_(False)
    dense_modules = {
        name: copy.deepcopy(module)
        for name, module in _adaptive_qfp_modules(
            dense_teacher, completed_task_id
        ).items()
    }
    dense_layout = dense_teacher.rssm.adaptive_compression_layout()
    expected_dense_widths = {
        "recurrent": config.task_mechanism_recurrent_width,
        "posterior": config.task_mechanism_representation_width,
        "prior": config.task_mechanism_transition_width,
    }
    observed_dense_widths = {
        name: values[completed_task_id]
        for name, values in dense_layout.items()
    }
    if observed_dense_widths != expected_dense_widths:
        raise ValueError(
            "The just-completed task was not acquired at full Dense width: "
            f"{observed_dense_widths} != {expected_dense_widths}"
        )

    world_model_parameters_before = sum(
        parameter.numel() for parameter in wm.parameters()
    )
    candidates: list[dict[str, Any]] = []
    best_modules: dict[str, torch.nn.Module] | None = None
    best_fraction = 1.0
    best_evaluation: dict[str, float] | None = None
    optimizer_updates = 0
    try:
        # The adaptive phase must not change the online trainer's Python,
        # NumPy/replay, or torch sampling streams. Its own fixed computation is
        # fully reflected in counters and artifacts instead.
        with _preserve_training_rng_state():
            wm.eval()
            dense_evaluation = evaluate_condition(dense_teacher)
            candidate_python_state = random.getstate()
            candidate_numpy_state = np.random.get_state()
            candidate_torch_state = torch.random.get_rng_state()
            candidate_cuda_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            source_aco = (
                actor_critic_bank.get(completed_task_id)
                if actor_critic_bank is not None
                else aco
            )
            frozen_actor = copy.deepcopy(source_aco.ac.actor).eval()
            if hasattr(frozen_actor, "set_task_route"):
                frozen_actor.set_task_route(completed_task_id)
            frozen_actor.requires_grad_(False)
            for fraction in config.adaptive_compression_width_fractions:
                # Equal update counts are not sufficient for a controlled width
                # comparison: give each candidate the exact same LTDM indices
                # and stochastic-latent draws as well.
                random.setstate(candidate_python_state)
                np.random.set_state(candidate_numpy_state)
                _restore_sampling_rng(
                    candidate_torch_state,
                    candidate_cuda_states,
                )
                selection, installed = _structured_adaptive_qfp_candidate(
                    wm=wm,
                    dense_teacher=dense_teacher,
                    task_id=completed_task_id,
                    fraction=float(fraction),
                )
                wm.requires_grad_(False)
                trainable = [
                    parameter
                    for module in installed.values()
                    for parameter in module.parameters()
                ]
                for parameter in trainable:
                    parameter.requires_grad_(True)
                optimizer = Adam(
                    trainable,
                    lr=config.adaptive_compression_lr,
                    fused=fused_adam,
                )
                losses: list[float] = []
                for _ in range(config.adaptive_compression_steps_per_candidate):
                    batch = replay_buffer.minibatch_for_task(
                        completed_task_id,
                        config.mb_t_size,
                        config.mb_n_size,
                        source="ltdm",
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss, _metrics = _evolving_memory_loss(
                        config=config,
                        wm=wm,
                        teacher=dense_teacher,
                        frozen_actor=frozen_actor,
                        batch=batch,
                        task_id=completed_task_id,
                        mechanism_output_scale=(
                            config.adaptive_compression_qfp_distill_scale
                        ),
                    )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            "Adaptive Q/F/P compression produced a non-finite loss"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1000)
                    optimizer.step()
                    optimizer_updates += 1
                    losses.append(float(loss.detach().float().cpu()))
                wm.eval()
                evaluation = evaluate_condition(wm)
                relative_drop = (
                    dense_evaluation["raw_mean"] - evaluation["raw_mean"]
                ) / max(abs(dense_evaluation["raw_mean"]), 1.0)
                passed, task_drops = _adaptive_qfp_validation_gate(
                    dense_evaluation, evaluation, config.adaptive_compression_max_return_drop,
                )
                candidate_record = {
                    "width_fraction": float(fraction),
                    "components": selection,
                    "optimizer_updates": len(losses),
                    "distillation_loss_mean": float(np.mean(losses)),
                    "distillation_loss_first": losses[0],
                    "distillation_loss_last": losses[-1],
                    "validation": evaluation,
                    "relative_raw_return_drop": relative_drop,
                    "per_validation_task_relative_raw_return_drop": task_drops,
                    "passed": passed,
                    "world_model_parameters": sum(
                        parameter.numel() for parameter in wm.parameters()
                    ),
                }
                candidates.append(candidate_record)
                if passed:
                    best_modules = {
                        name: copy.deepcopy(module)
                        for name, module in _adaptive_qfp_modules(
                            wm, completed_task_id
                        ).items()
                    }
                    best_fraction = float(fraction)
                    best_evaluation = evaluation

            expected_optimizer_updates = (
                len(config.adaptive_compression_width_fractions)
                * config.adaptive_compression_steps_per_candidate
            )
            if optimizer_updates != expected_optimizer_updates:
                raise RuntimeError(
                    "Adaptive Q/F/P compression did not execute its fixed "
                    f"candidate budget: {optimizer_updates} != "
                    f"{expected_optimizer_updates}"
                )
            selected_modules = dense_modules if best_modules is None else best_modules
            _install_adaptive_qfp_modules(
                wm, completed_task_id, selected_modules
            )
        selected_evaluation = (
            dense_evaluation if best_evaluation is None else best_evaluation
        )
        selected_layout = wm.rssm.adaptive_compression_layout()
        world_model_parameters_after = sum(
            parameter.numel() for parameter in wm.parameters()
        )
    except Exception:
        _install_adaptive_qfp_modules(wm, completed_task_id, dense_modules)
        raise
    finally:
        wm.train(was_training)
        wm.activate_task_expert(completed_task_id)

    artifact = recursive_python_scalars(
        {
            "schema_version": 1,
            "artifact_kind": "evolving_core_return_gated_qfp_compression",
            "method": config.continual_method,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "completed_task_id": completed_task_id,
            "world_model_update_start": global_step,
            "world_model_update_stop": global_step + optimizer_updates,
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": (
                len(config.adaptive_compression_width_fractions)
                * config.adaptive_compression_steps_per_candidate
            ),
            "candidate_compute_is_fixed": True,
            "candidate_sampling_stream": (
                "identical restored Python/NumPy/torch state for every width"
            ),
            "candidate_initialization": (
                "independent structured channel pruning from one frozen full-width "
                "post-consolidation Dense teacher"
            ),
            "recovery_replay": "completed-task LTDM only",
            "recovery_sequences_per_update": config.mb_n_size,
            "learning_rate": config.adaptive_compression_lr,
            "qfp_output_distillation_scale": (
                config.adaptive_compression_qfp_distill_scale
            ),
            "seed_cohort": "fixed_pruning_validation",
            "validation_seed": validation_seed,
            "all_seen_validation_seeds": list(seen_validation_seeds) if autoroute else None,
            "validation_policy": "auto_routed_all_seen" if autoroute else "oracle_current_task",
            "rollouts_per_evaluation": config.adaptive_compression_rollouts,
            "dense_teacher_validation": dense_evaluation,
            "maximum_relative_raw_return_drop": (
                config.adaptive_compression_max_return_drop
            ),
            "candidates": candidates,
            "selected_width_fraction": best_fraction,
            "selected_dense_fallback": best_modules is None,
            "selected_validation": selected_evaluation,
            "dense_layout": dense_layout,
            "selected_layout": selected_layout,
            "world_model_parameters_before": world_model_parameters_before,
            "world_model_parameters_after": world_model_parameters_after,
            "world_model_parameters_removed": (
                world_model_parameters_before - world_model_parameters_after
            ),
            "training_only_dense_teacher_discarded": True,
            "completed_task_private_optimizer_retirable": True,
            "evaluation_transitions_enter_replay": False,
            "heldout_final_data_used": False,
        }
    )
    output_dir = log_dir / "adaptive_qfp_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"task_{completed_task_id:02d}_boundary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    writer.add_scalar(
        "AdaptiveQFP/selected_width_fraction", best_fraction, global_step
    )
    writer.add_scalar(
        "AdaptiveQFP/world_model_parameters_removed",
        artifact["world_model_parameters_removed"],
        global_step,
    )
    return artifact


class _TaskLtdmReplayView:
    """Expose one completed task's LTDM partition through Replay.minibatch.

    Actor-Critic compression is an offline boundary phase.  Its imagined
    trajectories may use real replay only to infer their initial latent state,
    and this view prevents an accidental fallback to FIFO or another task.
    """

    def __init__(self, replay_buffer, task_id: int) -> None:
        if task_id < 0:
            raise ValueError("LTDM replay views require a non-negative task id")
        self.replay_buffer = replay_buffer
        self.task_id = int(task_id)

    def minibatch(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str | torch.device = "cuda",
        task_id: Optional[int] = None,
    ):
        if task_id is not None and int(task_id) != self.task_id:
            raise ValueError(
                "A completed-task LTDM view cannot serve another task route"
            )
        return self.replay_buffer.minibatch_for_task(
            self.task_id,
            mb_t,
            mb_n,
            source="ltdm",
            mb_device=mb_device,
        )


def _adaptive_behavior_modules(
    actor_critic: torch.nn.Module, task_id: int
) -> dict[str, torch.nn.Module]:
    if not getattr(actor_critic, "adaptive_behavior_residuals", False):
        raise ValueError("Actor-Critic does not own adaptive behavior residuals")
    modules: dict[str, torch.nn.Module] = {}
    for name in ("actor", "critic"):
        head = getattr(actor_critic, name)
        modules[name] = head.mechanism_for(task_id)
    return modules


def _install_adaptive_behavior_modules(
    actor_critic: torch.nn.Module,
    task_id: int,
    modules: Mapping[str, torch.nn.Module],
) -> list[dict[str, Any]]:
    if set(modules) != {"actor", "critic"}:
        raise ValueError(
            "Adaptive behavior replacement requires actor and critic residuals"
        )
    reference = next(actor_critic.parameters())
    reports: list[dict[str, Any]] = []
    for name in ("actor", "critic"):
        head = getattr(actor_critic, name)
        module = modules[name].to(device=reference.device, dtype=reference.dtype)
        report = head.install_task_mechanism(task_id, module)
        report["component"] = name
        reports.append(report)
    actor_critic.set_task_route(task_id)
    return reports


def _structured_adaptive_behavior_candidate(
    *,
    actor_critic: torch.nn.Module,
    dense_teacher: torch.nn.Module,
    task_id: int,
    fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.nn.Module]]:
    from clworldmodel.models.mechanism_bank import ResidualMechanism

    candidates: dict[str, torch.nn.Module] = {}
    selection_reports: list[dict[str, Any]] = []
    for name in ("actor", "critic"):
        student_head = getattr(actor_critic, name)
        teacher_head = getattr(dense_teacher, name)
        source = teacher_head.mechanism_for(task_id)
        if not isinstance(source, ResidualMechanism):
            raise TypeError(
                "Adaptive behavior compression teacher must own Dense residuals"
            )
        if source.hidden_features != student_head.hidden_features:
            raise ValueError(
                "Adaptive behavior compression must begin at the declared full width"
            )
        target_width = _adaptive_compression_target_width(
            student_head.hidden_features,
            fraction,
            student_head.num_atoms,
        )
        candidate, selected = ResidualMechanism.structured_pruned_copy(
            source, hidden_features=target_width
        )
        candidates[name] = candidate
        selection_bytes = ",".join(str(index) for index in selected).encode("ascii")
        selection_reports.append(
            {
                "component": name,
                "dense_hidden_features": source.hidden_features,
                "candidate_hidden_features": target_width,
                "selected_channel_count": len(selected),
                "selected_channel_sha256": hashlib.sha256(
                    selection_bytes
                ).hexdigest(),
            }
        )
    install_reports = _install_adaptive_behavior_modules(
        actor_critic, task_id, candidates
    )
    installed_by_name = {report["component"]: report for report in install_reports}
    for report in selection_reports:
        report.update(installed_by_name[report["component"]])
    return selection_reports, _adaptive_behavior_modules(actor_critic, task_id)


def _evaluate_adaptive_behavior_task(
    *,
    config: Config,
    wm: WorldModel,
    actor_critic: torch.nn.Module,
    task_id: int,
    eval_env_fns,
    validation_seed: int,
) -> dict[str, float]:
    actor_critic.set_task_route(task_id)
    with _preserve_training_rng_state():
        scaled_mean, scaled_std = evaluate(
            config.n_sync,
            wm=wm,
            ac=actor_critic,
            env_fns=eval_env_fns,
            env_repeat=config.env_repeat,
            n_rollouts=config.adaptive_behavior_rollouts,
            seed=validation_seed,
            task_id=task_id,
            deterministic_policy=True,
        )
    raw_mean, raw_std = _raw_return_statistics(
        [config.esc.env_configs[task_id]],
        [scaled_mean],
        [scaled_std],
    )
    return {
        "scaled_mean": float(scaled_mean),
        "scaled_std": float(scaled_std),
        "raw_mean": raw_mean[0],
        "raw_std": raw_std[0],
    }


def _compress_evolving_task_actor_critic(
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    replay_buffer,
    completed_task_id: int,
    eval_env_fns,
    validation_seed: int,
    epoch: int,
    actor_critic_updates: int,
    compression_updates_before: int,
    log_dir: Path,
    writer,
    fused_adam: bool,
) -> dict[str, Any]:
    """Compress the completed task's Actor/Critic residuals behind a raw gate.

    The shared MLP bases, older residuals, and reuse routes are frozen.  Every
    fixed-width candidate is independently structured-pruned from the same
    full-width post-task teacher, receives the same imagined-state
    distillation budget, and is evaluated on one dedicated real-environment
    cohort.  Failure of all candidates keeps the original Dense residuals.
    """

    from clworldmodel.continual import recursive_python_scalars

    if config.continual_method != _ADAPTIVE_QFP_AC_COMPRESSION_METHOD:
        raise ValueError("Adaptive Actor-Critic compression requires its named method")
    if not config.uses_adaptive_behavior_compression:
        raise ValueError("Adaptive Actor-Critic compression is not enabled")
    if not 0 <= completed_task_id < len(config.esc.env_configs):
        raise ValueError("Completed task is outside the adaptive curriculum")
    if not getattr(aco.ac, "adaptive_behavior_residuals", False):
        raise ValueError("Actor-Critic is not an adaptive residual topology")
    if aco.slow_critic is not None:
        raise ValueError(
            "Adaptive Actor-Critic compression does not support a second slow "
            "critic topology"
        )

    ac_was_training = aco.ac.training
    wm_was_training = wm.training
    dense_teacher = copy.deepcopy(aco.ac).eval()
    dense_teacher.requires_grad_(False)
    # Retain the actual objects so Dense fallback preserves their online Adam
    # moments.  Candidates and the frozen teacher are separate copies.
    dense_modules = _adaptive_behavior_modules(aco.ac, completed_task_id)
    dense_layout = dense_teacher.adaptive_behavior_layout()
    observed_dense_widths = {
        name: widths[completed_task_id] for name, widths in dense_layout.items()
    }
    expected_dense_widths = {
        "actor": config.adaptive_behavior_hidden_features,
        "critic": config.adaptive_behavior_hidden_features,
    }
    if observed_dense_widths != expected_dense_widths:
        raise ValueError(
            "The just-completed Actor-Critic residuals were not acquired at full "
            f"Dense width: {observed_dense_widths} != {expected_dense_widths}"
        )

    parameters_before = sum(parameter.numel() for parameter in aco.ac.parameters())
    candidates: list[dict[str, Any]] = []
    best_modules: dict[str, torch.nn.Module] | None = None
    best_fraction = 1.0
    best_evaluation: dict[str, float] | None = None
    optimizer_updates = 0
    imagined_states = 0
    replay_view = _TaskLtdmReplayView(replay_buffer, completed_task_id)
    try:
        with _preserve_training_rng_state():
            wm.eval()
            aco.ac.eval()
            dense_evaluation = _evaluate_adaptive_behavior_task(
                config=config,
                wm=wm,
                actor_critic=dense_teacher,
                task_id=completed_task_id,
                eval_env_fns=eval_env_fns,
                validation_seed=validation_seed,
            )
            candidate_python_state = random.getstate()
            candidate_numpy_state = np.random.get_state()
            candidate_torch_state = torch.random.get_rng_state()
            candidate_cuda_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            for fraction in config.adaptive_behavior_width_fractions:
                random.setstate(candidate_python_state)
                np.random.set_state(candidate_numpy_state)
                _restore_sampling_rng(
                    candidate_torch_state,
                    candidate_cuda_states,
                )
                selection, installed = _structured_adaptive_behavior_candidate(
                    actor_critic=aco.ac,
                    dense_teacher=dense_teacher,
                    task_id=completed_task_id,
                    fraction=float(fraction),
                )
                aco.ac.requires_grad_(False)
                trainable = [
                    parameter
                    for module in installed.values()
                    for parameter in module.parameters()
                ]
                for parameter in trainable:
                    parameter.requires_grad_(True)
                optimizer = Adam(
                    trainable,
                    lr=config.adaptive_behavior_lr,
                    fused=fused_adam,
                )
                actor_losses: list[float] = []
                critic_losses: list[float] = []
                total_losses: list[float] = []
                for _ in range(config.adaptive_behavior_steps_per_candidate):
                    with torch.no_grad():
                        dense_teacher.set_task_route(completed_task_id)
                        states, *_ = dream_rollout(
                            wm,
                            dense_teacher,
                            replay_view,
                            n_sync=config.mb_n_size,
                            n_steps=config.ac_dream_steps,
                            discount=config.ac_discount,
                            lam=config.ac_lambda,
                            n_ctx_frames=4,
                            task_id=completed_task_id,
                        )
                        teacher_actor_logs = dense_teacher.actor(states).float()
                        teacher_critic_logs = dense_teacher.critic(states).float()
                    aco.ac.set_task_route(completed_task_id)
                    optimizer.zero_grad(set_to_none=True)
                    actor_loss = actor_policy_kl(
                        aco.ac.actor, states, teacher_actor_logs
                    )
                    critic_loss = actor_policy_kl(
                        aco.ac.critic, states, teacher_critic_logs
                    )
                    loss = (
                        config.adaptive_behavior_actor_distill_scale * actor_loss
                        + config.adaptive_behavior_critic_distill_scale * critic_loss
                    )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            "Adaptive Actor-Critic compression produced a "
                            "non-finite distillation loss"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, config.ac_grad_clip)
                    optimizer.step()
                    optimizer_updates += 1
                    imagined_states += int(states.shape[0] * states.shape[1])
                    actor_losses.append(float(actor_loss.detach().float().cpu()))
                    critic_losses.append(float(critic_loss.detach().float().cpu()))
                    total_losses.append(float(loss.detach().float().cpu()))

                aco.ac.eval()
                evaluation = _evaluate_adaptive_behavior_task(
                    config=config,
                    wm=wm,
                    actor_critic=aco.ac,
                    task_id=completed_task_id,
                    eval_env_fns=eval_env_fns,
                    validation_seed=validation_seed,
                )
                relative_drop = (
                    dense_evaluation["raw_mean"] - evaluation["raw_mean"]
                ) / max(abs(dense_evaluation["raw_mean"]), 1.0)
                passed = _adaptive_compression_candidate_passes(
                    teacher_return=dense_evaluation["raw_mean"],
                    candidate_return=evaluation["raw_mean"],
                    maximum_relative_drop=(
                        config.adaptive_behavior_max_return_drop
                    ),
                )
                candidates.append(
                    {
                        "width_fraction": float(fraction),
                        "components": selection,
                        "optimizer_updates": len(total_losses),
                        "imagined_states": int(
                            len(total_losses)
                            * config.mb_n_size
                            * config.ac_dream_steps
                        ),
                        "actor_kl_mean": float(np.mean(actor_losses)),
                        "actor_kl_first": actor_losses[0],
                        "actor_kl_last": actor_losses[-1],
                        "critic_categorical_kl_mean": float(
                            np.mean(critic_losses)
                        ),
                        "critic_categorical_kl_first": critic_losses[0],
                        "critic_categorical_kl_last": critic_losses[-1],
                        "distillation_loss_mean": float(np.mean(total_losses)),
                        "validation": evaluation,
                        "relative_raw_return_drop": relative_drop,
                        "passed": passed,
                        "actor_critic_parameters": sum(
                            parameter.numel()
                            for parameter in aco.ac.parameters()
                        ),
                    }
                )
                if passed:
                    best_modules = {
                        name: copy.deepcopy(module)
                        for name, module in _adaptive_behavior_modules(
                            aco.ac, completed_task_id
                        ).items()
                    }
                    best_fraction = float(fraction)
                    best_evaluation = evaluation

            expected_optimizer_updates = (
                len(config.adaptive_behavior_width_fractions)
                * config.adaptive_behavior_steps_per_candidate
            )
            expected_imagined_states = (
                expected_optimizer_updates
                * config.mb_n_size
                * config.ac_dream_steps
            )
            if optimizer_updates != expected_optimizer_updates:
                raise RuntimeError(
                    "Adaptive Actor-Critic compression did not execute its fixed "
                    f"candidate budget: {optimizer_updates} != "
                    f"{expected_optimizer_updates}"
                )
            if imagined_states != expected_imagined_states:
                raise RuntimeError(
                    "Adaptive Actor-Critic compression imagined-state accounting "
                    f"changed: {imagined_states} != {expected_imagined_states}"
                )
            selected_modules = (
                dense_modules if best_modules is None else best_modules
            )
            _install_adaptive_behavior_modules(
                aco.ac, completed_task_id, selected_modules
            )
        selected_evaluation = (
            dense_evaluation if best_evaluation is None else best_evaluation
        )
        selected_layout = aco.ac.adaptive_behavior_layout()
        parameters_after = sum(
            parameter.numel() for parameter in aco.ac.parameters()
        )
    except Exception:
        _install_adaptive_behavior_modules(
            aco.ac, completed_task_id, dense_modules
        )
        raise
    finally:
        aco.ac.train(ac_was_training)
        wm.train(wm_was_training)
        # Nothing is optimized between this boundary and the next task
        # activation.  Keep the completed residual genuinely frozen; the next
        # epoch's ``activate_training_task`` reopens only its new route.
        aco.ac.requires_grad_(False)
        aco.ac.set_task_route(completed_task_id)
        _refresh_actor_critic_optimizer_parameters(aco)

    artifact = recursive_python_scalars(
        {
            "schema_version": 1,
            "artifact_kind": (
                "evolving_core_return_gated_actor_critic_residual_compression"
            ),
            "method": config.continual_method,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "completed_task_id": completed_task_id,
            "online_actor_critic_update_count": actor_critic_updates,
            "behavior_compression_update_start": compression_updates_before,
            "behavior_compression_update_stop": (
                compression_updates_before + optimizer_updates
            ),
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": (
                len(config.adaptive_behavior_width_fractions)
                * config.adaptive_behavior_steps_per_candidate
            ),
            "imagined_states": imagined_states,
            "expected_imagined_states": (
                len(config.adaptive_behavior_width_fractions)
                * config.adaptive_behavior_steps_per_candidate
                * config.mb_n_size
                * config.ac_dream_steps
            ),
            "candidate_compute_is_fixed": True,
            "candidate_sampling_stream": (
                "identical restored Python/NumPy/torch state for every width"
            ),
            "candidate_initialization": (
                "independent structured channel pruning from one frozen full-width "
                "post-task Actor-Critic residual teacher"
            ),
            "shared_actor_critic_bases_frozen_during_compression": True,
            "older_task_residuals_and_routes_frozen_during_compression": True,
            "recovery_state_context": "completed-task LTDM only",
            "recovery_context_frames": 4,
            "recovery_sequences_per_update": config.mb_n_size,
            "dream_steps_per_update": config.ac_dream_steps,
            "learning_rate": config.adaptive_behavior_lr,
            "actor_policy_kl_scale": (
                config.adaptive_behavior_actor_distill_scale
            ),
            "critic_categorical_kl_scale": (
                config.adaptive_behavior_critic_distill_scale
            ),
            "seed_cohort": "fixed_behavior_pruning_validation",
            "validation_seed": validation_seed,
            "rollouts_per_evaluation": config.adaptive_behavior_rollouts,
            "dense_teacher_validation": dense_evaluation,
            "maximum_relative_raw_return_drop": (
                config.adaptive_behavior_max_return_drop
            ),
            "candidates": candidates,
            "selected_width_fraction": best_fraction,
            "selected_dense_fallback": best_modules is None,
            "selected_validation": selected_evaluation,
            "dense_layout": dense_layout,
            "selected_layout": selected_layout,
            "actor_critic_parameters_before": parameters_before,
            "actor_critic_parameters_after": parameters_after,
            "actor_critic_parameters_removed": parameters_before - parameters_after,
            "training_only_dense_teacher_discarded": True,
            "completed_task_residual_frozen_after_boundary": True,
            "evaluation_transitions_enter_replay": False,
            "heldout_final_data_used": False,
        }
    )
    output_dir = log_dir / "adaptive_behavior_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"task_{completed_task_id:02d}_boundary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    writer.add_scalar(
        "AdaptiveBehavior/selected_width_fraction",
        best_fraction,
        actor_critic_updates,
    )
    writer.add_scalar(
        "AdaptiveBehavior/actor_critic_parameters_removed",
        artifact["actor_critic_parameters_removed"],
        actor_critic_updates,
    )
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_analysis_snapshot(
    snapshot_dir: Path,
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    epoch: int,
    world_model_updates: int,
    total_env_steps: int,
    reason: str,
    task_metadata: Optional[dict] = None,
    dream_rehearsal_actor_updates: int = 0,
) -> Path:
    """Save portable weights for offline diagnosis, not resumable training state."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if reason == "task_boundary":
        if task_metadata is None:
            raise ValueError("task-boundary snapshot requires task metadata")
        filename = (
            f"boundary_{task_metadata['boundary_index']:02d}_"
            f"task_{task_metadata['task_index']:02d}_epoch_{epoch:04d}.pt"
        )
    elif reason == "milestone":
        filename = f"milestone_completed_{epoch + 1:04d}_epoch_{epoch:04d}.pt"
    elif reason == "final":
        filename = f"final_epoch_{epoch:04d}.pt"
    else:
        raise ValueError(f"Unknown analysis snapshot reason: {reason}")

    path = snapshot_dir / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis snapshot: {path}")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "artifact_kind": "analysis_snapshot",
        "resumable": False,
        "reason": reason,
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "world_model_updates": world_model_updates,
        # Preserve the historical field as the matched base Dreamer update
        # counter.  Rehearsal is actor-only and is accounted separately so a
        # downstream report cannot silently mistake the extra compute for the
        # baseline budget.
        "actor_critic_updates": (epoch + 1) * config.ac_train_steps,
        "dream_rehearsal_actor_updates": dream_rehearsal_actor_updates,
        "actor_updates_total": (
            (epoch + 1) * config.ac_train_steps
            + dream_rehearsal_actor_updates
        ),
        "total_raw_environment_frames": total_env_steps,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "task": task_metadata,
        "config": config.to_dict(),
        "world_model_state_dict": _cpu_state_dict(wm),
        "actor_critic_state_dict": _cpu_state_dict(aco.ac),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

    digest = _sha256(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest_path = digest_path.with_suffix(digest_path.suffix + ".tmp")
    temporary_digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary_digest_path, digest_path)
    print(
        f"[analysis-snapshot] reason={reason} epoch={epoch} "
        f"path={path} sha256={digest}"
    )
    return path


def _save_task_bank_evaluation_snapshot(
    snapshot_dir: Path,
    *,
    config: Config,
    wm: WorldModel,
    actor_critic_bank,
    aco: Optional[ActorCriticOpt] = None,
    completed_epochs: int,
    world_model_updates: int,
    actor_critic_updates: int,
    total_env_steps: int,
    task_seeds: Sequence[int],
    scaled_means: Sequence[float],
    scaled_stds: Sequence[float],
    raw_means: Sequence[float],
    raw_stds: Sequence[float],
    cohort: str,
    eligible_task_count: Optional[int] = None,
) -> Path:
    """Save the exact task-bank weights evaluated by a fixed seed cohort."""
    uses_shared_actor = config.uses_shared_actor
    if getattr(config, "uses_reconstruction_task_inference", False) and (
        eligible_task_count is None or not 1 <= eligible_task_count <= config.rssm_num_experts
    ):
        raise ValueError("Auto-routed inference snapshots must persist acquired route eligibility")
    if uses_shared_actor:
        if actor_critic_bank is not None or aco is None:
            raise ValueError(
                "Shared-actor evaluation snapshots require exactly one actor-critic"
            )
    elif actor_critic_bank is None:
        raise ValueError("Task-bank evaluation snapshots require an actor bank")
    lengths = {
        len(task_seeds),
        len(scaled_means),
        len(scaled_stds),
        len(raw_means),
        len(raw_stds),
    }
    if len(lengths) != 1:
        raise ValueError("Evaluation snapshot metrics must have matching lengths")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{cohort}_completed_{completed_epochs:04d}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation snapshot: {path}")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "artifact_kind": "task_bank_evaluation_snapshot",
        "resumable": False,
        "cohort": cohort,
        "completed_epochs": completed_epochs,
        "world_model_updates": world_model_updates,
        "actor_critic_updates": actor_critic_updates,
        "total_raw_environment_frames": total_env_steps,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "task_base_seeds": [int(seed) for seed in task_seeds],
        "evaluation": [
            {
                "task_index": task_index,
                "scaled_return_mean": float(scaled_mean),
                "scaled_return_std": float(scaled_std),
                "raw_return_mean": float(raw_mean),
                "raw_return_std": float(raw_std),
            }
            for task_index, (
                scaled_mean,
                scaled_std,
                raw_mean,
                raw_std,
            ) in enumerate(zip(scaled_means, scaled_stds, raw_means, raw_stds))
        ],
        "omitted_state": [
            "optimizers",
            "replay",
            "RNG",
            "environment schedule",
            "step schedulers",
        ],
        "config": config.to_dict(),
        "world_model_state_dict": _cpu_state_dict(wm),
        "actor_topology": (
            "single_shared_actor_critic"
            if uses_shared_actor
            else "per_task_actor_critic_bank"
        ),
    }
    if getattr(config, "uses_reconstruction_task_inference", False):
        payload["inference_routing"] = {
            "mode": config.task_route_inference,
            "eligible_route_ids": list(range(eligible_task_count)),
            "task_identity_input": False,
        }
    if uses_shared_actor:
        payload["actor_critic_state_dict"] = _cpu_state_dict(aco.ac)
    else:
        payload["actor_critic_bank_state_dict"] = (
            actor_critic_bank.inference_state_dict()
        )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    digest = _sha256(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest_path = digest_path.with_suffix(digest_path.suffix + ".tmp")
    temporary_digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary_digest_path, digest_path)
    print(
        "[evaluation-snapshot] "
        f"cohort={cohort} completed_epochs={completed_epochs} path={path} "
        f"sha256={digest}"
    )
    return path


def _save_task_bank_boundary_snapshot(
    snapshot_dir: Path,
    *,
    config: Config,
    wm: WorldModel,
    actor_critic_bank,
    aco: Optional[ActorCriticOpt] = None,
    epoch: int,
    world_model_updates: int,
    total_env_steps: int,
    task_metadata: dict,
    project_git_commit: str,
) -> Path:
    """Save one complete task bank immediately after a task's final update."""
    uses_shared_actor = config.uses_shared_actor
    if uses_shared_actor:
        if actor_critic_bank is not None or aco is None:
            raise ValueError(
                "Shared-actor boundary snapshots require exactly one actor-critic"
            )
    elif actor_critic_bank is None:
        raise ValueError("Task-bank boundary snapshots require an actor bank")
    if len(project_git_commit) != 40:
        raise ValueError("Project Git commit must be a full 40-character hash")
    try:
        int(project_git_commit, 16)
    except ValueError as exc:
        raise ValueError("Project Git commit must be hexadecimal") from exc
    required_task_fields = {
        "boundary_index",
        "task_index",
        "task_name",
        "task_reward_scale",
    }
    if not required_task_fields.issubset(task_metadata):
        raise ValueError("Task-boundary metadata is incomplete")
    task_id = int(task_metadata["task_index"])
    completed_actor = aco if uses_shared_actor else actor_critic_bank.get(task_id)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    completed_epochs = epoch + 1
    path = snapshot_dir / (
        f"boundary_{int(task_metadata['boundary_index']):02d}_"
        f"task_{task_id:02d}_completed_{completed_epochs:04d}.pt"
    )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite task-boundary snapshot: {path}")
    index_path = snapshot_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("project_git_commit") != project_git_commit:
            raise RuntimeError("Task-boundary snapshot index commit changed mid-run")
        snapshots = index.get("snapshots")
        if not isinstance(snapshots, list):
            raise ValueError("Task-boundary snapshot index has invalid snapshots")
        if any(
            int(record.get("boundary_index", -1))
            == int(task_metadata["boundary_index"])
            for record in snapshots
        ):
            raise FileExistsError(
                "Refusing to duplicate task-boundary index entry: "
                f"{task_metadata['boundary_index']}"
            )
    else:
        index = {
            "schema_version": 1,
            "artifact_kind": "task_bank_boundary_snapshot_index",
            "project_git_commit": project_git_commit,
            "resumable": False,
            "snapshots": [],
        }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "artifact_kind": "task_bank_boundary_inference_snapshot",
        "resumable": False,
        "saved_after_final_task_update_before_schedule_advance": True,
        "project_git_commit": project_git_commit,
        "epoch": epoch,
        "completed_epochs": completed_epochs,
        "world_model_updates": world_model_updates,
        "actor_critic_updates": completed_epochs * config.ac_train_steps,
        "total_raw_environment_frames": total_env_steps,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "completed_task": dict(task_metadata),
        "omitted_state": [
            "optimizers",
            "replay",
            "RNG",
            "environment schedule",
            "step schedulers",
        ],
        "config": config.to_dict(),
        "world_model_state_dict": _cpu_state_dict(wm),
        "actor_topology": (
            "single_shared_actor_critic"
            if uses_shared_actor
            else "per_task_actor_critic_bank"
        ),
    }
    if getattr(config, "uses_reconstruction_task_inference", False):
        payload["inference_routing"] = {
            "mode": config.task_route_inference,
            "eligible_route_ids": list(range(task_id + 1)),
            "task_identity_input": False,
        }
    if uses_shared_actor:
        payload["actor_critic_state_dict"] = _cpu_state_dict(completed_actor.ac)
    else:
        payload["actor_critic_bank_state_dict"] = (
            actor_critic_bank.inference_state_dict()
        )
        payload["completed_task_actor_critic_state_dict"] = _cpu_state_dict(
            completed_actor.ac
        )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

    digest = _sha256(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest_path = digest_path.with_suffix(digest_path.suffix + ".tmp")
    temporary_digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary_digest_path, digest_path)

    index["snapshots"].append(
        {
            "boundary_index": int(task_metadata["boundary_index"]),
            "task_index": task_id,
            "task_name": str(task_metadata["task_name"]),
            "completed_epochs": completed_epochs,
            "path": path.name,
            "sha256": digest,
        }
    )
    temporary_index_path = index_path.with_suffix(".json.tmp")
    temporary_index_path.write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_index_path, index_path)
    print(
        "[task-boundary-snapshot] "
        f"boundary={task_metadata['boundary_index']} task={task_id} "
        f"completed_epochs={completed_epochs} path={path} sha256={digest}"
    )
    return path


def _write_best_validation_snapshot(
    snapshot_dir: Path,
    *,
    snapshot_path: Path,
    completed_epochs: int,
    seen_task_count: int,
    seen_task_raw_mean: float,
) -> None:
    payload = {
        "schema_version": 1,
        "selection_data": "fixed_periodic_validation_cohort",
        "selection_rule": "maximum_mean_raw_return_over_seen_tasks",
        "final_evaluation_data_used": False,
        "snapshot": snapshot_path.name,
        "completed_epochs": completed_epochs,
        "seen_task_count": seen_task_count,
        "seen_task_raw_return_mean": seen_task_raw_mean,
    }
    path = snapshot_dir / "best_validation_snapshot.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _init_swanlab(
    project: Optional[str], experiment_name: Optional[str], config: Config
):
    if project is None:
        return None
    try:
        import swanlab
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--swanlab-project requires the optional swanlab package"
        ) from exc
    swanlab.sync_tensorboard_torch()
    return swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=config.to_dict(),
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument(
        "--arrow-replay-ratio",
        choices=["50-50", "25-75", "75-25"],
        default=None,
        help="ARROW: optional FIFO/LTDM capacity split override.",
    )
    parser.add_argument(
        "--observation-objective",
        choices=[
            "reconstruction",
            "r2",
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        ],
        default=None,
        help="Optional world-model observation-objective override.",
    )
    parser.add_argument(
        "--actor-network",
        choices=[
            "mlp",
            "relu_kan",
            "relu_kan_bounded",
            "relu_kan_adaptive",
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
            "fast_kan_ac_stable",
        ],
        default=None,
        help=(
            "Optional behavior architecture override; FastKAN variants replace both "
            "actor and critic, while ReLU-KAN variants replace only the actor."
        ),
    )
    parser.add_argument(
        "--actor-kan-trainable-grid",
        action="store_true",
        default=None,
        help="Enable learned ReLU-KAN basis anchors for relu_kan_adaptive only.",
    )
    parser.add_argument("--r2-barlow-loss-scale", type=float, default=None)
    parser.add_argument("--r2-redundancy-scale", type=float, default=None)
    parser.add_argument("--r2-normalization-eps", type=float, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional explicit training-epoch override for a named pilot protocol.",
    )
    parser.add_argument("--compile-world-model", action="store_true")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Synchronize at stage boundaries and print per-epoch wall times.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Explicit run directory for TensorBoard events and resolved config.",
    )
    parser.add_argument(
        "--analysis-snapshot-dir",
        type=Path,
        help=(
            "Save portable world-model and actor-critic weights at task boundaries "
            "and at training end. These snapshots do not contain replay, optimizers, "
            "or RNG state and are not resumable checkpoints."
        ),
    )
    parser.add_argument(
        "--evaluation-snapshot-dir",
        type=Path,
        help=(
            "Save the exact task-bank inference weights used for each periodic "
            "fixed-cohort evaluation and the held-out final evaluation. These "
            "snapshots omit replay, optimizers, RNG, and schedule state."
        ),
    )
    parser.add_argument(
        "--task-bank-snapshot-dir",
        type=Path,
        help=(
            "Save the complete world-model and Actor-Critic bank immediately "
            "after every task's final update. These inference snapshots omit "
            "replay, optimizers, RNG, and schedule state and are not resumable."
        ),
    )
    parser.add_argument(
        "--project-git-commit",
        help="Full project commit embedded in each task-boundary snapshot.",
    )
    parser.add_argument(
        "--init-analysis-snapshot",
        type=Path,
        help=(
            "Initialize a task-acquisition run from a non-resumable analysis "
            "snapshot. Replay, optimizer, RNG, and schedule state are reset."
        ),
    )
    parser.add_argument(
        "--init-task1-boundary-snapshot",
        type=Path,
        help=(
            "Seed a named CNN projector method from a completed Task-1 "
            "CNN-FullBank inference boundary. Replay, optimizers, RNG, and "
            "the environment schedule are deliberately restarted at Task 2."
        ),
    )
    parser.add_argument(
        "--init-evolving-task0-transition-checkpoint",
        type=Path,
        help=(
            "Initialize the named atomic-LoRA shared-head method at Task 1 from "
            "the exact post-Task-0 learned-base Evolving-Core checkpoint. This "
            "preserves Task-0 weights/replay/behavior/counters/RNG but resets "
            "world-model optimizers because future-task ownership changes."
        ),
    )
    parser.add_argument(
        "--resume-adaptation-mode",
        choices=sorted(RESUME_ADAPTATION_MODES),
        default=None,
        help=(
            "When initializing from a snapshot, train only KAN residuals or "
            "also open the small latent/behavior readout heads."
        ),
    )
    parser.add_argument(
        "--milestone-completed-epoch",
        action="append",
        type=int,
        default=[],
        help=(
            "Evaluate and save a diagnostic snapshot after this many completed "
            "epochs; may be supplied more than once."
        ),
    )
    parser.add_argument(
        "--swanlab-project",
        help="Optionally mirror TensorBoard scalars to this SwanLab project.",
    )
    parser.add_argument(
        "--swanlab-experiment-name",
        help="Optional SwanLab experiment name; credentials come only from SwanLab.",
    )
    parser.add_argument(
        "--evaluate-final",
        action="store_true",
        help="Evaluate the final frozen policy after all configured training epochs.",
    )
    args = parser.parse_args()

    save_nets = False
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    log_images = False
    analysis_snapshot_dir = (
        args.analysis_snapshot_dir.resolve()
        if args.analysis_snapshot_dir is not None
        else None
    )
    evaluation_snapshot_dir = (
        args.evaluation_snapshot_dir.resolve()
        if args.evaluation_snapshot_dir is not None
        else None
    )
    task_bank_snapshot_dir = (
        args.task_bank_snapshot_dir.resolve()
        if args.task_bank_snapshot_dir is not None
        else None
    )
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    if args.config is not None:
        config = Config.from_file(Path(args.config))
    else:
        raise ValueError("--config is required")

    config_overrides = config.to_dict()
    if args.arrow_replay_ratio is not None:
        config_overrides["arrow_replay_capacity_ratio"] = args.arrow_replay_ratio
    if args.observation_objective is not None:
        config_overrides["observation_objective"] = args.observation_objective
    if args.actor_network is not None:
        config_overrides["actor_network"] = args.actor_network
    if args.actor_kan_trainable_grid is not None:
        config_overrides["actor_kan_trainable_grid"] = args.actor_kan_trainable_grid
    if args.r2_barlow_loss_scale is not None:
        config_overrides["r2_barlow_loss_scale"] = args.r2_barlow_loss_scale
    if args.r2_redundancy_scale is not None:
        config_overrides["r2_redundancy_scale"] = args.r2_redundancy_scale
    if args.r2_normalization_eps is not None:
        config_overrides["r2_normalization_eps"] = args.r2_normalization_eps
    if args.epochs is not None:
        config_overrides["epochs"] = args.epochs
    config = Config.from_dict(config_overrides)
    from clworldmodel.distributed import (
        DistributedContext,
        DistributedReplaySampler,
    )

    distributed_context = DistributedContext.initialize(
        config.data_parallel_world_size
    )
    device = distributed_context.device
    if distributed_context.enabled and args.compile_world_model:
        raise ValueError(
            "--compile-world-model is not yet validated with multi-GPU DDP"
        )
    if distributed_context.enabled and log_dir is None:
        raise ValueError("multi-GPU training requires an explicit --log-dir")
    _require_cuda_compute_support(config.compute_dtype)
    if config.uses_evolving_atomic_rssm and args.compile_world_model:
        raise ValueError(
            "Evolving-Core component-wise autograd is not compatible with "
            "--compile-world-model"
        )
    if config.uses_task_experts and analysis_snapshot_dir is not None:
        raise ValueError(
            "Task-bank analysis snapshots are disabled until replay, all actor "
            "optimizers, and task-router state are saved resumably"
        )
    if config.uses_bounded_dream_rehearsal and log_dir is None:
        raise ValueError(
            "Bounded Dream Rehearsal requires --log-dir for fixed-capacity "
            "mapped replay and update accounting"
        )
    if evaluation_snapshot_dir is not None:
        if not config.uses_task_experts:
            raise ValueError(
                "Evaluation snapshots are currently implemented only for task-bank methods"
            )
        if config.evaluation_seed_protocol != "fixed_validation_heldout_final":
            raise ValueError(
                "Evaluation snapshots require fixed validation and held-out final seeds"
            )
    if config.continual_method in {
        "cnn_fullbank_arrow",
        "cnn_projector_lora_arrow",
        "cnn_compact_shared_actor_arrow",
        "cnn_mechanism_bank_arrow",
        "rec_rssm_arrow",
        "evolving_atomic_rssm_arrow",
        "evolving_atomic_rssm_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow",
        "evolving_atomic_rssm_atomic_lora_shared_heads_arrow",
        "evolving_atomic_rssm_learned_base_adapters_arrow",
        "evolving_atomic_rssm_shared_fastkan_arrow",
    }:
        if task_bank_snapshot_dir is None:
            raise ValueError(
                "CNN-FullBank-ARROW requires --task-bank-snapshot-dir"
            )
        if args.project_git_commit is None:
            raise ValueError(
                "CNN-FullBank-ARROW task snapshots require --project-git-commit"
            )
    elif task_bank_snapshot_dir is not None:
        raise ValueError(
            "Task-bank boundary snapshots are currently required only for "
            "CNN task-bank methods"
        )
    milestone_completed_epochs = set(args.milestone_completed_epoch)
    invalid_milestones = sorted(
        epoch
        for epoch in milestone_completed_epochs
        if epoch < 1 or epoch > config.epochs
    )
    if invalid_milestones:
        raise ValueError(
            "Milestone completed epochs must lie within the configured run: "
            f"{invalid_milestones}"
        )

    resume_payload = None
    task1_seed_payload = None
    task0_transition_payload = None
    task0_transition_metadata: dict[str, Any] = {}
    training_start_epoch = 0
    resume_mode = args.resume_adaptation_mode
    initialization_modes = sum(
        value is not None
        for value in (
            args.init_analysis_snapshot,
            args.init_task1_boundary_snapshot,
            args.init_evolving_task0_transition_checkpoint,
        )
    )
    if initialization_modes > 1:
        raise ValueError("Only one snapshot initialization mode may be selected")
    if args.init_analysis_snapshot is not None:
        if resume_mode is None:
            resume_mode = "kan_only"
        resume_payload = _load_analysis_snapshot(
            args.init_analysis_snapshot.expanduser().resolve()
        )
        if config.residual_correction != "kan":
            raise ValueError(
                "Snapshot adaptation currently requires residual_correction='kan'"
            )
        if config.fresh_ac:
            raise ValueError(
                "Snapshot adaptation requires fresh_ac=False so the loaded actor "
                "is preserved"
            )
        if config.shared_core_mode != "snapshot_adaptation":
            raise ValueError(
                "Snapshot initialization requires shared_core_mode="
                "snapshot_adaptation"
            )
    elif resume_mode is not None:
        raise ValueError("--resume-adaptation-mode requires --init-analysis-snapshot")
    elif config.shared_core_mode == "snapshot_adaptation":
        raise ValueError(
            "shared_core_mode=snapshot_adaptation requires --init-analysis-snapshot"
        )
    if args.init_task1_boundary_snapshot is not None:
        if config.continual_method not in {
            "cnn_projector_lora_arrow",
            "cnn_compact_shared_actor_arrow",
            "cnn_mechanism_bank_arrow",
            "rec_rssm_arrow",
        }:
            raise ValueError(
                "Task-1 boundary initialization requires "
                "a named CNN projector continual method"
            )
        task1_seed_payload = _load_task1_boundary_snapshot(
            args.init_task1_boundary_snapshot.expanduser().resolve()
        )
        source_config = task1_seed_payload.get("config")
        if not isinstance(source_config, Mapping) or source_config.get(
            "continual_method"
        ) != "cnn_fullbank_arrow":
            raise ValueError(
                "Task-1 incremental training must be seeded by CNN-FullBank"
            )
        training_start_epoch = int(task1_seed_payload["completed_epochs"])
        first_task_duration = _sequential_task_durations(config)[0]
        if training_start_epoch != first_task_duration:
            raise ValueError(
                "Task-1 snapshot completion must equal one task duration: "
                f"{training_start_epoch} != {first_task_duration}"
            )
        if config.epochs <= training_start_epoch:
            raise ValueError(
                "Incremental training must include at least one post-Task-1 epoch"
            )
    elif config.continual_method == "cnn_compact_shared_actor_arrow":
        raise ValueError(
            "CNN-Compact-SharedActor requires --init-task1-boundary-snapshot"
        )
    elif config.continual_method in {"cnn_mechanism_bank_arrow", "rec_rssm_arrow"}:
        raise ValueError(
            "Mechanism-bank methods require --init-task1-boundary-snapshot"
        )
    elif config.continual_method == "cnn_projector_lora_arrow":
        training_start_epoch = 0
    if args.init_evolving_task0_transition_checkpoint is not None:
        task0_transition_payload, task0_transition_metadata = (
            _load_evolving_task0_transition_checkpoint(
                args.init_evolving_task0_transition_checkpoint,
                config=config,
            )
        )
        training_start_epoch = int(
            task0_transition_payload["schedule"]["completed_epochs"]
        )
        if config.epochs <= training_start_epoch:
            raise ValueError(
                "Task-0 transition training must include at least one later-task epoch"
            )
    elif config.continual_method == _ATOMIC_LORA_SHARED_HEADS_METHOD:
        raise ValueError(
            "The v1 atomic-LoRA shared-head pilot requires "
            "--init-evolving-task0-transition-checkpoint"
        )

    if config.algorithm == "arrow":
        print(f"ARROW FIFO/LTDM capacity ratio: {config.arrow_replay_capacity_ratio}")
    print(f"World-model observation objective: {config.observation_objective}")
    print(f"Observation encoder: {config.observation_encoder}")
    print(f"Residual correction: {config.residual_correction}")
    print(f"Residual input mode: {config.residual_input_mode}")
    print(f"Residual consolidation: {config.residual_consolidation}")
    print(f"Shared core mode: {config.shared_core_mode}")
    print(f"Continual method: {config.continual_method}")
    print(
        "Data parallel execution: "
        f"world_size={distributed_context.world_size} rank={distributed_context.rank} "
        f"local_rank={distributed_context.local_rank} device={device} "
        "global_batch_unchanged=True"
    )
    if config.uses_bounded_dream_rehearsal:
        print(
            "Bounded Dream Rehearsal: "
            f"reservoir_slots={config.sac_dv3_data_n_max} "
            f"sequence_length={config.data_t} actor=single_shared "
            "task_id_network_input=False actor_only=True "
            f"interval_decisions={config.dream_rehearsal_interval_agent_decisions} "
            f"updates_per_prior_task="
            f"{config.dream_rehearsal_updates_per_prior_task}"
        )
    if config.continual_method == "moe_arrow":
        print(
            "MoE-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "warm_start=previous_task_once current_fraction="
            f"{config.moe_arrow_current_task_fraction}"
        )
    elif config.continual_method == "cnn_fullbank_arrow":
        print(
            "CNN-FullBank-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "world_model_warm_start=previous_task_once actor_init=fresh "
            "visual_encoder=per_task_dreamerv3_cnn observation=pixels "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method == "cnn_projector_lora_arrow":
        print(
            "CNN-Projector-LoRA-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "base=task0_frozen_after_acquisition encoder=task0_plus_projector "
            "rssm=task0_plus_lora actor_init=fresh "
            f"ranks={config.task_lora_recurrent_rank}/"
            f"{config.task_lora_representation_rank}/"
            f"{config.task_lora_transition_rank} "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method == "cnn_compact_shared_actor_arrow":
        print(
            "CNN-Compact-SharedActor-ARROW routing: "
            f"experts={config.rssm_num_experts} actor=single_shared "
            "base=task0_frozen_after_acquisition encoder=task0_plus_projector "
            "recurrent=gru_output_adapter "
            f"adapter_sizes={config.task_recurrent_output_adapter_features}/"
            f"{config.task_lora_representation_rank}/"
            f"{config.task_lora_transition_rank} "
            "old_policy_retention=frozen_route_imagination "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method in {
        "evolving_atomic_rssm_arrow",
        "evolving_atomic_rssm_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow",
        "evolving_atomic_rssm_atomic_lora_shared_heads_arrow",
        "evolving_atomic_rssm_learned_base_adapters_arrow",
    }:
        prediction_topology = (
            "frozen_task0_base_heads_plus_private_feature_adapters"
            if config.task_private_prediction_adapters
            else "single_shared_decoder_reward_continue"
            if config.uses_shared_prediction_heads
            else "per_task_decoder_reward_continue"
        )
        private_topology = (
            "projector_qfp_low_rank_prediction_adapters_actor_critic"
            if config.task_private_prediction_adapters
            else "projector_dense_acquire_return_gated_compact_qfp_and_behavior"
            if config.uses_adaptive_behavior_compression
            else "projector_dense_acquire_return_gated_compact_qfp_actor_critic"
            if config.uses_adaptive_qfp_compression
            else "projector_qfp_atoms_actor_critic"
        )
        behavior_topology = (
            "single_shared_mlp_plus_task_adaptive_residuals"
            if config.uses_adaptive_behavior_compression
            else "single_shared_fastkan_stable"
            if config.uses_replay_rehearsed_shared_behavior
            else "per_task_mlp_bank"
        )
        print(
            "Evolving-Core Atomic RSSM routing: "
            f"tasks={config.rssm_num_experts} behavior={behavior_topology} "
            "base=continually_updated_shared_cnn_rssm "
            f"prediction_heads={prediction_topology} "
            f"private={private_topology} "
            f"widths={config.task_mechanism_recurrent_width}/"
            f"{config.task_mechanism_representation_width}/"
            f"{config.task_mechanism_transition_width} "
            f"parameterization={config.task_mechanism_parameterization} "
            f"reuse={config.task_mechanism_reuse}"
        )
    elif config.continual_method == "cnn_mechanism_bank_arrow":
        print(
            "CNN-MechanismBank-ARROW routing: "
            f"tasks={config.rssm_num_experts} actor_bank=per_task "
            "base=task0_frozen_after_acquisition encoder=task0_plus_projector "
            "rssm=shared_base_plus_residual_mechanisms actor_init=fresh "
            f"widths={config.task_mechanism_recurrent_width}/"
            f"{config.task_mechanism_representation_width}/"
            f"{config.task_mechanism_transition_width} "
            f"residual_scale={config.task_mechanism_residual_scale} "
            f"reuse={config.task_mechanism_reuse} "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method == "rec_rssm_arrow":
        print(
            "REC-RSSM routing: "
            f"tasks={config.rssm_num_experts} actor_bank=per_task "
            "base=task0_frozen_after_acquisition encoder=task0_plus_projector "
            "rssm=reuse_expand_consolidate actor_init=fresh "
            f"widths={config.task_mechanism_recurrent_width}/"
            f"{config.task_mechanism_representation_width}/"
            f"{config.task_mechanism_transition_width} "
            f"atoms={config.task_mechanism_num_atoms} "
            f"parameterization={config.task_mechanism_parameterization} "
            f"reuse_probe_epochs={config.task_mechanism_reuse_probe_epochs} "
            f"route_lr_scale={config.task_mechanism_route_lr_scale} "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.uses_evolving_atomic_rssm:
        behavior_topology = (
            "single_shared_mlp_plus_task_adaptive_residuals"
            if config.uses_adaptive_behavior_compression
            else "single_shared_fastkan_stable"
            if config.uses_replay_rehearsed_shared_behavior
            else "per_task_mlp_bank"
        )
        print(
            "Evolving-Core Atomic RSSM routing: "
            f"tasks={config.rssm_num_experts} behavior={behavior_topology} "
            "core=continually_updated_cnn_and_base_rssm "
            "private=projector_atoms_and_heads "
            f"mechanism_parameterization="
            f"{config.task_mechanism_parameterization} "
            f"behavior_current_fraction="
            f"{config.evolving_shared_behavior_current_task_fraction}"
        )
    elif config.continual_method == "dino_fullbank_arrow":
        print(
            "DINO-FullBank-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "world_model_warm_start=previous_task_once actor_init=fresh "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method == "dino_patchbank_arrow":
        print(
            "DINO-PatchBank-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "world_model_warm_start=previous_task_once actor_init=fresh "
            "visual_input=complete_16x16x384_patches "
            "feature_source=on_the_fly_from_replay_observations "
            "observation=pixels "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    elif config.continual_method == "dino_convbank_arrow":
        print(
            "DINO-ConvBank-ARROW routing: "
            f"experts={config.rssm_num_experts} actor_bank=per_task "
            "world_model_warm_start=previous_task_once actor_init=fresh "
            "visual_input=complete_16x16x384_patches "
            "shared_adapter=conv3x3_stride2_384to64 "
            "posterior_embedding=8x8x64 observation=pixels "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
        )
    if resume_payload is not None:
        print(
            "Initializing from analysis snapshot: "
            f"{args.init_analysis_snapshot} mode={resume_mode}"
        )
    print(f"Actor network: {config.actor_network}")
    print(
        "Actor-critic training: "
        f"optimizer={config.ac_optimizer} lr={config.ac_lr} "
        f"schedule={config.ac_schedule} final_lr={config.ac_final_lr} "
        f"final_entropy_scale={config.ac_final_entropy_scale} "
        f"dream_steps={config.ac_dream_steps} agc={config.ac_agc_clip} "
        f"grad_clip={config.ac_grad_clip}"
    )

    if config.algorithm == "sac":
        exit(0)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    (
        collection_environment_seed_rng,
        validation_environment_seed_rng,
        final_environment_seed_rng,
    ) = _environment_seed_streams(config.seed)
    fixed_evaluation_cohorts = (
        config.evaluation_seed_protocol == "fixed_validation_heldout_final"
    )
    if fixed_evaluation_cohorts:
        for _ in range(config.evaluation_task_seed_offset):
            _next_environment_seed(validation_environment_seed_rng)
            _next_environment_seed(final_environment_seed_rng)
    validation_task_seeds = (
        tuple(
            _next_environment_seed(validation_environment_seed_rng)
            for _ in config.esc.env_configs
        )
        if fixed_evaluation_cohorts
        else ()
    )
    final_task_seeds = (
        tuple(
            _next_environment_seed(final_environment_seed_rng)
            for _ in config.esc.env_configs
        )
        if fixed_evaluation_cohorts
        else ()
    )
    compression_validation_task_seeds = (
        _adaptive_compression_task_seeds(
            config.seed, len(config.esc.env_configs)
        )
        if config.uses_adaptive_qfp_compression
        else ()
    )
    behavior_compression_validation_task_seeds = (
        _adaptive_behavior_compression_task_seeds(
            config.seed, len(config.esc.env_configs)
        )
        if config.uses_adaptive_behavior_compression
        else ()
    )
    print("Training with seed: ", config.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _print_cuda_memory("startup")
    wm = WorldModel(
        3,
        (32, 32),
        config.action_space,
        config.gru_units,
        config.cnn_depth,
        config.mlp_features,
        config.mlp_layers,
        config.wall_time_optimisation,
        compute_dtype=config.compute_dtype,
        observation_objective=config.observation_objective,
        r2_barlow_loss_scale=config.r2_barlow_loss_scale,
        r2_redundancy_scale=config.r2_redundancy_scale,
        r2_normalization_eps=config.r2_normalization_eps,
        observation_encoder=config.observation_encoder,
        dinov3_model_path=config.dinov3_model_path,
        dinov3_input_size=config.dinov3_input_size,
        dinov3_max_batch_size=config.dinov3_max_batch_size,
        dinov3_feature_loss_scale=config.dinov3_feature_loss_scale,
        dinov3_feature_mode=config.dinov3_feature_mode,
        dinov3_patch_pool_size=config.dinov3_patch_pool_size,
        dinov3_patch_feature_dim=config.dinov3_patch_feature_dim,
        dinov3_patch_projection=config.dinov3_patch_projection,
        dinov3_patch_projection_seed=config.dinov3_patch_projection_seed,
        dinov3_patch_adapter=config.dinov3_patch_adapter,
        dinov3_feature_loss_kind=config.dinov3_feature_loss_kind,
        dinov3_feature_std_floor=config.dinov3_feature_std_floor,
        residual_correction=config.residual_correction,
        residual_bottleneck_features=config.residual_bottleneck_features,
        residual_grid_size=config.residual_grid_size,
        residual_input_min=config.residual_input_min,
        residual_input_max=config.residual_input_max,
        residual_rms_norm_epsilon=config.residual_rms_norm_epsilon,
        residual_alpha=config.residual_alpha,
        residual_input_mode=config.residual_input_mode,
        residual_consolidation=config.residual_consolidation,
        num_task_experts=config.rssm_num_experts,
        full_task_experts=config.uses_full_task_experts,
        full_task_rssm_experts=config.uses_full_task_rssm_experts,
        task_private_heads=config.uses_task_private_heads,
        task_shared_prediction_heads=config.uses_shared_prediction_heads,
        task_private_prediction_adapters=(
            config.task_private_prediction_adapters
        ),
        prediction_adapter_rank=config.prediction_adapter_rank,
        prediction_adapter_residual_scale=(
            config.prediction_adapter_residual_scale
        ),
        freeze_shared_prediction_heads_after_task0=(
            config.freeze_shared_prediction_heads_after_task0
        ),
        evolving_shared_core=config.evolving_shared_core,
        task_banked_image_encoder=config.task_banked_image_encoder,
        task_projected_image_encoder=config.task_projected_image_encoder,
        task_symmetric_image_projectors=config.task_atomic_routes,
        task_projector_bottleneck_features=(
            config.task_projector_bottleneck_features
        ),
        task_lora_recurrent_rank=config.task_lora_recurrent_rank,
        task_lora_representation_rank=config.task_lora_representation_rank,
        task_lora_transition_rank=config.task_lora_transition_rank,
        task_recurrent_output_adapter_features=(
            config.task_recurrent_output_adapter_features
        ),
        task_mechanism_bank=config.task_mechanism_bank,
        task_mechanism_reuse=config.task_mechanism_reuse,
        task_mechanism_recurrent_width=config.task_mechanism_recurrent_width,
        task_mechanism_representation_width=(
            config.task_mechanism_representation_width
        ),
        task_mechanism_transition_width=config.task_mechanism_transition_width,
        task_mechanism_residual_scale=config.task_mechanism_residual_scale,
        task_mechanism_num_atoms=config.task_mechanism_num_atoms,
        task_mechanism_parameterization=(
            config.task_mechanism_parameterization
        ),
        task_mechanism_low_rank=config.task_mechanism_low_rank,
        task_symmetric_mechanisms=config.task_atomic_routes,
    ).to(device)
    resume_world_model_opened: list[str] = []
    resume_state_report: dict[str, dict[str, list[str]]] = {}
    task1_seed_world_model_report: dict[str, int] = {}
    task0_transition_world_model_report: dict[str, Any] = {}
    if resume_payload is not None:
        resume_state_report["world_model"] = _load_snapshot_state(
            wm,
            resume_payload["world_model_state_dict"],
            label="World-model",
        )
        resume_world_model_opened = _configure_resume_world_model(
            wm,
            str(resume_mode),
        )
        print(
            "Loaded world-model weights; KAN residuals are plastic. "
            f"Opened shared readouts: {resume_world_model_opened or 'none'}"
        )
    elif task1_seed_payload is not None:
        task1_seed_world_model_report = _seed_task1_world_model_from_fullbank(
            wm, task1_seed_payload
        )
        print(
            "Loaded the completed Task-1 CNN/RSSM/heads; later routes remain "
            "zero-effect projector/RSSM adaptations"
        )
    elif task0_transition_payload is not None:
        task0_transition_world_model_report = (
            _seed_atomic_lora_task0_world_model(
                wm, task0_transition_payload["world_model"]
            )
        )
        print(
            "Loaded the exact Task-0 shared/core/dense-QFP/head state; future "
            "Rank-128 atomic Q/F/P residuals and routes keep target initialization"
        )
    evolving_shared_optimizer: Optional[torch.optim.Optimizer] = None
    evolving_private_optimizers: dict[int, torch.optim.Optimizer] = {}
    evolving_route_optimizers: dict[int, torch.optim.Optimizer] = {}
    if config.uses_evolving_atomic_rssm:
        evolving_shared_optimizer = Adam(
            _evolving_shared_optimizer_parameter_groups(
                wm,
                core_lr=config.first_task_shared_core_lr,
                prediction_head_lr=config.task_private_lr,
            ),
            fused=args.fused_adam,
        )
        # Keep ``opt`` as the shared optimizer for generic accounting paths;
        # Evolving-Core updates step the explicit optimizer bank below.
        opt = evolving_shared_optimizer
    elif config.continual_method == "rec_rssm_arrow":
        opt = Adam(
            _rec_optimizer_parameter_groups(
                wm,
                wm_lr=config.wm_lr,
                route_lr_scale=config.task_mechanism_route_lr_scale,
            ),
            fused=args.fused_adam,
        )
    else:
        trainable_world_model_parameters = [
            parameter for parameter in wm.parameters() if parameter.requires_grad
        ]
        opt = Adam(
            trainable_world_model_parameters,
            lr=config.wm_lr,
            fused=args.fused_adam,
        )
    compute_world_model_loss = wm.compute_loss
    distributed_world_model = None
    distributed_world_model_task_id = None
    if args.compile_world_model:
        compute_world_model_loss = torch.compile(
            compute_world_model_loss, dynamic=False, mode="reduce-overhead"
        )
    local_torch_seed = distributed_context.seed_local_torch_stream(config.seed)
    print(f"Local torch sampling seed: {local_torch_seed}")

    envs = config.get_env_schedule()
    for _ in range(training_start_epoch):
        envs.step()
    if training_start_epoch:
        print(
            "Restarted the environment schedule at completed epoch "
            f"{training_start_epoch}; current_task={envs.current_task_index()}"
        )
    replay_storage_directory = None
    if config.continual_method in {
        "bounded_dream_rehearsal",
        "cnn_fullbank_arrow",
        "cnn_projector_lora_arrow",
        "cnn_compact_shared_actor_arrow",
        "cnn_mechanism_bank_arrow",
        "rec_rssm_arrow",
        "evolving_atomic_rssm_arrow",
        "evolving_atomic_rssm_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow",
        "evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow",
        "evolving_atomic_rssm_atomic_lora_shared_heads_arrow",
        "evolving_atomic_rssm_learned_base_adapters_arrow",
        "evolving_atomic_rssm_shared_fastkan_arrow",
        "dino_patchbank_arrow",
        "dino_convbank_arrow",
    } and (not distributed_context.enabled or distributed_context.is_primary):
        if log_dir is None:
            raise ValueError(
                "Mapped observation replay requires --log-dir"
            )
        mmap_root = log_dir / "mmap_replay"
        replay_storage_directory = mmap_root / "observations"
    authoritative_replay = (
        config.get_replay_buffer(replay_storage_directory)
        if not distributed_context.enabled or distributed_context.is_primary
        else None
    )
    if task0_transition_payload is not None:
        if distributed_context.enabled or authoritative_replay is None:
            raise ValueError(
                "Task-0 cross-topology transition is validated only on one GPU"
            )
        authoritative_replay.load_state_dict(task0_transition_payload["replay"])
        if authoritative_replay.available_task_ids() != (0,):
            raise ValueError(
                "Task-0 transition replay contains tasks beyond Task 0: "
                f"{authoritative_replay.available_task_ids()}"
            )
        print(
            "Restored exact Task-0 FIFO/LTDM replay into independent working mmaps: "
            f"n_valid={authoritative_replay.n_valid}"
        )
    replay = (
        DistributedReplaySampler(
            distributed_context,
            authoritative_replay,
            action_space=config.action_space,
            num_tasks=len(config.esc.env_configs),
            observation_shape=(3, config.img_size, config.img_size),
        )
        if distributed_context.enabled
        else authoritative_replay
    )
    feature_cache = None
    if config.observation_encoder == "dinov3_vits16":
        cache_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[config.dinov3_feature_cache_dtype]
        if config.dinov3_replay_feature_mode == "cached":
            from clworldmodel.replay import ArrowFrozenFeatureCache

            feature_cache = ArrowFrozenFeatureCache(
                replay,
                wm.rssm.image_embedder.output_size,
                dtype=cache_dtype,
            )
        else:
            from clworldmodel.replay import ArrowOnTheFlyFeatureSource

            feature_cache = ArrowOnTheFlyFeatureSource(
                replay,
                wm.rssm.image_embedder,
                wm.rssm.image_embedder.output_size,
                dtype=cache_dtype,
                consumer_dtype={
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }[config.compute_dtype],
            )
    if distributed_context.is_primary:
        _print_replay_buffer_debug(config, authoritative_replay)

    # Optional snapshot initialization. The loaded actor must exist before the
    # first Task-2 collection; otherwise the first epoch would silently use a
    # random policy and would not test acquisition from Task 1.
    aco: Optional[ActorCriticOpt] = None
    resume_actor_critic_opened: list[str] = []
    if resume_payload is not None:
        aco = build_actor_critic_opt(
            wm,
            lr=config.ac_lr,
            **_actor_critic_constructor_kwargs(config),
        )
        resume_state_report["actor_critic"] = _load_snapshot_state(
            aco.ac,
            resume_payload["actor_critic_state_dict"],
            label="Actor-critic",
        )
        resume_actor_critic_opened = _configure_resume_actor_critic(
            aco,
            str(resume_mode),
        )
        print(
            "Loaded actor-critic weights; KAN residuals are plastic. "
            f"Opened behavior readouts: {resume_actor_critic_opened or 'none'}"
        )

    actor_critic_bank = None
    shared_actor_teacher: Optional[torch.nn.Module] = None
    shared_actor_teacher_seen_tasks = 0
    shared_behavior_update_rng: Optional[np.random.Generator] = None
    if config.uses_task_experts:
        from clworldmodel.continual import (
            ActorCriticBank,
            allocate_task_updates,
            shuffled_task_schedule,
        )
        task_update_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 0x4D4F4541])
        )
        if config.uses_replay_rehearsed_shared_behavior:
            # Keep behavior-route shuffling independent from Evolving-Core's
            # old-task world-model sampling so replacing the behavior head does
            # not silently change the v2 world-model replay sequence.
            shared_behavior_update_rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 0x464B414E])
            )

        def build_task_actor_critic(task_id: int) -> ActorCriticOpt:
            # Task-bank construction must not perturb world-model sampling RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed + 1_000_003 * (task_id + 1))
                return build_actor_critic_opt(
                    wm,
                    lr=config.ac_lr,
                    **_actor_critic_constructor_kwargs(config),
                )

        if config.uses_shared_actor:
            aco = build_task_actor_critic(0)
            if task1_seed_payload is not None:
                actor_bank_state = task1_seed_payload[
                    "actor_critic_bank_state_dict"
                ]
                if not isinstance(actor_bank_state, Mapping):
                    raise ValueError("Task-1 actor bank state must be a mapping")
                task_states = actor_bank_state.get("tasks")
                if not isinstance(task_states, Mapping) or not isinstance(
                    task_states.get("0"), Mapping
                ):
                    raise ValueError("Task-1 actor state is missing")
                aco.ac.load_state_dict(task_states["0"], strict=True)
                shared_actor_teacher = copy.deepcopy(aco.ac.actor).eval()
                shared_actor_teacher.requires_grad_(False)
                shared_actor_teacher_seen_tasks = 1
                print(
                    "Loaded the completed Task-1 actor as the shared actor and "
                    "created one transient frozen teacher"
                )
        else:
            if config.continual_method == "cnn_fullbank_arrow":
                actor_bank_artifact_kind = (
                    "cnn_fullbank_arrow_actor_critic_bank_inference_state"
                )
            elif config.continual_method == "cnn_projector_lora_arrow":
                actor_bank_artifact_kind = (
                    "cnn_projector_lora_arrow_actor_critic_bank_inference_state"
                )
            elif config.continual_method == "cnn_mechanism_bank_arrow":
                actor_bank_artifact_kind = (
                    "cnn_mechanism_bank_arrow_actor_critic_bank_inference_state"
                )
            elif config.continual_method == "rec_rssm_arrow":
                actor_bank_artifact_kind = (
                    "rec_rssm_arrow_actor_critic_bank_inference_state"
                )
            elif config.continual_method == "dino_patchbank_arrow":
                actor_bank_artifact_kind = (
                    "dino_patchbank_arrow_actor_critic_bank_inference_state"
                )
            elif config.continual_method == "dino_convbank_arrow":
                actor_bank_artifact_kind = (
                    "dino_convbank_arrow_actor_critic_bank_inference_state"
                )
            elif config.uses_evolving_atomic_rssm:
                actor_bank_artifact_kind = (
                    "evolving_atomic_rssm_actor_critic_bank_resumable_state"
                )
            elif config.uses_full_task_experts:
                actor_bank_artifact_kind = (
                    "dino_fullbank_arrow_actor_critic_bank_inference_state"
                )
            else:
                actor_bank_artifact_kind = (
                    "moe_arrow_actor_critic_bank_inference_state"
                )
            actor_critic_bank = ActorCriticBank(
                artifact_kind=actor_bank_artifact_kind
            )
        if task1_seed_payload is not None and not config.uses_shared_actor:
            seeded_actor = actor_critic_bank.ensure(0, build_task_actor_critic)
            actor_bank_state = task1_seed_payload["actor_critic_bank_state_dict"]
            if not isinstance(actor_bank_state, Mapping):
                raise ValueError("Task-1 actor bank state must be a mapping")
            task_states = actor_bank_state.get("tasks")
            if not isinstance(task_states, Mapping) or not isinstance(
                task_states.get("0"), Mapping
            ):
                raise ValueError("Task-1 actor state is missing")
            seeded_actor.ac.load_state_dict(task_states["0"], strict=True)
            aco = seeded_actor
            print("Loaded and froze the completed Task-1 Actor-Critic bank entry")

        if task0_transition_payload is not None:
            if config.uses_shared_actor or actor_critic_bank is None:
                raise RuntimeError(
                    "Task-0 transition requires the private MLP Actor-Critic bank"
                )
            actor_critic_bank.load_resumable_state_dict(
                task0_transition_payload["optimizers"]["actor_critic_bank"],
                build_task_actor_critic,
            )
            if actor_critic_bank.task_ids() != (0,):
                raise ValueError(
                    "Task-0 transition Actor-Critic bank must contain only Task 0"
                )
            aco = actor_critic_bank.get(0)
            print("Restored exact Task-0 MLP Actor-Critic and optimizer state")

    task0_transition_state: dict[str, int] = {}
    task0_transition_boundary_teacher: Optional[WorldModel] = None
    if task0_transition_payload is not None:
        if not config.uses_task_experts:
            raise RuntimeError("Task-0 transition requires task experts")
        task0_transition_boundary_teacher = copy.deepcopy(wm).eval()
        _seed_atomic_lora_task0_world_model(
            task0_transition_boundary_teacher,
            task0_transition_payload["boundary_teacher"],
        )
        task0_transition_boundary_teacher.requires_grad_(False)
        _restore_task0_transition_rng(
            task0_transition_payload["rng"],
            task_update_rng=task_update_rng,
            collection_environment_seed_rng=collection_environment_seed_rng,
            validation_environment_seed_rng=validation_environment_seed_rng,
            final_environment_seed_rng=final_environment_seed_rng,
        )
        source_counters = task0_transition_payload["counters"]
        task0_transition_state = {
            "completed_epochs": training_start_epoch,
            "raw_environment_frames": int(
                source_counters["raw_environment_frames"]
            ),
            "world_model_updates": int(source_counters["world_model_updates"]),
            "actor_critic_updates": int(
                source_counters["actor_critic_updates"]
            ),
        }
        task0_transition_metadata["world_model_state_transfer"] = (
            task0_transition_world_model_report
        )
        # Release the source optimizer/model tensors before training while the
        # copied working replay and compact provenance record remain live.
        task0_transition_payload = None

    if log_dir is None:
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        job_id = os.getenv("SLURM_JOB_ID")
        run_name = f"{current_time}_{socket.gethostname()}_{config.seed}_{job_id}"
        # One env in the schedule → single-task; multiple → continual (sequential) training

        if len(config.esc.env_configs) == 1: 
            task_kind = "single"
        else:
            first_task_duration = (
                _sequential_task_durations(config)[0]
                if config.esc.env_schedule_type is SequentialEnvironments
                else None
            )
            if (
                config.esc.env_configs[0].name == "ALE/MsPacman-v5"
                and first_task_duration == 90
            ):
                task_kind = "cl_original"
            elif (
                config.esc.env_configs[0].name == "ALE/Enduro-v5"
                and first_task_duration == 90
            ):
                task_kind = "cl_reversed"
            else:
                task_kind = "cl_two_cycle"
        
        if config.algorithm == "arrow":
            ratio = config.arrow_replay_capacity_ratio.replace("-", "_")
            log_root = Path.cwd() / "runs" / task_kind / config.algorithm / ratio
        else:
            log_root = Path.cwd() / "runs" / task_kind / config.algorithm        

        log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] log_dir={log_dir}")
    else:
        if distributed_context.is_primary:
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] log_dir={log_dir} (explicit)")
        distributed_context.barrier()


    swanlab_run = (
        _init_swanlab(args.swanlab_project, args.swanlab_experiment_name, config)
        if distributed_context.is_primary
        else None
    )
    from torch.utils.tensorboard import SummaryWriter

    writer = (
        SummaryWriter(log_dir=log_dir)
        if distributed_context.is_primary
        else _NoOpWriter()
    )
    log_dir = Path(log_dir)
    if distributed_context.is_primary:
        config.save(log_dir / "config.json")
        evaluation_seed_manifest = {
            "schema_version": 1,
            "protocol": config.evaluation_seed_protocol,
            "task_seed_index_offset": config.evaluation_task_seed_offset,
            "periodic_validation": {
                "task_base_seeds": list(validation_task_seeds),
                "reused_at_every_checkpoint": fixed_evaluation_cohorts,
                "training_rng_state_restored": True,
            },
            "final_evaluation": {
                "task_base_seeds": list(final_task_seeds),
                "held_out_from_periodic_validation": fixed_evaluation_cohorts,
                "training_rng_state_restored": True,
            },
            "cohorts_disjoint_by_seed_sequence": fixed_evaluation_cohorts,
            "evaluation_transitions_enter_replay": False,
        }
        if config.uses_adaptive_qfp_compression:
            evaluation_seed_manifest["adaptive_compression_validation"] = {
                "task_base_seeds": list(compression_validation_task_seeds),
                "seed_sequence_spawn_index": 3,
                "reused_for_dense_teacher_and_every_candidate": True,
                "rollouts_per_condition": (
                    config.adaptive_compression_rollouts
                ),
                "used_for_width_selection": True,
                "disjoint_from_periodic_and_final_seed_domains": True,
                "training_rng_state_restored": True,
                "evaluation_transitions_enter_replay": False,
            }
            evaluation_seed_manifest["final_evaluation"][
                "used_for_adaptive_width_selection"
            ] = False
        if config.uses_adaptive_behavior_compression:
            evaluation_seed_manifest["adaptive_behavior_validation"] = {
                "task_base_seeds": list(
                    behavior_compression_validation_task_seeds
                ),
                "seed_sequence_spawn_index": 4,
                "reused_for_dense_teacher_and_every_candidate": True,
                "rollouts_per_condition": config.adaptive_behavior_rollouts,
                "used_for_actor_critic_width_selection": True,
                "disjoint_from_periodic_final_and_qfp_seed_domains": True,
                "training_rng_state_restored": True,
                "evaluation_transitions_enter_replay": False,
            }
            evaluation_seed_manifest["final_evaluation"][
                "used_for_actor_critic_width_selection"
            ] = False
        evaluation_seed_path = log_dir / "evaluation_seed_manifest.json"
        temporary_evaluation_seed_path = evaluation_seed_path.with_suffix(
            ".json.tmp"
        )
        temporary_evaluation_seed_path.write_text(
            json.dumps(evaluation_seed_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_evaluation_seed_path, evaluation_seed_path)
        if resume_payload is not None:
            resume_metadata = {
                "schema_version": 1,
                "artifact_kind": "task_acquisition_from_analysis_snapshot",
                "initial_snapshot": str(args.init_analysis_snapshot.expanduser().resolve()),
                "initial_snapshot_epoch": resume_payload.get("epoch"),
                "initial_snapshot_task": resume_payload.get("task"),
                "adaptation_mode": resume_mode,
                "replay_state": "reset_empty",
                "optimizer_state": "reset_new_optimizer",
                "rng_state": "reset_from_config_seed",
                "collection_policy": "loaded_actor_from_snapshot",
                "world_model_opened_modules": resume_world_model_opened,
                "actor_critic_opened_modules": resume_actor_critic_opened,
                "snapshot_state_report": resume_state_report,
            }
            (log_dir / "resume_initialization.json").write_text(
                json.dumps(resume_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        if task1_seed_payload is not None:
            seed_metadata = {
                "schema_version": 1,
                "artifact_kind": "task1_boundary_seeded_incremental_training",
                "initial_snapshot": str(
                    args.init_task1_boundary_snapshot.expanduser().resolve()
                ),
                "initial_snapshot_completed_epochs": training_start_epoch,
                "replay_state": "reset_empty",
                "optimizer_state": "reset_new_optimizers",
                "rng_state": "reset_from_config_seed",
                "environment_schedule_restart_task": 1,
                "source_task1_world_model_modules": task1_seed_world_model_report,
                "source_task1_actor_loaded": True,
                "actor_topology": (
                    "single_shared_actor_critic"
                    if config.uses_shared_actor
                    else "per_task_actor_critic_bank"
                ),
                "old_real_replay_used": False,
                "old_policy_protection": (
                    "frozen old-route world-model imagination with one transient "
                    "previous shared-actor teacher"
                    if config.uses_shared_actor
                    else "frozen task-specific actor entries"
                ),
                "source_snapshot_resumable": False,
                "scientific_scope": (
                    "snapshot-seeded Task-2/3 acquisition; not an equivalent "
                    "resume of the source run"
                ),
            }
            (log_dir / "task1_seed_initialization.json").write_text(
                json.dumps(seed_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        if task0_transition_metadata:
            transition_path = log_dir / "task0_transition_initialization.json"
            temporary_transition_path = transition_path.with_suffix(".json.tmp")
            temporary_transition_path.write_text(
                json.dumps(task0_transition_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_transition_path, transition_path)
        if feature_cache is not None:
            feature_accounting_path = log_dir / "feature_cache_storage_accounting.json"
            temporary_feature_accounting_path = feature_accounting_path.with_suffix(
                ".json.tmp"
            )
            temporary_feature_accounting_path.write_text(
                json.dumps(feature_cache.storage_accounting(), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_feature_accounting_path, feature_accounting_path)
        if replay_storage_directory is not None:
            replay_accounting_path = log_dir / "replay_mmap_storage_accounting.json"
            temporary_replay_accounting_path = replay_accounting_path.with_suffix(
                ".json.tmp"
            )
            temporary_replay_accounting_path.write_text(
                json.dumps(
                    _mapped_replay_storage_accounting(authoritative_replay), indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_replay_accounting_path, replay_accounting_path)
        parameter_accounting_path = log_dir / "model_parameter_accounting.json"
        temporary_accounting_path = parameter_accounting_path.with_suffix(".json.tmp")
        temporary_accounting_path.write_text(
            json.dumps(_world_model_parameter_accounting(wm), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_accounting_path, parameter_accounting_path)
    actor_accounting_path = log_dir / "actor_critic_parameter_accounting.json"
    shared_actor_distillation_counters = {
        "optimizer_updates": 0,
        "distillation_batches": 0,
        "distilled_states": 0,
        "burnin_state_uses": 0,
    }
    shared_behavior_replay_updates: dict[int, int] = {}
    adaptive_behavior_compression_updates = 0
    profile_stages = args.profile_stages and distributed_context.is_primary

    
    total_env_steps = (
        task0_transition_state["raw_environment_frames"]
        if task0_transition_state
        else int(task1_seed_payload.get("total_raw_environment_frames", 0))
        if task1_seed_payload is not None
        else 0
    )  # number of *real* environment interactions so far
    if total_env_steps % config.env_repeat:
        raise ValueError("Raw environment-frame counter is not decision aligned")
    total_agent_decisions = total_env_steps // config.env_repeat
    dream_rehearsal_intervals = 0
    dream_rehearsal_updates = 0
    dream_rehearsal_updates_by_task: dict[int, int] = {}
    dream_rehearsal_dreamed_trajectories = 0
    dream_rehearsal_selected_trajectories = 0
    encountered_replay_task_ids = set(replay.available_task_ids())

    best_rews_mean = float("-inf")
    best_validation_seen_task_raw_mean = float("-inf")
    global_step = (
        task0_transition_state["world_model_updates"]
        if task0_transition_state
        else int(task1_seed_payload.get("world_model_updates", 0))
        if task1_seed_payload is not None
        else 0
    )  # gradient updates so far
    shared_core_frozen = resume_payload is not None
    boundary_teacher: Optional[WorldModel] = task0_transition_boundary_teacher
    capture_kan_parameter_values = None
    protect_kan_parameter_updates = None
    if config.residual_consolidation == "replay_functional":
        from clworldmodel.continual import (
            capture_kan_parameter_values,
            protect_kan_parameter_updates,
        )

    for epoch in range(training_start_epoch, config.epochs):
        print("Starting Epoch ", epoch)
        agent_decisions_before_epoch = total_agent_decisions
        current_task_id = None
        if config.uses_task_experts:
            current_task_id = envs.current_task_index()
            mechanism_phase = "full"
            if config.continual_method == "rec_rssm_arrow":
                _, task_epoch = _sequential_task_position(config, epoch)
                task_local_epoch = task_epoch - 1
                if (
                    current_task_id >= 2
                    and task_local_epoch < config.task_mechanism_reuse_probe_epochs
                ):
                    mechanism_phase = "reuse_probe"
            warm_start_from = (
                0
                if config.continual_method in {
                    "cnn_projector_lora_arrow",
                    "cnn_compact_shared_actor_arrow",
                }
                and current_task_id > 0
                else current_task_id - 1
                if current_task_id > 0
                else None
            )
            if config.uses_shared_actor:
                if aco is None:
                    raise RuntimeError("Shared actor was not initialized")
                if warm_start_from is not None:
                    initialized = wm.initialize_task_expert(
                        current_task_id, warm_start_from
                    )
                    if initialized:
                        print(
                            f"Warm-started world-model expert {current_task_id} "
                            f"from expert {warm_start_from}"
                        )
                if current_task_id > shared_actor_teacher_seen_tasks:
                    if current_task_id != shared_actor_teacher_seen_tasks + 1:
                        raise RuntimeError(
                            "Shared-actor teacher tasks must advance sequentially"
                        )
                    shared_actor_teacher = copy.deepcopy(aco.ac.actor).eval()
                    shared_actor_teacher.requires_grad_(False)
                    shared_actor_teacher_seen_tasks = current_task_id
                    print(
                        "Refreshed the one transient shared-actor teacher before "
                        f"task {current_task_id}; old routes={tuple(range(current_task_id))}"
                    )
                aco.ac.activate_training_task(current_task_id)
                wm.activate_task_expert(current_task_id)
            elif current_task_id not in actor_critic_bank:
                if warm_start_from is not None:
                    if warm_start_from not in actor_critic_bank:
                        raise RuntimeError(
                            f"Task {current_task_id} arrived before task "
                            f"{warm_start_from} was initialized"
                        )
                    initialized = wm.initialize_task_expert(
                        current_task_id, warm_start_from
                    )
                    if initialized:
                        print(
                            f"Warm-started world-model expert {current_task_id} "
                            f"from expert {warm_start_from}"
                        )
                actor_critic_bank.ensure(
                    current_task_id,
                    build_task_actor_critic,
                    warm_start_from=(
                        warm_start_from
                        if config.continual_method == "moe_arrow"
                        else None
                    ),
                )
                print(
                    f"Initialized independent actor-critic for task {current_task_id}"
                )
            if config.uses_task_private_heads or config.uses_evolving_atomic_rssm:
                wm.activate_task_expert(
                    current_task_id, mechanism_phase=mechanism_phase
                )
                if actor_critic_bank is not None:
                    actor_critic_bank.activate(current_task_id)
                if config.continual_method == "rec_rssm_arrow":
                    writer.add_scalar(
                        "RECRSSM/reuse_probe_active",
                        int(mechanism_phase == "reuse_probe"),
                        global_step,
                    )
                    print(
                        "REC-RSSM mechanism phase: "
                        f"task={current_task_id} phase={mechanism_phase}"
                    )
            if actor_critic_bank is not None:
                aco = actor_critic_bank.get(current_task_id)
            if config.uses_evolving_atomic_rssm:
                if evolving_shared_optimizer is None:
                    raise RuntimeError("Evolving-Core shared optimizer was not initialized")
                _set_evolving_shared_optimizer_learning_rates(
                    evolving_shared_optimizer,
                    core_lr=(
                        config.first_task_shared_core_lr
                        if current_task_id == 0
                        else config.shared_core_lr
                    ),
                    prediction_head_lr=config.task_private_lr,
                )
                _ensure_evolving_private_optimizers(
                    wm=wm,
                    task_id=current_task_id,
                    private_optimizers=evolving_private_optimizers,
                    route_optimizers=evolving_route_optimizers,
                    private_lr=config.task_private_lr,
                    route_lr=config.task_route_lr,
                    fused=args.fused_adam,
                )
        replay_task_id = (
            envs.current_task_index()
            if config.uses_bounded_dream_rehearsal
            else current_task_id
        )
        task_boundary = epoch > 0 and envs.is_new_env()
        if config.residual_consolidation == "replay_functional" and task_boundary:
            if aco is None:
                raise RuntimeError(
                    "The actor-critic must be initialized before KAN consolidation"
                )
            diagnostics = _consolidate_kan_from_replay(
                config=config,
                wm=wm,
                aco=aco,
                feature_cache=feature_cache,
                epoch=epoch,
                global_step=global_step,
                log_dir=log_dir,
                writer=writer,
            )
            print(
                "Consolidated replay-important KAN coefficients at task boundary "
                f"{_sequential_task_position(config, epoch)[0]}: "
                f"modules={len(diagnostics)}"
            )
        if (
            config.shared_core_mode == "freeze_after_first_task"
            and not shared_core_frozen
            and task_boundary
        ):
            if aco is None:
                raise RuntimeError(
                    "The actor-critic must be initialized before freezing the shared core"
                )
            wm.freeze_shared_core()
            aco.ac.freeze_shared_core()
            if config.residual_consolidation == "replay_functional":
                from clworldmodel.continual import freeze_kan_coordinate_maps

                freeze_kan_coordinate_maps(
                    {"world_model": wm, "actor_critic": aco.ac}
                )
            _restrict_optimizer_to_trainable(opt, wm)
            _restrict_optimizer_to_trainable(aco.opt, aco.ac)
            shared_core_frozen = True
            print("Frozen shared world-model and actor-critic cores after task 1")
            writer.add_scalar("Continual/shared_core_frozen", 1, global_step)
        epoch_started = _stage_clock(profile_stages)
        collect_started = _stage_clock(profile_stages)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if distributed_context.is_primary:
            if resume_payload is not None:
                random_policy = False
            elif config.random_policy == "first":
                random_policy = epoch == 0
            elif config.random_policy == "new":
                random_policy = envs.is_new_env()
            for _ in range(
                config.pretrain_data_multiplier
                if random_policy and config.pretrain_enabled
                else 1
            ):
                collection_routing = {}
                collection_behavior = None if random_policy else (
                    _autorouted_behavior(config, aco, actor_critic_bank, current_task_id + 1)
                    if config.uses_reconstruction_task_inference else aco.ac
                )
                _acts, _obss, _rews, _conts, _resets = reinterpret_nt_to_t_n(
                    *generate_trajectories(
                        config.n_sync * config.gen_seq_len,
                        config.n_sync,
                        wm=wm,
                        ac=collection_behavior,
                        env_fns=envs.funcs(),
                        env_repeat=config.env_repeat,
                        seed=_next_environment_seed(collection_environment_seed_rng),
                        task_id=None if config.uses_reconstruction_task_inference else current_task_id,
                        eligible_route_ids=(tuple(range(current_task_id + 1))
                                            if config.uses_reconstruction_task_inference else None),
                        routing_diagnostics=collection_routing,

                    ),
                    config.data_t,
                    config.data_n,
                )
                if config.uses_reconstruction_task_inference:
                    from clworldmodel.routing import routing_audit

                    collection_routing["audit"] = routing_audit(
                        collection_routing["routing_events"], true_task_id=current_task_id,
                        task_count=config.rssm_num_experts,
                    )
                    collection_routing["epoch"] = epoch
                    collection_routing["world_model_updates"] = global_step
                    collection_routing["replay_label_source"] = "training scheduler, never inferred route"
                    _write_routing_diagnostic(
                        log_dir / "task_routing" / f"collection_epoch_{epoch:04d}_batch_{_:02d}.json",
                        collection_routing,
                    )
                frozen_features = None
                if feature_cache is not None and feature_cache.requires_recording:
                    encoder = wm.rssm.image_embedder
                    if getattr(encoder, "requires_projection_fit", False):
                        if epoch != 0 or not random_policy or global_step != 0:
                            raise RuntimeError(
                                "Task-1 patch PCA must be fitted before model training"
                            )
                        projection_metadata = _fit_dinov3_patch_projection(
                            wm,
                            _obss,
                            calibration_frames=config.dinov3_patch_projection_frames,
                        )
                        projection_path = log_dir / "dinov3_patch_projection.json"
                        temporary_projection_path = projection_path.with_suffix(
                            ".json.tmp"
                        )
                        temporary_projection_path.write_text(
                            json.dumps(projection_metadata, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        os.replace(temporary_projection_path, projection_path)
                        writer.add_scalar(
                            "DINOv3/patch_projection_explained_variance",
                            projection_metadata["explained_variance_ratio"],
                            global_step,
                        )
                        print(
                            "Fitted and froze Task-1 DINOv3 patch PCA: "
                            f"{projection_metadata}"
                        )
                    frozen_features = _encode_frozen_observation_features(
                        wm,
                        _obss,
                        batch_size=config.dinov3_max_batch_size,
                    )
                write_slots = replay.add(
                    _acts,
                    _obss,
                    _rews,
                    _conts,
                    _resets,
                    task_id=replay_task_id,
                )
                if replay_task_id is not None:
                    encountered_replay_task_ids.add(replay_task_id)
                if feature_cache is not None and feature_cache.requires_recording:
                    feature_cache.record(write_slots, frozen_features)
                print(f"{replay.n_valid=}")
                num_new_env_steps = (
                    _acts.shape[0] * _acts.shape[1] * config.env_repeat
                )
                total_agent_decisions += _acts.shape[0] * _acts.shape[1]
                total_env_steps += num_new_env_steps
                writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)
                writer.add_scalar(
                    "Counters/agent_decisions", total_agent_decisions, global_step
                )

            rews_eps_mean = _rews.sum().item() / _resets.sum().item()
            writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
            len_eps_mean = (
                config.gen_seq_len / _resets.sum().item() * config.env_repeat
            )
            writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)
            if rews_eps_mean >= best_rews_mean:
                best_rews_mean = rews_eps_mean
                if save_nets and aco is not None:
                    print(f"Saving best rews eps mean {rews_eps_mean=}")
                    torch.save(wm.state_dict(), log_dir / "save_wm_best.pt")
                    torch.save(aco.ac.state_dict(), log_dir / "save_ac_best.pt")
                    if actor_critic_bank is not None:
                        torch.save(
                            actor_critic_bank.inference_state_dict(),
                            log_dir / "save_ac_bank_best.pt",
                        )

        collect_seconds = _stage_elapsed(collect_started, profile_stages)
        distributed_context.barrier()

        # Evaluation games
        eval_started = _stage_clock(profile_stages)
        if epoch % 10 == 0 or epoch + 1 in milestone_completed_epochs:
            periodic_task_seeds = (
                validation_task_seeds
                if fixed_evaluation_cohorts
                else tuple(
                    _next_environment_seed(validation_environment_seed_rng)
                    for _ in envs.eval_funcs()
                )
            )
            periodic_routing = []
            eval_results_mean, eval_results_std = _evaluate_policy_tasks(
                config,
                wm,
                aco,
                envs.eval_funcs(),
                periodic_task_seeds,
                actor_critic_bank=actor_critic_bank,
                distributed_context=distributed_context,
                eligible_task_count=current_task_id + 1,
                routing_diagnostics=periodic_routing,
            )
            if config.uses_reconstruction_task_inference:
                _write_routing_diagnostic(
                    log_dir / "task_routing" / f"periodic_epoch_{epoch:04d}.json",
                    {"epoch": epoch, "world_model_updates": global_step,
                     "evaluation_transitions_enter_replay": False, "tasks": periodic_routing},
                )
            eval_raw_mean, eval_raw_std = _raw_return_statistics(
                config.esc.env_configs, eval_results_mean, eval_results_std
            )
            if distributed_context.is_primary:
                writer.add_scalars(
                    "Perf/eval_rew_eps_mean",
                    {f"{i}": m for i, m in enumerate(eval_results_mean)},
                    global_step,
                )
                writer.add_scalars(
                    "Perf/eval_rew_eps_std",
                    {f"{i}": s for i, s in enumerate(eval_results_std)},
                    global_step,
                )
                writer.add_scalars(
                    "Perf/eval_raw_return_mean",
                    {f"{i}": mean for i, mean in enumerate(eval_raw_mean)},
                    global_step,
                )
                writer.add_scalars(
                    "Perf/eval_raw_return_std",
                    {f"{i}": std for i, std in enumerate(eval_raw_std)},
                    global_step,
                )
                print("Eval for epoch: ",epoch)
                print(f"Eval means: {eval_results_mean}")
                print(f"Eval stds: {eval_results_std}")
                print(f"Eval raw means: {eval_raw_mean}")
                print(f"Eval raw stds: {eval_raw_std}")
                if evaluation_snapshot_dir is not None and epoch > 0:
                    snapshot_path = _save_task_bank_evaluation_snapshot(
                        evaluation_snapshot_dir,
                        config=config,
                        wm=wm,
                        actor_critic_bank=actor_critic_bank,
                        aco=aco,
                        completed_epochs=epoch,
                        world_model_updates=global_step,
                        actor_critic_updates=epoch * config.ac_train_steps,
                        total_env_steps=total_env_steps,
                        task_seeds=periodic_task_seeds,
                        scaled_means=eval_results_mean,
                        scaled_stds=eval_results_std,
                        raw_means=eval_raw_mean,
                        raw_stds=eval_raw_std,
                        cohort="periodic_validation",
                        eligible_task_count=current_task_id + 1,
                    )
                    seen_task_count = min(
                        len(eval_raw_mean),
                        _sequential_seen_task_count(config, epoch),
                    )
                    seen_task_raw_mean = float(
                        np.mean(eval_raw_mean[:seen_task_count])
                    )
                    if (
                        seen_task_raw_mean
                        > best_validation_seen_task_raw_mean
                    ):
                        best_validation_seen_task_raw_mean = seen_task_raw_mean
                        _write_best_validation_snapshot(
                            evaluation_snapshot_dir,
                            snapshot_path=snapshot_path,
                            completed_epochs=epoch,
                            seen_task_count=seen_task_count,
                            seen_task_raw_mean=seen_task_raw_mean,
                        )

        eval_seconds = _stage_elapsed(eval_started, profile_stages)

        if (
            distributed_context.enabled
            and distributed_world_model_task_id != current_task_id
        ):
            distributed_context.barrier()
            distributed_world_model = distributed_context.wrap_module(wm)
            compute_world_model_loss = distributed_world_model
            distributed_world_model_task_id = current_task_id

        world_model_started = _stage_clock(profile_stages)
        world_model_updates_this_epoch = (
            config.steps_per_batch
            if epoch > 0 or not config.pretrain_enabled
            else config.pretrain_steps
        )
        world_model_task_schedule = None
        if config.uses_task_experts:
            available_task_ids = replay.available_task_ids()
            world_model_allocation = allocate_task_updates(
                world_model_updates_this_epoch,
                current_task_id=current_task_id,
                available_task_ids=available_task_ids,
                current_task_fraction=config.task_update_fraction,
            )
            world_model_task_schedule = shuffled_task_schedule(
                world_model_allocation, task_update_rng
            )
            for task_id, update_count in world_model_allocation.items():
                writer.add_scalar(
                    (
                        f"CNNFullBankArrow/world_model_updates_task_{task_id}"
                        if config.continual_method == "cnn_fullbank_arrow"
                        else f"CNNProjectorLoraArrow/world_model_updates_task_{task_id}"
                        if config.continual_method == "cnn_projector_lora_arrow"
                        else f"CNNCompactSharedActor/world_model_updates_task_{task_id}"
                        if config.continual_method == "cnn_compact_shared_actor_arrow"
                        else f"CNNMechanismBank/world_model_updates_task_{task_id}"
                        if config.continual_method == "cnn_mechanism_bank_arrow"
                        else f"RECRSSM/world_model_updates_task_{task_id}"
                        if config.continual_method == "rec_rssm_arrow"
                        else f"EvolvingCore/world_model_updates_task_{task_id}"
                        if config.uses_evolving_atomic_rssm
                        else f"DINOPatchBankArrow/world_model_updates_task_{task_id}"
                        if config.continual_method == "dino_patchbank_arrow"
                        else f"DINOConvBankArrow/world_model_updates_task_{task_id}"
                        if config.continual_method == "dino_convbank_arrow"
                        else f"DINOFullBankArrow/world_model_updates_task_{task_id}"
                        if config.uses_full_task_experts
                        else f"MoEArrow/world_model_updates_task_{task_id}"
                    ),
                    update_count,
                    global_step,
                )
        progbar = trange(
            world_model_updates_this_epoch,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            disable = True,
        )
        for update_index in progbar:
            update_task_id = (
                world_model_task_schedule[update_index]
                if world_model_task_schedule is not None
                else None
            )
            if args.compile_world_model:
                torch.compiler.cudagraph_mark_step_begin()
            observation_features = None
            if epoch > 0 or not config.pretrain_enabled:
                mb_t_size = config.mb_t_size
                global_mb_n_size = config.mb_n_size
            else:
                mb_t_size = config.pretrain_mb_t_size
                global_mb_n_size = config.pretrain_mb_n_size
            mb_n_size = distributed_context.local_sequences(global_mb_n_size)
            if config.uses_evolving_atomic_rssm:
                if current_task_id is None or evolving_shared_optimizer is None:
                    raise RuntimeError("Evolving-Core requires an active task and optimizer")
                private_optimizer = evolving_private_optimizers[current_task_id]
                route_optimizer = evolving_route_optimizers.get(current_task_id)
                memory_task_id = (
                    int(task_update_rng.integers(0, current_task_id))
                    if current_task_id > 0
                    else None
                )
                metrics, projection_diagnostics, grad_norm = (
                    _evolving_world_model_update(
                        config=config,
                        wm=wm,
                        boundary_teacher=boundary_teacher,
                        actor_critic_bank=actor_critic_bank,
                        frozen_actor=(
                            shared_actor_teacher
                            if config.uses_replay_rehearsed_shared_behavior
                            else None
                        ),
                        replay_buffer=replay,
                        current_task_id=current_task_id,
                        memory_task_id=memory_task_id,
                        sequence_length=mb_t_size,
                        shared_optimizer=evolving_shared_optimizer,
                        private_optimizer=private_optimizer,
                        route_optimizer=route_optimizer,
                        materialize_diagnostics=(
                            global_step % config.log_frequency == 0
                        ),
                    )
                )
                if global_step % config.log_frequency == 0:
                    writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
                    for metric_key, metric_value in metrics.items():
                        writer.add_scalar(metric_key, metric_value, global_step)
                    for component, diagnostic in projection_diagnostics.items():
                        writer.add_scalar(
                            f"EvolvingCore/{component}/gradient_dot",
                            diagnostic.dot_product,
                            global_step,
                        )
                        writer.add_scalar(
                            f"EvolvingCore/{component}/gradient_conflict",
                            int(diagnostic.conflicted),
                            global_step,
                        )
                        writer.add_scalar(
                            f"EvolvingCore/{component}/projected_current_norm",
                            diagnostic.projected_current_norm,
                            global_step,
                        )
                    if memory_task_id is not None:
                        writer.add_scalar(
                            "EvolvingCore/memory_task_id",
                            memory_task_id,
                            global_step,
                        )
                global_step += 1
                continue
            replay_sample_kwargs = {}
            if update_task_id is not None:
                replay_sample_kwargs["task_id"] = update_task_id
            if feature_cache is None:
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets = replay.minibatch(
                    mb_t_size, mb_n_size, **replay_sample_kwargs
                )
            else:
                (
                    mb_acts,
                    mb_obss,
                    observation_features,
                    mb_rews,
                    mb_conts,
                    mb_resets,
                ) = feature_cache.minibatch(
                    mb_t_size,
                    mb_n_size,
                    **replay_sample_kwargs,
                )

            world_model_loss_kwargs = {
                "observation_features": observation_features,
            }
            if update_task_id is not None:
                world_model_loss_kwargs["task_id"] = update_task_id
            with _autocast_context(mb_acts.device, config.compute_dtype):
                loss, metrics = compute_world_model_loss(
                    mb_acts,
                    mb_obss,
                    mb_rews,
                    mb_conts,
                    mb_resets,
                    **world_model_loss_kwargs,
                )

            protected_values = None
            if shared_core_frozen and capture_kan_parameter_values is not None:
                protected_values = capture_kan_parameter_values(wm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                _optimizer_parameters(opt), 1000
            )
            opt.step()
            if protected_values is not None:
                protect_kan_parameter_updates(wm, protected_values)

            # Optional progress bar logging
            # if global_step % 10 == 0:
            #     progbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.items()})

            if global_step % config.log_frequency == 0:
                metrics = distributed_context.mean_tensor_mapping(metrics)
                writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
                with torch.no_grad():
                    for metric_key, metric_value in metrics.items():
                        writer.add_scalar(metric_key, metric_value, global_step)

                    if (
                        distributed_context.is_primary
                        and log_images
                        and config.observation_objective == "reconstruction"
                    ):
                        original = _obss[:16, 0:2].to(device)
                        writer.add_images(
                            "original", original.swapaxes(0, 1).flatten(0, 1), global_step
                        )

                        init_z, init_h = wm.rssm.initial_state(original.shape[1])
                        no_resets = torch.zeros(*original.shape[:2], 1, device=init_z.device)
                        image_log_rssm_kwargs = {}
                        if current_task_id is not None:
                            image_log_rssm_kwargs["task_id"] = current_task_id
                        z_posts, z, h = wm.rssm(
                            init_z,
                            _acts[:, 0:2].to(device),
                            init_h,
                            original,
                            no_resets,
                            **image_log_rssm_kwargs,
                        )
                        zhs = wm.zh_transform(z, h)
                        recon = torch.stack(
                            [
                                wm.predict_observation(zh, current_task_id)
                                for zh in zhs
                            ]
                        )

                        writer.add_images(
                            "reconstructed",
                            recon.clip(0, 1).swapaxes(0, 1).flatten(0, 1),
                            global_step,
                        )
                        writer.add_images(
                            "latent",
                            z_posts.exp().swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
                        writer.add_images(
                            "latent sample",
                            z.swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
            global_step += 1

        world_model_seconds = _stage_elapsed(world_model_started, profile_stages)
        actor_started = _stage_clock(profile_stages)

        scheduled_ac_lr, scheduled_entropy_scale, actor_task_epoch = (
            _actor_critic_schedule_values(config, epoch)
        )
        actor_update_counter_before_epoch = epoch * config.ac_train_steps
        writer.add_scalar(
            "ActorCriticSchedule/learning_rate",
            scheduled_ac_lr,
            actor_update_counter_before_epoch,
        )
        writer.add_scalar(
            "ActorCriticSchedule/entropy_scale",
            scheduled_entropy_scale,
            actor_update_counter_before_epoch,
        )
        writer.add_scalar(
            "ActorCriticSchedule/task_epoch",
            actor_task_epoch,
            actor_update_counter_before_epoch,
        )
        if distributed_context.is_primary:
            print(
                "[actor-schedule] "
                f"epoch={epoch} task_epoch={actor_task_epoch} "
                f"kind={config.ac_schedule} lr={scheduled_ac_lr:.8g} "
                f"entropy_scale={scheduled_entropy_scale:.8g}"
            )

        actor_critic_kwargs = {
            "dream_steps": config.ac_dream_steps,
            "actor_network": config.actor_network,
            "actor_kan_hidden_features": config.actor_kan_hidden_features,
            "actor_kan_grid_size": config.actor_kan_grid_size,
            "actor_kan_spline_order": config.actor_kan_spline_order,
            "actor_kan_input_min": config.actor_kan_input_min,
            "actor_kan_input_max": config.actor_kan_input_max,
            "actor_kan_normalize_recurrent_state": (
                config.actor_kan_normalize_recurrent_state
            ),
            "fastkan_hidden_features": config.fastkan_hidden_features,
            "fastkan_hidden_layers": config.fastkan_hidden_layers,
            "fastkan_grid_size": config.fastkan_grid_size,
            "fastkan_input_min": config.fastkan_input_min,
            "fastkan_input_max": config.fastkan_input_max,
            "fastkan_rms_norm_epsilon": config.fastkan_rms_norm_epsilon,
            "fastkan_actor_output_scale": config.fastkan_actor_output_scale,
            "fastkan_actor_unimix": config.fastkan_actor_unimix,
            "optimizer_name": config.ac_optimizer,
            "optimizer_eps": config.ac_optimizer_eps,
            "optimizer_beta1": config.ac_optimizer_beta1,
            "optimizer_beta2": config.ac_optimizer_beta2,
            "optimizer_warmup_steps": config.ac_optimizer_warmup_steps,
            "agc_clip": config.ac_agc_clip,
            "grad_clip": config.ac_grad_clip,
            "discount": config.ac_discount,
            "lam": config.ac_lambda,
            "entropy_scale": scheduled_entropy_scale,
            "return_norm_decay": config.ac_return_norm_decay,
            "persistent_return_norm": config.ac_persistent_return_norm,
            "slow_critic_regularizer": config.ac_slow_critic_regularizer,
            "slow_critic_decay": config.ac_slow_critic_decay,
            "replay_critic_loss_scale": config.ac_replay_critic_loss_scale,
            "use_slow_critic_targets": config.ac_use_slow_critic_targets,
            "corrected_imagination_bootstrap": (
                config.ac_corrected_imagination_bootstrap
            ),
            "residual_correction": config.residual_correction,
            "residual_bottleneck_features": config.residual_bottleneck_features,
            "residual_grid_size": config.residual_grid_size,
            "residual_input_min": config.residual_input_min,
            "residual_input_max": config.residual_input_max,
            "residual_rms_norm_epsilon": config.residual_rms_norm_epsilon,
            "residual_alpha": config.residual_alpha,
            "residual_input_mode": config.residual_input_mode,
            "residual_consolidation": config.residual_consolidation,
            "adaptive_behavior_residuals": config.adaptive_behavior_residuals,
            "adaptive_behavior_num_tasks": config.rssm_num_experts,
            "adaptive_behavior_hidden_features": (
                config.adaptive_behavior_hidden_features
            ),
            "adaptive_behavior_residual_scale": (
                config.adaptive_behavior_residual_scale
            ),
            "adaptive_behavior_num_atoms": config.adaptive_behavior_num_atoms,
            "adaptive_behavior_reuse": config.adaptive_behavior_reuse,
            "protect_residual_updates": (
                shared_core_frozen
                and config.residual_consolidation == "replay_functional"
            ),
            "feature_cache": feature_cache,
            "distributed_context": distributed_context,
        }

        local_ac_train_sync = distributed_context.local_sequences(
            config.ac_train_sync
        )

        if config.uses_replay_rehearsed_shared_behavior:
            if (
                current_task_id is None
                or aco is None
                or shared_behavior_update_rng is None
            ):
                raise RuntimeError(
                    "Shared behavior replay rehearsal requires an active route and "
                    "actor-critic plus its independent schedule RNG"
                )
            replay_task_ids = replay.available_task_ids()
            expected_replay_task_ids = tuple(range(current_task_id + 1))
            if replay_task_ids != expected_replay_task_ids:
                raise RuntimeError(
                    "Shared behavior fixed-budget rehearsal requires Replay coverage "
                    "for every seen task route: "
                    f"available={replay_task_ids}, expected={expected_replay_task_ids}"
                )
            behavior_allocation = allocate_task_updates(
                config.ac_train_steps,
                current_task_id=current_task_id,
                available_task_ids=replay_task_ids,
                current_task_fraction=(
                    config.evolving_shared_behavior_current_task_fraction
                ),
            )
            behavior_schedule = shuffled_task_schedule(
                behavior_allocation, shared_behavior_update_rng
            )
            behavior_metric_namespace = (
                "EvolvingCoreAdaptiveBehavior"
                if config.uses_adaptive_behavior_compression
                else "EvolvingCoreSharedFastKAN"
            )
            for task_id, task_steps in behavior_allocation.items():
                writer.add_scalar(
                    f"{behavior_metric_namespace}/actor_critic_updates_task_{task_id}",
                    task_steps,
                    (epoch + 1) * config.ac_train_steps,
                )
                shared_behavior_replay_updates[task_id] = (
                    shared_behavior_replay_updates.get(task_id, 0) + task_steps
                )
            aco, approx_perf, actor_critic_metrics = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                local_ac_train_sync,
                aco=aco,
                lr=scheduled_ac_lr,
                task_id_schedule=behavior_schedule,
                training_task_id=(
                    current_task_id
                    if config.uses_adaptive_behavior_compression
                    else None
                ),
                **actor_critic_kwargs,
            )
        elif config.uses_shared_actor:
            if current_task_id is None:
                raise RuntimeError("Shared-actor training requires a current task route")
            distillation_kwargs = {}
            if current_task_id > 0:
                if shared_actor_teacher is None:
                    raise RuntimeError(
                        "Old-task actor distillation requires the frozen teacher"
                    )
                distillation_kwargs = {
                    "actor_teacher": shared_actor_teacher,
                    "actor_distill_task_ids": tuple(range(current_task_id)),
                    "actor_distill_scale": config.shared_actor_distill_scale,
                    "actor_distill_interval": config.shared_actor_distill_interval,
                    "actor_distill_n_sync": config.shared_actor_distill_n_sync,
                    "actor_distill_burnin_steps": (
                        config.shared_actor_distill_burnin_steps
                    ),
                    "actor_distill_steps": config.shared_actor_distill_steps,
                }
            aco, approx_perf, actor_critic_metrics = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                local_ac_train_sync,
                aco=aco,
                lr=scheduled_ac_lr,
                task_id=current_task_id,
                **distillation_kwargs,
                **actor_critic_kwargs,
            )
            writer.add_scalar(
                f"CNNCompactSharedActor/actor_updates_task_{current_task_id}",
                config.ac_train_steps,
                (epoch + 1) * config.ac_train_steps,
            )
            shared_actor_distillation_counters["optimizer_updates"] += (
                config.ac_train_steps
            )
            shared_actor_distillation_counters["distillation_batches"] += int(
                actor_critic_metrics["shared_actor_distillation_batches"]
            )
            shared_actor_distillation_counters["distilled_states"] += int(
                actor_critic_metrics["shared_actor_distillation_states"]
            )
            shared_actor_distillation_counters["burnin_state_uses"] += int(
                actor_critic_metrics[
                    "shared_actor_distillation_burnin_state_uses"
                ]
            )
        elif config.uses_task_experts:
            replay_task_ids = replay.available_task_ids()
            actor_available_tasks = tuple(
                task_id
                for task_id in actor_critic_bank.task_ids()
                if task_id in replay_task_ids
            )
            actor_allocation = allocate_task_updates(
                config.ac_train_steps,
                current_task_id=current_task_id,
                available_task_ids=actor_available_tasks,
                current_task_fraction=config.task_update_fraction,
            )
            actor_metric_totals: dict[str, float] = {}
            approx_perf_total = 0.0
            for task_id, task_steps in actor_allocation.items():
                writer.add_scalar(
                    (
                        f"CNNFullBankArrow/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "cnn_fullbank_arrow"
                        else f"CNNProjectorLoraArrow/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "cnn_projector_lora_arrow"
                        else f"CNNMechanismBank/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "cnn_mechanism_bank_arrow"
                        else f"RECRSSM/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "rec_rssm_arrow"
                        else f"EvolvingCore/actor_critic_updates_task_{task_id}"
                        if config.uses_evolving_atomic_rssm
                        else f"DINOPatchBankArrow/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "dino_patchbank_arrow"
                        else f"DINOConvBankArrow/actor_critic_updates_task_{task_id}"
                        if config.continual_method == "dino_convbank_arrow"
                        else f"DINOFullBankArrow/actor_critic_updates_task_{task_id}"
                        if config.uses_full_task_experts
                        else f"MoEArrow/actor_critic_updates_task_{task_id}"
                    ),
                    task_steps,
                    (epoch + 1) * config.ac_train_steps,
                )
                if task_steps == 0:
                    continue
                task_aco, task_perf, task_metrics = train_ac_from_wm(
                    wm,
                    replay,
                    task_steps,
                    local_ac_train_sync,
                    aco=actor_critic_bank.get(task_id),
                    lr=scheduled_ac_lr,
                    task_id=task_id,
                    **actor_critic_kwargs,
                )
                if task_aco is not actor_critic_bank.get(task_id):
                    raise RuntimeError("Actor training replaced a task-bank entry")
                approx_perf_total += float(task_perf.detach().item()) * task_steps
                for metric_name, metric_value in task_metrics.items():
                    actor_metric_totals[metric_name] = (
                        actor_metric_totals.get(metric_name, 0.0)
                        + metric_value * task_steps
                    )
                    writer.add_scalar(
                        f"ActorCriticTask/{task_id}/{metric_name}",
                        metric_value,
                        (epoch + 1) * config.ac_train_steps,
                    )
            approx_perf = approx_perf_total / config.ac_train_steps
            actor_critic_metrics = {
                name: total / config.ac_train_steps
                for name, total in actor_metric_totals.items()
            }
            aco = actor_critic_bank.get(current_task_id)
        elif config.fresh_ac and epoch % config.fresh_ac == 0:
            aco, approx_perf, actor_critic_metrics = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                local_ac_train_sync,
                lr=(
                    config.ac_fresh_lr
                    if config.ac_schedule == "constant"
                    else scheduled_ac_lr
                ),
                **actor_critic_kwargs,
            )
        else:
            aco, approx_perf, actor_critic_metrics = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                local_ac_train_sync,
                aco=aco,
                lr=scheduled_ac_lr,
                **actor_critic_kwargs,
            )

        if config.uses_bounded_dream_rehearsal:
            from clworldmodel.continual.dream_rehearsal import (
                crossed_rehearsal_intervals,
                rehearsal_update_allocation,
            )

            if replay_task_id is None or aco is None:
                raise RuntimeError(
                    "Bounded Dream Rehearsal requires its scheduler task and "
                    "persistent Actor-Critic"
                )
            interval_count = crossed_rehearsal_intervals(
                agent_decisions_before_epoch,
                total_agent_decisions,
                config.dream_rehearsal_interval_agent_decisions,
            )
            # Exclude only the task supplying this epoch's real data.  This
            # remains correct for reversed curricula and for the published
            # 541st epoch, where the cyclic schedule revisits its first task.
            prior_task_ids = tuple(
                sorted(encountered_replay_task_ids - {replay_task_id})
            )
            available_task_ids = replay.available_task_ids()
            missing_task_ids = sorted(
                set(prior_task_ids) - set(available_task_ids)
            )
            if missing_task_ids:
                raise RuntimeError(
                    "The fixed-capacity reservoir lost all rehearsal starts for "
                    f"prior tasks {missing_task_ids}; available={available_task_ids}"
                )
            allocation = rehearsal_update_allocation(
                interval_count,
                prior_task_ids,
                config.dream_rehearsal_updates_per_prior_task,
            )
            rehearsal_metric_totals: dict[str, float] = {}
            for old_task_id, task_updates in allocation.items():
                if not task_updates:
                    continue
                task_metrics = train_bounded_dream_rehearsal(
                    wm,
                    replay,
                    aco,
                    replay_task_id=old_task_id,
                    updates=task_updates,
                    n_sync=config.dream_rehearsal_batch_sequences,
                    context_steps=config.dream_rehearsal_context_steps,
                    dream_steps=config.dream_rehearsal_horizon,
                    discount=config.ac_discount,
                    top_fraction=config.dream_rehearsal_top_fraction,
                    realized_threshold=(
                        config.dream_rehearsal_realized_threshold
                    ),
                    realized_bonus=config.dream_rehearsal_realized_bonus,
                    grad_clip=config.dream_rehearsal_grad_clip,
                )
                dream_rehearsal_updates_by_task[old_task_id] = (
                    dream_rehearsal_updates_by_task.get(old_task_id, 0)
                    + task_updates
                )
                dream_rehearsal_updates += task_updates
                dream_rehearsal_dreamed_trajectories += int(
                    task_metrics["dreamed_trajectories"]
                )
                dream_rehearsal_selected_trajectories += int(
                    task_metrics["selected_trajectories"]
                )
                for metric_name, metric_value in task_metrics.items():
                    writer.add_scalar(
                        f"BoundedDreamRehearsal/task_{old_task_id}/{metric_name}",
                        metric_value,
                        dream_rehearsal_updates,
                    )
                    rehearsal_metric_totals[metric_name] = (
                        rehearsal_metric_totals.get(metric_name, 0.0)
                        + metric_value * task_updates
                    )
            dream_rehearsal_intervals += interval_count
            writer.add_scalar(
                "Counters/dream_rehearsal_actor_updates",
                dream_rehearsal_updates,
                total_agent_decisions,
            )
            writer.add_scalar(
                "Counters/dream_rehearsal_intervals",
                dream_rehearsal_intervals,
                total_agent_decisions,
            )
            updates_this_epoch = sum(allocation.values())
            if updates_this_epoch:
                actor_critic_metrics.update(
                    {
                        f"dream_rehearsal_{name}": total / updates_this_epoch
                        for name, total in rehearsal_metric_totals.items()
                        if name
                        not in {
                            "actor_updates",
                            "selected_trajectories",
                            "dreamed_trajectories",
                        }
                    }
                )
            if distributed_context.is_primary:
                accounting = {
                    "schema_version": 1,
                    "artifact_kind": "bounded_dream_rehearsal_accounting",
                    "method": "Bounded-Dream-Rehearsal-v1-Atari",
                    "replay": {
                        "retention": "uniform_random_key_reservoir",
                        "trajectory_slots": replay.n,
                        "sequence_length": replay.t,
                        "transition_capacity": replay.n * replay.t,
                        "valid_trajectory_slots": replay.n_valid,
                        "observation_dtype": config.replay_observation_dtype,
                        "task_ids_are_replay_metadata_only": True,
                        "task_ids_exposed_to_model_or_actor": False,
                        "available_task_ids": list(available_task_ids),
                    },
                    "schedule": {
                        "interval_agent_decisions": (
                            config.dream_rehearsal_interval_agent_decisions
                        ),
                        "updates_per_prior_task_per_interval": (
                            config.dream_rehearsal_updates_per_prior_task
                        ),
                        "collection_granularity_agent_decisions": (
                            total_agent_decisions - agent_decisions_before_epoch
                        ),
                        "due_updates_are_batched_after_the_epoch_actor_update": True,
                    },
                    "counters": {
                        "agent_decisions": total_agent_decisions,
                        "completed_rehearsal_intervals": (
                            dream_rehearsal_intervals
                        ),
                        "extra_actor_updates": dream_rehearsal_updates,
                        "extra_actor_updates_by_prior_task": dict(
                            sorted(dream_rehearsal_updates_by_task.items())
                        ),
                        "dreamed_trajectories": (
                            dream_rehearsal_dreamed_trajectories
                        ),
                        "selected_trajectories": (
                            dream_rehearsal_selected_trajectories
                        ),
                    },
                    "actor_only_rehearsal": True,
                    "world_model_and_critic_updated_by_rehearsal": False,
                    "evaluation_transitions_enter_training": False,
                }
                accounting_path = log_dir / "bounded_dream_rehearsal_accounting.json"
                temporary_accounting_path = accounting_path.with_suffix(
                    ".json.tmp"
                )
                temporary_accounting_path.write_text(
                    json.dumps(accounting, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_accounting_path, accounting_path)

        actor_seconds = _stage_elapsed(actor_started, profile_stages)
        if distributed_context.is_primary and (
            actor_critic_bank is not None
            or config.uses_shared_actor
            or not actor_accounting_path.exists()
        ):
            actor_accounting = _active_actor_critic_parameter_accounting(
                config=config,
                aco=aco,
                actor_critic_bank=actor_critic_bank,
                shared_actor_teacher=shared_actor_teacher,
            )
            _write_json_atomically(actor_accounting_path, actor_accounting)
            if config.uses_replay_rehearsed_shared_behavior:
                replay_accounting = {
                    "schema_version": 1,
                    "artifact_kind": (
                        "evolving_core_shared_behavior_replay_accounting"
                    ),
                    "method": config.continual_method,
                    "topology": (
                        "single_shared_mlp_plus_task_adaptive_residuals"
                        if config.uses_adaptive_behavior_compression
                        else "single_shared_fastkan_actor_critic"
                    ),
                    "fixed_optimizer_updates_per_epoch": config.ac_train_steps,
                    "optimizer_updates_are_extra": False,
                    "current_task_fraction_when_old_tasks_exist": (
                        config.evolving_shared_behavior_current_task_fraction
                    ),
                    "old_task_allocation": "uniform_over_available_completed_tasks",
                    "route_schedule": "independently_seeded_shuffled_exact_counts",
                    "replay_source": (
                        "task-conditioned ARROW mixed replay; unchanged FIFO/LTDM "
                        "weights are renormalized over sub-buffers containing the "
                        "requested task"
                    ),
                    "real_old_task_replay_used": any(
                        task_id < current_task_id
                        and update_count > 0
                        for task_id, update_count in shared_behavior_replay_updates.items()
                    ),
                    "evaluation_transitions_enter_training": False,
                    "actor_and_critic_both_updated_on_rehearsal": True,
                    "world_model_interface_teacher": (
                        "one transient frozen cumulative shared actor from the "
                        "previous task boundary"
                    ),
                    "actor_imagination_distillation": False,
                    "optimizer_updates_by_task_route": dict(
                        sorted(shared_behavior_replay_updates.items())
                    ),
                    "optimizer_updates_total": sum(
                        shared_behavior_replay_updates.values()
                    ),
                }
                replay_accounting_path = (
                    log_dir / "shared_behavior_replay_accounting.json"
                )
                temporary_replay_accounting_path = (
                    replay_accounting_path.with_suffix(".json.tmp")
                )
                temporary_replay_accounting_path.write_text(
                    json.dumps(replay_accounting, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(
                    temporary_replay_accounting_path,
                    replay_accounting_path,
                )
            elif config.uses_shared_actor:
                distillation_accounting = {
                    "schema_version": 1,
                    "method": config.continual_method,
                    "real_old_task_replay_samples": 0,
                    "evaluation_transitions_enter_training": False,
                    "old_state_source": (
                        "zero_initialized_frozen_world_model_routes"
                    ),
                    "teacher_topology": "one transient previous shared actor",
                    "teacher_persistent": False,
                    "counts": dict(shared_actor_distillation_counters),
                }
                distillation_accounting_path = (
                    log_dir / "shared_actor_distillation_accounting.json"
                )
                temporary_distillation_accounting_path = (
                    distillation_accounting_path.with_suffix(".json.tmp")
                )
                temporary_distillation_accounting_path.write_text(
                    json.dumps(distillation_accounting, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(
                    temporary_distillation_accounting_path,
                    distillation_accounting_path,
                )
        writer.add_scalar("Perf/approx_perf", approx_perf, global_step)
        actor_critic_updates = (epoch + 1) * config.ac_train_steps
        writer.add_scalar(
            "Counters/actor_critic_updates",
            actor_critic_updates,
            actor_critic_updates,
        )
        writer.add_scalar(
            "Counters/actor_updates_total_including_dream_rehearsal",
            actor_critic_updates + dream_rehearsal_updates,
            total_agent_decisions,
        )
        for metric_name, metric_value in actor_critic_metrics.items():
            writer.add_scalar(
                f"ActorCritic/{metric_name}",
                metric_value,
                actor_critic_updates,
            )
        if distributed_context.is_primary:
            _print_cuda_memory(f"epoch_end_{epoch}")

        boundary_snapshot_metadata = (
            _task_boundary_metadata(config, epoch)
            if task_bank_snapshot_dir is not None
            else None
        )
        if (
            distributed_context.is_primary
            and config.uses_evolving_atomic_rssm
            and boundary_snapshot_metadata is not None
        ):
            completed_task_id = int(boundary_snapshot_metadata["task_index"])
            if current_task_id != completed_task_id:
                raise RuntimeError(
                    "Evolving-Core boundary task does not match the active route: "
                    f"{completed_task_id} != {current_task_id}"
                )
            if evolving_shared_optimizer is None:
                raise RuntimeError(
                    "Evolving-Core boundary requires its shared optimizer"
                )
            if config.uses_replay_rehearsed_shared_behavior:
                if (
                    actor_critic_bank is not None
                    or aco is None
                    or shared_behavior_update_rng is None
                ):
                    raise RuntimeError(
                        "Shared behavior boundary requires one actor-critic and its "
                        "route-schedule RNG"
                    )
            elif actor_critic_bank is None:
                raise RuntimeError(
                    "Private-behavior Evolving-Core boundary requires an actor bank"
                )
            provisional_teacher = copy.deepcopy(wm).eval()
            provisional_teacher.requires_grad_(False)
            checkpoint_dir = log_dir / "evolving_core_checkpoints"
            pre_checkpoint = checkpoint_dir / (
                f"task_{completed_task_id:02d}_pre_consolidation.pt"
            )
            _save_evolving_resumable_checkpoint(
                pre_checkpoint,
                config=config,
                wm=wm,
                boundary_teacher=provisional_teacher,
                shared_optimizer=evolving_shared_optimizer,
                private_optimizers=evolving_private_optimizers,
                route_optimizers=evolving_route_optimizers,
                actor_critic_bank=actor_critic_bank,
                aco=(
                    aco if config.uses_replay_rehearsed_shared_behavior else None
                ),
                shared_behavior_update_rng=(
                    shared_behavior_update_rng
                    if config.uses_replay_rehearsed_shared_behavior
                    else None
                ),
                shared_behavior_replay_updates=(
                    shared_behavior_replay_updates
                    if config.uses_replay_rehearsed_shared_behavior
                    else None
                ),
                replay_buffer=replay,
                environment_schedule=envs,
                epoch=epoch,
                current_task_id=completed_task_id,
                world_model_updates=global_step,
                actor_critic_updates=actor_critic_updates,
                adaptive_behavior_compression_updates=(
                    adaptive_behavior_compression_updates
                ),
                total_env_steps=total_env_steps,
                task_update_rng=task_update_rng,
                collection_environment_seed_rng=collection_environment_seed_rng,
                validation_environment_seed_rng=validation_environment_seed_rng,
                final_environment_seed_rng=final_environment_seed_rng,
            )
            consolidation_succeeded = False
            boundary_shared_before = wm.shared_core_state_dict()
            boundary_optimizer_before = copy.deepcopy(
                evolving_shared_optimizer.state_dict()
            )
            try:
                consolidation = _consolidate_evolving_shared_core(
                    config=config,
                    wm=wm,
                    shared_optimizer=evolving_shared_optimizer,
                    replay_buffer=replay,
                    actor_critic_bank=actor_critic_bank,
                    aco=(
                        aco
                        if config.uses_replay_rehearsed_shared_behavior
                        else None
                    ),
                    completed_task_id=completed_task_id,
                    eval_funcs=envs.eval_funcs(),
                    validation_task_seeds=validation_task_seeds,
                    epoch=epoch,
                    global_step=global_step,
                    log_dir=log_dir,
                    writer=writer,
                )
                global_step += config.boundary_consolidation_steps
                consolidation_succeeded = True
                print(
                    "Consolidated Evolving-Core shared RSSM: "
                    f"task={completed_task_id} "
                    f"rollback={consolidation['rollback']}"
                )
            except Exception as exc:
                # Consolidation is an explicitly isolated post-training phase.
                # Its helper restores shared weights/Adam state before raising;
                # preserve the completed pre-consolidation checkpoint and record
                # the failure rather than corrupting the completed task.
                wm.load_shared_core_state_dict(boundary_shared_before)
                evolving_shared_optimizer.load_state_dict(
                    boundary_optimizer_before
                )
                wm.activate_task_expert(completed_task_id)
                failure = {
                    "schema_version": 1,
                    "artifact_kind": "evolving_core_consolidation_failure",
                    "task_id": completed_task_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "pre_consolidation_checkpoint": str(pre_checkpoint),
                    "formal_training_state_preserved": True,
                }
                failure_path = checkpoint_dir / (
                    f"task_{completed_task_id:02d}_consolidation_failure.json"
                )
                temporary_failure = failure_path.with_suffix(".json.tmp")
                temporary_failure.write_text(
                    json.dumps(failure, indent=2) + "\n", encoding="utf-8"
                )
                os.replace(temporary_failure, failure_path)
                if config.uses_reconstruction_task_inference:
                    raise
                print(
                    "Evolving-Core consolidation failed after safe rollback; "
                    f"continuing from completed task state: {type(exc).__name__}: {exc}"
                )
            if config.uses_adaptive_qfp_compression:
                compression_dense_state = _cpu_state_dict(wm)
                try:
                    compression = _compress_evolving_task_qfp(
                        config=config,
                        wm=wm,
                        replay_buffer=replay,
                        actor_critic_bank=actor_critic_bank,
                        aco=(
                            aco
                            if config.uses_replay_rehearsed_shared_behavior
                            else None
                        ),
                        completed_task_id=completed_task_id,
                        eval_env_fns=envs.eval_funcs()[completed_task_id],
                        validation_seed=(
                            compression_validation_task_seeds[completed_task_id]
                        ),
                        epoch=epoch,
                        global_step=global_step,
                        log_dir=log_dir,
                        writer=writer,
                        fused_adam=args.fused_adam,
                        seen_eval_env_fns=envs.eval_funcs()[:completed_task_id + 1],
                        seen_validation_seeds=compression_validation_task_seeds[:completed_task_id + 1],
                    )
                except Exception as exc:
                    wm.load_state_dict(compression_dense_state, strict=True)
                    wm.activate_task_expert(completed_task_id)
                    failure = {
                        "schema_version": 1,
                        "artifact_kind": "adaptive_qfp_compression_failure",
                        "task_id": completed_task_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "pre_consolidation_checkpoint": str(pre_checkpoint),
                        "dense_topology_restored_before_failure": True,
                        "training_stopped": True,
                        "final_heldout_data_used": False,
                    }
                    failure_dir = log_dir / "adaptive_qfp_compression"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure_path = failure_dir / (
                        f"task_{completed_task_id:02d}_failure.json"
                    )
                    temporary_failure = failure_path.with_suffix(".json.tmp")
                    temporary_failure.write_text(
                        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary_failure, failure_path)
                    raise
                global_step += int(compression["optimizer_updates"])
                compression_dense_state = None
                retired_private_optimizer = evolving_private_optimizers.pop(
                    completed_task_id, None
                )
                if retired_private_optimizer is None:
                    raise RuntimeError(
                        "Adaptive compression could not retire the completed "
                        "task's stale Dense private optimizer"
                    )
                evolving_route_optimizers.pop(completed_task_id, None)
                # Drop loop-local references as well: the installed compact
                # module owns new Parameter objects and no completed-task Adam
                # state is ever used again in this single-pass curriculum.
                private_optimizer = None
                route_optimizer = None
                retired_private_optimizer = None
                writer.add_scalar(
                    "Counters/world_model_updates", global_step, global_step
                )
                print(
                    "Compressed completed task Q/F/P after Dense acquisition: "
                    f"task={completed_task_id} "
                    f"selected_fraction={compression['selected_width_fraction']} "
                    f"dense_fallback={compression['selected_dense_fallback']} "
                    f"layout={compression['selected_layout']}"
                )
            if config.uses_adaptive_behavior_compression:
                if aco is None:
                    raise RuntimeError(
                        "Adaptive behavior compression requires its shared "
                        "Actor-Critic"
                    )
                behavior_dense_state = copy.deepcopy(
                    _actor_critic_opt_resumable_state_dict(aco)
                )
                try:
                    behavior_compression = _compress_evolving_task_actor_critic(
                        config=config,
                        wm=wm,
                        aco=aco,
                        replay_buffer=replay,
                        completed_task_id=completed_task_id,
                        eval_env_fns=envs.eval_funcs()[completed_task_id],
                        validation_seed=(
                            behavior_compression_validation_task_seeds[
                                completed_task_id
                            ]
                        ),
                        epoch=epoch,
                        actor_critic_updates=actor_critic_updates,
                        compression_updates_before=(
                            adaptive_behavior_compression_updates
                        ),
                        log_dir=log_dir,
                        writer=writer,
                        fused_adam=args.fused_adam,
                    )
                except Exception as exc:
                    _load_actor_critic_opt_resumable_state_dict(
                        aco, behavior_dense_state
                    )
                    aco.ac.set_task_route(completed_task_id)
                    failure = {
                        "schema_version": 1,
                        "artifact_kind": (
                            "adaptive_actor_critic_compression_failure"
                        ),
                        "task_id": completed_task_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "pre_consolidation_checkpoint": str(pre_checkpoint),
                        "dense_topology_and_optimizer_restored_before_failure": True,
                        "training_stopped": True,
                        "final_heldout_data_used": False,
                    }
                    failure_dir = log_dir / "adaptive_behavior_compression"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure_path = failure_dir / (
                        f"task_{completed_task_id:02d}_failure.json"
                    )
                    temporary_failure = failure_path.with_suffix(".json.tmp")
                    temporary_failure.write_text(
                        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary_failure, failure_path)
                    raise
                adaptive_behavior_compression_updates += int(
                    behavior_compression["optimizer_updates"]
                )
                behavior_dense_state = None
                writer.add_scalar(
                    "Counters/adaptive_behavior_compression_updates",
                    adaptive_behavior_compression_updates,
                    actor_critic_updates,
                )
                print(
                    "Compressed completed task Actor/Critic residuals after "
                    "Dense acquisition: "
                    f"task={completed_task_id} "
                    "selected_fraction="
                    f"{behavior_compression['selected_width_fraction']} "
                    "dense_fallback="
                    f"{behavior_compression['selected_dense_fallback']} "
                    f"layout={behavior_compression['selected_layout']}"
                )
            # The per-epoch artifacts above describe the full-width acquisition
            # topology. Rewrite them after both selectors so the boundary (and,
            # for the last task, final) ledger describes the modules that are
            # actually retained online rather than a stale pre-pruning count.
            if distributed_context.is_primary:
                if wm.rssm.task_mechanism_bank_enabled:
                    _write_json_atomically(
                        log_dir / "model_parameter_accounting.json",
                        _world_model_parameter_accounting(wm),
                    )
                _write_json_atomically(
                    actor_accounting_path,
                    _active_actor_critic_parameter_accounting(
                        config=config,
                        aco=aco,
                        actor_critic_bank=actor_critic_bank,
                        shared_actor_teacher=shared_actor_teacher,
                    ),
                )
            boundary_teacher = copy.deepcopy(wm).eval()
            boundary_teacher.requires_grad_(False)
            post_checkpoint = checkpoint_dir / (
                f"task_{completed_task_id:02d}_post_consolidation.pt"
            )
            _save_evolving_resumable_checkpoint(
                post_checkpoint,
                config=config,
                wm=wm,
                boundary_teacher=boundary_teacher,
                shared_optimizer=evolving_shared_optimizer,
                private_optimizers=evolving_private_optimizers,
                route_optimizers=evolving_route_optimizers,
                actor_critic_bank=actor_critic_bank,
                aco=(
                    aco if config.uses_replay_rehearsed_shared_behavior else None
                ),
                shared_behavior_update_rng=(
                    shared_behavior_update_rng
                    if config.uses_replay_rehearsed_shared_behavior
                    else None
                ),
                shared_behavior_replay_updates=(
                    shared_behavior_replay_updates
                    if config.uses_replay_rehearsed_shared_behavior
                    else None
                ),
                replay_buffer=replay,
                environment_schedule=envs,
                epoch=epoch,
                current_task_id=completed_task_id,
                world_model_updates=global_step,
                actor_critic_updates=actor_critic_updates,
                adaptive_behavior_compression_updates=(
                    adaptive_behavior_compression_updates
                ),
                total_env_steps=total_env_steps,
                task_update_rng=task_update_rng,
                collection_environment_seed_rng=collection_environment_seed_rng,
                validation_environment_seed_rng=validation_environment_seed_rng,
                final_environment_seed_rng=final_environment_seed_rng,
            )
            retention = _apply_evolving_checkpoint_retention(
                checkpoint_dir,
                completed_task_id=completed_task_id,
                retention=config.evolving_checkpoint_retention,
            )
            if retention["removed_older_artifacts"]:
                print(
                    "Applied Evolving-Core checkpoint retention: "
                    f"mode={config.evolving_checkpoint_retention} "
                    f"completed_task={completed_task_id} "
                    f"removed={len(retention['removed_older_artifacts'])}"
                )
            writer.add_scalar(
                "EvolvingCoreConsolidation/succeeded",
                int(consolidation_succeeded),
                global_step,
            )
        if (
            distributed_context.is_primary
            and config.continual_method == "rec_rssm_arrow"
            and boundary_snapshot_metadata is not None
            and int(boundary_snapshot_metadata["task_index"]) >= 1
        ):
            completed_task_id = int(boundary_snapshot_metadata["task_index"])
            if current_task_id != completed_task_id:
                raise RuntimeError(
                    "REC-RSSM boundary task does not match the active route: "
                    f"{completed_task_id} != {current_task_id}"
                )
            if aco is None:
                raise RuntimeError(
                    "REC-RSSM consolidation requires the completed task actor"
                )
            consolidation = _consolidate_rec_routes(
                config=config,
                wm=wm,
                aco=aco,
                replay_buffer=replay,
                completed_task_id=completed_task_id,
                eval_env_fns=envs.eval_funcs()[completed_task_id],
                validation_seed=validation_task_seeds[completed_task_id],
                epoch=epoch,
                global_step=global_step,
                log_dir=log_dir,
                writer=writer,
            )
            print(
                "Consolidated REC-RSSM atom route: "
                f"task={completed_task_id} "
                f"candidates={consolidation['candidate_count']} "
                f"rollback={consolidation['rollback']}"
            )

        if distributed_context.is_primary and (
            save_nets
            or (config.uses_task_experts and epoch == config.epochs - 1)
        ):
            torch.save(wm.state_dict(), log_dir / "save_wm.pt")
            torch.save(aco.ac.state_dict(), log_dir / "save_ac.pt")
            if actor_critic_bank is not None:
                torch.save(
                    actor_critic_bank.inference_state_dict(),
                    log_dir / "save_ac_bank.pt",
                )
            if wm.rssm.task_mechanism_bank_enabled:
                final_accounting_path = log_dir / "model_parameter_accounting.json"
                _write_json_atomically(
                    final_accounting_path,
                    _world_model_parameter_accounting(wm),
                )
            _write_json_atomically(
                actor_accounting_path,
                _active_actor_critic_parameter_accounting(
                    config=config,
                    aco=aco,
                    actor_critic_bank=actor_critic_bank,
                    shared_actor_teacher=shared_actor_teacher,
                ),
            )

        if (
            distributed_context.is_primary
            and boundary_snapshot_metadata is not None
        ):
            _save_task_bank_boundary_snapshot(
                task_bank_snapshot_dir,
                config=config,
                wm=wm,
                actor_critic_bank=actor_critic_bank,
                aco=aco,
                epoch=epoch,
                world_model_updates=global_step,
                total_env_steps=total_env_steps,
                task_metadata=boundary_snapshot_metadata,
                project_git_commit=str(args.project_git_commit),
            )
            writer.flush()
        if boundary_snapshot_metadata is not None:
            distributed_context.barrier()

        if distributed_context.is_primary and analysis_snapshot_dir is not None:
            boundary_metadata = _task_boundary_metadata(config, epoch)
            if boundary_metadata is not None:
                _save_analysis_snapshot(
                    analysis_snapshot_dir,
                    config=config,
                    wm=wm,
                    aco=aco,
                    epoch=epoch,
                    world_model_updates=global_step,
                    total_env_steps=total_env_steps,
                    reason="task_boundary",
                    task_metadata=boundary_metadata,
                    dream_rehearsal_actor_updates=dream_rehearsal_updates,
                )
                writer.flush()
            if (
                epoch + 1 in milestone_completed_epochs
                and boundary_metadata is None
                and epoch != config.epochs - 1
            ):
                _save_analysis_snapshot(
                    analysis_snapshot_dir,
                    config=config,
                    wm=wm,
                    aco=aco,
                    epoch=epoch,
                    world_model_updates=global_step,
                    total_env_steps=total_env_steps,
                    reason="milestone",
                    dream_rehearsal_actor_updates=dream_rehearsal_updates,
                )
                writer.flush()
            if epoch == config.epochs - 1:
                _save_analysis_snapshot(
                    analysis_snapshot_dir,
                    config=config,
                    wm=wm,
                    aco=aco,
                    epoch=epoch,
                    world_model_updates=global_step,
                    total_env_steps=total_env_steps,
                    reason="final",
                    dream_rehearsal_actor_updates=dream_rehearsal_updates,
                )
                writer.flush()

        envs.step()
        torch.cuda.empty_cache()
        if profile_stages:
            epoch_seconds = _stage_elapsed(epoch_started, True)
            measured = (
                collect_seconds + eval_seconds + world_model_seconds + actor_seconds
            )
            print(
                "[stage-time] "
                f"epoch={epoch} collect={collect_seconds:.3f}s "
                f"eval={eval_seconds:.3f}s "
                f"world_model={world_model_seconds:.3f}s "
                f"actor={actor_seconds:.3f}s "
                f"overhead={max(0.0, epoch_seconds - measured):.3f}s "
                f"total={epoch_seconds:.3f}s"
            )

    if args.evaluate_final:
        eval_funcs = envs.eval_funcs()
        task_configs = config.esc.env_configs
        if config.esc.env_schedule_type is SequentialEnvironments:
            seen_tasks = min(
                len(task_configs),
                _sequential_seen_task_count(config, config.epochs),
            )
            eval_funcs = eval_funcs[:seen_tasks]
            task_configs = task_configs[:seen_tasks]
        final_eval_task_seeds = (
            final_task_seeds[: len(eval_funcs)]
            if fixed_evaluation_cohorts
            else tuple(
                _next_environment_seed(validation_environment_seed_rng)
                for _ in eval_funcs
            )
        )
        final_routing = []
        final_scaled_means, final_scaled_stds = _evaluate_policy_tasks(
            config,
            wm,
            aco,
            eval_funcs,
            final_eval_task_seeds,
            actor_critic_bank=actor_critic_bank,
            distributed_context=distributed_context,
            eligible_task_count=len(eval_funcs),
            routing_diagnostics=final_routing,
        )
        final_raw_means, final_raw_stds = _raw_return_statistics(
            task_configs, final_scaled_means, final_scaled_stds
        )
        final_evaluation = {
            "schema_version": 1,
            "evaluation_after_completed_epochs": config.epochs,
            "seed_cohort": (
                "heldout_final"
                if fixed_evaluation_cohorts
                else "advancing_evaluation_stream"
            ),
            "seed_cohort_used_for_periodic_validation": (
                not fixed_evaluation_cohorts
            ),
            "policy": (
                "deterministic_argmax_and_latent_mode"
                if config.uses_task_experts
                else "stochastic"
            ),
            "rollouts_per_task": 16,
            "tasks": [
                {
                    "task_index": index,
                    "task_name": task.name,
                    "reward_scale": task.rew_scale,
                    "scaled_return_mean": scaled_mean,
                    "scaled_return_std": scaled_std,
                    "raw_return_mean": raw_mean,
                    "raw_return_std": raw_std,
                }
                for index, (
                    task,
                    scaled_mean,
                    scaled_std,
                    raw_mean,
                    raw_std,
                ) in enumerate(
                    zip(
                        task_configs,
                        final_scaled_means,
                        final_scaled_stds,
                        final_raw_means,
                        final_raw_stds,
                    )
                )
            ],
        }
        if config.uses_reconstruction_task_inference:
            final_evaluation.update({
                "policy": "first_frame_reconstruction_episode_lock_argmax_latent_mode",
                "task_identity_exposed_during_inference": False,
                "task_aware_training": True,
                "eligible_route_ids": list(range(len(eval_funcs))),
                "episode_count_mode": "exact", "routing": final_routing,
            })
        if distributed_context.is_primary:
            final_evaluation_path = log_dir / "final_evaluation.json"
            temporary_final_evaluation_path = final_evaluation_path.with_suffix(
                ".json.tmp"
            )
            temporary_final_evaluation_path.write_text(
                json.dumps(final_evaluation, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_final_evaluation_path, final_evaluation_path)
            writer.add_scalars(
                "Perf/final_eval_rew_eps_mean",
                {f"{i}": mean for i, mean in enumerate(final_scaled_means)},
                global_step,
            )
            writer.add_scalars(
                "Perf/final_eval_rew_eps_std",
                {f"{i}": std for i, std in enumerate(final_scaled_stds)},
                global_step,
            )
            writer.add_scalars(
                "Perf/final_eval_raw_return_mean",
                {f"{i}": mean for i, mean in enumerate(final_raw_means)},
                global_step,
            )
            writer.add_scalars(
                "Perf/final_eval_raw_return_std",
                {f"{i}": std for i, std in enumerate(final_raw_stds)},
                global_step,
            )
            print(f"Final eval scaled means: {final_scaled_means}")
            print(f"Final eval scaled stds: {final_scaled_stds}")
            print(f"Final eval raw means: {final_raw_means}")
            print(f"Final eval raw stds: {final_raw_stds}")
            if evaluation_snapshot_dir is not None:
                _save_task_bank_evaluation_snapshot(
                    evaluation_snapshot_dir,
                    config=config,
                    wm=wm,
                    actor_critic_bank=actor_critic_bank,
                    aco=aco,
                    completed_epochs=config.epochs,
                    world_model_updates=global_step,
                    actor_critic_updates=config.epochs * config.ac_train_steps,
                    total_env_steps=total_env_steps,
                    task_seeds=final_eval_task_seeds,
                    scaled_means=final_scaled_means,
                    scaled_stds=final_scaled_stds,
                    raw_means=final_raw_means,
                    raw_stds=final_raw_stds,
                    cohort="heldout_final",
                )
    writer.close()
    if swanlab_run is not None:
        import swanlab

        swanlab.finish()
    if distributed_context.is_primary:
        _print_cuda_memory("training_end")
    distributed_context.barrier()
    distributed_context.close()
