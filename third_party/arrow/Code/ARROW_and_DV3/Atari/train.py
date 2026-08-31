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
    build_actor_critic_opt,
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


def _mapped_replay_storage_accounting(buf: MultiTypeReplay) -> dict[str, object]:
    buffers = []
    observation_dtypes = set()
    for index, sub_replay in enumerate(buf.replays):
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


def _evaluate_policy_tasks(
    config: Config,
    wm: WorldModel,
    aco: Optional[ActorCriticOpt],
    eval_funcs,
    task_seeds: Sequence[int],
    actor_critic_bank=None,
    distributed_context=None,
) -> tuple[list[float], list[float]]:
    if len(task_seeds) != len(eval_funcs):
        raise ValueError(
            "Evaluation task functions and fixed task seeds must have equal length"
        )
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
    replay_buffer,
    environment_schedule,
    epoch: int,
    current_task_id: int,
    world_model_updates: int,
    actor_critic_updates: int,
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
    payload = {
        "schema_version": 1,
        "artifact_kind": "evolving_core_atomic_rssm_resumable_checkpoint",
        "resumable": True,
        "config": config.to_dict(),
        "world_model": wm.state_dict(),
        "boundary_teacher": boundary_teacher.state_dict(),
        "optimizers": {
            "shared": shared_optimizer.state_dict(),
            "private_by_task": _optimizer_bank_state_dict(private_optimizers),
            "route_by_task": _optimizer_bank_state_dict(route_optimizers),
            "actor_critic_bank": actor_critic_bank.resumable_state_dict(),
        },
        "replay": replay_state,
        "rng": {
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
        },
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
        },
        "replay_checkpoint_semantics": (
            "mapped observations are copied into immutable checkpoint-owned assets; "
            "all other replay tensors and retention indices are embedded"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    temporary_checksum = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    temporary_checksum.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    os.replace(temporary_checksum, checksum_path)
    return path


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
    replay_buffer,
    environment_schedule,
    task_update_rng: np.random.Generator,
    collection_environment_seed_rng: np.random.Generator,
    validation_environment_seed_rng: np.random.Generator,
    final_environment_seed_rng: np.random.Generator,
) -> dict[str, int]:
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
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "evolving_core_atomic_rssm_resumable_checkpoint"
        or payload.get("resumable") is not True
    ):
        raise ValueError("Checkpoint is not resumable Evolving-Core schema v1")
    if payload.get("config") != config.to_dict():
        raise ValueError("Resolved config changed across Evolving-Core resume")

    wm.load_state_dict(payload["world_model"], strict=True)
    boundary_teacher.load_state_dict(payload["boundary_teacher"], strict=True)
    optimizers = payload["optimizers"]
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
    schedule = payload["schedule"]
    environment_schedule._step = int(schedule["environment_step"])
    counters = payload["counters"]
    return {
        "completed_epochs": int(schedule["completed_epochs"]),
        "current_task_id": int(schedule["current_task_id"]),
        "raw_environment_frames": int(counters["raw_environment_frames"]),
        "world_model_updates": int(counters["world_model_updates"]),
        "actor_critic_updates": int(counters["actor_critic_updates"]),
    }


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
    accounting["topology"] = "single_shared_actor_critic"
    accounting["persistent_actor_copies"] = 1
    accounting["per_task_actor_growth"] = 0
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute old-task Dreamer and Q/H/policy interface protection losses."""

    from clworldmodel.continual import interface_distillation_losses

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
        total = (
            dreamer_loss
            + config.interface_q_scale * interface["posterior"]
            + config.interface_h_scale * interface["hidden"]
            + config.interface_actor_scale * interface["actor"]
        )
    metrics = {
        **dreamer_metrics,
        "Loss/evolving_interface_q": interface["posterior"].detach(),
        "Loss/evolving_interface_h": interface["hidden"].detach(),
        "Loss/evolving_interface_actor": interface["actor"].detach(),
        "Loss/evolving_memory_total": total.detach(),
    }
    return total, metrics


def _evolving_world_model_update(
    *,
    config: Config,
    wm: WorldModel,
    boundary_teacher: Optional[WorldModel],
    actor_critic_bank,
    replay_buffer,
    current_task_id: int,
    memory_task_id: Optional[int],
    sequence_length: int,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizer: torch.optim.Optimizer,
    route_optimizer: Optional[torch.optim.Optimizer],
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
        memory_loss, memory_metrics = _evolving_memory_loss(
            config=config,
            wm=wm,
            teacher=boundary_teacher,
            frozen_actor=actor_critic_bank.get(memory_task_id).ac.actor,
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

    pre_scaled_mean, pre_scaled_std = _evaluate_policy_tasks(
        config,
        wm,
        actor_critic_bank.get(completed_task_id),
        seen_eval_funcs,
        seen_validation_seeds,
        actor_critic_bank=actor_critic_bank,
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
            actor_critic_bank.get(completed_task_id),
            seen_eval_funcs,
            seen_validation_seeds,
            actor_critic_bank=actor_critic_bank,
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
        "actor_critic_updates": (epoch + 1) * config.ac_train_steps,
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
) -> Path:
    """Save the exact task-bank weights evaluated by a fixed seed cohort."""
    uses_shared_actor = config.uses_shared_actor
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
    training_start_epoch = 0
    resume_mode = args.resume_adaptation_mode
    if (
        args.init_analysis_snapshot is not None
        and args.init_task1_boundary_snapshot is not None
    ):
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
    elif config.uses_shared_actor:
        raise ValueError(
            "CNN-Compact-SharedActor requires --init-task1-boundary-snapshot"
        )
    elif config.continual_method in {"cnn_mechanism_bank_arrow", "rec_rssm_arrow"}:
        raise ValueError(
            "Mechanism-bank methods require --init-task1-boundary-snapshot"
        )
    elif config.continual_method == "cnn_projector_lora_arrow":
        training_start_epoch = 0

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
            f"reuse_probe_epochs={config.task_mechanism_reuse_probe_epochs} "
            f"route_lr_scale={config.task_mechanism_route_lr_scale} "
            f"current_fraction={config.dino_fullbank_current_task_fraction}"
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
        task_symmetric_mechanisms=config.task_atomic_routes,
    ).to(device)
    resume_world_model_opened: list[str] = []
    resume_state_report: dict[str, dict[str, list[str]]] = {}
    task1_seed_world_model_report: dict[str, int] = {}
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
    evolving_shared_optimizer: Optional[torch.optim.Optimizer] = None
    evolving_private_optimizers: dict[int, torch.optim.Optimizer] = {}
    evolving_route_optimizers: dict[int, torch.optim.Optimizer] = {}
    if config.uses_evolving_atomic_rssm:
        evolving_shared_optimizer = Adam(
            _flatten_parameter_groups(wm.shared_parameter_groups()),
            lr=config.first_task_shared_core_lr,
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
        "cnn_fullbank_arrow",
        "cnn_projector_lora_arrow",
        "cnn_compact_shared_actor_arrow",
        "cnn_mechanism_bank_arrow",
        "rec_rssm_arrow",
        "evolving_atomic_rssm_arrow",
        "dino_patchbank_arrow",
        "dino_convbank_arrow",
    } and (not distributed_context.enabled or distributed_context.is_primary):
        if log_dir is None:
            raise ValueError(
                "Pixel task banks require --log-dir for mapped observation replay"
            )
        mmap_root = log_dir / "mmap_replay"
        replay_storage_directory = mmap_root / "observations"
    authoritative_replay = (
        config.get_replay_buffer(replay_storage_directory)
        if not distributed_context.enabled or distributed_context.is_primary
        else None
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
    if config.uses_task_experts:
        from clworldmodel.continual import (
            ActorCriticBank,
            allocate_task_updates,
            shuffled_task_schedule,
        )
        task_update_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 0x4D4F4541])
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
    profile_stages = args.profile_stages and distributed_context.is_primary

    
    total_env_steps = (
        int(task1_seed_payload.get("total_raw_environment_frames", 0))
        if task1_seed_payload is not None
        else 0
    )  # number of *real* environment interactions so far

    best_rews_mean = float("-inf")
    best_validation_seen_task_raw_mean = float("-inf")
    global_step = (
        int(task1_seed_payload.get("world_model_updates", 0))
        if task1_seed_payload is not None
        else 0
    )  # gradient updates so far
    shared_core_frozen = resume_payload is not None
    boundary_teacher: Optional[WorldModel] = None
    capture_kan_parameter_values = None
    protect_kan_parameter_updates = None
    if config.residual_consolidation == "replay_functional":
        from clworldmodel.continual import (
            capture_kan_parameter_values,
            protect_kan_parameter_updates,
        )

    for epoch in range(training_start_epoch, config.epochs):
        print("Starting Epoch ", epoch)
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
            if config.uses_task_private_heads:
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
                _set_optimizer_learning_rate(
                    evolving_shared_optimizer,
                    config.first_task_shared_core_lr
                    if current_task_id == 0
                    else config.shared_core_lr,
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
                _acts, _obss, _rews, _conts, _resets = reinterpret_nt_to_t_n(
                    *generate_trajectories(
                        config.n_sync * config.gen_seq_len,
                        config.n_sync,
                        wm=wm,
                        ac=None if random_policy else aco.ac,
                        env_fns=envs.funcs(),
                        env_repeat=config.env_repeat,
                        seed=_next_environment_seed(collection_environment_seed_rng),
                        task_id=current_task_id,
                    ),
                    config.data_t,
                    config.data_n,
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
                    task_id=current_task_id,
                )
                if feature_cache is not None and feature_cache.requires_recording:
                    feature_cache.record(write_slots, frozen_features)
                print(f"{replay.n_valid=}")
                num_new_env_steps = (
                    _acts.shape[0] * _acts.shape[1] * config.env_repeat
                )
                total_env_steps += num_new_env_steps
                writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)

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
            eval_results_mean, eval_results_std = _evaluate_policy_tasks(
                config,
                wm,
                aco,
                envs.eval_funcs(),
                periodic_task_seeds,
                actor_critic_bank=actor_critic_bank,
                distributed_context=distributed_context,
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
                        replay_buffer=replay,
                        current_task_id=current_task_id,
                        memory_task_id=memory_task_id,
                        sequence_length=mb_t_size,
                        shared_optimizer=evolving_shared_optimizer,
                        private_optimizer=private_optimizer,
                        route_optimizer=route_optimizer,
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
                            [wm.decoder_for(current_task_id)(zh) for zh in zhs]
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

        if config.uses_shared_actor:
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

        actor_seconds = _stage_elapsed(actor_started, profile_stages)
        if distributed_context.is_primary and (
            actor_critic_bank is not None
            or config.uses_shared_actor
            or not actor_accounting_path.exists()
        ):
            temporary_actor_accounting_path = actor_accounting_path.with_suffix(
                ".json.tmp"
            )
            actor_accounting = (
                _actor_critic_bank_parameter_accounting(actor_critic_bank)
                if actor_critic_bank is not None
                else _shared_actor_parameter_accounting(
                    aco, shared_actor_teacher
                )
                if config.uses_shared_actor
                else _actor_critic_parameter_accounting(aco)
            )
            temporary_actor_accounting_path.write_text(
                json.dumps(actor_accounting, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_actor_accounting_path, actor_accounting_path)
            if config.uses_shared_actor:
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
            if evolving_shared_optimizer is None or actor_critic_bank is None:
                raise RuntimeError(
                    "Evolving-Core boundary requires shared and Actor-Critic optimizers"
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
                replay_buffer=replay,
                environment_schedule=envs,
                epoch=epoch,
                current_task_id=completed_task_id,
                world_model_updates=global_step,
                actor_critic_updates=actor_critic_updates,
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
                print(
                    "Evolving-Core consolidation failed after safe rollback; "
                    f"continuing from completed task state: {type(exc).__name__}: {exc}"
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
                replay_buffer=replay,
                environment_schedule=envs,
                epoch=epoch,
                current_task_id=completed_task_id,
                world_model_updates=global_step,
                actor_critic_updates=actor_critic_updates,
                total_env_steps=total_env_steps,
                task_update_rng=task_update_rng,
                collection_environment_seed_rng=collection_environment_seed_rng,
                validation_environment_seed_rng=validation_environment_seed_rng,
                final_environment_seed_rng=final_environment_seed_rng,
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
                temporary_final_accounting_path = final_accounting_path.with_suffix(
                    ".json.tmp"
                )
                temporary_final_accounting_path.write_text(
                    json.dumps(_world_model_parameter_accounting(wm), indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(
                    temporary_final_accounting_path,
                    final_accounting_path,
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
        final_scaled_means, final_scaled_stds = _evaluate_policy_tasks(
            config,
            wm,
            aco,
            eval_funcs,
            final_eval_task_seeds,
            actor_critic_bank=actor_critic_bank,
            distributed_context=distributed_context,
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
