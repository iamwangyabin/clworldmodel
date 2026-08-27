#!/usr/bin/env python3
"""Fit task-specific RSSM LoRA routes from a finished CNN-FullBank checkpoint.

This is a posthoc compression probe, not a continual-training result.  It keeps
the Task 1 CNN encoder and RSSM weights frozen, starts each later task from a
zero-effect LoRA route, optionally continues training that task's spatial
projector, and gives every task an independent actor initialized from the
checkpoint.  A frozen native route and actor provide functional distillation
targets on a private training trajectory cohort.  Validation and held-out
environment transitions never enter the optimization cohort or Replay.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from evaluate_cnn_fullbank_oracle_lora import (
    ROOT,
    _autocast,
    _build_actor,
    _build_adapter,
    _build_world_model,
    _evaluate,
    _install_shared_encoder_adapter,
    _load_exact,
    _sha256,
    _vendor_modules,
    _write_json_atomic,
    _write_text_atomic,
)
from git_provenance import git_state, require_synced_training_git_state


class LowRankWeightDelta(nn.Module):
    """Add a trainable low-rank matrix to a frozen affine weight."""

    def __init__(
        self,
        weight: torch.Tensor,
        rank: int,
        *,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("LoRA weights must be matrices")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        out_features, in_features = weight.shape
        self.rank = min(rank, out_features, in_features)
        self.alpha = float(self.rank if alpha is None else alpha)
        self.a = nn.Parameter(weight.new_empty(self.rank, in_features))
        self.b = nn.Parameter(weight.new_zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + (self.b @ self.a) * self.scale


class ExactVectorDelta(nn.Module):
    """Store a small task-specific bias or normalization delta exactly."""

    def __init__(self, original: torch.Tensor) -> None:
        super().__init__()
        if original.ndim != 1:
            raise ValueError("Exact vector deltas require one-dimensional tensors")
        self.delta = nn.Parameter(torch.zeros_like(original))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.delta


@dataclass(frozen=True)
class TrajectoryCohort:
    """Worker-major uint8 observations and aligned action/reset columns."""

    observations: torch.Tensor
    action_indices: torch.Tensor
    resets: torch.Tensor

    def __post_init__(self) -> None:
        if self.observations.dtype != torch.uint8:
            raise TypeError("Trajectory observations must be uint8")
        if self.action_indices.dtype != torch.uint8:
            raise TypeError("Trajectory action indices must be uint8")
        if self.resets.dtype != torch.bool:
            raise TypeError("Trajectory reset flags must be bool")
        if self.observations.shape[:2] != self.action_indices.shape:
            raise ValueError("Observation and action worker/time axes must match")
        if self.resets.shape != self.action_indices.shape:
            raise ValueError("Reset and action worker/time axes must match")

    @property
    def workers(self) -> int:
        return int(self.action_indices.shape[0])

    @property
    def frames_per_worker(self) -> int:
        return int(self.action_indices.shape[1])

    @property
    def frames(self) -> int:
        return self.workers * self.frames_per_worker

    def sample(
        self,
        *,
        sequence_length: int,
        batch_size: int,
        action_space: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sequence_length < 1 or sequence_length > self.frames_per_worker:
            raise ValueError("Sequence length is outside the trajectory cohort")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        worker = torch.randint(
            self.workers, (batch_size,), generator=generator
        )
        max_start = self.frames_per_worker - sequence_length + 1
        start = torch.randint(max_start, (batch_size,), generator=generator)
        offset = torch.arange(sequence_length)
        time_index = start[:, None] + offset[None, :]
        worker_index = worker[:, None].expand_as(time_index)

        observations = self.observations[worker_index, time_index]
        action_indices = self.action_indices[worker_index, time_index]
        resets = self.resets[worker_index, time_index]
        observations = observations.transpose(0, 1).to(device=device).float().div_(255)
        actions = F.one_hot(
            action_indices.transpose(0, 1).to(device=device).long(),
            action_space,
        ).float()
        resets = (
            resets.transpose(0, 1).to(device=device).float().unsqueeze(-1)
        )
        return actions, observations, resets


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("Seeds must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameterize_affines(module: nn.Module, rank: int) -> dict[str, Any]:
    """Install zero-effect LoRA matrices and exact vector deltas in one block."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    layers: dict[str, Any] = {}
    matrix_parameters = 0
    vector_parameters = 0
    supported = (nn.Linear, nn.GRUCell, nn.LayerNorm)
    for module_name, child in list(module.named_modules()):
        if not isinstance(child, supported):
            continue
        candidate_names: Iterable[str]
        if isinstance(child, nn.GRUCell):
            candidate_names = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        else:
            candidate_names = ("weight", "bias")
        for parameter_name in candidate_names:
            parameter = getattr(child, parameter_name, None)
            if parameter is None:
                continue
            label = ".".join(part for part in (module_name, parameter_name) if part)
            if parameter.ndim == 2:
                delta = LowRankWeightDelta(parameter, rank)
                parametrize.register_parametrization(child, parameter_name, delta)
                original = getattr(child.parametrizations, parameter_name).original
                original.requires_grad_(False)
                count = delta.a.numel() + delta.b.numel()
                matrix_parameters += count
                layers[label] = {
                    "shape": list(parameter.shape),
                    "rank": delta.rank,
                    "alpha": delta.alpha,
                    "parameters": count,
                }
            elif parameter.ndim == 1:
                delta = ExactVectorDelta(parameter)
                parametrize.register_parametrization(child, parameter_name, delta)
                original = getattr(child.parametrizations, parameter_name).original
                original.requires_grad_(False)
                vector_parameters += delta.delta.numel()
                layers[label] = {
                    "shape": list(parameter.shape),
                    "stored_as_exact_delta": True,
                    "parameters": delta.delta.numel(),
                }
            else:
                raise TypeError(
                    f"Unsupported affine parameter {label} with shape {parameter.shape}"
                )
    if not layers:
        raise ValueError("No supported RSSM affine parameters were found")
    return {
        "rank": rank,
        "matrix_lora_parameters": matrix_parameters,
        "exact_vector_parameters": vector_parameters,
        "trainable_parameters": matrix_parameters + vector_parameters,
        "layers": layers,
    }


def _install_lora_route(
    world_model: nn.Module,
    *,
    base_task: int,
    target_task: int,
    recurrent_rank: int,
    representation_rank: int,
    transition_rank: int,
) -> dict[str, Any]:
    if base_task != 0:
        raise ValueError("This probe currently anchors the shared RSSM at Task 1")
    if target_task <= base_task:
        raise ValueError("LoRA target task must be later than the base task")
    rssm = world_model.rssm
    target_slot = target_task - 1
    rssm.recurrent_experts[target_slot] = copy.deepcopy(
        rssm.recurrent_for(base_task)
    )
    rssm.representation_experts[target_slot] = copy.deepcopy(
        rssm.representation_for(base_task)
    )
    rssm.transition_experts[target_slot] = copy.deepcopy(
        rssm.transition_for(base_task)
    )
    reports = {
        "recurrent": _parameterize_affines(
            rssm.recurrent_for(target_task), recurrent_rank
        ),
        "representation": _parameterize_affines(
            rssm.representation_for(target_task), representation_rank
        ),
        "transition": _parameterize_affines(
            rssm.transition_for(target_task), transition_rank
        ),
    }
    reports["trainable_parameters"] = sum(
        report["trainable_parameters"] for report in reports.values()
    )
    return reports


def _trainable_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _load_trainable_state(
    module: nn.Module, state: Mapping[str, torch.Tensor], label: str
) -> None:
    trainable = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != set(state):
        raise RuntimeError(
            f"{label} trainable state mismatch: "
            f"missing={sorted(set(trainable) - set(state))} "
            f"unexpected={sorted(set(state) - set(trainable))}"
        )
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def _collect_cohort(
    vendor: Any,
    config: Any,
    world_model: nn.Module,
    actor: nn.Module,
    *,
    task: int,
    frames: int,
    seed: int,
) -> TrajectoryCohort:
    if frames < config.n_sync or frames % config.n_sync:
        raise ValueError("Trajectory frames must be a positive multiple of n_sync")
    task_config = config.esc.env_configs[task]
    actions, observations, _, _, resets = vendor.generate.generate_trajectories(
        frames,
        config.n_sync,
        world_model,
        actor,
        [task_config.get_function() for _ in range(config.n_sync)],
        config.env_repeat,
        None,
        no_images=False,
        seed=seed,
        task_id=task,
        deterministic_policy=True,
    )
    if observations is None:
        raise RuntimeError("Trajectory collection did not return observations")
    per_worker = frames // config.n_sync
    uint8_observations = (
        observations.mul(255).round().clamp(0, 255).to(torch.uint8)
    )
    return TrajectoryCohort(
        observations=uint8_observations.reshape(
            config.n_sync, per_worker, *uint8_observations.shape[1:]
        ).contiguous(),
        action_indices=actions.argmax(-1).to(torch.uint8).reshape(
            config.n_sync, per_worker
        ).contiguous(),
        resets=resets.squeeze(-1).bool().reshape(
            config.n_sync, per_worker
        ).contiguous(),
    )


def _categorical_kl(
    teacher_log_probs: torch.Tensor, student_log_probs: torch.Tensor
) -> torch.Tensor:
    teacher = teacher_log_probs.detach().float()
    student = student_log_probs.float()
    return (teacher.exp() * (teacher - student)).sum(-1)


def _distillation_step(
    vendor: Any,
    teacher: nn.Module,
    student: nn.Module,
    teacher_actor: nn.Module,
    student_actor: nn.Module,
    actions: torch.Tensor,
    observations: torch.Tensor,
    resets: torch.Tensor,
    *,
    task: int,
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = actions.shape[1]
    device = actions.device
    with torch.no_grad(), _autocast(torch, device, teacher.compute_dtype):
        teacher_embeddings = teacher.rssm.embed_observations(
            observations, task_id=task
        )
        teacher_z0, teacher_h0 = teacher.rssm.initial_state(batch_size)
        teacher_posts, teacher_z, teacher_h = teacher.rssm.observe_embeddings(
            teacher_z0,
            actions,
            teacher_h0,
            teacher_embeddings,
            resets,
            stochastic=False,
            task_id=task,
        )
        teacher_priors = teacher.rssm.prior(teacher_h, task_id=task)
        teacher_action_logs = teacher_actor.actor(
            vendor.ac.zh_to_ac_state(teacher_z, teacher_h)
        ).float()

    with _autocast(torch, device, student.compute_dtype):
        student_embeddings = student.rssm.embed_observations(
            observations, task_id=task
        )
        student_z0, student_h0 = student.rssm.initial_state(batch_size)
        student_posts, student_z, student_h = student.rssm.observe_embeddings(
            student_z0,
            actions,
            student_h0,
            student_embeddings,
            resets,
            stochastic=False,
            task_id=task,
        )
        student_priors = student.rssm.prior(student_h, task_id=task)
        student_action_logs = student_actor.actor(
            vendor.ac.zh_to_ac_state(student_z, student_h)
        ).float()

    posterior_kl = _categorical_kl(teacher_posts, student_posts).sum(-1).mean()
    prior_kl = _categorical_kl(teacher_priors, student_priors).sum(-1).mean()
    actor_kl = _categorical_kl(
        teacher_action_logs, student_action_logs
    ).mean()
    teacher_actions = teacher_action_logs.argmax(-1, keepdim=True)
    teacher_action_nll = -student_action_logs.gather(-1, teacher_actions).mean()
    hidden_mse = F.mse_loss(student_h.float(), teacher_h.detach().float())
    feature_mse = F.mse_loss(
        student_embeddings.float(), teacher_embeddings.detach().float()
    )
    losses = {
        "posterior_kl": posterior_kl,
        "prior_kl": prior_kl,
        "actor_kl": actor_kl,
        "teacher_action_nll": teacher_action_nll,
        "hidden_mse": hidden_mse,
        "feature_mse": feature_mse,
    }
    total = sum(loss_weights[name] * loss for name, loss in losses.items())
    if not torch.isfinite(total):
        raise FloatingPointError("RSSM LoRA distillation produced a non-finite loss")

    with torch.no_grad():
        metrics = {
            "loss": float(total.detach()),
            **{name: float(value.detach()) for name, value in losses.items()},
            "posterior_categorical_agreement": float(
                (teacher_posts.argmax(-1) == student_posts.argmax(-1))
                .float()
                .mean()
            ),
            "actor_argmax_agreement": float(
                (
                    teacher_action_logs.argmax(-1)
                    == student_action_logs.argmax(-1)
                )
                .float()
                .mean()
            ),
        }
    return total, metrics


def _average_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty metric collection")
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }


@torch.no_grad()
def _validate(
    vendor: Any,
    teacher: nn.Module,
    student: nn.Module,
    teacher_actor: nn.Module,
    student_actor: nn.Module,
    cohort: TrajectoryCohort,
    *,
    task: int,
    sequence_length: int,
    batch_size: int,
    action_space: int,
    batches: int,
    sampling_seed: int,
    device: torch.device,
    loss_weights: Mapping[str, float],
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(sampling_seed)
    rows = []
    for _ in range(batches):
        actions, observations, resets = cohort.sample(
            sequence_length=sequence_length,
            batch_size=batch_size,
            action_space=action_space,
            generator=generator,
            device=device,
        )
        _, metrics = _distillation_step(
            vendor,
            teacher,
            student,
            teacher_actor,
            student_actor,
            actions,
            observations,
            resets,
            task=task,
            loss_weights=loss_weights,
        )
        rows.append(metrics)
    return _average_metrics(rows)


def _selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    return (
        metrics["actor_argmax_agreement"],
        metrics["posterior_categorical_agreement"],
        -metrics["actor_kl"],
        -metrics["loss"],
    )


def _full_route_parameters(world_model: nn.Module, task: int) -> int:
    modules = (
        world_model.rssm.recurrent_for(task),
        world_model.rssm.representation_for(task),
        world_model.rssm.transition_for(task),
    )
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-task", type=int, default=0)
    parser.add_argument("--target-task", type=int, required=True)
    parser.add_argument("--recurrent-rank", type=int, default=128)
    parser.add_argument("--representation-rank", type=int, default=128)
    parser.add_argument("--transition-rank", type=int, default=32)
    parser.add_argument("--train-frames", type=int, default=8192)
    parser.add_argument("--validation-frames", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1800)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--lora-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, required=True)
    parser.add_argument("--evaluation-decisions", type=int, default=32768)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-adapter", action="store_true")
    parser.add_argument("--freeze-actor", action="store_true")
    parser.add_argument("--skip-teacher-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_integers = {
        "recurrent_rank": args.recurrent_rank,
        "representation_rank": args.representation_rank,
        "transition_rank": args.transition_rank,
        "train_frames": args.train_frames,
        "validation_frames": args.validation_frames,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "validation_interval": args.validation_interval,
        "validation_batches": args.validation_batches,
        "evaluation_decisions": args.evaluation_decisions,
        "cpu_threads": args.cpu_threads,
    }
    invalid = [name for name, value in positive_integers.items() if value < 1]
    if invalid:
        raise ValueError(f"Positive values required for: {', '.join(invalid)}")
    for name in (
        "lora_learning_rate",
        "adapter_learning_rate",
        "actor_learning_rate",
        "gradient_clip",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.target_task <= args.base_task:
        raise ValueError("target_task must be later than base_task")
    seeds = (args.train_seed, args.validation_seed, args.heldout_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training, validation, and held-out seeds must be distinct")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    planned = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.dry_run:
        print(
            json.dumps(
                {"planned": planned, "project_git": git_state(ROOT)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    git = require_synced_training_git_state(ROOT)
    vendor = _vendor_modules()
    protocol = (
        "posthoc-shared-task1-encoder-rssm-lora-distillation-v1"
        if args.freeze_actor
        else "posthoc-shared-task1-encoder-rssm-lora-actor-distillation-v1"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(
        args.output_dir / "resolved_probe_config.json",
        {
            "schema_version": 1,
            "artifact_kind": "cnn_fullbank_rssm_lora_distillation_config",
            "official_result": False,
            "protocol": protocol,
            "project_git": git,
            **planned,
            "adapter_mode": "direct",
            "teacher_policy": "native deterministic checkpoint policy",
            "training_transitions_enter_replay": False,
            "validation_transitions_enter_replay": False,
            "evaluation_transitions_enter_replay": False,
            "task_identity_available": True,
            "actor_training": not args.freeze_actor,
            "claim_scope": "posthoc task-aware compression probe only",
        },
    )
    _write_json_atomic(
        args.output_dir / "run_status.json",
        {"complete": False, "state": "running", "project_git": git},
    )

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    _seed_everything(args.sampling_seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter_payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_kind") != "task_bank_boundary_inference_snapshot":
        raise ValueError("Expected a CNN-FullBank task boundary snapshot")
    if int(adapter_payload.get("source_task", -1)) != args.base_task:
        raise ValueError("Adapter source task does not match --base-task")
    if int(adapter_payload.get("target_task", -1)) != args.target_task:
        raise ValueError("Adapter target task does not match --target-task")
    config = vendor.config.Config.from_dict(checkpoint["config"])
    if not config.uses_full_task_experts or not config.task_banked_image_encoder:
        raise ValueError("Checkpoint is not a complete CNN-FullBank task bank")
    if args.target_task >= config.rssm_num_experts:
        raise ValueError("Target task is outside the checkpoint task bank")
    for frames in (args.train_frames, args.validation_frames):
        if frames % config.n_sync:
            raise ValueError("Cohort frame counts must be divisible by n_sync")
        if frames // config.n_sync < args.sequence_length:
            raise ValueError("Each worker needs at least one complete sequence")

    teacher = _build_world_model(torch, vendor, config, device)
    _load_exact(teacher, checkpoint["world_model_state_dict"], "Teacher world model")
    student = _build_world_model(torch, vendor, config, device)
    _load_exact(student, checkpoint["world_model_state_dict"], "Student world model")
    actor_bank = checkpoint["actor_critic_bank_state_dict"]["tasks"]
    teacher_actor = _build_actor(
        vendor, teacher, config, actor_bank[str(args.target_task)]
    )
    student_actor = _build_actor(
        vendor, student, config, actor_bank[str(args.target_task)]
    )
    teacher.requires_grad_(False).eval()
    student.requires_grad_(False).eval()
    teacher_actor.requires_grad_(False).eval()
    student_actor.requires_grad_(False).eval()

    adapter = _build_adapter(
        torch, adapter_payload["state_dict"], residual=False
    ).to(device)
    _install_shared_encoder_adapter(
        torch, student, adapter, target_task=args.target_task
    )
    student.requires_grad_(False)
    lora_report = _install_lora_route(
        student,
        base_task=args.base_task,
        target_task=args.target_task,
        recurrent_rank=args.recurrent_rank,
        representation_rank=args.representation_rank,
        transition_rank=args.transition_rank,
    )
    adapter.requires_grad_(not args.freeze_adapter)
    student_actor.actor.requires_grad_(not args.freeze_actor)
    lora_parameters = [
        parameter
        for module in (
            student.rssm.recurrent_for(args.target_task),
            student.rssm.representation_for(args.target_task),
            student.rssm.transition_for(args.target_task),
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    actor_parameters = [
        parameter
        for parameter in student_actor.actor.parameters()
        if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in lora_parameters) != lora_report[
        "trainable_parameters"
    ]:
        raise RuntimeError("LoRA trainable parameter accounting mismatch")
    optimizer_groups = [
        {"params": lora_parameters, "lr": args.lora_learning_rate}
    ]
    if adapter_parameters:
        optimizer_groups.append(
            {"params": adapter_parameters, "lr": args.adapter_learning_rate}
        )
    if actor_parameters:
        optimizer_groups.append(
            {"params": actor_parameters, "lr": args.actor_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=args.weight_decay
    )

    task_config = config.esc.env_configs[args.target_task]
    full_route_parameters = _full_route_parameters(teacher, args.target_task)
    accounting = {
        "task_index": args.target_task,
        "task_name": task_config.name,
        "full_native_rssm_route_parameters": full_route_parameters,
        "lora": lora_report,
        "adapter_trainable_parameters": sum(
            parameter.numel() for parameter in adapter_parameters
        ),
        "actor_critic_parameters": sum(
            parameter.numel() for parameter in student_actor.parameters()
        ),
        "actor_trainable_parameters": sum(
            parameter.numel() for parameter in actor_parameters
        ),
        "critic_trainable_parameters": 0,
        "lora_to_full_rssm_ratio": lora_report["trainable_parameters"]
        / full_route_parameters,
    }
    _write_json_atomic(args.output_dir / "parameter_accounting.json", accounting)
    print(
        "[setup] "
        f"task={args.target_task}:{task_config.name} "
        f"rssm_lora={lora_report['trainable_parameters']:,} "
        f"full_rssm={full_route_parameters:,} "
        f"adapter={accounting['adapter_trainable_parameters']:,} "
        f"actor_trainable={accounting['actor_trainable_parameters']:,}"
    )

    cohort_started = time.perf_counter()
    train_cohort = _collect_cohort(
        vendor,
        config,
        teacher,
        teacher_actor,
        task=args.target_task,
        frames=args.train_frames,
        seed=args.train_seed,
    )
    validation_cohort = _collect_cohort(
        vendor,
        config,
        teacher,
        teacher_actor,
        task=args.target_task,
        frames=args.validation_frames,
        seed=args.validation_seed,
    )
    cohort_seconds = time.perf_counter() - cohort_started
    loss_weights = {
        "posterior_kl": 1.0,
        "prior_kl": 0.5,
        "actor_kl": 1.0,
        "teacher_action_nll": 0.1,
        "hidden_mse": 0.5,
        "feature_mse": 0.5,
    }
    generator = torch.Generator().manual_seed(args.sampling_seed)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_update = -1
    train_started = time.perf_counter()

    for update in range(args.updates + 1):
        if update % args.validation_interval == 0 or update == args.updates:
            metrics = _validate(
                vendor,
                teacher,
                student,
                teacher_actor,
                student_actor,
                validation_cohort,
                task=args.target_task,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                action_space=config.action_space,
                batches=args.validation_batches,
                sampling_seed=args.validation_seed + 1_000_000,
                device=device,
                loss_weights=loss_weights,
            )
            record = {"update": update, "validation": metrics}
            history.append(record)
            print(
                "[validation] "
                f"update={update}/{args.updates} loss={metrics['loss']:.5f} "
                f"actor_agree={metrics['actor_argmax_agreement']:.4f} "
                f"post_agree={metrics['posterior_categorical_agreement']:.4f} "
                f"actor_kl={metrics['actor_kl']:.5f}"
            )
            if best_metrics is None or _selection_key(metrics) > _selection_key(
                best_metrics
            ):
                best_metrics = metrics
                best_update = update
                best_state = {
                    "recurrent": _trainable_state(
                        student.rssm.recurrent_for(args.target_task)
                    ),
                    "representation": _trainable_state(
                        student.rssm.representation_for(args.target_task)
                    ),
                    "transition": _trainable_state(
                        student.rssm.transition_for(args.target_task)
                    ),
                    "adapter": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in adapter.state_dict().items()
                    },
                    "actor": _cpu_state_dict(student_actor.actor),
                }
        if update == args.updates:
            break

        actions, observations, resets = train_cohort.sample(
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            action_space=config.action_space,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _distillation_step(
            vendor,
            teacher,
            student,
            teacher_actor,
            student_actor,
            actions,
            observations,
            resets,
            task=args.target_task,
            loss_weights=loss_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*lora_parameters, *adapter_parameters, *actor_parameters],
            args.gradient_clip,
        )
        optimizer.step()

    training_seconds = time.perf_counter() - train_started
    if best_state is None or best_metrics is None:
        raise RuntimeError("No validation state was selected")
    _load_trainable_state(
        student.rssm.recurrent_for(args.target_task),
        best_state["recurrent"],
        "Recurrent LoRA",
    )
    _load_trainable_state(
        student.rssm.representation_for(args.target_task),
        best_state["representation"],
        "Representation LoRA",
    )
    _load_trainable_state(
        student.rssm.transition_for(args.target_task),
        best_state["transition"],
        "Transition LoRA",
    )
    adapter.load_state_dict(best_state["adapter"])
    student_actor.actor.load_state_dict(best_state["actor"])
    _write_json_atomic(
        args.output_dir / "distillation_metrics.json",
        {
            "schema_version": 1,
            "loss_weights": loss_weights,
            "selection": (
                "lexicographic actor argmax agreement, posterior categorical "
                "agreement, negative actor KL, negative total loss"
            ),
            "best_update": best_update,
            "best_validation": best_metrics,
            "history": history,
            "cohort_seconds": cohort_seconds,
            "training_seconds": training_seconds,
        },
    )

    compact_checkpoint = args.output_dir / f"rssm_lora_task_{args.target_task}.pt"
    _atomic_torch_save(
        compact_checkpoint,
        {
            "schema_version": 1,
            "artifact_kind": "cnn_fullbank_task_rssm_lora_actor_inference_state",
            "official_result": False,
            "resumable": False,
            "project_git": git,
            "protocol": protocol,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256(args.checkpoint),
            "source_adapter": str(args.adapter.resolve()),
            "source_adapter_sha256": _sha256(args.adapter),
            "base_task": args.base_task,
            "target_task": args.target_task,
            "target_task_name": task_config.name,
            "ranks": {
                "recurrent": args.recurrent_rank,
                "representation": args.representation_rank,
                "transition": args.transition_rank,
            },
            "lora_trainable_state": {
                "recurrent": best_state["recurrent"],
                "representation": best_state["representation"],
                "transition": best_state["transition"],
            },
            "adapter_state_dict": best_state["adapter"],
            "actor_critic_state_dict": _cpu_state_dict(student_actor),
            "actor_initialization": "complete target-task checkpoint actor",
            "actor_trained": not args.freeze_actor,
            "critic_trained": False,
            "best_update": best_update,
            "best_validation": best_metrics,
            "parameter_accounting": accounting,
            "omitted_state": [
                "optimizer",
                "RNG",
                "training and validation trajectory frames",
                "environment state",
                "Replay",
                "scheduler and task position",
            ],
        },
    )
    checkpoint_digest = _sha256(compact_checkpoint)
    _write_text_atomic(
        compact_checkpoint.with_suffix(compact_checkpoint.suffix + ".sha256"),
        f"{checkpoint_digest}  {compact_checkpoint.name}\n",
    )

    del train_cohort, validation_cohort, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluation: dict[str, Any] = {
        "heldout_seed": args.heldout_seed,
        "agent_decisions": args.evaluation_decisions,
        "deterministic_policy": True,
        "evaluation_transitions_enter_replay": False,
        "conditions": {},
    }
    if not args.skip_teacher_evaluation:
        evaluation["conditions"]["full_native_target_rssm_and_actor"] = _evaluate(
            torch,
            vendor,
            config,
            teacher,
            teacher_actor,
            task=args.target_task,
            seed=args.heldout_seed,
            decisions=args.evaluation_decisions,
        )
    student_condition = (
        "trained_lora_rssm_full_target_actor"
        if args.freeze_actor
        else "trained_lora_rssm_task_actor"
    )
    evaluation["conditions"][student_condition] = _evaluate(
        torch,
        vendor,
        config,
        student,
        student_actor,
        task=args.target_task,
        seed=args.heldout_seed,
        decisions=args.evaluation_decisions,
    )
    evaluation.update(
        {
            "schema_version": 1,
            "artifact_kind": "cnn_fullbank_trained_rssm_lora_policy_evaluation",
            "official_result": False,
            "complete": True,
            "task_index": args.target_task,
            "task_name": task_config.name,
            "best_update": best_update,
            "compact_checkpoint": compact_checkpoint.name,
            "compact_checkpoint_sha256": checkpoint_digest,
        }
    )
    evaluation_path = args.output_dir / "policy_evaluation.json"
    _write_json_atomic(evaluation_path, evaluation)
    evaluation_digest = _sha256(evaluation_path)
    _write_text_atomic(
        evaluation_path.with_suffix(evaluation_path.suffix + ".sha256"),
        f"{evaluation_digest}  {evaluation_path.name}\n",
    )
    _write_json_atomic(
        args.output_dir / "run_status.json",
        {
            "complete": True,
            "state": "complete",
            "return_code": 0,
            "project_git": git,
            "compact_checkpoint_sha256": checkpoint_digest,
            "policy_evaluation_sha256": evaluation_digest,
        },
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    print(f"compact_checkpoint_sha256={checkpoint_digest}")
    print(f"policy_evaluation_sha256={evaluation_digest}")


if __name__ == "__main__":
    main()
