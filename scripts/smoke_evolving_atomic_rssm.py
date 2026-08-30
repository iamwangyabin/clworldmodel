#!/usr/bin/env python3
"""Run one production-shaped Evolving-Core optimizer update on a CUDA device.

This is a synthetic execution smoke: it performs no environment interaction and
does not produce a performance claim.  It exercises the exact 12-current / 4-LTDM
memory update, interface losses, component gradient projection, and all three
world-model optimizer classes used by the named Atari protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
ATARI_ROOT = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)
for path in (ROOT, ROOT / "src", ROOT / "scripts", ATARI_ROOT):
    sys.path.insert(0, str(path))

from config import Config  # noqa: E402
from replay import FifoReplay, LongTermReplay, MultiTypeReplay  # noqa: E402
import train  # noqa: E402
from wm import WorldModel  # noqa: E402

from run_evolving_atomic_rssm import (  # noqa: E402
    COMPACT_MECHANISM_PROFILE,
    DEFAULT_MECHANISM_PROFILE,
    MECHANISM_PROFILE_WIDTHS,
    _resolved_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--mechanism-profile",
        choices=tuple(MECHANISM_PROFILE_WIDTHS),
        default=DEFAULT_MECHANISM_PROFILE,
    )
    return parser


def _source_config() -> Path:
    return (
        ROOT
        / "third_party"
        / "arrow"
        / "Configs"
        / "Atari configs"
        / "CL-task configs"
        / "Original Order"
        / (
            "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,"
            "ALE_Seaquest,ALE_Enduro-s0-arrow.json"
        )
    )


def _config(mechanism_profile: str = DEFAULT_MECHANISM_PROFILE) -> Config:
    source = Config.from_file(_source_config()).to_dict()
    task_order = (
        "arrow-original-six"
        if mechanism_profile == COMPACT_MECHANISM_PROFILE
        else "mspacman-boxing-crazyclimber"
    )
    return Config.from_dict(
        _resolved_config(
            source,
            task_order=task_order,
            mechanism_profile=mechanism_profile,
        )
    )


def _world_model(config: Config, device: torch.device) -> WorldModel:
    return WorldModel(
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
        task_mechanism_transition_width=(
            config.task_mechanism_transition_width
        ),
        task_mechanism_residual_scale=config.task_mechanism_residual_scale,
        task_mechanism_num_atoms=config.task_mechanism_num_atoms,
        task_symmetric_mechanisms=config.task_atomic_routes,
    ).to(device)


def _synthetic_batch(
    config: Config,
    *,
    task_id: int,
    time: int = 2,
    sequences: int = 16,
) -> tuple[torch.Tensor, ...]:
    action_ids = torch.arange(time * sequences).reshape(time, sequences)
    action_ids = (action_ids + task_id) % config.action_space
    actions = torch.nn.functional.one_hot(
        action_ids,
        config.action_space,
    ).float()
    observations = torch.full(
        (time, sequences, 3, config.img_size, config.img_size),
        0.15 + 0.1 * task_id,
    )
    rewards = torch.linspace(-0.5, 0.5, time * sequences).reshape(
        time, sequences, 1
    )
    continues = (
        torch.arange(time * sequences).reshape(time, sequences, 1) % 3 != 0
    ).float()
    resets = torch.zeros(time, sequences, 1)
    resets[0, ::5] = 1
    return actions, observations, rewards, continues, resets


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = {
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    }
    if steps != {1}:
        raise RuntimeError(f"Expected every initialized Adam state at step 1: {steps}")
    return 1


def main() -> int:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Evolving-Core target smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    config = _config(args.mechanism_profile)
    world_model = _world_model(config, device)
    world_model.activate_task_expert(0)
    boundary_teacher = copy.deepcopy(world_model).eval()
    boundary_teacher.requires_grad_(False)
    if not world_model.initialize_task_expert(1, 0):
        raise RuntimeError("Task 1 private topology did not initialize")
    world_model.activate_task_expert(1)

    replay = MultiTypeReplay(
        FifoReplay(
            2,
            32,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        LongTermReplay(
            2,
            32,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        sampling_weights=(0.5, 0.5),
    )
    for task_id in (0, 1):
        replay.add(*_synthetic_batch(config, task_id=task_id), task_id=task_id)

    shared_optimizer = torch.optim.Adam(
        train._flatten_parameter_groups(world_model.shared_parameter_groups()),
        lr=config.shared_core_lr,
        fused=True,
    )
    private_optimizer = torch.optim.Adam(
        world_model.private_parameters(1),
        lr=config.task_private_lr,
        fused=True,
    )
    route_optimizer = torch.optim.Adam(
        world_model.route_parameters(1),
        lr=config.task_route_lr,
        fused=True,
    )
    actor = nn.Sequential(
        nn.Linear(world_model.zh_transform.out_features, config.action_space),
        nn.LogSoftmax(dim=-1),
    ).to(device)
    actor.requires_grad_(False)
    actor_bank = SimpleNamespace(
        get=lambda task_id: SimpleNamespace(
            ac=SimpleNamespace(actor=actor)
        )
        if task_id == 0
        else (_ for _ in ()).throw(ValueError("Smoke has only old task 0"))
    )
    old_private = tuple(world_model.private_parameters(0))

    metrics, diagnostics, gradient_norm = train._evolving_world_model_update(
        config=config,
        wm=world_model,
        boundary_teacher=boundary_teacher,
        actor_critic_bank=actor_bank,
        replay_buffer=replay,
        current_task_id=1,
        memory_task_id=0,
        sequence_length=2,
        shared_optimizer=shared_optimizer,
        private_optimizer=private_optimizer,
        route_optimizer=route_optimizer,
    )
    checked_metrics = {
        "current_loss": metrics["Loss/evolving_current_total"],
        "memory_loss": metrics["Memory/Loss/evolving_memory_total"],
        "gradient_norm": gradient_norm,
    }
    nonfinite = [
        name
        for name, value in checked_metrics.items()
        if not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite smoke values: {nonfinite}")
    expected_components = {
        "encoder",
        "posterior",
        "recurrent",
        "prior",
        "latent_interface",
    }
    if set(diagnostics) != expected_components:
        raise RuntimeError(f"Projection groups changed: {sorted(diagnostics)}")
    if any(parameter.grad is not None for parameter in old_private):
        raise RuntimeError("A completed task-private parameter received a gradient")

    result = {
        "schema_version": 1,
        "classification": "smoke",
        "synthetic_data": True,
        "environment_interaction": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_dtype": config.compute_dtype,
        "mechanism_profile": config.task_mechanism_capacity_profile,
        "mechanism_widths": [
            config.task_mechanism_recurrent_width,
            config.task_mechanism_representation_width,
            config.task_mechanism_transition_width,
        ],
        "current_sequences": config.current_batch_n,
        "memory_sequences": config.memory_batch_n,
        "current_loss": float(checked_metrics["current_loss"].detach().cpu()),
        "memory_loss": float(checked_metrics["memory_loss"].detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "projection_conflicts": {
            name: diagnostic.conflicted
            for name, diagnostic in diagnostics.items()
        },
        "optimizer_steps": {
            "shared": _optimizer_step(shared_optimizer),
            "private": _optimizer_step(private_optimizer),
            "route": _optimizer_step(route_optimizer),
        },
        "old_private_gradients_are_none": True,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
