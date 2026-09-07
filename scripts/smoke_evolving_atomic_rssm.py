#!/usr/bin/env python3
"""Run one production-shaped Evolving-Core optimizer update on a CUDA device.

This is a synthetic execution smoke: it performs no environment interaction and
does not produce a performance claim. It exercises the exact 12-current / 4-LTDM
memory update, interface losses, component gradient projection, and all three
world-model optimizer classes. D-AutoRoute also checks inferred policy routing.
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
    ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR,
    BEHAVIOR_PROFILES,
    COMPACT_MECHANISM_PROFILE,
    DEFAULT_MECHANISM_PROFILE,
    DENSE_PRIVATE_PARAMETERIZATION,
    MECHANISM_PROFILE_WIDTHS,
    MECHANISM_PARAMETERIZATIONS,
    PREDICTION_HEAD_PROFILES,
    PRIVATE_MLP_BEHAVIOR,
    PRIVATE_MLP_AUTOROUTE_BEHAVIOR,
    PRIVATE_PREDICTION_HEADS_PROFILE,
    SHARED_DISTILLED_HEADS_PROFILE,
    SHARED_DOWN_PARAMETERIZATION,
    _resolved_config,
)


LEGACY_METHOD_PROFILE = "legacy"
ATOMIC_LORA_SHARED_HEADS_PROFILE = "atomic_lora_shared_heads"
ADAPTIVE_QFP_COMPRESSION_PROFILE = "adaptive_qfp_compression"
ADAPTIVE_QFP_AC_COMPRESSION_PROFILE = "adaptive_qfp_ac_compression"

D_AUTOROUTE_PROFILE = "d_autoroute"
METHOD_PROFILES = (D_AUTOROUTE_PROFILE, LEGACY_METHOD_PROFILE, ATOMIC_LORA_SHARED_HEADS_PROFILE, ADAPTIVE_QFP_COMPRESSION_PROFILE, ADAPTIVE_QFP_AC_COMPRESSION_PROFILE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--method-profile",
        choices=METHOD_PROFILES,
        default=ADAPTIVE_QFP_COMPRESSION_PROFILE,
    )
    parser.add_argument(
        "--mechanism-profile",
        choices=tuple(MECHANISM_PROFILE_WIDTHS),
        default=DEFAULT_MECHANISM_PROFILE,
    )
    parser.add_argument(
        "--mechanism-parameterization",
        choices=MECHANISM_PARAMETERIZATIONS,
        default=DENSE_PRIVATE_PARAMETERIZATION,
    )
    parser.add_argument(
        "--behavior-profile",
        choices=BEHAVIOR_PROFILES,
        default=PRIVATE_MLP_BEHAVIOR,
    )
    parser.add_argument(
        "--prediction-head-profile",
        choices=PREDICTION_HEAD_PROFILES,
        default=SHARED_DISTILLED_HEADS_PROFILE,
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


def _config(
    mechanism_profile: str = DEFAULT_MECHANISM_PROFILE,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
    behavior_profile: str = PRIVATE_MLP_BEHAVIOR,
    prediction_head_profile: str = SHARED_DISTILLED_HEADS_PROFILE,
    method_profile: str = ADAPTIVE_QFP_COMPRESSION_PROFILE,
) -> Config:
    profiles = {ADAPTIVE_QFP_COMPRESSION_PROFILE: PRIVATE_MLP_BEHAVIOR, D_AUTOROUTE_PROFILE: PRIVATE_MLP_AUTOROUTE_BEHAVIOR}
    if method_profile not in profiles:
        raise ValueError(f"Unknown smoke method profile: {method_profile!r}")
    if behavior_profile != PRIVATE_MLP_BEHAVIOR:
        raise ValueError("Select smoke behavior through --method-profile")
    source = Config.from_file(_source_config()).to_dict()
    return Config.from_dict(_resolved_config(
        source, task_order="arrow-original-six", mechanism_profile=mechanism_profile,
        mechanism_parameterization=mechanism_parameterization,
        prediction_head_profile=prediction_head_profile,
        behavior_profile=profiles[method_profile], adaptive_qfp_compression=True,
    ))


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
        observation_encoder=config.observation_encoder,
        num_task_experts=config.rssm_num_experts,
        task_shared_prediction_heads=config.uses_shared_prediction_heads,
        evolving_shared_core=config.evolving_shared_core,
        task_projected_image_encoder=config.task_projected_image_encoder,
        task_symmetric_image_projectors=config.task_atomic_routes,
        task_projector_bottleneck_features=(
            config.task_projector_bottleneck_features
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
        task_mechanism_parameterization=config.task_mechanism_parameterization,
        task_symmetric_mechanisms=config.task_atomic_routes,
    ).to(device)


def _synthetic_batch(
    config: Config,
    *,
    task_id: int,
    time: int = 4,
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
    config = _config(
        args.mechanism_profile,
        args.mechanism_parameterization,
        args.behavior_profile,
        args.prediction_head_profile,
        args.method_profile,
    )
    world_model = _world_model(config, device)
    train._world_model_parameter_accounting(world_model)
    world_model.activate_task_expert(0)
    boundary_teacher = copy.deepcopy(world_model).eval()
    boundary_teacher.requires_grad_(False)
    if not world_model.initialize_task_expert(1, 0):
        raise RuntimeError("Task 1 private topology did not initialize")
    world_model.activate_task_expert(1)

    replay = MultiTypeReplay(
        FifoReplay(
            4,
            32,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        LongTermReplay(
            4,
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
        train._evolving_shared_optimizer_parameter_groups(
            world_model,
            core_lr=config.shared_core_lr,
            prediction_head_lr=config.task_private_lr,
        ),
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
    shared_aco = None
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
    routing_smoke = None
    if config.uses_reconstruction_task_inference:
        from clworldmodel.routing import TwoFrameReconstructionRouter
        from generate_trajectory import _routed_policy_step

        router = TwoFrameReconstructionRouter((0, 1))
        with train._preserve_training_rng_state():
            routing_aco = shared_aco
            routing_bank = None
            if config.task_private_actor_critic:
                from ac import build_actor_critic_opt
                from clworldmodel.continual import ActorCriticBank

                routing_bank = ActorCriticBank()
                for task in range(2):
                    routing_bank.ensure(task, lambda _: build_actor_critic_opt(
                        world_model, lr=config.ac_lr,
                        **train._actor_critic_constructor_kwargs(config),
                    ))
            routing_policy = train._autorouted_behavior(config, routing_aco, routing_bank, 2)
            z, h = world_model.rssm.initial_state(2)
            previous = torch.nn.functional.one_hot(
                torch.zeros(2, dtype=torch.long, device=device), config.action_space
            )
            for t in range(3):
                z, h, actions = _routed_policy_step(
                    world_model, routing_policy, router,
                    torch.zeros(2, 3, 64, 64, device=device), z, h, previous,
                    torch.full((2, 1), float(t == 0), device=device), stochastic=False,
                )
                previous = torch.nn.functional.one_hot(actions, config.action_space)
        if actions.shape != (2,) or len(router.events) != 4 or router.counts.tolist() != [2, 2]:
            raise RuntimeError("Two-frame autoroute smoke did not cover both workers")
        routing_smoke = {"events": router.events, "accuracy_claimed": False}
    old_private = tuple(world_model.private_parameters(0))

    metrics, diagnostics, gradient_norm = train._evolving_world_model_update(
        config=config,
        wm=world_model,
        boundary_teacher=boundary_teacher,
        actor_critic_bank=actor_bank,
        frozen_actor=(
            (None)
        ),
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
    if config.uses_shared_prediction_heads:
        expected_components.update(
            {"observation_head", "reward_head", "continue_head"}
        )
    if set(diagnostics) != expected_components:
        raise RuntimeError(f"Projection groups changed: {sorted(diagnostics)}")
    if any(parameter.grad is not None for parameter in old_private):
        raise RuntimeError("A completed task-private parameter received a gradient")

    behavior_metrics = None

    adaptive_compression_smoke = None
    if config.uses_adaptive_qfp_compression:
        dense_teacher = copy.deepcopy(world_model).eval()
        dense_teacher.requires_grad_(False)
        dense_parameter_count = sum(
            parameter.numel() for parameter in world_model.parameters()
        )
        selection, installed = train._structured_adaptive_qfp_candidate(
            wm=world_model,
            dense_teacher=dense_teacher,
            task_id=1,
            fraction=config.adaptive_compression_width_fractions[0],
        )
        world_model.requires_grad_(False)
        compact_parameters = [
            parameter
            for module in installed.values()
            for parameter in module.parameters()
        ]
        for parameter in compact_parameters:
            parameter.requires_grad_(True)
        compression_optimizer = torch.optim.Adam(
            compact_parameters,
            lr=config.adaptive_compression_lr,
            fused=True,
        )
        compression_batch = replay.minibatch_for_task(
            1,
            2,
            config.mb_n_size,
            source="ltdm",
            mb_device=str(device),
        )
        compression_optimizer.zero_grad(set_to_none=True)
        compression_loss, compression_metrics = train._evolving_memory_loss(
            config=config,
            wm=world_model,
            teacher=dense_teacher,
            frozen_actor=actor,
            batch=compression_batch,
            task_id=1,
            mechanism_output_scale=(
                config.adaptive_compression_qfp_distill_scale
            ),
        )
        if not bool(torch.isfinite(compression_loss).item()):
            raise FloatingPointError("Adaptive Q/F/P smoke loss is non-finite")
        compression_loss.backward()
        compression_optimizer.step()
        compact_parameter_count = sum(
            parameter.numel() for parameter in world_model.parameters()
        )
        compact_state = copy.deepcopy(world_model.state_dict())
        restored_world_model = _world_model(config, device)
        restored_world_model.load_state_dict(compact_state, strict=True)
        expected_layout = world_model.rssm.adaptive_compression_layout()
        if restored_world_model.rssm.adaptive_compression_layout() != expected_layout:
            raise RuntimeError("Adaptive Q/F/P state-dict topology did not rebuild")
        adaptive_compression_smoke = {
            "task_id": 1,
            "width_fraction": config.adaptive_compression_width_fractions[0],
            "selection": selection,
            "layout": expected_layout,
            "loss": float(compression_loss.detach().cpu()),
            "qfp_distillation_loss": float(
                compression_metrics[
                    "Loss/evolving_qfp_distill_scaled"
                ].detach().cpu()
            ),
            "optimizer_step": _optimizer_step(compression_optimizer),
            "world_model_parameters_before": dense_parameter_count,
            "world_model_parameters_after": compact_parameter_count,
            "state_dict_dynamic_topology_round_trip": True,
        }

    adaptive_behavior_compression_smoke = None

    result = {
        "schema_version": 1,
        "classification": "smoke",
        "synthetic_data": True,
        "environment_interaction": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_dtype": config.compute_dtype,
        "behavior_profile": (args.behavior_profile),
        "method_profile": args.method_profile,
        "prediction_head_profile": (
            SHARED_DISTILLED_HEADS_PROFILE
            if config.uses_shared_prediction_heads
            else args.prediction_head_profile
        ),
        "mechanism_profile": config.task_mechanism_capacity_profile,
        "mechanism_parameterization": config.task_mechanism_parameterization,
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
        "world_model_parameter_accounting": train._world_model_parameter_accounting(world_model),
        "shared_behavior_route_schedule": (
            [0, 1] if behavior_metrics is not None else None
        ),
        "shared_behavior_metrics": behavior_metrics,
        "two_frame_autoroute_smoke": routing_smoke,
        "adaptive_compression_smoke": adaptive_compression_smoke,
        "adaptive_behavior_compression_smoke": (
            adaptive_behavior_compression_smoke
        ),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
