#!/usr/bin/env python3
"""CUDA smoke for the learned-base Rank-32 Evolving-Core pilot.

The smoke uses deterministic synthetic replay only.  It performs no environment
interaction and supports no performance claim.  It verifies the exact six-task
parameter topology and executes one Task-1 world-model update with Task-0 LTDM
protection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

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

from run_evolving_learned_base_adapters import (  # noqa: E402
    _parameter_manifest,
    _resolved_config,
)
from smoke_evolving_atomic_rssm import (  # noqa: E402
    _source_config,
    _synthetic_batch,
    _world_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path)
    return parser


def _parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _write_result(path: Path, result: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The learned-base target smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    source = Config.from_file(_source_config()).to_dict()
    resolved = _resolved_config(source)
    config = Config.from_dict(resolved)
    expected = _parameter_manifest(resolved)
    world_model = _world_model(config, device)
    actual_world_model_parameters = _parameters(world_model)
    if actual_world_model_parameters != expected["world_model_parameters"]:
        raise RuntimeError(
            "World-model parameter ledger drift: "
            f"actual={actual_world_model_parameters} "
            f"expected={expected['world_model_parameters']}"
        )

    actor_critics = []
    for task_id in range(config.rssm_num_experts):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 1_000_003 * (task_id + 1))
            actor_critics.append(
                train.build_actor_critic_opt(
                    world_model,
                    lr=config.ac_lr,
                    **train._actor_critic_constructor_kwargs(config),
                )
            )
    actual_behavior_parameters = sum(
        _parameters(actor_critic.ac) for actor_critic in actor_critics
    )
    if actual_behavior_parameters != expected["behavior_parameters"]:
        raise RuntimeError(
            "Actor-Critic parameter ledger drift: "
            f"actual={actual_behavior_parameters} "
            f"expected={expected['behavior_parameters']}"
        )
    actual_online_parameters = (
        actual_world_model_parameters + actual_behavior_parameters
    )
    if actual_online_parameters != expected["online_parameters"]:
        raise RuntimeError(
            "Online parameter ledger drift: "
            f"actual={actual_online_parameters} "
            f"expected={expected['online_parameters']}"
        )

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
    if world_model.route_parameters(1):
        raise RuntimeError("Learned-base v1 unexpectedly exposed route parameters")
    old_private = tuple(world_model.private_parameters(0))

    metrics, diagnostics, gradient_norm = train._evolving_world_model_update(
        config=config,
        wm=world_model,
        boundary_teacher=boundary_teacher,
        actor_critic_bank=None,
        frozen_actor=actor_critics[0].ac.actor,
        replay_buffer=replay,
        current_task_id=1,
        memory_task_id=0,
        sequence_length=2,
        shared_optimizer=shared_optimizer,
        private_optimizer=private_optimizer,
        route_optimizer=None,
    )
    checked = {
        "current_loss": metrics["Loss/evolving_current_total"],
        "memory_loss": metrics["Memory/Loss/evolving_memory_total"],
        "gradient_norm": gradient_norm,
    }
    nonfinite = [
        name
        for name, value in checked.items()
        if not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite smoke values: {nonfinite}")
    expected_projection_groups = {
        "encoder",
        "posterior",
        "recurrent",
        "prior",
        "latent_interface",
    }
    if set(diagnostics) != expected_projection_groups:
        raise RuntimeError(f"Projection groups changed: {sorted(diagnostics)}")
    if any(parameter.grad is not None for parameter in old_private):
        raise RuntimeError("A completed Task-0 private parameter received a gradient")

    result: dict[str, object] = {
        "schema_version": 1,
        "classification": "smoke",
        "synthetic_data": True,
        "environment_interaction": False,
        "performance_claim": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "method": config.continual_method,
        "mechanism_parameterization": config.task_mechanism_parameterization,
        "mechanism_rank": config.task_mechanism_low_rank,
        "prediction_adapter_rank": config.prediction_adapter_rank,
        "world_model_parameters": actual_world_model_parameters,
        "behavior_parameters": actual_behavior_parameters,
        "online_parameters": actual_online_parameters,
        "current_loss": float(checked["current_loss"].detach().cpu()),
        "memory_loss": float(checked["memory_loss"].detach().cpu()),
        "gradient_norm_before_clip": float(
            checked["gradient_norm"].detach().cpu()
        ),
        "projection_groups": sorted(diagnostics),
        "old_private_gradients_are_none": True,
        "route_optimizer": None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    if args.output is not None:
        _write_result(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
