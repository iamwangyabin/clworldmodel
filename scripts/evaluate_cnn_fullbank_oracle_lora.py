#!/usr/bin/env python3
"""Evaluate an oracle low-rank compression of a CNN-FullBank task route.

This diagnostic does not train LoRA parameters.  It computes the best truncated
SVD approximation of the already-trained target-minus-base weights, installs
the resulting base-plus-low-rank weights in memory, and evaluates the frozen
deterministic policy.  It therefore tests whether a rank budget can preserve a
known route, not whether a particular LoRA optimizer can learn that route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from git_provenance import git_state


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _vendor_modules() -> SimpleNamespace:
    for source_path in (PROJECT_SRC, VENDORED_ATARI):
        rendered = str(source_path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    return SimpleNamespace(
        ac=importlib.import_module("ac"),
        config=importlib.import_module("config"),
        generate=importlib.import_module("generate_trajectory"),
        train=importlib.import_module("train"),
        wm=importlib.import_module("wm"),
    )


def _require_synced_git() -> dict[str, int | str | bool | None]:
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Oracle evaluation requires a clean worktree")
    if state["upstream"] is None or state["ahead"] or state["behind"]:
        raise RuntimeError(
            "Oracle evaluation requires HEAD to match its configured upstream: "
            f"{state}"
        )
    return state


def _build_world_model(
    torch: Any, vendor: SimpleNamespace, config: Any, device: Any
) -> Any:
    model = vendor.wm.WorldModel(
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
        task_banked_image_encoder=config.task_banked_image_encoder,
    ).to(device)
    return model


def _load_exact(module: Any, state: Mapping[str, Any], label: str) -> None:
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _build_actor(
    vendor: SimpleNamespace,
    world_model: Any,
    config: Any,
    state: Mapping[str, Any],
) -> Any:
    bundle = vendor.ac.build_actor_critic_opt(
        world_model,
        lr=config.ac_lr,
        **vendor.train._actor_critic_constructor_kwargs(config),
    )
    _load_exact(bundle.ac, state, "Actor-critic")
    return bundle.ac


def _build_adapter(torch: Any, state: Mapping[str, Any], *, residual: bool) -> Any:
    nn = torch.nn

    class SpatialAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(32, 256)
            self.project = nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.SiLU(),
                nn.Conv2d(64, 64, 3, padding=1, groups=64),
                nn.SiLU(),
                nn.Conv2d(64, 256, 1),
            )

        def forward(self, features: Any) -> Any:
            spatial = features.reshape(features.shape[0], 256, 4, 4)
            projected = self.project(self.norm(spatial))
            if residual:
                projected = spatial + projected
            return projected.flatten(1)

    adapter = SpatialAdapter()
    _load_exact(adapter, state, "Feature adapter")
    return adapter


def _install_shared_encoder_adapter(
    torch: Any,
    world_model: Any,
    adapter: Any,
    *,
    target_task: int,
) -> None:
    if target_task <= 0:
        raise ValueError("The target task must use an expert slot")
    nn = torch.nn

    class SharedEncoderAdapter(nn.Module):
        def __init__(self, encoder: Any, converter: Any) -> None:
            super().__init__()
            self.encoder = encoder
            self.converter = converter
            self.output_size = encoder.output_size

        def forward(self, images: Any) -> Any:
            return self.converter(self.encoder(images))

    wrapped = SharedEncoderAdapter(world_model.rssm.image_embedder, adapter)
    world_model.rssm.image_embedder_experts[target_task - 1] = wrapped


def _truncated_delta(
    torch: Any, base: Any, target: Any, rank: int
) -> tuple[Any, float]:
    """Return base plus the best rank-r target-minus-base approximation."""
    if base.shape != target.shape or base.ndim < 2:
        raise ValueError("Low-rank projection requires equal matrix-like tensors")
    delta = (target.float() - base.float()).reshape(target.shape[0], -1)
    total_energy = float(delta.double().square().sum())
    if total_energy == 0:
        return base.detach().clone(), 1.0
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    used_rank = min(rank, singular.numel())
    approximation = (u[:, :used_rank] * singular[:used_rank]) @ vh[:used_rank]
    captured = float(singular[:used_rank].double().square().sum()) / total_energy
    reconstructed = base.float() + approximation.reshape_as(target)
    return reconstructed.to(dtype=target.dtype), min(captured, 1.0)


def _install_oracle_state_delta(
    torch: Any,
    base_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    if set(base_state) != set(target_state):
        raise RuntimeError("Base and target states have different keys")
    captured_numerator = 0.0
    captured_denominator = 0.0
    matrix_parameters = 0
    lora_parameters = 0
    layers: dict[str, Any] = {}
    with torch.no_grad():
        for name, target in target_state.items():
            base = base_state[name].to(device=target.device)
            if not target.is_floating_point():
                continue
            delta_energy = float((target.double() - base.double()).square().sum())
            if target.ndim >= 2:
                reconstructed, captured = _truncated_delta(torch, base, target, rank)
                target.copy_(reconstructed)
                matrix = target.reshape(target.shape[0], -1)
                used_rank = min(rank, min(matrix.shape))
                full = target.numel()
                overhead = min(full, used_rank * sum(matrix.shape))
                matrix_parameters += full
                lora_parameters += overhead
                captured_numerator += captured * delta_energy
                captured_denominator += delta_energy
                layers[name] = {
                    "shape": list(target.shape),
                    "rank": used_rank,
                    "delta_energy_capture": captured,
                    "lora_parameters": overhead,
                }
            else:
                # A practical route keeps its small bias and normalization vectors.
                captured_numerator += delta_energy
                captured_denominator += delta_energy
                lora_parameters += target.numel()
                layers[name] = {
                    "shape": list(target.shape),
                    "stored_exactly": True,
                    "parameters": target.numel(),
                }
    return {
        "rank": rank,
        "matrix_parameters": matrix_parameters,
        "lora_parameters_including_exact_vectors": lora_parameters,
        "delta_energy_capture_including_exact_vectors": (
            captured_numerator / captured_denominator
            if captured_denominator
            else 1.0
        ),
        "layers": layers,
    }


def _install_oracle_module_delta(
    torch: Any,
    base_module: Any,
    target_module: Any,
    rank: int,
) -> dict[str, Any]:
    return _install_oracle_state_delta(
        torch, base_module.state_dict(), target_module.state_dict(), rank
    )


def _install_oracle_actor_delta(
    torch: Any,
    actor: Any,
    base_state: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    target_state = actor.state_dict()
    actor_keys = sorted(name for name in target_state if name.startswith("actor."))
    if any(name not in base_state for name in actor_keys):
        raise RuntimeError("Base actor state is incomplete")

    return _install_oracle_state_delta(
        torch,
        {name: base_state[name] for name in actor_keys},
        {name: target_state[name] for name in actor_keys},
        rank,
    )


def _autocast(torch: Any, device: Any, compute_dtype: str) -> Any:
    if compute_dtype == "float32":
        return nullcontext()
    if compute_dtype != "bfloat16":
        raise ValueError(f"Unsupported compute dtype {compute_dtype!r}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _synthetic_forward(
    torch: Any,
    vendor: SimpleNamespace,
    world_model: Any,
    actor: Any,
    task: int,
) -> None:
    batch = 2
    z, h = world_model.rssm.initial_state(batch)
    actions = torch.zeros(batch, world_model.a_dim, device=z.device)
    actions[:, 0] = 1
    images = torch.zeros(batch, 3, 64, 64, device=z.device)
    resets = torch.ones(batch, 1, device=z.device)
    with torch.no_grad(), _autocast(torch, z.device, world_model.compute_dtype):
        _, z, h = world_model.rssm(
            z, actions, h, images, resets, task_id=task, stochastic=False
        )
        logits = actor.actor(vendor.ac.zh_to_ac_state(z, h)).float()
    if logits.shape != (batch, world_model.a_dim) or not torch.isfinite(logits).all():
        raise RuntimeError("Synthetic oracle policy forward is invalid")


def _episode_statistics(
    rewards: Any,
    continues: Any,
    *,
    n_sync: int,
    reward_scale: float,
) -> dict[str, Any]:
    raw_rewards = rewards.cpu().numpy().reshape(n_sync, -1) / reward_scale
    continuation = continues.cpu().numpy().reshape(n_sync, -1)
    episode_returns: list[float] = []
    for worker_rewards, worker_continues in zip(raw_rewards, continuation):
        start = 0
        for end in np.flatnonzero(worker_continues == 0):
            episode_returns.append(float(worker_rewards[start : end + 1].sum()))
            start = int(end) + 1
    horizon = [float(worker.sum()) for worker in raw_rewards]
    return {
        "completed_episodes": len(episode_returns),
        "raw_returns": episode_returns,
        "raw_return_mean": (
            float(np.mean(episode_returns)) if episode_returns else None
        ),
        "raw_return_std": (
            float(np.std(episode_returns)) if episode_returns else None
        ),
        "fixed_horizon_raw_reward_per_worker": horizon,
        "fixed_horizon_raw_reward_worker_mean": float(np.mean(horizon)),
    }


def _evaluate(
    torch: Any,
    vendor: SimpleNamespace,
    config: Any,
    world_model: Any,
    actor: Any,
    *,
    task: int,
    seed: int,
    decisions: int,
) -> dict[str, Any]:
    task_config = config.esc.env_configs[task]
    _, _, rewards, continues, _ = vendor.generate.generate_trajectories(
        decisions,
        config.n_sync,
        world_model,
        actor,
        [task_config.get_function() for _ in range(config.n_sync)],
        config.env_repeat,
        None,
        no_images=True,
        seed=seed,
        task_id=task,
        deterministic_policy=True,
    )
    result = _episode_statistics(
        rewards,
        continues,
        n_sync=config.n_sync,
        reward_scale=task_config.rew_scale,
    )
    result.update(
        {
            "task_index": task,
            "task_name": task_config.name,
            "heldout_seed": seed,
            "agent_decisions": decisions,
            "deterministic_policy": True,
        }
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--base-task", type=int, default=0)
    parser.add_argument("--target-task", type=int, default=2)
    parser.add_argument("--heldout-seed", type=int, required=True)
    parser.add_argument("--decisions", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--adapter-mode", choices=("residual", "direct"), default="residual"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rank < 1 or args.decisions < 1:
        raise ValueError("Rank and decision budget must be positive")
    if args.output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    git = git_state(ROOT) if args.dry_run else _require_synced_git()
    vendor = _vendor_modules()
    import torch

    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter_payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_kind") != "task_bank_boundary_inference_snapshot":
        raise ValueError("Expected a task-bank boundary snapshot")
    if int(adapter_payload.get("source_task", -1)) != args.base_task:
        raise ValueError("Adapter source task does not match --base-task")
    if int(adapter_payload.get("target_task", -1)) != args.target_task:
        raise ValueError("Adapter target task does not match --target-task")
    config = vendor.config.Config.from_dict(checkpoint["config"])
    if not config.uses_full_task_experts or not config.task_banked_image_encoder:
        raise ValueError("Checkpoint is not a CNN-FullBank task bank")

    world_model = _build_world_model(torch, vendor, config, device)
    _load_exact(world_model, checkpoint["world_model_state_dict"], "World model")
    actor_bank = checkpoint["actor_critic_bank_state_dict"]["tasks"]
    actor = _build_actor(
        vendor, world_model, config, actor_bank[str(args.target_task)]
    )
    adapter = _build_adapter(
        torch,
        adapter_payload["state_dict"],
        residual=args.adapter_mode == "residual",
    ).to(device)
    _install_shared_encoder_adapter(
        torch, world_model, adapter, target_task=args.target_task
    )
    world_model.eval()
    actor.eval()
    _synthetic_forward(torch, vendor, world_model, actor, args.target_task)

    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "cnn_fullbank_oracle_lora_policy_evaluation",
        "official_result": False,
        "analysis_kind": "oracle_truncated_svd_not_trained_lora",
        "project_git": git,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": _sha256(args.adapter),
        "adapter_mode": args.adapter_mode,
        "base_task": args.base_task,
        "target_task": args.target_task,
        "rank": args.rank,
        "heldout_seed": args.heldout_seed,
        "decision_budget": args.decisions,
        "evaluation_transitions_enter_replay": False,
        "gradient_updates": 0,
        "conditions": {},
    }
    if not args.dry_run:
        output["conditions"]["full_target_rssm_and_actor"] = _evaluate(
            torch,
            vendor,
            config,
            world_model,
            actor,
            task=args.target_task,
            seed=args.heldout_seed,
            decisions=args.decisions,
        )

    projection = {
        "recurrent": _install_oracle_module_delta(
            torch,
            world_model.rssm.recurrent_for(args.base_task),
            world_model.rssm.recurrent_for(args.target_task),
            args.rank,
        ),
        "representation": _install_oracle_module_delta(
            torch,
            world_model.rssm.representation_for(args.base_task),
            world_model.rssm.representation_for(args.target_task),
            args.rank,
        ),
        "transition": _install_oracle_module_delta(
            torch,
            world_model.rssm.transition_for(args.base_task),
            world_model.rssm.transition_for(args.target_task),
            args.rank,
        ),
        "actor": _install_oracle_actor_delta(
            torch, actor, actor_bank[str(args.base_task)], args.rank
        ),
    }
    _synthetic_forward(torch, vendor, world_model, actor, args.target_task)
    output["projection"] = projection
    if args.dry_run:
        output["dry_run"] = True
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    output["conditions"]["oracle_rank_route"] = _evaluate(
        torch,
        vendor,
        config,
        world_model,
        actor,
        task=args.target_task,
        seed=args.heldout_seed,
        decisions=args.decisions,
    )
    output["complete"] = True
    _write_json_atomic(args.output, output)
    digest = _sha256(args.output)
    _write_text_atomic(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{digest}  {args.output.name}\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"output_sha256={digest}")


if __name__ == "__main__":
    main()
