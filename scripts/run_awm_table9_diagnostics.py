#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run frozen-checkpoint diagnostics needed by paper Table 9.

This command never trains or writes into replay.  It loads the final AWM-AutoRoute
boundary snapshot, reuses the run's held-out final evaluation cohort for paired
oracle-route and reuse-off returns, and measures open-loop prediction error on a
separate deterministic held-out trajectory cohort.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

from artifact_io import write_json_atomic, write_sha256_sidecar
from git_provenance import require_synced_training_git_state


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
HORIZONS = (1, 2, 4, 8, 16)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenRun:
    run_dir: Path
    snapshot_path: Path
    snapshot_sha256: str
    snapshot: Mapping[str, Any]
    config: Any
    world_model: Any
    actor_bank: Any
    vendor: SimpleNamespace


class _InferenceActorBank:
    def __init__(self, entries: Mapping[int, Any]) -> None:
        self._entries = dict(entries)

    def get(self, task_id: int) -> Any:
        return self._entries[task_id]

    def get_optional(self, task_id: int) -> Any | None:
        return self._entries.get(task_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vendor_modules() -> SimpleNamespace:
    for source_path in (PROJECT_SRC, VENDORED_ATARI):
        rendered = str(source_path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    return SimpleNamespace(
        torch=importlib.import_module("torch"),
        ac=importlib.import_module("ac"),
        config=importlib.import_module("config"),
        generate=importlib.import_module("generate_trajectory"),
        train=importlib.import_module("train"),
        wm=importlib.import_module("wm"),
    )


def _final_snapshot(run_dir: Path) -> tuple[Path, str]:
    index_path = run_dir / "task_boundary_snapshots" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    snapshots = index.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError(f"No task-boundary snapshots listed in {index_path}")
    record = max(snapshots, key=lambda row: int(row["completed_epochs"]))
    snapshot_path = index_path.parent / str(record["path"])
    observed = _sha256(snapshot_path)
    expected = str(record["sha256"])
    if observed != expected:
        raise ValueError(f"Snapshot checksum mismatch: {snapshot_path}")
    return snapshot_path, observed


def _build_world_model(vendor: SimpleNamespace, config: Any, device: Any) -> Any:
    return vendor.wm.WorldModel(
        3,
        (32, 32),
        config.action_space,
        config.gru_units,
        config.cnn_depth,
        config.mlp_features,
        config.mlp_layers,
        config.wall_time_optimisation,
        compute_dtype=config.compute_dtype,
        observation_objective="reconstruction",
        observation_encoder="cnn",
        num_task_experts=config.rssm_num_experts,
        task_shared_prediction_heads=config.task_shared_prediction_heads,
        evolving_shared_core=config.evolving_shared_core,
        task_projected_image_encoder=config.task_projected_image_encoder,
        task_symmetric_image_projectors=config.task_atomic_routes,
        task_projector_bottleneck_features=config.task_projector_bottleneck_features,
        task_mechanism_bank=config.task_mechanism_bank,
        task_mechanism_reuse=config.task_mechanism_reuse,
        task_mechanism_recurrent_width=config.task_mechanism_recurrent_width,
        task_mechanism_representation_width=config.task_mechanism_representation_width,
        task_mechanism_transition_width=config.task_mechanism_transition_width,
        task_mechanism_residual_scale=config.task_mechanism_residual_scale,
        task_mechanism_num_atoms=config.task_mechanism_num_atoms,
        task_mechanism_parameterization=config.task_mechanism_parameterization,
        task_symmetric_mechanisms=config.task_atomic_routes,
    ).to(device)


def _load_frozen_run(run_dir: Path, device_name: str) -> FrozenRun:
    vendor = _vendor_modules()
    torch = vendor.torch
    snapshot_path, snapshot_sha256 = _final_snapshot(run_dir)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if snapshot.get("artifact_kind") != "task_bank_boundary_inference_snapshot":
        raise ValueError(f"Unexpected snapshot kind in {snapshot_path}")
    if snapshot.get("resumable") is not False:
        raise ValueError("Table 9 requires a frozen inference snapshot")
    config = vendor.config.Config.from_dict(dict(snapshot["config"]))
    if not config.uses_reconstruction_task_inference:
        raise ValueError("Table 9 diagnostics require AWM-AutoRoute v4")
    if int(snapshot["completed_epochs"]) != config.epochs:
        raise ValueError("Snapshot is not the completed final checkpoint")

    device = torch.device(device_name)
    world_model = _build_world_model(vendor, config, device)
    world_model.load_state_dict(snapshot["world_model_state_dict"], strict=True)
    world_model.eval().requires_grad_(False)

    actor_state = snapshot["actor_critic_bank_state_dict"]
    if actor_state.get("resumable") is not False:
        raise ValueError("Expected inference-only Actor-Critic bank state")
    entries: dict[int, Any] = {}
    for task_key, state in sorted(
        actor_state["tasks"].items(), key=lambda item: int(item[0])
    ):
        task_id = int(task_key)
        actor_critic = vendor.ac.ActorCritic(
            math.prod(world_model.ls) + world_model.h_dim,
            world_model.a_dim,
            actor_network=config.actor_network,
            h_dim=world_model.h_dim,
        ).to(device)
        actor_critic.load_state_dict(state, strict=True)
        actor_critic.eval().requires_grad_(False)
        entries[task_id] = SimpleNamespace(ac=actor_critic)
    expected_ids = set(range(config.rssm_num_experts))
    if set(entries) != expected_ids:
        raise ValueError(f"Actor bank tasks changed: {sorted(entries)}")

    return FrozenRun(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha256,
        snapshot=snapshot,
        config=config,
        world_model=world_model,
        actor_bank=_InferenceActorBank(entries),
        vendor=vendor,
    )


def _reuse_banks(world_model: Any) -> tuple[Any, ...]:
    rssm = world_model.rssm
    return (
        rssm.recurrent_mechanism_bank,
        rssm.representation_mechanism_bank,
        rssm.transition_mechanism_bank,
    )


@contextlib.contextmanager
def reuse_disabled(world_model: Any) -> Iterator[None]:
    """Temporarily remove historical atom contributions without changing weights."""
    banks = _reuse_banks(world_model)
    original = tuple(bool(bank.reuse_enabled) for bank in banks)
    try:
        for bank in banks:
            bank.reuse_enabled = False
        yield
    finally:
        for bank, enabled in zip(banks, original):
            bank.reuse_enabled = enabled


def _state_versions(frozen: FrozenRun) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    wm_versions = tuple(tensor._version for tensor in frozen.world_model.state_dict().values())
    actor_versions = tuple(
        tuple(tensor._version for tensor in frozen.actor_bank.get(task).ac.state_dict().values())
        for task in range(frozen.config.rssm_num_experts)
    )
    return wm_versions, actor_versions


def _condition_rows(
    config: Any, means: Sequence[float], stds: Sequence[float]
) -> list[dict[str, Any]]:
    raw_means, raw_stds = config_raw_returns(config, means, stds)
    return [
        {
            "task_index": task_id,
            "task_name": task.name,
            "reward_scale": task.rew_scale,
            "scaled_return_mean": float(means[task_id]),
            "scaled_return_std": float(stds[task_id]),
            "raw_return_mean": raw_means[task_id],
            "raw_return_std": raw_stds[task_id],
        }
        for task_id, task in enumerate(config.esc.env_configs)
    ]


def config_raw_returns(
    config: Any, means: Sequence[float], stds: Sequence[float]
) -> tuple[list[float], list[float]]:
    if not (len(config.esc.env_configs) == len(means) == len(stds)):
        raise ValueError("Task and return vectors must have equal length")
    raw_means, raw_stds = [], []
    for task, mean, std in zip(config.esc.env_configs, means, stds):
        if task.rew_scale == 0:
            raise ValueError(f"Zero reward scale for {task.name}")
        raw_means.append(float(mean) / float(task.rew_scale))
        raw_stds.append(float(std) / abs(float(task.rew_scale)))
    return raw_means, raw_stds


def paired_differences(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, name: str
) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError("Paired conditions must contain the same tasks")
    rows = []
    for left_row, right_row in zip(left, right):
        if left_row["task_index"] != right_row["task_index"]:
            raise ValueError("Paired task order changed")
        rows.append(
            {
                "task_index": left_row["task_index"],
                "task_name": left_row["task_name"],
                name: float(left_row["raw_return_mean"])
                - float(right_row["raw_return_mean"]),
            }
        )
    return rows


def _source_auto_rows(run_dir: Path, config: Any) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / "final_evaluation.json").read_text(encoding="utf-8"))
    if int(payload["evaluation_after_completed_epochs"]) != config.epochs:
        raise ValueError("Saved auto-route evaluation is not from the final checkpoint")
    if payload.get("task_identity_exposed_during_inference") is not False:
        raise ValueError("Saved final evaluation is not task-ID-free")
    rows = payload["tasks"]
    expected = [task.name for task in config.esc.env_configs]
    if [row["task_name"] for row in rows] != expected:
        raise ValueError("Saved auto-route task order changed")
    return rows


def _return_diagnostics(frozen: FrozenRun) -> dict[str, Any]:
    config = frozen.config
    seed_manifest = json.loads(
        (frozen.run_dir / "evaluation_seed_manifest.json").read_text(encoding="utf-8")
    )
    task_seeds = [int(seed) for seed in seed_manifest["final_evaluation"]["task_base_seeds"]]
    eval_funcs = config.get_env_schedule().eval_funcs()
    oracle_means, oracle_stds = frozen.vendor.train._evaluate_policy_tasks(
        config,
        frozen.world_model,
        None,
        eval_funcs,
        task_seeds,
        actor_critic_bank=frozen.actor_bank,
        eligible_task_count=config.rssm_num_experts,
        oracle_routes=True,
    )
    oracle = _condition_rows(config, oracle_means, oracle_stds)
    with reuse_disabled(frozen.world_model):
        no_reuse_means, no_reuse_stds = frozen.vendor.train._evaluate_policy_tasks(
            config,
            frozen.world_model,
            None,
            eval_funcs,
            task_seeds,
            actor_critic_bank=frozen.actor_bank,
            eligible_task_count=config.rssm_num_experts,
            oracle_routes=True,
        )
    no_reuse = _condition_rows(config, no_reuse_means, no_reuse_stds)
    auto = _source_auto_rows(frozen.run_dir, config)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "awm_table9_paired_return_diagnostics",
        "complete": True,
        "heldout_task_base_seeds": task_seeds,
        "rollouts_per_task_per_condition": 16,
        "conditions": {
            "auto_route_saved_final": auto,
            "oracle_route": oracle,
            "oracle_route_reuse_disabled": no_reuse,
        },
        "paired_raw_return_differences": {
            "auto_minus_oracle": paired_differences(
                auto, oracle, name="auto_minus_oracle_raw_return"
            ),
            "reuse_on_minus_off_oracle_route": paired_differences(
                oracle, no_reuse, name="reuse_on_minus_off_raw_return"
            ),
        },
        "guardrails": {
            "same_frozen_checkpoint": True,
            "same_evaluation_seed_cohort": True,
            "same_rollout_budget": True,
            "reuse_intervention_changes_weights": False,
            "reuse_intervention_uses_oracle_route_to_avoid_router_confound": True,
            "evaluation_transitions_enter_replay": False,
        },
    }


def _predictive_seed(training_seed: int, task_id: int) -> int:
    import numpy as np

    sequence = np.random.SeedSequence([training_seed, 0x54414239, task_id])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _trajectory_chunks(
    frozen: FrozenRun,
    task_id: int,
    *,
    chunks: int,
    steps_per_worker: int,
    burn_in: int,
) -> tuple[dict[str, Any], int]:
    import numpy as np

    config = frozen.config
    seed = _predictive_seed(config.seed, task_id)
    actor = frozen.actor_bank.get(task_id).ac
    env_fns = config.get_env_schedule().eval_funcs()[task_id]
    arrays = frozen.vendor.generate.generate_trajectories(
        config.n_sync * steps_per_worker,
        config.n_sync,
        frozen.world_model,
        actor,
        env_fns,
        config.env_repeat,
        seed=seed,
        task_id=task_id,
        deterministic_policy=True,
    )
    actions, observations, rewards, continues, resets = arrays
    time_steps = actions.shape[0] // config.n_sync
    reshaped = [
        value.reshape(config.n_sync, time_steps, *value.shape[1:])
        for value in (actions, observations, rewards, continues, resets)
    ]
    length = burn_in + max(HORIZONS) + 1
    candidates: list[tuple[int, int]] = []
    cont_array = reshaped[3]
    reset_array = reshaped[4]
    anchor = burn_in - 1
    for worker in range(config.n_sync):
        for start in range(0, time_steps - length + 1):
            target_start = start + anchor
            target_stop = target_start + max(HORIZONS)
            if bool(cont_array[worker, target_start:target_stop].all()) and not bool(
                reset_array[worker, target_start + 1 : target_stop + 1].any()
            ):
                candidates.append((worker, start))
    if len(candidates) < chunks:
        raise RuntimeError(
            f"Task {task_id} yielded {len(candidates)} valid chunks; need {chunks}"
        )
    selector = np.random.default_rng(seed ^ 0x515046)
    selected = selector.choice(len(candidates), size=chunks, replace=False)
    result: dict[str, Any] = {}
    for name, value in zip(
        ("actions", "observations", "rewards", "continues", "resets"), reshaped
    ):
        result[name] = frozen.vendor.torch.stack(
            [value[candidates[int(index)][0], candidates[int(index)][1] : candidates[int(index)][1] + length]
             for index in selected]
        )
    result["resets"][:, 0] = 1
    return result, seed


def _open_loop_metrics(
    frozen: FrozenRun, task_id: int, chunks: Mapping[str, Any], burn_in: int
) -> list[dict[str, Any]]:
    torch = frozen.vendor.torch
    wm = frozen.world_model
    device = next(wm.parameters()).device
    actions = chunks["actions"].to(device).swapaxes(0, 1)
    observations = chunks["observations"].to(device).swapaxes(0, 1)
    rewards = chunks["rewards"].to(device).swapaxes(0, 1)
    resets = chunks["resets"].to(device).swapaxes(0, 1)
    batch = actions.shape[1]
    anchor = burn_in - 1
    reward_scale = float(frozen.config.esc.env_configs[task_id].rew_scale)
    rows = []
    with torch.inference_mode():
        z0, h0 = wm.rssm.initial_state(batch)
        _, posterior_z, hiddens = wm.rssm(
            z0,
            actions[: burn_in],
            h0,
            observations[: burn_in],
            resets[: burn_in],
            task_id=task_id,
            stochastic=False,
        )
        open_z = posterior_z[anchor]
        open_h = hiddens[anchor]
        no_reset = torch.zeros(batch, 1, device=device)
        for horizon in range(1, max(HORIZONS) + 1):
            _, open_z, open_h = wm.rssm(
                open_z,
                actions[anchor + horizon],
                open_h,
                None,
                no_reset,
                task_id=task_id,
                stochastic=False,
            )
            if horizon not in HORIZONS:
                continue
            state = wm.zh_transform(open_z, open_h)
            predicted_observation = wm.decoder_for(task_id)(state)
            predicted_reward = frozen.vendor.wm.symexp(
                wm.predict_reward_symlog(state, task_id)
            )
            visual = (
                predicted_observation - observations[anchor + horizon]
            ).square().mean(dim=(1, 2, 3))
            reward = (
                predicted_reward / reward_scale
                - rewards[anchor + horizon] / reward_scale
            ).abs().squeeze(-1)
            rows.append(
                {
                    "horizon": horizon,
                    "visual_mse_mean": float(visual.mean().cpu()),
                    "visual_mse_std": float(visual.std(unbiased=True).cpu()),
                    "raw_reward_mae_mean": float(reward.mean().cpu()),
                    "raw_reward_mae_std": float(reward.std(unbiased=True).cpu()),
                    "chunks": batch,
                }
            )
    return rows


def _predictive_diagnostics(
    frozen: FrozenRun, *, chunks: int, steps_per_worker: int, burn_in: int
) -> dict[str, Any]:
    tasks = []
    for task_id, task in enumerate(frozen.config.esc.env_configs):
        dataset, seed = _trajectory_chunks(
            frozen,
            task_id,
            chunks=chunks,
            steps_per_worker=steps_per_worker,
            burn_in=burn_in,
        )
        tasks.append(
            {
                "task_index": task_id,
                "task_name": task.name,
                "reward_scale": task.rew_scale,
                "heldout_collection_seed": seed,
                "metrics": _open_loop_metrics(frozen, task_id, dataset, burn_in),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "awm_table9_predictive_retention",
        "complete": True,
        "horizons": list(HORIZONS),
        "burn_in": burn_in,
        "chunks_per_task": chunks,
        "collection_steps_per_worker_per_task": steps_per_worker,
        "tasks": tasks,
        "guardrails": {
            "frozen_final_checkpoint": True,
            "oracle_task_route_for_model_diagnostic": True,
            "deterministic_checkpoint_policy_collection": True,
            "held_out_from_training_replay": True,
            "evaluation_transitions_enter_replay": False,
            "parameter_updates": 0,
        },
    }


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, payload)
    write_sha256_sidecar(path)


def _runtime_environment(torch: Any, device: Any) -> dict[str, Any]:
    accelerator = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        accelerator = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "visible_device_index": device.index if device.index is not None else 0,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "accelerator": accelerator,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("all", "returns", "predictive"), default="all")
    parser.add_argument("--predictive-chunks", type=int, default=64)
    parser.add_argument("--predictive-steps-per-worker", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.predictive_chunks < 2 or args.predictive_steps_per_worker < 64:
        raise ValueError("Predictive audit budgets are too small")
    if args.burn_in < 1:
        raise ValueError("burn-in must be positive")
    git = require_synced_training_git_state(ROOT)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _load_frozen_run(run_dir, args.device)
    versions_before = _state_versions(frozen)
    produced = []
    if args.mode in {"all", "returns"}:
        path = output_dir / "paired_returns.json"
        _write_result(path, _return_diagnostics(frozen))
        produced.append(path.name)
    if args.mode in {"all", "predictive"}:
        path = output_dir / "predictive_retention.json"
        _write_result(
            path,
            _predictive_diagnostics(
                frozen,
                chunks=args.predictive_chunks,
                steps_per_worker=args.predictive_steps_per_worker,
                burn_in=args.burn_in,
            ),
        )
        produced.append(path.name)
    versions_after = _state_versions(frozen)
    if versions_before != versions_after:
        raise RuntimeError("Frozen checkpoint state changed during Table 9 diagnostics")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "awm_table9_diagnostic_manifest",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git": git,
        "source_run": str(run_dir),
        "source_training_commit": frozen.snapshot["project_git_commit"],
        "source_artifacts": {
            "final_evaluation_sha256": _sha256(run_dir / "final_evaluation.json"),
            "evaluation_seed_manifest_sha256": _sha256(
                run_dir / "evaluation_seed_manifest.json"
            ),
        },
        "snapshot": {
            "path": str(frozen.snapshot_path),
            "sha256": frozen.snapshot_sha256,
            "completed_epochs": frozen.snapshot["completed_epochs"],
        },
        "training_seed": frozen.config.seed,
        "runtime_environment": _runtime_environment(
            frozen.vendor.torch, next(frozen.world_model.parameters()).device
        ),
        "outputs": produced,
        "parameter_versions_unchanged": True,
        "training_or_optimizer_steps": 0,
        "evaluation_transitions_enter_replay": False,
    }
    _write_result(output_dir / "manifest.json", manifest)
    print(f"[table9] complete seed={frozen.config.seed} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
