#!/usr/bin/env python3
"""Measure learned mechanism reuse without changing checkpoint weights.

The diagnostic has two parts.  First, it evaluates the final Task-3 policy
under transient in-memory gate ablations on the fixed held-out cohort.  Second,
it replays one fixed full-policy trajectory through every ablation to measure
mechanism contribution ratios and latent/reward drift.  It also completes the
epoch-260/270 by periodic/held-out cross-cohort table.

No transition collected here is written to Replay and no gradient update is
performed.  The source snapshots remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

from git_provenance import git_state


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)
PROTOCOL = (
    "CNN-MechanismBank-RSSM-ARROW-v1-"
    "Task1SnapshotSeeded-Atari-TaskAware"
)
CONDITIONS = (
    "full_reuse",
    "no_reuse",
    "no_recurrent_reuse",
    "no_posterior_reuse",
    "no_prior_reuse",
)
BANK_NAMES = ("recurrent", "posterior", "prior")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_checksum(path: Path) -> str:
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Snapshot checksum is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields or fields[0] != digest:
        raise RuntimeError(f"Snapshot checksum mismatch: {path}")
    return digest


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
        raise RuntimeError("Reuse evaluation requires a clean worktree")
    if state["upstream"] is None or state["ahead"] or state["behind"]:
        raise RuntimeError(
            "Reuse evaluation requires HEAD to match its configured upstream: "
            f"{state}"
        )
    return state


def _load_exact(module: Any, state: Mapping[str, Any], label: str) -> None:
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _build_world_model(
    torch: Any, vendor: SimpleNamespace, config: Any, device: Any
) -> Any:
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
        task_projected_image_encoder=config.task_projected_image_encoder,
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
    ).to(device)


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


def _validate_snapshot(payload: Mapping[str, Any], *, completed_epochs: int) -> None:
    if payload.get("artifact_kind") != "task_bank_evaluation_snapshot":
        raise ValueError("Expected a task-bank evaluation snapshot")
    if bool(payload.get("resumable", True)):
        raise ValueError("Evaluation snapshot must be explicitly non-resumable")
    if int(payload.get("completed_epochs", -1)) != completed_epochs:
        raise ValueError(
            f"Expected completed_epochs={completed_epochs}, got "
            f"{payload.get('completed_epochs')}"
        )
    config = payload.get("config", {})
    if not config.get("task_mechanism_bank"):
        raise ValueError("Snapshot does not use a mechanism bank")
    if not config.get("task_mechanism_reuse"):
        raise ValueError("Source snapshot is not the learned-reuse method")


def _banks(world_model: Any) -> dict[str, Any]:
    rssm = world_model.rssm
    return {
        "recurrent": rssm.recurrent_mechanism_bank,
        "posterior": rssm.representation_mechanism_bank,
        "prior": rssm.transition_mechanism_bank,
    }


def _route_values(world_model: Any, task_id: int) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in bank.route_values(task_id)]
        for name, bank in _banks(world_model).items()
    }


def _disabled_banks(condition: str) -> frozenset[str]:
    mapping = {
        "full_reuse": frozenset(),
        "no_reuse": frozenset(BANK_NAMES),
        "no_recurrent_reuse": frozenset(("recurrent",)),
        "no_posterior_reuse": frozenset(("posterior",)),
        "no_prior_reuse": frozenset(("prior",)),
    }
    try:
        return mapping[condition]
    except KeyError as error:
        raise ValueError(f"Unknown reuse condition: {condition}") from error


@contextmanager
def _gate_condition(
    torch: Any, world_model: Any, task_id: int, condition: str
) -> Iterator[None]:
    disabled = _disabled_banks(condition)
    task_route_index = task_id - 1
    originals: dict[str, Any] = {}
    if task_route_index < 1:
        raise ValueError("Gate ablation requires at least one reusable mechanism")
    try:
        with torch.no_grad():
            for name, bank in _banks(world_model).items():
                logits = bank.routes[task_route_index].logits
                if logits is None:
                    raise RuntimeError(f"{name} route has no reusable gate")
                originals[name] = logits.detach().clone()
                if name in disabled:
                    logits.zero_()
        yield
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                _banks(world_model)[name].routes[task_route_index].logits.copy_(
                    original
                )


def _state_digest(torch: Any, module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous().reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _autocast(torch: Any, device: Any, compute_dtype: str) -> Any:
    if compute_dtype == "float32":
        return nullcontext()
    if compute_dtype != "bfloat16":
        raise ValueError(f"Unsupported compute dtype {compute_dtype!r}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _evaluate_policy(
    vendor: SimpleNamespace,
    config: Any,
    world_model: Any,
    actor: Any,
    *,
    task_id: int,
    seed: int,
    n_rollouts: int,
) -> dict[str, Any]:
    task = config.esc.env_configs[task_id]
    scaled_mean, scaled_std = vendor.generate.evaluate(
        config.n_sync,
        wm=world_model,
        ac=actor,
        env_fns=[task.get_function() for _ in range(config.n_sync)],
        env_repeat=config.env_repeat,
        n_rollouts=n_rollouts,
        seed=seed,
        task_id=task_id,
        deterministic_policy=True,
    )
    raw_mean = float(scaled_mean) / float(task.rew_scale)
    raw_std = float(scaled_std) / float(task.rew_scale)
    return {
        "task_index": task_id,
        "task_name": task.name,
        "base_seed": int(seed),
        "rollouts": int(n_rollouts),
        "policy": "deterministic_argmax_and_latent_mode",
        "scaled_return_mean": float(scaled_mean),
        "scaled_return_std": float(scaled_std),
        "raw_return_mean": raw_mean,
        "raw_return_std": raw_std,
        "raw_return_95ci_normal": [
            raw_mean - 1.96 * raw_std / math.sqrt(n_rollouts),
            raw_mean + 1.96 * raw_std / math.sqrt(n_rollouts),
        ],
    }


def _saved_evaluation(payload: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    for result in payload["evaluation"]:
        if int(result["task_index"]) == task_id:
            return dict(result)
    raise KeyError(f"Snapshot has no saved evaluation for task {task_id}")


class _NormAccumulator:
    def __init__(self) -> None:
        self.sums: list[Any] = []
        self.count = 0

    def hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        norms = output.detach().float().norm(dim=-1)
        self.sums.append(norms.sum())
        self.count += norms.numel()

    def mean(self, torch: Any) -> float:
        if not self.sums or self.count == 0:
            raise RuntimeError("Mechanism norm hook observed no activations")
        return float(torch.stack(self.sums).sum().item() / self.count)


@contextmanager
def _mechanism_norm_hooks(
    world_model: Any, task_id: int
) -> Iterator[dict[str, dict[str, _NormAccumulator]]]:
    current_index = task_id - 1
    if current_index < 1:
        raise ValueError("Contribution ratios require a reusable old mechanism")
    accumulators: dict[str, dict[str, _NormAccumulator]] = {}
    handles = []
    try:
        for name, bank in _banks(world_model).items():
            old = _NormAccumulator()
            current = _NormAccumulator()
            accumulators[name] = {"old": old, "current": current}
            handles.append(bank.mechanisms[0].register_forward_hook(old.hook))
            handles.append(
                bank.mechanisms[current_index].register_forward_hook(current.hook)
            )
        yield accumulators
    finally:
        for handle in handles:
            handle.remove()


def _functional_ratios(
    torch: Any,
    world_model: Any,
    task_id: int,
    accumulators: Mapping[str, Mapping[str, _NormAccumulator]],
) -> dict[str, Any]:
    gates = _route_values(world_model, task_id)
    result: dict[str, Any] = {}
    for name in BANK_NAMES:
        if len(gates[name]) != 1:
            raise ValueError("The three-task pilot must have exactly one old route")
        gate = float(gates[name][0])
        old_mean = accumulators[name]["old"].mean(torch)
        current_mean = accumulators[name]["current"].mean(torch)
        weighted_old_mean = abs(gate) * old_mean
        result[name] = {
            "gate": gate,
            "mean_old_mechanism_l2": old_mean,
            "mean_abs_gate_times_old_l2": weighted_old_mean,
            "mean_current_mechanism_l2": current_mean,
            "eta": (
                weighted_old_mean / current_mean
                if current_mean > 0
                else None
            ),
        }
    return result


def _normalize_log_probs(torch: Any, values: Any) -> Any:
    values = values.float()
    return values - torch.logsumexp(values, dim=-1, keepdim=True)


def _categorical_kl(torch: Any, log_p: Any, log_q: Any) -> Any:
    log_p = _normalize_log_probs(torch, log_p)
    log_q = _normalize_log_probs(torch, log_q)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def _categorical_js(torch: Any, log_p: Any, log_q: Any) -> Any:
    log_p = _normalize_log_probs(torch, log_p)
    log_q = _normalize_log_probs(torch, log_q)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return 0.5 * (
        _categorical_kl(torch, log_p, log_m)
        + _categorical_kl(torch, log_q, log_m)
    )


def _drift_summary(torch: Any, full: Any, changed: Any) -> dict[str, float]:
    difference = (full.float() - changed.float()).flatten(0, -2)
    l2 = difference.norm(dim=-1)
    return {
        "mean_l2": float(l2.mean()),
        "rms_per_element": float(difference.square().mean().sqrt()),
        "max_l2": float(l2.max()),
    }


def _collect_diagnostic_trajectory(
    vendor: SimpleNamespace,
    config: Any,
    world_model: Any,
    actor: Any,
    *,
    task_id: int,
    seed: int,
    decisions: int,
) -> tuple[Any, Any, Any]:
    task = config.esc.env_configs[task_id]
    actions, images, _rewards, _continues, resets = (
        vendor.generate.generate_trajectories(
            decisions,
            config.n_sync,
            world_model,
            actor,
            [task.get_function() for _ in range(config.n_sync)],
            config.env_repeat,
            None,
            no_images=False,
            seed=seed,
            task_id=task_id,
            deterministic_policy=True,
        )
    )
    if images is None or actions.shape[0] % config.n_sync:
        raise RuntimeError("Diagnostic trajectory has an invalid packed shape")
    steps = actions.shape[0] // config.n_sync
    actions = actions.reshape(config.n_sync, steps, -1).swapaxes(0, 1)
    images = images.reshape(config.n_sync, steps, *images.shape[1:]).swapaxes(0, 1)
    resets = resets.reshape(config.n_sync, steps, -1).swapaxes(0, 1)
    return actions, images, resets


def _embed_images(
    torch: Any,
    world_model: Any,
    images: Any,
    *,
    task_id: int,
    device: Any,
    chunk_steps: int,
) -> Any:
    embeddings = []
    with torch.no_grad():
        for start in range(0, images.shape[0], chunk_steps):
            image_chunk = images[start : start + chunk_steps].to(device)
            with _autocast(torch, device, world_model.compute_dtype):
                embeddings.append(
                    world_model.rssm.embed_observations(
                        image_chunk, task_id=task_id
                    )
                )
    return torch.cat(embeddings, dim=0)


def _sequence_states(
    torch: Any,
    world_model: Any,
    actions: Any,
    embeddings: Any,
    resets: Any,
    *,
    task_id: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    actions = actions.to(device)
    resets = resets.to(device)
    initial_z, initial_h = world_model.rssm.initial_state(actions.shape[1])
    with torch.no_grad(), _autocast(torch, device, world_model.compute_dtype):
        posterior, z, h = world_model.rssm.observe_embeddings(
            initial_z,
            actions,
            initial_h,
            embeddings,
            resets,
            stochastic=False,
            task_id=task_id,
        )
        prior = world_model.rssm.prior(h, task_id=task_id)
        model_state = world_model.zh_transform(z, h)
        reward_symlog = world_model.predict_reward_symlog(
            model_state, task_id=task_id
        )
    return posterior, prior, z, h, reward_symlog


def _anchor_indices(torch: Any, count: int, maximum: int, device: Any) -> Any:
    used = min(count, maximum)
    if used < 1:
        raise ValueError("At least one imagination anchor is required")
    if used == count:
        return torch.arange(count, device=device)
    return torch.linspace(0, count - 1, steps=used, device=device).long()


def _full_imagination(
    torch: Any,
    vendor: SimpleNamespace,
    world_model: Any,
    actor: Any,
    z: Any,
    h: Any,
    *,
    task_id: int,
    horizon: int,
    max_anchors: int,
    device: Any,
) -> dict[str, Any]:
    flat_z = z.flatten(0, 1)
    flat_h = h.flatten(0, 1)
    indices = _anchor_indices(torch, flat_h.shape[0], max_anchors, device)
    z_step = flat_z.index_select(0, indices)
    h_step = flat_h.index_select(0, indices)
    reset = torch.zeros(h_step.shape[0], 1, device=device)
    actions = []
    hidden = []
    rewards = []
    with torch.no_grad():
        for _ in range(horizon):
            with _autocast(torch, device, world_model.compute_dtype):
                policy_logits = actor.actor(
                    vendor.ac.zh_to_ac_state(z_step, h_step)
                ).float()
                action_index = policy_logits.argmax(dim=-1)
                action = torch.nn.functional.one_hot(
                    action_index, world_model.a_dim
                ).float()
                _, z_step, h_step = world_model.rssm(
                    z_step,
                    action,
                    h_step,
                    None,
                    reset,
                    stochastic=False,
                    task_id=task_id,
                )
                reward = world_model.predict_reward_symlog(
                    world_model.zh_transform(z_step, h_step), task_id=task_id
                )
            actions.append(action)
            hidden.append(h_step)
            rewards.append(reward)
    return {
        "indices": indices,
        "actions": actions,
        "hidden": torch.stack(hidden),
        "reward_symlog": torch.stack(rewards),
    }


def _condition_imagination(
    torch: Any,
    world_model: Any,
    z: Any,
    h: Any,
    full_imagination: Mapping[str, Any],
    *,
    task_id: int,
    device: Any,
) -> tuple[Any, Any]:
    flat_z = z.flatten(0, 1)
    flat_h = h.flatten(0, 1)
    indices = full_imagination["indices"]
    z_step = flat_z.index_select(0, indices)
    h_step = flat_h.index_select(0, indices)
    reset = torch.zeros(h_step.shape[0], 1, device=device)
    hidden = []
    rewards = []
    with torch.no_grad():
        for action in full_imagination["actions"]:
            with _autocast(torch, device, world_model.compute_dtype):
                _, z_step, h_step = world_model.rssm(
                    z_step,
                    action,
                    h_step,
                    None,
                    reset,
                    stochastic=False,
                    task_id=task_id,
                )
                reward = world_model.predict_reward_symlog(
                    world_model.zh_transform(z_step, h_step), task_id=task_id
                )
            hidden.append(h_step)
            rewards.append(reward)
    return torch.stack(hidden), torch.stack(rewards)


def _reward_drift(
    torch: Any,
    vendor: SimpleNamespace,
    full_symlog: Any,
    changed_symlog: Any,
    *,
    reward_scale: float,
) -> dict[str, float]:
    symlog_delta = (full_symlog.float() - changed_symlog.float()).abs()
    full_raw = vendor.wm.symexp(full_symlog.float()) / reward_scale
    changed_raw = vendor.wm.symexp(changed_symlog.float()) / reward_scale
    raw_delta = (full_raw - changed_raw).abs()
    return {
        "mean_abs_symlog": float(symlog_delta.mean()),
        "mean_abs_raw_reward": float(raw_delta.mean()),
        "max_abs_raw_reward": float(raw_delta.max()),
    }


def _condition_diagnostics(
    torch: Any,
    vendor: SimpleNamespace,
    world_model: Any,
    actor: Any,
    actions: Any,
    embeddings: Any,
    resets: Any,
    full_states: tuple[Any, Any, Any, Any, Any],
    full_imagination: Mapping[str, Any],
    *,
    condition: str,
    task_id: int,
    reward_scale: float,
    device: Any,
) -> dict[str, Any]:
    full_post, full_prior, _full_z, full_h, full_reward = full_states
    with _gate_condition(torch, world_model, task_id, condition):
        changed = _sequence_states(
            torch,
            world_model,
            actions,
            embeddings,
            resets,
            task_id=task_id,
            device=device,
        )
        post, prior, z, h, reward = changed
        imagined_h, imagined_reward = _condition_imagination(
            torch,
            world_model,
            z,
            h,
            full_imagination,
            task_id=task_id,
            device=device,
        )

    posterior_kl = _categorical_kl(torch, full_post, post).sum(dim=-1)
    posterior_reverse_kl = _categorical_kl(torch, post, full_post).sum(dim=-1)
    prior_js = _categorical_js(torch, full_prior, prior).sum(dim=-1)
    imagined_drift = (
        full_imagination["hidden"].float() - imagined_h.float()
    ).norm(dim=-1)
    imagined_reward_drift = _reward_drift(
        torch,
        vendor,
        full_imagination["reward_symlog"],
        imagined_reward,
        reward_scale=reward_scale,
    )
    return {
        "posterior_kl_full_to_condition_mean": float(posterior_kl.mean()),
        "posterior_kl_condition_to_full_mean": float(
            posterior_reverse_kl.mean()
        ),
        "prior_js_mean": float(prior_js.mean()),
        "observed_hidden_state_drift": _drift_summary(torch, full_h, h),
        "observed_reward_prediction_drift": _reward_drift(
            torch,
            vendor,
            full_reward,
            reward,
            reward_scale=reward_scale,
        ),
        "imagined_hidden_state_drift_mean_l2": float(imagined_drift.mean()),
        "imagined_hidden_state_drift_final_l2": float(imagined_drift[-1].mean()),
        "imagined_hidden_state_drift_l2_by_horizon": [
            float(value) for value in imagined_drift.mean(dim=1)
        ],
        "imagined_reward_prediction_drift": imagined_reward_drift,
    }


def _load_model_and_actor(
    torch: Any,
    vendor: SimpleNamespace,
    payload: Mapping[str, Any],
    *,
    task_id: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    config = vendor.config.Config.from_dict(payload["config"])
    if not config.task_mechanism_bank or not config.task_mechanism_reuse:
        raise ValueError("Expected the learned-reuse MB-RSSM config")
    world_model = _build_world_model(torch, vendor, config, device)
    _load_exact(world_model, payload["world_model_state_dict"], "World model")
    actor_bank = payload["actor_critic_bank_state_dict"]["tasks"]
    actor = _build_actor(vendor, world_model, config, actor_bank[str(task_id)])
    world_model.eval()
    actor.eval()
    return config, world_model, actor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch270-checkpoint", type=Path, required=True)
    parser.add_argument("--epoch260-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-task", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--diagnostic-decisions", type=int, default=4096)
    parser.add_argument("--diagnostic-chunk-steps", type=int, default=64)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument("--max-imagination-anchors", type=int, default=256)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--classification", choices=("smoke", "pilot"), default="pilot"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    positive = {
        "cpu_threads": args.cpu_threads,
        "n_rollouts": args.n_rollouts,
        "diagnostic_decisions": args.diagnostic_decisions,
        "diagnostic_chunk_steps": args.diagnostic_chunk_steps,
        "imagination_horizon": args.imagination_horizon,
        "max_imagination_anchors": args.max_imagination_anchors,
    }
    if any(value < 1 for value in positive.values()):
        raise ValueError(f"All numerical budgets must be positive: {positive}")
    if args.classification == "pilot":
        if args.n_rollouts != 16 or tuple(args.conditions) != CONDITIONS:
            raise ValueError("Pilot evaluation fixes 16 rollouts and all conditions")
    epoch270_path = args.epoch270_checkpoint.expanduser().resolve()
    epoch260_path = args.epoch260_checkpoint.expanduser().resolve()
    for path in (epoch270_path, epoch260_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    epoch270_sha = _require_checksum(epoch270_path)
    epoch260_sha = _require_checksum(epoch260_path)

    vendor = _vendor_modules()
    import torch

    epoch270 = torch.load(epoch270_path, map_location="cpu", weights_only=False)
    epoch260 = torch.load(epoch260_path, map_location="cpu", weights_only=False)
    _validate_snapshot(epoch270, completed_epochs=270)
    _validate_snapshot(epoch260, completed_epochs=260)
    if epoch270["config"] != epoch260["config"]:
        raise RuntimeError("Epoch-260 and epoch-270 resolved configs differ")
    task_id = int(args.target_task)
    if task_id < 2:
        raise ValueError("The reuse diagnostic targets the third task")
    periodic_seed = int(epoch260["task_base_seeds"][task_id])
    heldout_seed = int(epoch270["task_base_seeds"][task_id])

    git = git_state(ROOT) if args.dry_run else _require_synced_git()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "mechanism_bank_reuse_gate_ablation_manifest",
        "classification": args.classification,
        "protocol": PROTOCOL,
        "official_result": False,
        "single_seed_pilot": True,
        "task_agnostic_claimed": False,
        "project_git": git,
        "source_snapshots": {
            "epoch260": {"path": str(epoch260_path), "sha256": epoch260_sha},
            "epoch270": {"path": str(epoch270_path), "sha256": epoch270_sha},
        },
        "target_task": task_id,
        "periodic_validation_seed": periodic_seed,
        "heldout_final_seed": heldout_seed,
        "conditions": list(args.conditions),
        "rollouts_per_condition": args.n_rollouts,
        "diagnostic_decisions": args.diagnostic_decisions,
        "diagnostic_trajectory_policy": "full-reuse deterministic policy",
        "diagnostic_trajectory_shared_across_conditions": True,
        "imagination_horizon": args.imagination_horizon,
        "max_imagination_anchors": args.max_imagination_anchors,
        "evaluation_transitions_enter_replay": False,
        "gradient_updates": 0,
        "checkpoint_parameters_persistently_modified": False,
        "status": "dry_run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    try:
        config, world_model, actor = _load_model_and_actor(
            torch, vendor, epoch270, task_id=task_id, device=device
        )
        initial_digest = _state_digest(torch, world_model)
        learned_routes = _route_values(world_model, task_id)
        heldout_conditions: dict[str, Any] = {}
        for condition in args.conditions:
            with _gate_condition(torch, world_model, task_id, condition):
                heldout_conditions[condition] = _evaluate_policy(
                    vendor,
                    config,
                    world_model,
                    actor,
                    task_id=task_id,
                    seed=heldout_seed,
                    n_rollouts=args.n_rollouts,
                )
                heldout_conditions[condition]["effective_routes"] = (
                    _route_values(world_model, task_id)
                )

        with _gate_condition(torch, world_model, task_id, "full_reuse"):
            epoch270_periodic = _evaluate_policy(
                vendor,
                config,
                world_model,
                actor,
                task_id=task_id,
                seed=periodic_seed,
                n_rollouts=args.n_rollouts,
            )

        actions, images, resets = _collect_diagnostic_trajectory(
            vendor,
            config,
            world_model,
            actor,
            task_id=task_id,
            seed=heldout_seed,
            decisions=args.diagnostic_decisions,
        )
        embeddings = _embed_images(
            torch,
            world_model,
            images,
            task_id=task_id,
            device=device,
            chunk_steps=args.diagnostic_chunk_steps,
        )
        with _mechanism_norm_hooks(world_model, task_id) as norm_accumulators:
            full_states = _sequence_states(
                torch,
                world_model,
                actions,
                embeddings,
                resets,
                task_id=task_id,
                device=device,
            )
        eta = _functional_ratios(
            torch, world_model, task_id, norm_accumulators
        )
        full_imagination = _full_imagination(
            torch,
            vendor,
            world_model,
            actor,
            full_states[2],
            full_states[3],
            task_id=task_id,
            horizon=args.imagination_horizon,
            max_anchors=args.max_imagination_anchors,
            device=device,
        )
        diagnostics: dict[str, Any] = {}
        for condition in args.conditions:
            diagnostics[condition] = _condition_diagnostics(
                torch,
                vendor,
                world_model,
                actor,
                actions,
                embeddings,
                resets,
                full_states,
                full_imagination,
                condition=condition,
                task_id=task_id,
                reward_scale=float(config.esc.env_configs[task_id].rew_scale),
                device=device,
            )
        final_digest = _state_digest(torch, world_model)
        if final_digest != initial_digest:
            raise RuntimeError("Transient gate evaluation mutated model state")

        del actor, world_model, embeddings, images, full_states, full_imagination
        if device.type == "cuda":
            torch.cuda.empty_cache()

        config260, world_model260, actor260 = _load_model_and_actor(
            torch, vendor, epoch260, task_id=task_id, device=device
        )
        with _gate_condition(torch, world_model260, task_id, "full_reuse"):
            epoch260_heldout = _evaluate_policy(
                vendor,
                config260,
                world_model260,
                actor260,
                task_id=task_id,
                seed=heldout_seed,
                n_rollouts=args.n_rollouts,
            )

        saved270 = _saved_evaluation(epoch270, task_id)
        saved260 = _saved_evaluation(epoch260, task_id)
        full_reproduction = None
        if args.n_rollouts == 16 and "full_reuse" in heldout_conditions:
            difference = (
                heldout_conditions["full_reuse"]["raw_return_mean"]
                - float(saved270["raw_return_mean"])
            )
            full_reproduction = {
                "saved_raw_return_mean": float(saved270["raw_return_mean"]),
                "reevaluated_raw_return_mean": heldout_conditions["full_reuse"][
                    "raw_return_mean"
                ],
                "difference": difference,
                "passed_atol_1e-3": abs(difference) <= 1e-3,
            }
            if not full_reproduction["passed_atol_1e-3"]:
                raise RuntimeError(
                    "Full-reuse heldout reevaluation did not reproduce the saved metric"
                )

        results: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "mechanism_bank_reuse_gate_ablation_results",
            "classification": args.classification,
            "complete": True,
            "source_manifest": "manifest.json",
            "learned_routes": learned_routes,
            "heldout_gate_ablation": heldout_conditions,
            "functional_contribution_ratios": eta,
            "shared_trajectory_diagnostics": {
                "decision_budget": args.diagnostic_decisions,
                "base_seed": heldout_seed,
                "posterior_kl_direction": "full_reuse_to_condition",
                "prior_metric": "categorical_Jensen-Shannon_divergence",
                "reward_prediction_units": "raw_environment_reward",
                "conditions": diagnostics,
            },
            "cross_cohort": {
                "epoch260": {
                    "periodic_validation_saved": saved260,
                    "heldout_final_reevaluated": epoch260_heldout,
                },
                "epoch270": {
                    "periodic_validation_reevaluated": epoch270_periodic,
                    "heldout_final_saved": saved270,
                },
            },
            "integrity": {
                "world_model_state_sha256_before": initial_digest,
                "world_model_state_sha256_after": final_digest,
                "state_unchanged": initial_digest == final_digest,
                "saved_full_metric_reproduction": full_reproduction,
                "gradient_updates": 0,
                "evaluation_transitions_enter_replay": False,
            },
            "interpretation_limits": [
                "posthoc gate ablation tests dependence of the trained policy, not retraining without reuse",
                "single seed does not establish a paper-level reuse claim",
                "task identity selects the route and policy",
            ],
        }
        results_path = output_dir / "results.json"
        _write_json_atomic(results_path, results)
        results_sha = _sha256(results_path)
        _write_text_atomic(
            results_path.with_suffix(".json.sha256"),
            f"{results_sha}  {results_path.name}\n",
        )
        manifest["status"] = "complete"
        manifest["results_sha256"] = results_sha
        _write_json_atomic(output_dir / "manifest.json", manifest)
        _write_json_atomic(
            output_dir / "run_status.json",
            {"complete": True, "return_code": 0, "results_sha256": results_sha},
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        print(f"results_sha256={results_sha}")
        return 0
    except BaseException as error:
        _write_json_atomic(
            output_dir / "run_status.json",
            {
                "complete": False,
                "return_code": 1,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
