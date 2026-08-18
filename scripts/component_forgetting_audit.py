#!/usr/bin/env python3
"""Collect and evaluate fixed checkpoint-differencing audit data for ARROW/DV3.

The script has two deliberately separate phases:

* ``collect`` interacts with environments using only a boundary snapshot and
  writes held-out diagnostic chunks.  Those transitions never enter replay and
  no model parameters are updated.
* ``evaluate`` consumes only snapshots and the frozen chunks.  It performs
  deterministic teacher-forced and open-loop forwards, so it can be repeated
  without environment interaction.

The initial target is the completed DV3/FIFO pilot.  The file is method-agnostic
at the snapshot boundary so the same evaluator can later be used for ARROW-50.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from component_audit_metrics import (
    linear_cka,
    mean_and_episode_bootstrap_ci,
    symmetric_kl_from_log_probs,
)
from git_provenance import git_state, require_synced_training_git_state


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
SCHEMA_VERSION = 1
DEFAULT_HORIZONS = (1, 2, 4, 8, 16)
DEFAULT_BURN_IN = 16
DEFAULT_CHUNK_LENGTH = 64
DEFAULT_DIAGNOSTIC_CHUNKS = 256
DEFAULT_EVENT_CHUNKS = 32
DEFAULT_BOOTSTRAP_REPETITIONS = 1_000
EVAL_RE = re.compile(
    r"Eval for epoch:\s+(?P<epoch>\d+)\s*\n"
    r"Eval means: (?P<means>\[[^\n]+\])\s*\n"
    r"Eval stds: (?P<stds>\[[^\n]+\])"
)


@dataclass(frozen=True)
class SnapshotSpec:
    """Portable metadata for one analysis snapshot."""

    path: Path
    sha256: str
    epoch: int
    reason: str
    task_index: int | None
    task_name: str | None
    payload: Mapping[str, Any]

    @property
    def label(self) -> str:
        if self.task_index is None:
            return f"Cfinal_e{self.epoch}"
        return f"C{self.task_index + 1}_e{self.epoch}"


@dataclass
class ModelBundle:
    """A world model and actor-critic instantiated from one snapshot."""

    world_model: Any
    actor_critic: Any
    config: Mapping[str, Any]
    device: Any
    vendor: SimpleNamespace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_sha256_sidecar(path: Path) -> str:
    digest = _sha256(path)
    _write_text_atomic(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def _vendor_modules() -> SimpleNamespace:
    """Load the pinned vendored modules using their upstream import layout.

    The vendored ARROW Atari source is intentionally not rewritten as a project
    package.  Its modules use upstream top-level imports (``from rssm import``),
    so this explicit, local bootstrap is the narrow compatibility boundary.
    """

    for source_path in (PROJECT_SRC, VENDORED_ATARI):
        rendered = str(source_path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)

    ac = importlib.import_module("ac")
    wm = importlib.import_module("wm")
    return SimpleNamespace(
        ActorCritic=ac.ActorCritic,
        zh_to_ac_state=ac.zh_to_ac_state,
        WorldModel=wm.WorldModel,
        categorical_kl=wm.categorical_kl,
        symexp=wm.symexp,
        symlog=wm.symlog,
    )


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - local CPU-only UX
        raise RuntimeError(
            "This command needs the experiment PyTorch environment. "
            "Run it on the configured GPU server."
        ) from error
    return torch


def _require_snapshot(payload: Mapping[str, Any], path: Path) -> None:
    required = {
        "artifact_kind",
        "resumable",
        "reason",
        "epoch",
        "config",
        "world_model_state_dict",
        "actor_critic_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is not a valid analysis snapshot; missing {missing}")
    if payload["artifact_kind"] != "analysis_snapshot" or payload["resumable"]:
        raise ValueError(f"{path} is not a non-resumable analysis snapshot")
    if payload["reason"] not in {"task_boundary", "final"}:
        raise ValueError(f"{path} has unsupported snapshot reason {payload['reason']!r}")


def load_snapshot_specs(snapshot_dir: Path) -> list[SnapshotSpec]:
    """Load and validate the immutable snapshot sequence for one run."""
    torch = _torch()
    snapshot_dir = snapshot_dir.resolve()
    paths = sorted(snapshot_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No analysis snapshots found in {snapshot_dir}")

    specs: list[SnapshotSpec] = []
    for path in paths:
        digest = _sha256(path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing checksum sidecar for {path.name}")
        declared = sidecar.read_text(encoding="ascii").split()[0]
        if declared != digest:
            raise ValueError(f"Checksum mismatch for {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Snapshot payload is not a mapping: {path}")
        _require_snapshot(payload, path)
        task = payload.get("task")
        task_index = None if task is None else int(task["task_index"])
        task_name = None if task is None else str(task["task_name"])
        specs.append(
            SnapshotSpec(
                path=path,
                sha256=digest,
                epoch=int(payload["epoch"]),
                reason=str(payload["reason"]),
                task_index=task_index,
                task_name=task_name,
                payload=payload,
            )
        )

    boundaries = sorted(
        (spec for spec in specs if spec.reason == "task_boundary"),
        key=lambda spec: (spec.task_index if spec.task_index is not None else -1, spec.epoch),
    )
    if not boundaries:
        raise ValueError(f"No task-boundary snapshots found in {snapshot_dir}")
    finals = [spec for spec in specs if spec.reason == "final"]
    if len(finals) != 1:
        raise ValueError(f"Expected exactly one final snapshot in {snapshot_dir}")
    if any(spec.task_index != index for index, spec in enumerate(boundaries)):
        raise ValueError("Boundary snapshots must cover task indices 0..N-1 exactly once")
    if finals[0].epoch < boundaries[-1].epoch:
        raise ValueError("Final snapshot cannot precede the last task boundary")
    return [*boundaries, finals[0]]


def _model_bundle(spec: SnapshotSpec, device_name: str) -> ModelBundle:
    torch = _torch()
    vendor = _vendor_modules()
    config = spec.payload["config"]
    device = torch.device(device_name)
    world_model = vendor.WorldModel(
        3,
        (32, 32),
        int(config["action_space"]),
        int(config["gru_units"]),
        int(config["cnn_depth"]),
        int(config["mlp_features"]),
        int(config["mlp_layers"]),
        bool(config["wall_time_optimisation"]),
        observation_objective=str(config.get("observation_objective", "reconstruction")),
        r2_barlow_loss_scale=float(config.get("r2_barlow_loss_scale", 0.05)),
        r2_redundancy_scale=float(config.get("r2_redundancy_scale", 5e-4)),
        r2_normalization_eps=float(config.get("r2_normalization_eps", 1e-8)),
    ).to(device)
    world_model.load_state_dict(spec.payload["world_model_state_dict"], strict=True)
    actor_critic = vendor.ActorCritic(
        int(np.prod(world_model.ls)) + world_model.h_dim,
        world_model.a_dim,
        actor_network=str(config.get("actor_network", "mlp")),
        h_dim=world_model.h_dim,
        kan_hidden_features=int(config.get("actor_kan_hidden_features", 64)),
        kan_grid_size=int(config.get("actor_kan_grid_size", 5)),
        kan_spline_order=int(config.get("actor_kan_spline_order", 3)),
        kan_input_min=float(config.get("actor_kan_input_min", 0.0)),
        kan_input_max=float(config.get("actor_kan_input_max", 1.0)),
        kan_normalize_recurrent_state=bool(
            config.get("actor_kan_normalize_recurrent_state", True)
        ),
    ).to(device)
    actor_critic.load_state_dict(spec.payload["actor_critic_state_dict"], strict=True)
    world_model.eval()
    actor_critic.eval()
    return ModelBundle(world_model, actor_critic, config, device, vendor)


def _normalise_observation(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.uint8)
    if obs.shape != (64, 64, 3):
        raise ValueError(f"Expected Atari RGB observation (64, 64, 3), got {obs.shape}")
    return obs


def _one_hot(action: int, action_space: int) -> np.ndarray:
    if not 0 <= action < action_space:
        raise ValueError(f"Action {action} is outside 0..{action_space - 1}")
    encoded = np.zeros(action_space, dtype=np.uint8)
    encoded[action] = 1
    return encoded


def _make_atari_environment(task: Mapping[str, Any], env_repeat: int) -> Any:
    try:
        import gymnasium as gym
        import ale_py
        from gymnasium.wrappers import AtariPreprocessing
    except ModuleNotFoundError as error:  # pragma: no cover - server dependency
        raise RuntimeError("Gymnasium Atari dependencies are required for collection") from error
    gym.register_envs(ale_py)
    raw = gym.make(
        str(task["name"]),
        frameskip=1,
        repeat_action_probability=0,
        full_action_space=True,
        **dict(task.get("kwargs", {})),
    )
    return AtariPreprocessing(raw, frame_skip=env_repeat, screen_size=64, grayscale_obs=False)


def _episode_arrays(
    model: ModelBundle,
    task: Mapping[str, Any],
    *,
    reset_seed: int,
    policy_seed: int,
    max_decisions: int,
) -> tuple[dict[str, np.ndarray], bool]:
    """Collect one bounded episode segment and report whether it really ended.

    A collector cap must not fabricate a terminal transition.  A capped segment
    is valid for the natural, all-continue diagnostic set, but never for the
    terminal-event subset.  The caller records the distinction in the manifest.
    """
    torch = _torch()
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    environment = _make_atari_environment(task, int(model.config["env_repeat"]))
    try:
        observation, _ = environment.reset(seed=reset_seed)
        observation = _normalise_observation(observation)
        action_space = int(model.config["action_space"])
        reward_scale = float(task.get("rew_scale", 1.0))
        actions = [_one_hot(0, action_space)]
        observations = [observation]
        raw_rewards = [0.0]
        scaled_rewards = [0.0]
        continues = [1]
        resets = [1]
        terminateds = [0]
        truncateds = [0]

        z, h = model.world_model.rssm.initial_state(1)
        previous_action = torch.zeros(1, action_space, device=model.device)
        previous_action[:, 0] = 1
        reset_flag = torch.ones(1, 1, device=model.device)
        completed = False
        with torch.no_grad():
            for _ in range(max_decisions):
                image = (
                    torch.from_numpy(observation)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(model.device, dtype=torch.float32)
                    / 255.0
                )
                _, z, h = model.world_model.rssm(
                    z,
                    previous_action,
                    h,
                    image,
                    reset_flag,
                    stochastic=True,
                )
                action_log_probs = model.actor_critic.actor(model.vendor.zh_to_ac_state(z, h))
                action = int(torch.distributions.Categorical(logits=action_log_probs).sample().item())
                next_observation, raw_reward, terminated, truncated, _ = environment.step(action)
                observation = _normalise_observation(next_observation)
                finished = bool(terminated or truncated)

                actions.append(_one_hot(action, action_space))
                observations.append(observation)
                raw_rewards.append(float(raw_reward))
                scaled_rewards.append(float(raw_reward) * reward_scale)
                continues.append(0 if finished else 1)
                resets.append(0)
                terminateds.append(int(bool(terminated)))
                truncateds.append(int(bool(truncated)))
                previous_action = torch.from_numpy(actions[-1]).to(
                    model.device, dtype=torch.float32
                ).unsqueeze(0)
                reset_flag = torch.zeros(1, 1, device=model.device)
                if finished:
                    completed = True
                    break
    finally:
        environment.close()

    return (
        {
            "actions": np.asarray(actions, dtype=np.uint8),
            "observations": np.asarray(observations, dtype=np.uint8).transpose(0, 3, 1, 2),
            "raw_rewards": np.asarray(raw_rewards, dtype=np.float32)[:, None],
            "scaled_rewards": np.asarray(scaled_rewards, dtype=np.float32)[:, None],
            "continues": np.asarray(continues, dtype=np.uint8)[:, None],
            "resets": np.asarray(resets, dtype=np.uint8)[:, None],
            "terminated": np.asarray(terminateds, dtype=np.uint8)[:, None],
            "truncated": np.asarray(truncateds, dtype=np.uint8)[:, None],
        },
        completed,
    )


def _slice_chunk(
    episode: Mapping[str, np.ndarray], start: int, length: int, *, episode_id: int
) -> dict[str, np.ndarray | int]:
    stop = start + length
    if start < 0 or stop > len(episode["actions"]):
        raise ValueError("Chunk lies outside the source episode")
    chunk = {key: value[start:stop].copy() for key, value in episode.items()}
    # Each chunk begins a fresh RSSM inference; the fixed burn-in removes this
    # artificial reset from all reported metrics.
    chunk["resets"][0, 0] = 1
    chunk["episode_id"] = episode_id
    chunk["start_index"] = start
    return chunk


def _natural_candidates(
    episode: Mapping[str, np.ndarray],
    *,
    chunk_length: int,
    episode_id: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray | int]]:
    total = len(episode["actions"])
    if total < chunk_length:
        return []
    max_start = total - chunk_length
    offset = int(rng.integers(0, min(chunk_length, max_start + 1)))
    starts = list(range(offset, max_start + 1, chunk_length))
    if not starts:
        starts = [0]
    candidates = []
    for start in starts:
        stop = start + chunk_length
        if bool(np.all(episode["continues"][start:stop])):
            candidates.append(_slice_chunk(episode, start, chunk_length, episode_id=episode_id))
    return candidates


def _event_candidate(
    episode: Mapping[str, np.ndarray], *, chunk_length: int, episode_id: int
) -> dict[str, np.ndarray | int] | None:
    total = len(episode["actions"])
    if total < chunk_length or int(episode["continues"][-1, 0]) != 0:
        return None
    return _slice_chunk(episode, total - chunk_length, chunk_length, episode_id=episode_id)


def _stack_chunks(chunks: Sequence[Mapping[str, np.ndarray | int]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("Cannot write an empty diagnostic set")
    array_keys = (
        "actions",
        "observations",
        "raw_rewards",
        "scaled_rewards",
        "continues",
        "resets",
        "terminated",
        "truncated",
    )
    return {
        **{key: np.stack([chunk[key] for chunk in chunks]) for key in array_keys},
        "episode_ids": np.asarray([chunk["episode_id"] for chunk in chunks], dtype=np.int64),
        "start_indices": np.asarray([chunk["start_index"] for chunk in chunks], dtype=np.int32),
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return _write_sha256_sidecar(path)


def _task_output_name(task_index: int, role: str) -> str:
    return f"task_{task_index:02d}_{role}.npz"


def _collection_provenance(spec: SnapshotSpec, model: ModelBundle, task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "held_out_component_audit_dataset",
        "resumable": False,
        "snapshot": {
            "label": spec.label,
            "name": spec.path.name,
            "sha256": spec.sha256,
            "epoch": spec.epoch,
        },
        "task": {
            "index": spec.task_index,
            "name": task["name"],
            "reward_scale": task.get("rew_scale", 1.0),
        },
        "observation": {"dtype": "uint8", "shape": [3, 64, 64], "preprocessing": "AtariPreprocessing(screen_size=64, frame_skip=config.env_repeat, grayscale_obs=False)"},
        "action": {"representation": "one_hot", "dtype": "uint8", "action_space": int(model.config["action_space"])},
        "reward": {"raw": "environment reward before configured scaling", "scaled": "raw_reward * task.rew_scale"},
        "evaluation_isolation": "Collected after training and never inserted into replay or used for parameter updates.",
    }


def collect_diagnostic_sets(args: argparse.Namespace) -> None:
    """Collect a frozen natural/event diagnostic set for every task boundary."""
    if args.chunk_length <= args.burn_in + max(args.horizons):
        raise ValueError("chunk length must leave at least one post-burn-in open-loop target")
    if args.chunks < 1 or args.event_chunks < 0:
        raise ValueError("chunk counts must be non-negative, with natural chunks positive")
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit output {output_dir}")
    git = require_synced_training_git_state(ROOT)
    specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    boundaries = [spec for spec in specs if spec.reason == "task_boundary"]
    expected_tasks = len(boundaries)
    config = boundaries[0].payload["config"]
    tasks = config["esc"]["env_configs"]
    if len(tasks) != expected_tasks:
        raise ValueError("Snapshot task count does not match its config")

    _seed_everything(args.collection_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for task_index, boundary in enumerate(boundaries):
        if boundary.task_index != task_index:
            raise ValueError("Boundary/task order changed while collecting audit data")
        model = _model_bundle(boundary, args.device)
        task = tasks[task_index]
        selector = np.random.default_rng(args.chunk_selection_seed + task_index)
        natural_candidates: list[dict[str, np.ndarray | int]] = []
        event_candidates: list[dict[str, np.ndarray | int]] = []
        target_natural_candidates = max(args.chunks * 2, args.chunks)
        completed_segments = 0
        capped_segments = 0
        segments_started = 0
        for episode_id in range(args.max_episodes):
            episode, completed = _episode_arrays(
                model,
                task,
                reset_seed=args.environment_seed + task_index * 100_000 + episode_id,
                policy_seed=args.policy_seed + task_index * 100_000 + episode_id,
                max_decisions=args.max_episode_decisions,
            )
            segments_started += 1
            if completed:
                completed_segments += 1
            else:
                capped_segments += 1
            natural_candidates.extend(
                _natural_candidates(
                    episode,
                    chunk_length=args.chunk_length,
                    episode_id=episode_id,
                    rng=selector,
                )
            )
            event = (
                _event_candidate(episode, chunk_length=args.chunk_length, episode_id=episode_id)
                if completed
                else None
            )
            if event is not None:
                event_candidates.append(event)
            if len(natural_candidates) >= target_natural_candidates and (
                args.event_chunks == 0 or len(event_candidates) >= args.event_chunks
            ):
                break
        else:
            raise RuntimeError(
                f"Task {task_index} did not yield enough diagnostic chunks within "
                f"{args.max_episodes} collection segments (natural={len(natural_candidates)}, "
                f"event={len(event_candidates)}, capped={capped_segments})"
            )

        natural_indices = selector.choice(len(natural_candidates), size=args.chunks, replace=False)
        natural = [natural_candidates[int(index)] for index in natural_indices]
        natural_path = output_dir / _task_output_name(task_index, "natural")
        natural_digest = _write_npz_atomic(natural_path, _stack_chunks(natural))
        task_entry: dict[str, Any] = {
            **_collection_provenance(boundary, model, task),
            "natural": {
                "path": natural_path.name,
                "sha256": natural_digest,
                "chunks": args.chunks,
                "chunk_length": args.chunk_length,
                "burn_in": args.burn_in,
                "candidate_count": len(natural_candidates),
            },
            "collection_segments": {
                "started": segments_started,
                "completed_with_environment_terminal": completed_segments,
                "capped_nonterminal": capped_segments,
                "max_decisions": args.max_episode_decisions,
                "capped_segments_event_excluded": True,
            },
        }
        if args.event_chunks:
            event_indices = selector.choice(
                len(event_candidates), size=args.event_chunks, replace=False
            )
            event_chunks = [event_candidates[int(index)] for index in event_indices]
            event_path = output_dir / _task_output_name(task_index, "event")
            event_digest = _write_npz_atomic(event_path, _stack_chunks(event_chunks))
            task_entry["event"] = {
                "path": event_path.name,
                "sha256": event_digest,
                "chunks": args.event_chunks,
                "chunk_length": args.chunk_length,
                "candidate_count": len(event_candidates),
            }
        datasets.append(task_entry)
        print(
            f"[audit-collect] task={task_index} natural={args.chunks} "
            f"event={args.event_chunks} candidates={len(natural_candidates)} "
            f"terminal_segments={completed_segments} capped_segments={capped_segments}"
        )
        del model
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "component_forgetting_audit_collection",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "role": args.role,
        "project_git": git,
        "source_run": str(run_dir),
        "source_run_manifest_sha256": _sha256(run_dir / "launch.json"),
        "collection_seeds": {
            "global": args.collection_seed,
            "environment_base": args.environment_seed,
            "policy_base": args.policy_seed,
            "chunk_selection_base": args.chunk_selection_seed,
        },
        "protocol": {
            "chunk_length": args.chunk_length,
            "burn_in": args.burn_in,
            "natural_chunks_per_task": args.chunks,
            "event_chunks_per_task": args.event_chunks,
            "horizons": list(args.horizons),
            "model_collection_policy": "stochastic categorical posterior and actor sampling",
            "evaluation_transitions_enter_replay": False,
        },
        "datasets": datasets,
    }
    manifest_path = output_dir / "collection_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _write_sha256_sidecar(manifest_path)
    print(f"[audit-collect] complete output={output_dir}")


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Diagnostic data checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "actions",
        "observations",
        "raw_rewards",
        "scaled_rewards",
        "continues",
        "resets",
        "episode_ids",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Malformed audit data {path}; missing {missing}")
    if data["observations"].ndim != 5 or data["observations"].shape[2:] != (3, 64, 64):
        raise ValueError(f"Unexpected observation shape in {path}: {data['observations'].shape}")
    return data


def _to_time_batch(torch: Any, array: np.ndarray, device: Any, *, dtype: Any) -> Any:
    return torch.from_numpy(array).to(device=device, dtype=dtype).swapaxes(0, 1)


def _fixed_returns_torch(rewards: Any, continues: Any, discount: float) -> Any:
    torch = _torch()
    output = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[time_index] + discount * continues[time_index] * running
        output[time_index] = running
    return output


def _evaluate_one_checkpoint(
    model: ModelBundle,
    dataset: Mapping[str, np.ndarray],
    *,
    burn_in: int,
    horizons: Sequence[int],
    batch_size: int,
    event_anchor: bool,
    reward_scale: float,
) -> dict[str, Any]:
    """Evaluate absolute per-chunk metrics for one snapshot on one frozen dataset."""
    if not hasattr(model.world_model, "decoder"):
        raise ValueError(
            "The full component audit includes reconstruction and visual rollout metrics, "
            "so it is not defined for decoder-free R2 snapshots. Use the fixed-input "
            "encoder/RSSM/actor audits for ARROW-R2Rep-50."
        )
    torch = _torch()
    import torch.nn.functional as functional

    n_chunks, sequence_length = dataset["actions"].shape[:2]
    if sequence_length <= burn_in + max(horizons):
        raise ValueError("Diagnostic sequence is too short for burn-in plus max horizon")
    if reward_scale == 0:
        raise ValueError("Reward scale must be non-zero")
    metric_batches: dict[str, list[np.ndarray]] = {}
    feature_batches: list[np.ndarray] = []
    actor_batches: list[np.ndarray] = []
    for start in range(0, n_chunks, batch_size):
        stop = min(start + batch_size, n_chunks)
        actions = _to_time_batch(torch, dataset["actions"][start:stop], model.device, dtype=torch.float32)
        observations = _to_time_batch(
            torch, dataset["observations"][start:stop], model.device, dtype=torch.float32
        ) / 255.0
        rewards = _to_time_batch(
            torch, dataset["scaled_rewards"][start:stop], model.device, dtype=torch.float32
        )
        raw_rewards = _to_time_batch(
            torch, dataset["raw_rewards"][start:stop], model.device, dtype=torch.float32
        )
        continues = _to_time_batch(
            torch, dataset["continues"][start:stop], model.device, dtype=torch.float32
        )
        resets = _to_time_batch(
            torch, dataset["resets"][start:stop], model.device, dtype=torch.float32
        )
        chunk_count = stop - start
        with torch.no_grad():
            z0, h0 = model.world_model.rssm.initial_state(chunk_count)
            posterior_logits, posterior_z, hiddens = model.world_model.rssm(
                z0, actions, h0, observations, resets, stochastic=False
            )
            prior_logits = model.world_model.rssm.transition(hiddens)
            zhs = model.world_model.zh_transform(posterior_z, hiddens)
            flattened = zhs.reshape(-1, zhs.shape[-1])
            reconstruction = model.world_model.decoder(flattened).view_as(observations)
            reward_prediction_symlog = model.world_model.reward_fc(zhs)
            continue_prediction = model.world_model.continue_fc(zhs)
            actor_log_probs = model.actor_critic.actor(
                model.vendor.zh_to_ac_state(posterior_z, hiddens)
            )
            _, critic_prediction = model.actor_critic(
                model.vendor.zh_to_ac_state(posterior_z, hiddens)
            )
            fixed_returns = _fixed_returns_torch(rewards, continues, discount=0.997)
            metric_tensors = {
                "teacher_forced.reconstruction_mse": (reconstruction - observations)
                .square()
                .mean(dim=(2, 3, 4)),
                "teacher_forced.posterior_prior_kl": model.vendor.categorical_kl(
                    posterior_logits, prior_logits
                ).sum(dim=-1),
                "teacher_forced.reward_symlog_mse": (
                    reward_prediction_symlog - model.vendor.symlog(rewards)
                )
                .square()
                .squeeze(-1),
                "teacher_forced.reward_scaled_mae": (
                    model.vendor.symexp(reward_prediction_symlog) - rewards
                )
                .abs()
                .squeeze(-1),
                "teacher_forced.reward_raw_mae": (
                    model.vendor.symexp(reward_prediction_symlog) / reward_scale - raw_rewards
                )
                .abs()
                .squeeze(-1),
                "teacher_forced.continue_bce": functional.binary_cross_entropy(
                    continue_prediction, continues, reduction="none"
                ).squeeze(-1),
                "teacher_forced.continue_brier": (continue_prediction - continues)
                .square()
                .squeeze(-1),
                "critic.anchored_return_mae": (critic_prediction - fixed_returns)
                .abs()
                .squeeze(-1),
            }
            for name, values in metric_tensors.items():
                metric_batches.setdefault(name, []).append(
                    values[burn_in:].mean(dim=0).detach().cpu().numpy()
                )

            anchor = sequence_length - max(horizons) - 1 if event_anchor else burn_in - 1
            open_z = posterior_z[anchor]
            open_h = hiddens[anchor]
            no_reset = torch.zeros(chunk_count, 1, device=model.device)
            for horizon in range(1, max(horizons) + 1):
                _, open_z, open_h = model.world_model.rssm(
                    open_z,
                    actions[anchor + horizon],
                    open_h,
                    None,
                    no_reset,
                    stochastic=False,
                )
                if horizon not in horizons:
                    continue
                open_zh = model.world_model.zh_transform(open_z, open_h)
                predicted_observation = model.world_model.decoder(open_zh)
                predicted_reward_symlog = model.world_model.reward_fc(open_zh)
                predicted_continue = model.world_model.continue_fc(open_zh)
                target_observation = observations[anchor + horizon]
                target_reward = rewards[anchor + horizon]
                target_raw_reward = raw_rewards[anchor + horizon]
                target_continue = continues[anchor + horizon]
                prefix = f"open_loop.h{horizon}"
                open_metrics = {
                    f"{prefix}.visual_mse": (predicted_observation - target_observation)
                    .square()
                    .mean(dim=(1, 2, 3)),
                    f"{prefix}.reward_symlog_mse": (
                        predicted_reward_symlog - model.vendor.symlog(target_reward)
                    )
                    .square()
                    .squeeze(-1),
                    f"{prefix}.reward_scaled_mae": (
                        model.vendor.symexp(predicted_reward_symlog) - target_reward
                    )
                    .abs()
                    .squeeze(-1),
                    f"{prefix}.reward_raw_mae": (
                        model.vendor.symexp(predicted_reward_symlog) / reward_scale
                        - target_raw_reward
                    )
                    .abs()
                    .squeeze(-1),
                    f"{prefix}.continue_bce": functional.binary_cross_entropy(
                        predicted_continue, target_continue, reduction="none"
                    ).squeeze(-1),
                    f"{prefix}.continue_brier": (predicted_continue - target_continue)
                    .square()
                    .squeeze(-1),
                }
                for name, values in open_metrics.items():
                    metric_batches.setdefault(name, []).append(values.detach().cpu().numpy())

            feature_batches.append(
                zhs[burn_in:].permute(1, 0, 2).reshape(chunk_count, -1, zhs.shape[-1])
                .detach()
                .cpu()
                .numpy()
            )
            actor_batches.append(
                actor_log_probs[burn_in:].permute(1, 0, 2).detach().cpu().numpy()
            )

    per_chunk = {name: np.concatenate(values) for name, values in metric_batches.items()}
    return {
        "per_chunk": per_chunk,
        "features": np.concatenate(feature_batches, axis=0),
        "actor_log_probs": np.concatenate(actor_batches, axis=0),
    }


def _metric_direction(name: str) -> str:
    if name in {"actor.top1_agreement", "representation.linear_cka"}:
        return "higher_is_better"
    return "lower_is_better"


def _summary_record(
    *,
    task_index: int,
    task_name: str,
    role: str,
    checkpoint: SnapshotSpec,
    metric: str,
    values: np.ndarray,
    episode_ids: np.ndarray,
    baseline_mean: float | None,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary = mean_and_episode_bootstrap_ci(
        values,
        episode_ids,
        seed=bootstrap_seed,
        repetitions=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    direction = _metric_direction(metric)
    forgetting = None
    if baseline_mean is not None:
        forgetting = (
            summary["mean"] - baseline_mean
            if direction == "lower_is_better"
            else baseline_mean - summary["mean"]
        )
    return {
        "task_index": task_index,
        "task_name": task_name,
        "dataset_role": role,
        "checkpoint": checkpoint.label,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_sha256": checkpoint.sha256,
        "metric": metric,
        "direction": direction,
        "boundary_relative_forgetting": forgetting,
        **summary,
    }


def _global_summary_record(
    *,
    task_index: int,
    task_name: str,
    checkpoint: SnapshotSpec,
    metric: str,
    value: float,
    baseline_value: float,
) -> dict[str, Any]:
    direction = _metric_direction(metric)
    forgetting = value - baseline_value if direction == "lower_is_better" else baseline_value - value
    return {
        "task_index": task_index,
        "task_name": task_name,
        "dataset_role": "natural",
        "checkpoint": checkpoint.label,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_sha256": checkpoint.sha256,
        "metric": metric,
        "direction": direction,
        "boundary_relative_forgetting": float(forgetting),
        "mean": float(value),
        "n_chunks": None,
        "n_episodes": None,
        "ci_low": None,
        "ci_high": None,
    }


def _append_actor_comparisons(
    output: dict[str, Any], baseline: Mapping[str, Any]
) -> None:
    base_log_probs = baseline["actor_log_probs"]
    current_log_probs = output["actor_log_probs"]
    symmetric_kl = symmetric_kl_from_log_probs(base_log_probs, current_log_probs).mean(axis=1)
    agreement = (
        base_log_probs.argmax(axis=-1) == current_log_probs.argmax(axis=-1)
    ).mean(axis=1)
    output["per_chunk"]["actor.symmetric_kl"] = symmetric_kl
    output["per_chunk"]["actor.top1_agreement"] = agreement
    output["representation.linear_cka"] = linear_cka(
        baseline["features"].reshape(-1, baseline["features"].shape[-1]),
        output["features"].reshape(-1, output["features"].shape[-1]),
    )


def _return_matrix(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract raw end-to-end returns at the post-boundary evaluations from the log."""
    log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    evaluations = []
    for match in EVAL_RE.finditer(log_text):
        evaluations.append(
            {
                "epoch": int(match.group("epoch")),
                "means": ast.literal_eval(match.group("means")),
                "stds": ast.literal_eval(match.group("stds")),
            }
        )
    if not evaluations:
        raise ValueError("No complete evaluation records found in train.log")
    by_epoch = {row["epoch"]: row for row in evaluations}
    tasks = config["esc"]["env_configs"]
    swap = int(config["esc"]["kwargs"]["swap_sched"])
    boundary_epochs = [(index + 1) * swap for index in range(len(tasks))]
    missing = [epoch for epoch in boundary_epochs if epoch not in by_epoch]
    if missing:
        raise ValueError(f"Missing post-boundary evaluations at epochs {missing}")
    rows = []
    for task_index, task in enumerate(tasks):
        scale = float(task.get("rew_scale", 1.0))
        history = [
            {"epoch": epoch, "raw_return": float(by_epoch[epoch]["means"][task_index]) / scale}
            for epoch in boundary_epochs[task_index:]
        ]
        acquisition = history[0]["raw_return"]
        final = history[-1]["raw_return"]
        best = max(point["raw_return"] for point in history)
        rows.append(
            {
                "task_index": task_index,
                "task_name": task["name"],
                "acquisition_evaluation_epoch": boundary_epochs[task_index],
                "acquisition_raw_return": acquisition,
                "best_historical_raw_return": best,
                "c6_raw_return": final,
                "boundary_forgetting": acquisition - final,
                "max_forgetting": best - final,
                "history": history,
            }
        )
    return {
        "evaluation_alignment": (
            "Epoch k*swap is evaluated before its gradient updates; it therefore evaluates "
            "the preceding task-boundary weights. Epoch 540 evaluates C6, not Cfinal."
        ),
        "rows": rows,
    }


def _write_metrics_npz(path: Path, records: Sequence[Mapping[str, Any]], values: Sequence[np.ndarray]) -> str:
    if len(records) != len(values):
        raise ValueError("Metric metadata and values must have identical lengths")
    offsets = [0]
    flattened_values = []
    for value in values:
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        flattened_values.append(row)
        offsets.append(offsets[-1] + len(row))
    numeric = np.concatenate(flattened_values) if flattened_values else np.asarray([], dtype=np.float32)
    metadata = np.asarray([json.dumps(record, sort_keys=True) for record in records])
    return _write_npz_atomic(
        path,
        {
            "records": metadata,
            "offsets": np.asarray(offsets, dtype=np.int64),
            "values": numeric,
        },
    )


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# DreamerV3/FIFO Component Forgetting Audit (Pilot P1)",
        "",
        "This report is a single-seed pilot. It is diagnostic evidence, not an official baseline or causal intervention result.",
        "",
        "## End-to-End Returns",
        "",
        "| Task | Acquisition raw return | Best historical raw return | C6 raw return | Boundary forgetting | Max forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["end_to_end_returns"]["rows"]:
        lines.append(
            "| {task} | {acq:.4g} | {best:.4g} | {final:.4g} | {boundary:.4g} | {maximum:.4g} |".format(
                task=row["task_name"],
                acq=row["acquisition_raw_return"],
                best=row["best_historical_raw_return"],
                final=row["c6_raw_return"],
                boundary=row["boundary_forgetting"],
                maximum=row["max_forgetting"],
            )
        )
    lines.extend(
        [
            "",
            "## Final Component Metrics",
            "",
            "Positive boundary-relative forgetting means degradation by the metric's declared direction. CKA has no bootstrap interval because it is a global paired-feature statistic.",
            "",
            "| Task | Checkpoint | Metric | Absolute value | Boundary-relative forgetting | 95% episode-cluster CI |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    latest_by_task_metric: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in payload["summary_records"]:
        if row["dataset_role"] != "natural":
            continue
        key = (int(row["task_index"]), str(row["metric"]))
        existing = latest_by_task_metric.get(key)
        if existing is None or int(row["checkpoint_epoch"]) > int(existing["checkpoint_epoch"]):
            latest_by_task_metric[key] = row
    for _, row in sorted(latest_by_task_metric.items()):
        ci = (
            "n/a"
            if row["ci_low"] is None
            else f"[{row['ci_low']:.4g}, {row['ci_high']:.4g}]"
        )
        forgetting = row["boundary_relative_forgetting"]
        rendered_forgetting = "n/a" if forgetting is None else f"{forgetting:.4g}"
        lines.append(
            f"| {row['task_name']} | {row['checkpoint']} | {row['metric']} | "
            f"{row['mean']:.4g} | {rendered_forgetting} | {ci} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Teacher-forced stability with open-loop deterioration is evidence for a long-horizon dynamics candidate, not a causal proof.",
            "- A large actor divergence includes latent-to-actor interface drift; it does not by itself isolate actor parameters.",
            "- These snapshots are world-model/actor analysis weights only. They contain no optimizer, replay, RNG, or schedule state and are not resumable checkpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_diagnostic_sets(args: argparse.Namespace) -> None:
    """Run deterministic component metrics on every valid checkpoint/data pair."""
    run_dir = args.run_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation output {output_dir}")
    manifest = json.loads((audit_dir / "collection_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Diagnostic collection manifest is not complete")
    specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    boundaries = [spec for spec in specs if spec.reason == "task_boundary"]
    final = next(spec for spec in specs if spec.reason == "final")
    config = boundaries[0].payload["config"]
    task_entries = manifest.get("datasets", [])
    if len(task_entries) != len(boundaries):
        raise ValueError("Collection manifest does not match the boundary snapshot count")
    if args.burn_in != int(manifest["protocol"]["burn_in"]):
        raise ValueError("Evaluation burn-in must match the frozen diagnostic dataset")
    if tuple(args.horizons) != tuple(manifest["protocol"]["horizons"]):
        raise ValueError("Evaluation horizons must match the frozen diagnostic dataset")

    _seed_everything(args.evaluation_seed)
    output_dir.mkdir(parents=True)
    summary_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    raw_values: list[np.ndarray] = []
    determinism_checked = False
    for task_index, boundary in enumerate(boundaries):
        task_entry = task_entries[task_index]
        if int(task_entry["task"]["index"]) != task_index:
            raise ValueError("Collection task order does not match the run snapshots")
        task_name = str(task_entry["task"]["name"])
        reward_scale = float(task_entry["task"]["reward_scale"])
        natural_info = task_entry["natural"]
        natural = _load_dataset(audit_dir / natural_info["path"], natural_info["sha256"])
        event_info = task_entry.get("event")
        event = (
            _load_dataset(audit_dir / event_info["path"], event_info["sha256"])
            if event_info is not None
            else None
        )
        eligible = [*boundaries[task_index:], final]
        baseline_output: dict[str, Any] | None = None
        baseline_means: dict[str, float] = {}
        event_baseline_means: dict[str, float] = {}
        for checkpoint in eligible:
            model = _model_bundle(checkpoint, args.device)
            natural_output = _evaluate_one_checkpoint(
                model,
                natural,
                burn_in=args.burn_in,
                horizons=args.horizons,
                batch_size=args.batch_size,
                event_anchor=False,
                reward_scale=reward_scale,
            )
            if not determinism_checked:
                repeat_output = _evaluate_one_checkpoint(
                    model,
                    natural,
                    burn_in=args.burn_in,
                    horizons=args.horizons,
                    batch_size=args.batch_size,
                    event_anchor=False,
                    reward_scale=reward_scale,
                )
                for metric, values in natural_output["per_chunk"].items():
                    if not np.array_equal(values, repeat_output["per_chunk"][metric]):
                        raise RuntimeError(f"Deterministic repeat failed for {metric}")
                determinism_checked = True
            if baseline_output is None:
                baseline_output = natural_output
                _append_actor_comparisons(natural_output, natural_output)
                baseline_means = {
                    metric: float(values.mean())
                    for metric, values in natural_output["per_chunk"].items()
                }
                baseline_means["representation.linear_cka"] = float(
                    natural_output["representation.linear_cka"]
                )
            else:
                _append_actor_comparisons(natural_output, baseline_output)

            for metric, values in natural_output["per_chunk"].items():
                record = _summary_record(
                    task_index=task_index,
                    task_name=task_name,
                    role="natural",
                    checkpoint=checkpoint,
                    metric=metric,
                    values=values,
                    episode_ids=natural["episode_ids"],
                    baseline_mean=baseline_means.get(metric),
                    bootstrap_seed=args.bootstrap_seed + task_index * 10_000 + checkpoint.epoch,
                )
                summary_records.append(record)
                raw_records.append(
                    {
                        "task_index": task_index,
                        "dataset_role": "natural",
                        "checkpoint": checkpoint.label,
                        "checkpoint_epoch": checkpoint.epoch,
                        "metric": metric,
                    }
                )
                raw_values.append(values)
            cka_value = float(natural_output["representation.linear_cka"])
            summary_records.append(
                _global_summary_record(
                    task_index=task_index,
                    task_name=task_name,
                    checkpoint=checkpoint,
                    metric="representation.linear_cka",
                    value=cka_value,
                    baseline_value=baseline_means["representation.linear_cka"],
                )
            )

            if event is not None:
                event_output = _evaluate_one_checkpoint(
                    model,
                    event,
                    burn_in=args.burn_in,
                    horizons=args.horizons,
                    batch_size=args.batch_size,
                    event_anchor=True,
                    reward_scale=reward_scale,
                )
                for metric in (
                    "teacher_forced.continue_bce",
                    "teacher_forced.continue_brier",
                    *[
                        item
                        for horizon in args.horizons
                        for item in (
                            f"open_loop.h{horizon}.continue_bce",
                            f"open_loop.h{horizon}.continue_brier",
                        )
                    ],
                ):
                    values = event_output["per_chunk"][metric]
                    key = f"event:{metric}"
                    baseline_event_mean = None
                    if checkpoint.path == boundary.path:
                        baseline_event_mean = float(values.mean())
                    else:
                        baseline_event_mean = event_baseline_means[key]
                    summary_records.append(
                        _summary_record(
                            task_index=task_index,
                            task_name=task_name,
                            role="event",
                            checkpoint=checkpoint,
                            metric=metric,
                            values=values,
                            episode_ids=event["episode_ids"],
                            baseline_mean=baseline_event_mean,
                            bootstrap_seed=args.bootstrap_seed
                            + 100_000
                            + task_index * 10_000
                            + checkpoint.epoch,
                        )
                    )
                    raw_records.append(
                        {
                            "task_index": task_index,
                            "dataset_role": "event",
                            "checkpoint": checkpoint.label,
                            "checkpoint_epoch": checkpoint.epoch,
                            "metric": metric,
                        }
                    )
                    raw_values.append(values)
                    if checkpoint.path == boundary.path:
                        event_baseline_means[key] = float(values.mean())

            del model
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[audit-evaluate] task={task_index} checkpoint={checkpoint.label}")

    metrics_path = output_dir / "per_chunk_metrics.npz"
    metrics_digest = _write_metrics_npz(metrics_path, raw_records, raw_values)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "component_forgetting_audit_results",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "project_git": git_state(ROOT),
        "source_run": str(run_dir),
        "source_audit": str(audit_dir),
        "evaluation": {
            "device": args.device,
            "seed": args.evaluation_seed,
            "deterministic_posterior_and_prior": True,
            "horizons": list(args.horizons),
            "burn_in": args.burn_in,
            "batch_size": args.batch_size,
            "environment_interaction": False,
            "gradient_updates": False,
        },
        "snapshot_sequence": [
            {
                "label": spec.label,
                "name": spec.path.name,
                "epoch": spec.epoch,
                "sha256": spec.sha256,
            }
            for spec in specs
        ],
        "end_to_end_returns": _return_matrix(run_dir, config),
        "summary_records": summary_records,
        "raw_metrics": {"path": metrics_path.name, "sha256": metrics_digest},
    }
    result_path = output_dir / "results.json"
    _write_json_atomic(result_path, payload)
    _write_sha256_sidecar(result_path)
    report_path = output_dir / "REPORT.md"
    _write_text_atomic(report_path, _markdown_report(payload))
    _write_sha256_sidecar(report_path)
    print(f"[audit-evaluate] complete output={output_dir}")


def _parse_horizons(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in value.split(",") if part)
    if not parts or any(part < 1 for part in parts) or tuple(sorted(set(parts))) != parts:
        raise argparse.ArgumentTypeError("horizons must be ascending, unique positive integers")
    return parts


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect frozen held-out Atari diagnostic chunks")
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--label", default="dv3_fifo_pilot_p1")
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--chunks", type=_positive_int, default=DEFAULT_DIAGNOSTIC_CHUNKS)
    collect.add_argument("--event-chunks", type=int, default=DEFAULT_EVENT_CHUNKS)
    collect.add_argument("--chunk-length", type=_positive_int, default=DEFAULT_CHUNK_LENGTH)
    collect.add_argument("--burn-in", type=_positive_int, default=DEFAULT_BURN_IN)
    collect.add_argument("--horizons", type=_parse_horizons, default=DEFAULT_HORIZONS)
    collect.add_argument("--collection-seed", type=int, default=20260815)
    collect.add_argument("--environment-seed", type=int, default=410_000)
    collect.add_argument("--policy-seed", type=int, default=510_000)
    collect.add_argument("--chunk-selection-seed", type=int, default=610_000)
    collect.add_argument("--max-episodes", type=_positive_int, default=2_000)
    collect.add_argument("--max-episode-decisions", type=_positive_int, default=20_000)
    collect.add_argument("--role", choices=("smoke", "pilot"), default="pilot")
    collect.set_defaults(handler=collect_diagnostic_sets)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate snapshots on frozen diagnostic chunks")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--audit-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=_positive_int, default=32)
    evaluate.add_argument("--burn-in", type=_positive_int, default=DEFAULT_BURN_IN)
    evaluate.add_argument("--horizons", type=_parse_horizons, default=DEFAULT_HORIZONS)
    evaluate.add_argument("--evaluation-seed", type=int, default=710_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=810_000)
    evaluate.add_argument("--role", choices=("smoke", "pilot"), default="pilot")
    evaluate.set_defaults(handler=evaluate_diagnostic_sets)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
