#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen-checkpoint CoinRun routing diagnostic; never trains or writes Replay.

Launcher-only vendor bridge. All routers score the same saved episode prefixes.
Labels enter environment construction and reporting, never route selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from artifact_io import sha256_file, write_json_atomic, write_sha256_sidecar
from git_provenance import require_synced_training_git_state

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "CoinRun-Frozen-Sequence-Routing-Diagnostic-v1"


@dataclass(frozen=True)
class ProbeConfig:
    episodes_per_task: int = 16
    prefix_decisions: int = 32
    batch_size: int = 16
    cpu_threads: int = 2
    seed: int = 20260907
    device: str = "cuda:0"
    windows: tuple[int, ...] = (1, 4, 8, 16, 32)
    cohorts: tuple[str, ...] = ("random", "first_frame_policy")

    def __post_init__(self):
        if min(self.episodes_per_task, self.prefix_decisions, self.batch_size, self.cpu_threads) < 1:
            raise ValueError("Episode, decision, batch and thread budgets must be positive")
        if not 0 <= self.seed < 2**31:
            raise ValueError("Probe seed must be in [0, 2**31)")


def rooted(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def verify_launch(receipt: dict, state: dict) -> None:
    """Controller fetch + checksum-verified Git bundle, as in this campaign."""
    for key in ("commit", "upstream", "ahead", "behind"):
        if receipt.get(key) != state[key]:
            raise ValueError(f"Fetched upstream receipt differs from launch state: {key}")
    if receipt.get("origin") != subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Receipt does not match configured GitHub origin")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(receipt["fetched_at_utc"])).total_seconds()
    if not 0 <= age <= 3600 or not receipt.get("bundle_sha256"):
        raise ValueError("Require a fresh (<= 1 h) controller Git fetch and bundle digest")


def load_model(checkpoint: Path, device: torch.device):
    # This is a reference launcher, not a clworldmodel library import convention.
    for path in (ROOT / "src", ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"):
        sys.path.insert(0, str(path))
    from config import Config
    from smoke_evolving_atomic_rssm import _world_model
    from ac import build_actor_critic_opt
    from train import _actor_critic_constructor_kwargs
    from clworldmodel.routing import RoutedActorBank

    digest = sha256_file(checkpoint)
    if checkpoint.with_suffix(checkpoint.suffix + ".sha256").read_text().split()[0] != digest:
        raise ValueError("Checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("artifact_kind") != "task_bank_boundary_inference_snapshot":
        raise ValueError("Expected a task-boundary inference snapshot")
    config = Config.from_dict(payload["config"])
    if not (config.uses_reconstruction_task_inference and config.task_private_actor_critic
            and config.benchmark == "procgen_coinrun"):
        raise ValueError("Expected the private-MLP D-AutoRoute CoinRun checkpoint")
    routes = tuple(payload["inference_routing"]["eligible_route_ids"])
    if routes != tuple(range(payload["completed_task"]["task_index"] + 1)) or len(routes) < 2:
        raise ValueError("Need at least two completed, contiguous eligible task routes")
    states = payload["actor_critic_bank_state_dict"]["tasks"]
    if set(states) != {str(k) for k in routes}:
        raise ValueError("Actor bank and route registry differ")
    wm = _world_model(config, device)
    wm.load_state_dict(payload["world_model_state_dict"], strict=True)
    actors = {}
    for k in routes:
        # Reuse the production constructor's config mapping. Its optimizer is
        # discarded without any update or state allocation.
        ac = build_actor_critic_opt(wm, lr=config.ac_lr,
                                    **_actor_critic_constructor_kwargs(config)).ac
        ac.load_state_dict(states[str(k)], strict=True)
        ac.requires_grad_(False).eval()
        actors[k] = ac.actor
    bank = RoutedActorBank(actors).requires_grad_(False).eval()
    wm.requires_grad_(False).eval()
    return wm, bank, config, routes, payload, digest


def weight_digest(*modules) -> str:
    digest = hashlib.sha256()
    for index, module in enumerate(modules):
        for name, value in module.state_dict().items():
            digest.update(f"{index}:{name}".encode())
            if torch.is_tensor(value):
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def seed_for(seed: int, domain: int, episode: int) -> int:
    return int(np.random.SeedSequence([seed, domain, episode]).generate_state(1)[0] % (2**31))


def append_transition(observations, actions, rewards, returned, action, reward, done):
    """Native terminal observation is an autoreset frame, not a target."""
    rewards.append(float(reward))
    if done:
        return False
    actions.append(int(action))
    observations.append(returned.copy())
    return True


@torch.inference_mode()
def collect(wm, actors, routes, cfg: ProbeConfig, task_names: tuple[str, ...], output: Path):
    from clworldmodel.environments.coinrun import CoinRunFactory
    from clworldmodel.routing import EpisodeReconstructionRouter
    from generate_trajectory import _routed_policy_step

    device = next(wm.parameters()).device
    rows, metadata_rows = [], []
    for cohort in cfg.cohorts:
        for task, name in enumerate(task_names):
            factory = CoinRunFactory(name)
            for episode in range(cfg.episodes_per_task):
                env_seed = seed_for(cfg.seed, 1001, episode)
                action_seed = seed_for(cfg.seed, 2001, episode)
                env = factory.prepare(1, action_seed)
                rng = np.random.default_rng(action_seed)
                router = EpisodeReconstructionRouter(routes)
                z, h = wm.rssm.initial_state(1)
                previous = torch.nn.functional.one_hot(
                    torch.tensor([factory.dummy_previous_action], device=device), wm.a_dim
                ).float()
                try:
                    obs, _ = env.reset(seed=env_seed)
                    observations, actions, rewards = [obs.copy()], [], []
                    done = False
                    for decision in range(cfg.prefix_decisions):
                        if cohort == "random":
                            action = int(rng.integers(wm.a_dim))
                        else:
                            x = torch.from_numpy(obs).permute(2, 0, 1)[None].to(device).float() / 255
                            z, h, a = _routed_policy_step(
                                wm, actors, router, x, z, h, previous,
                                torch.full((1, 1), float(decision == 0), device=device),
                                stochastic=False, dummy_previous_action=factory.dummy_previous_action,
                            )
                            action = int(a.item())
                            previous = torch.nn.functional.one_hot(a, wm.a_dim).float()
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = bool(terminated or truncated)
                        if not append_transition(observations, actions, rewards, obs, action, reward, done):
                            break
                    rows.append((observations, actions))
                    metadata_rows.append({
                        "task_index_for_audit_only": task, "task_name": name,
                        "cohort": cohort, "episode_index": episode, "environment_seed": env_seed,
                        "action_seed": action_seed, "collected_agent_decisions": len(rewards),
                        "valid_nonterminal_transitions": len(actions), "ended": done,
                        "raw_rewards": rewards, "partial_reward_sum_not_episode_return": sum(rewards),
                        "initial_policy_route": int(router.routes[0]) if router.routes is not None else None,
                    })
                finally:
                    env.close()
            print(json.dumps({"phase": "collection", "cohort": cohort, "task": name,
                              "prefixes": cfg.episodes_per_task}), flush=True)
    shape = rows[0][0][0].shape
    xs = np.zeros((len(rows), cfg.prefix_decisions + 1, *shape), dtype=np.uint8)
    acts = np.full((len(rows), cfg.prefix_decisions), 4, dtype=np.int64)
    valid = np.zeros((len(rows), cfg.prefix_decisions + 1), dtype=np.bool_)
    for i, (observations, actions) in enumerate(rows):
        xs[i, :len(observations)] = np.stack(observations)
        acts[i, :len(actions)] = actions
        valid[i, :len(observations)] = True
    shuffled = shuffle_actions(acts, valid, cfg.seed)
    path = output / "trajectories.npz"
    np.savez_compressed(path, observations=xs, actions=acts, valid_observations=valid, shuffled_actions=shuffled)
    write_sha256_sidecar(path)
    write_json_atomic(output / "episodes.json", {"episodes": metadata_rows})
    return xs, acts, shuffled, valid, metadata_rows


def shuffle_actions(actions: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    shuffled = actions.copy()
    for i in range(len(actions)):
        n = int(valid[i].sum()) - 1
        rng = np.random.default_rng(seed_for(seed, 3001, i))
        shuffled[i, :n] = actions[i, rng.permutation(n)]
    return shuffled


@torch.inference_mode()
def score_history(wm, observations: torch.Tensor, actions: torch.Tensor, routes: tuple[int, ...],
                  *, dummy_previous_action: int = 4):
    """Inputs [B,T,C,H,W] and [B,T-1,A]; outputs [B,T,K]. No task labels.

    Posterior-mode filtering is candidate-private. Prior-mode predictions are
    formed before supplying the current image; t=0 has no dynamics score.
    """
    from ac import zh_to_ac_state
    from rssm import straight_through_one_hot
    from wm import categorical_kl
    if (observations.ndim != 5 or actions.ndim != 3
            or actions.shape[:2] != (len(observations), observations.shape[1] - 1)):
        raise ValueError("Expected observations [B,T,C,H,W] and actions [B,T-1,A]")
    if not routes or tuple(sorted(set(routes))) != routes:
        raise ValueError("Routes must be nonempty, sorted and unique")
    batch, times = observations.shape[:2]
    scores = {key: observations.new_zeros((batch, times, len(routes)), dtype=torch.float32)
              for key in ("reconstruction", "prediction", "posterior_prior_kl")}
    reset = observations.new_ones((batch, 1))
    dummy = torch.nn.functional.one_hot(
        torch.full((batch,), dummy_previous_action, device=observations.device), actions.shape[-1]
    ).float()
    for column, route in enumerate(routes):
        z, h = wm.rssm.initial_state(batch)
        for t in range(times):
            if t == 0:
                q, z, h = wm.rssm(z, dummy, h, observations[:, t], reset,
                                  task_id=route, stochastic=False)
            else:
                # Candidate state comes only from this candidate's earlier posteriors.
                p, prior_z, h = wm.rssm(z, actions[:, t - 1], h, None, reset * 0,
                                       task_id=route, stochastic=False)
                predicted = wm.decoder_for(route)(zh_to_ac_state(prior_z, h))
                scores["prediction"][:, t, column] = (
                    predicted.float() - observations[:, t].float()
                ).square().flatten(1).mean(1)
                e = wm.rssm.adapt_observation_embeddings(
                    wm.rssm.image_embedder_for(route)(observations[:, t]), task_id=route
                )
                q = wm.rssm.posterior_step(e, h, task_id=route)
                q, z = straight_through_one_hot(q, stochastic=False)
                scores["posterior_prior_kl"][:, t, column] = categorical_kl(q, p).sum(-1)
            reconstruction = wm.decoder_for(route)(zh_to_ac_state(z, h))
            scores["reconstruction"][:, t, column] = (
                reconstruction.float() - observations[:, t].float()
            ).square().flatten(1).mean(1)
    if any(not bool(torch.isfinite(s).all()) for s in scores.values()):
        raise FloatingPointError("Nonfinite router score")
    return scores


def score_dataset(wm, xs, actions, routes, cfg):
    from generate_trajectory import _autocast_context
    device = next(wm.parameters()).device
    parts = {key: [] for key in ("reconstruction", "prediction", "posterior_prior_kl")}
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(xs), cfg.batch_size):
        x = torch.from_numpy(xs[start:start + cfg.batch_size]).permute(0, 1, 4, 2, 3).to(device).float() / 255
        a = torch.nn.functional.one_hot(torch.from_numpy(actions[start:start + cfg.batch_size]).to(device), wm.a_dim).float()
        with _autocast_context(device, wm.compute_dtype):
            result = score_history(wm, x, a, routes)
        for key, values in result.items(): parts[key].append(values.cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize(device)
    return {k: np.concatenate(v) for k, v in parts.items()}, time.perf_counter() - started


def summarize(scores, shuffled, valid, episodes, routes, cfg):
    results = []
    labels = np.array([e["task_index_for_audit_only"] for e in episodes])
    first = np.asarray(routes)[scores["reconstruction"][:, 0].argmin(-1)]
    for cohort in cfg.cohorts:
        cohort_mask = np.array([e["cohort"] == cohort for e in episodes])
        for window in cfg.windows:
            if window > cfg.prefix_decisions: continue
            selected = cohort_mask & valid[:, window]
            candidates = {
                "A_first_frame": first,
                "B_multiframe_reconstruction": np.asarray(routes)[scores["reconstruction"][:, :window + 1].mean(1).argmin(-1)],
                "C_action_conditioned_prediction": np.asarray(routes)[scores["prediction"][:, 1:window + 1].mean(1).argmin(-1)],
                "C_shuffled_actions": np.asarray(routes)[shuffled["prediction"][:, 1:window + 1].mean(1).argmin(-1)],
            }
            for name, predicted in candidates.items():
                matrix = np.zeros((len(routes), len(routes)), np.int64)
                np.add.at(matrix, (labels[selected], predicted[selected]), 1)
                denominator = int(matrix.sum())
                results.append({"cohort": cohort, "window_transitions": window, "router": name,
                    "sample_count": denominator, "short_prefixes_excluded": int(cohort_mask.sum() - denominator),
                    "accuracy": float(np.trace(matrix) / denominator) if denominator else None,
                    "per_task_accuracy": [float(matrix[i, i] / row.sum()) if row.sum() else None for i, row in enumerate(matrix)],
                    "confusion_matrix": matrix.tolist(),
                    "wrong_first_frame_corrected": int(((first != labels) & (predicted == labels) & selected).sum()),
                    "correct_first_frame_broken": int(((first == labels) & (predicted != labels) & selected).sum())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-verification", type=Path, required=True)
    for key in ("episodes_per_task", "prefix_decisions", "batch_size", "cpu_threads", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    checkpoint, output, verification = (rooted(args.pop(k)) for k in ("checkpoint", "output_dir", "upstream_verification"))
    cfg = ProbeConfig(**args)
    git = require_synced_training_git_state(ROOT)
    receipt = json.loads(verification.read_text())
    verify_launch(receipt, git)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)  # Initialize allocator before resetting its counters.
            torch.cuda.reset_peak_memory_stats(device)
        wm, actors, model_config, routes, payload, digest = load_model(checkpoint, device)
        from clworldmodel.environments.coinrun import COINRUN_TASKS, CoinRunFactory, PROCGEN_COMMIT
        task_names = COINRUN_TASKS[:len(routes)]
        before = weight_digest(wm, actors)
        manifest = {
            "protocol": PROTOCOL, "classification": "debug", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_git": git, "upstream_verification": receipt,
            "arrow_base_commit": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
            "vendored_manifest_sha256": sha256_file(ROOT / "third_party/arrow/MANIFEST.sha256"),
            "metric_schema_version": 1,
            "checkpoint": {"path": str(checkpoint), "sha256": digest, "project_git_commit": payload["project_git_commit"],
                           "completed_epochs": payload["completed_epochs"], "seed": payload["seed"]},
            "resolved_probe_config": asdict(cfg), "resolved_training_config": model_config.to_dict(),
            "eligible_route_ids": list(routes), "task_names": task_names,
            "environment_options": {name: CoinRunFactory(name).options for name in task_names}, "procgen_commit": PROCGEN_COMMIT,
            "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "packages": {k: metadata.version(k) for k in ("torch", "numpy", "gymnasium", "gym3", "procgen")},
                        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu": str(torch.cuda.get_device_properties(device)) if device.type == "cuda" else None,
                        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True) if device.type == "cuda" else None},
            "determinism": {"latent": "mode", "policy": "argmax", "cudnn_benchmark": False,
                            "cudnn_deterministic": True, "tf32": False, "strict_deterministic_algorithms": False,
                            "caveat": "Same-runtime seeded diagnostic, not a cross-platform bitwise guarantee"},
            "budgets": {"max_agent_decisions": len(routes) * len(cfg.cohorts) * cfg.episodes_per_task * cfg.prefix_decisions,
                        "world_model_updates": 0, "actor_critic_updates": 0, "replay_writes": 0},
            "measurement": "Matched saved prefixes; no closed-loop rerouting or episode-return comparison",
            "prediction_score": "One-step prior-mode pixel MSE, not exact predictive likelihood",
            "terminal_handling": "Stop prefix at done; exclude native autoreset frame from prediction targets",
            "task_semantics": "NB/RT are visual variants, not a claim of different transition laws",
            "parameter_bytes": sum(p.numel() * p.element_size() for m in (wm, actors) for p in m.parameters()),
            "frozen_state_sha256_before": before,
        }
        write_json_atomic(output / "manifest.json", manifest)
        print(json.dumps({"phase": "loaded", "eligible_routes": routes, "parameter_updates": 0}), flush=True)
        xs, actions, shuffled_actions, valid, episodes = collect(wm, actors, routes, cfg, task_names, output)
        collected_at = time.perf_counter()
        scores, score_seconds = score_dataset(wm, xs, actions, routes, cfg)
        shuffled, shuffled_seconds = score_dataset(wm, xs, shuffled_actions, routes, cfg)
        np.savez_compressed(output / "scores.npz", **scores, shuffled_prediction=shuffled["prediction"],
                            shuffled_posterior_prior_kl=shuffled["posterior_prior_kl"])
        write_sha256_sidecar(output / "scores.npz")
        after = weight_digest(wm, actors)
        if after != before: raise RuntimeError("Frozen model/actor state changed during diagnostic")
        for i, episode in enumerate(episodes):
            route = episode["initial_policy_route"]
            if route is not None and route != routes[int(scores["reconstruction"][i, 0].argmin())]:
                raise RuntimeError("Offline first-frame scorer disagrees with the production collector")
        result = {"protocol": PROTOCOL, "classification": "debug", "complete": True,
                  "tables": summarize(scores, shuffled, valid, episodes, routes, cfg),
                  "actual_agent_decisions": sum(e["collected_agent_decisions"] for e in episodes),
                  "actual_raw_environment_frames": sum(e["collected_agent_decisions"] for e in episodes),
                  "valid_nonterminal_transitions": int(valid[:, 1:].sum()),
                  "shuffled_action_changed_fraction": float(((actions != shuffled_actions) & valid[:, 1:]).sum() / max(1, valid[:, 1:].sum())),
                  "collection_and_setup_seconds": collected_at - started,
                  "all_scores_seconds": score_seconds, "shuffled_scores_seconds": shuffled_seconds,
                  "elapsed_seconds": time.perf_counter() - started,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "timing_caveat": "Shared GPU; combined scorer timing is not per-router online FPS",
                  "frozen_state_sha256_after": after, "world_model_updates": 0, "actor_critic_updates": 0}
        write_json_atomic(output / "results.json", result)
        print(json.dumps({"complete": True, "output_dir": str(output), "tables": result["tables"]}), flush=True)
    except BaseException:
        write_json_atomic(output / "failure.json", {"complete": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
