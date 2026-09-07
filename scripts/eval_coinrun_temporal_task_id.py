#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task-ID accuracy only: frozen CoinRun features, disjoint router train/val/test.

No route switching, stopping rule, reward objective, WM/AC updates or Replay
writes. The first-frame policy supplies identical inputs to every classifier.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from torch import nn
from torch.nn import functional as F

from artifact_io import sha256_file, write_json_atomic, write_sha256_sidecar, write_text_atomic
from git_provenance import require_synced_training_git_state
from probe_coinrun_sequence_routing import (
    ROOT, ProbeConfig, collect, load_model, rooted, score_dataset, seed_for,
    verify_launch, weight_digest,
)

PROTOCOL = "CoinRun-Frozen-TaskID-PrefixClassifier-Pilot-v2"
SPLIT_DOMAINS = {"train": 6101, "validation": 6201, "test": 6301}


@dataclass(frozen=True)
class ExperimentConfig:
    train_episodes_per_task: int = 256
    validation_episodes_per_task: int = 128
    test_episodes_per_task: int = 512
    prefix_decisions: int = 16
    epochs: int = 100
    router_batch_size: int = 64
    score_batch_size: int = 1
    gru_hidden: int = 32
    mlp_hidden: int = 56
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    seed: int = 20260909
    router_seed_count: int = 3
    cpu_threads: int = 2
    device: str = "cuda:0"

    def __post_init__(self):
        integers = (self.train_episodes_per_task, self.validation_episodes_per_task,
                    self.test_episodes_per_task, self.prefix_decisions, self.epochs,
                    self.router_batch_size, self.score_batch_size, self.gru_hidden,
                    self.mlp_hidden, self.router_seed_count, self.cpu_threads)
        if any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("All episode, prefix, model and update counts must be positive integers")
        if not 0 <= self.seed < 2**31 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid seed or optimizer settings")
        if self.score_batch_size != 1:
            raise ValueError("This protocol matches the single-instance production routing batch exactly")


def split_config(cfg: ExperimentConfig, split: str) -> ProbeConfig:
    return ProbeConfig(
        episodes_per_task=getattr(cfg, f"{split}_episodes_per_task"),
        prefix_decisions=cfg.prefix_decisions, batch_size=cfg.score_batch_size,
        cpu_threads=cfg.cpu_threads, seed=seed_for(cfg.seed, SPLIT_DOMAINS[split], 0),
        device=cfg.device, cohorts=("first_frame_policy",),
        windows=tuple(range(1, cfg.prefix_decisions + 1)),
    )


def seed_plan(cfg: ExperimentConfig) -> dict:
    """Pair task variants by level seed; never split frames of one seed group."""
    plan = {}
    for split in SPLIT_DOMAINS:
        sc = split_config(cfg, split)
        plan[split] = [seed_for(sc.seed, 1001, i) for i in range(sc.episodes_per_task)]
    all_seeds = sum(plan.values(), [])
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Environment seed collision within or across data splits")
    return plan


def validate_features(errors: torch.Tensor, valid: torch.Tensor) -> None:
    """Float32 errors [N,T,K], bool valid [N,T]; right-padding only."""
    if (errors.ndim != 3 or errors.dtype != torch.float32 or min(errors.shape) < 1
            or valid.shape != errors.shape[:2] or valid.dtype != torch.bool
            or errors.device != valid.device):
        raise ValueError("Expected nonempty float32 [N,T,K] errors and colocated bool [N,T] mask")
    if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("Each episode must have a first frame and contiguous valid prefix")
    if not bool(torch.isfinite(errors).all()) or bool((errors < 0).any()):
        raise ValueError("Reconstruction errors must be finite and nonnegative")


def fit_input_calibration(errors, valid, labels):
    """Fit only on train. One common positive temperature preserves argmin."""
    validate_features(errors, valid)
    values = errors[valid]
    mean, std = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-8)
    temperatures = torch.logspace(-6, 0, 61)
    losses = torch.stack([F.cross_entropy(-errors[:, 0] / t, labels) for t in temperatures])
    return mean, std, float(temperatures[losses.argmin()])


class ErrorRouter(nn.Module):
    """Error-only classifiers; forward accepts no true ID, reward or future frame.

    GRU: initial logits + cumulative prefix correction; first frame is unchanged.
    MLP: current-step errors only, with no explicit temporal aggregation. These
    errors already contain the candidate RSSM's filtering history, so MLP is not
    a claim to be an RGB-only memoryless visual baseline.
    """
    def __init__(self, kind: str, mean: torch.Tensor, std: torch.Tensor,
                 temperature: float, hidden: int):
        super().__init__()
        if kind not in ("gru", "mlp") or temperature <= 0 or hidden < 1:
            raise ValueError("Invalid router kind, temperature or width")
        self.kind = kind
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.register_buffer("temperature", torch.tensor(temperature))
        k = len(mean)
        self.core = (nn.GRU(k, hidden, batch_first=True) if kind == "gru" else
                     nn.Sequential(nn.Linear(k, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU()))
        self.output = nn.Linear(hidden, k)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        if errors.ndim != 3 or errors.shape[-1] != len(self.mean):
            raise ValueError("Router input must be [N,T,K] for its acquired route count")
        x = (errors - self.mean) / self.std
        if self.kind == "gru":
            hidden, _ = self.core(x)
            delta = self.output(hidden)
            delta = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, 1:]), 1)
            return -errors[:, :1] / self.temperature + delta
        return -errors / self.temperature + self.output(self.core(x))


def prefix_loss(logits, labels, valid):
    """Equal episode weight, then equal weight for each actually observed prefix."""
    if logits.shape[:2] != valid.shape or labels.shape != valid.shape[:1]:
        raise ValueError("Prefix logits, label and mask shapes differ")
    losses = F.cross_entropy(logits.transpose(1, 2), labels[:, None].expand_as(valid), reduction="none")
    return ((losses * valid).sum(1) / valid.sum(1)).mean()


@torch.no_grad()
def selection_metrics(model, errors, valid, labels):
    logits = model(errors)
    # Select on all observed post-first-frame prefixes, not one favorable window.
    mask = valid.clone(); mask[:, 0] = False
    count = mask.sum(1)
    usable = count > 0
    if not bool(usable.any()):
        raise ValueError("Validation cohort has no post-first-frame observations")
    hit = logits.argmax(-1).eq(labels[:, None])
    accuracy = ((hit * mask).sum(1)[usable] / count[usable]).mean().item()
    nll = prefix_loss(logits, labels, valid).item()
    return {"mean_post_first_prefix_accuracy": accuracy, "prefix_nll": nll}


def atomic_router_snapshot(path: Path, model, cfg, seed, epoch, metrics):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"artifact_kind": "router_inference_only_not_resumable",
                "protocol": PROTOCOL, "config": asdict(cfg), "kind": model.kind,
                "router_seed": seed, "selected_epoch": epoch,
                "validation_metrics": metrics, "state_dict": model.state_dict()}, temporary)
    os.replace(temporary, path)
    write_sha256_sidecar(path)


def train_routers(train, validation, cfg, output):
    """Only this function performs parameter updates; test data are not accepted."""
    x, valid, y = train
    calibration = fit_input_calibration(x, valid, y)
    output.mkdir()
    models, records = {}, []
    for replica in range(cfg.router_seed_count):
        seed = seed_for(cfg.seed, 6401, replica)
        for kind, width in (("mlp", cfg.mlp_hidden), ("gru", cfg.gru_hidden)):
            torch.manual_seed(seed)
            model = ErrorRouter(kind, *calibration, width).cpu()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
            sampler = torch.Generator().manual_seed(seed_for(seed, 6501, 0))
            best_key, best_state, selected_epoch, history, updates = None, None, None, [], 0
            for epoch in range(1, cfg.epochs + 1):
                model.train()
                order = torch.randperm(len(x), generator=sampler)
                total_loss = 0.0
                for indices in order.split(cfg.router_batch_size):
                    loss = prefix_loss(model(x[indices]), y[indices], valid[indices])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite router training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step(); updates += 1
                    total_loss += loss.item() * len(indices)
                model.eval()
                metrics = selection_metrics(model, *validation)
                key = (metrics["mean_post_first_prefix_accuracy"], -metrics["prefix_nll"])
                history.append({"epoch": epoch, "updates": updates, "train_prefix_nll": total_loss / len(x), **metrics})
                if best_key is None or key > best_key:
                    best_key, selected_epoch = key, epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if epoch % 25 == 0:
                    print(json.dumps({"phase": "router_training", "kind": kind,
                                      "replica": replica, **history[-1]}), flush=True)
            model.load_state_dict(best_state)
            model.requires_grad_(False).eval()
            path = output / f"{kind}-seed{seed}.pt"
            atomic_router_snapshot(path, model, cfg, seed, selected_epoch, history[selected_epoch - 1])
            # Test the persisted inference contract before opening the test cohort.
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restored = ErrorRouter(kind, *calibration, width).eval()
            restored.load_state_dict(saved["state_dict"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(x[:4]), model(x[:4]), rtol=0, atol=0)
            models[(kind, seed)] = model
            records.append({"kind": kind, "router_seed": seed, "selected_epoch": selected_epoch,
                            "router_updates": updates, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                            "checkpoint": path.name, "sha256": sha256_file(path), "history": history})
    write_json_atomic(output / "training.json", {"temperature": calibration[2],
                      "feature_mean": calibration[0].tolist(), "feature_std": calibration[1].tolist(),
                      "runs": records, "test_not_collected_yet": True})
    return models, records


def collect_split(wm, actors, routes, names, cfg, split, output):
    sc = split_config(cfg, split)
    directory = output / split; directory.mkdir()
    xs, actions, _, valid, rows = collect(wm, actors, routes, sc, names, directory)
    scores, seconds = score_dataset(wm, xs, actions, routes, sc, reconstruction_only=True)
    errors = scores["reconstruction"]
    labels = np.array([r["task_index_for_audit_only"] for r in rows], dtype=np.int64)
    production_first = np.asarray([r["initial_policy_reconstruction_mse"] for r in rows], dtype=np.float32)
    first_score_difference = float(np.abs(errors[:, 0] - production_first).max())
    # BF16 near-tied categorical modes can amplify batch-size differences. Do
    # not relax this check or replace scores to conceal a changed latent path.
    np.testing.assert_array_equal(errors[:, 0], production_first)
    for i, row in enumerate(rows):
        if row["initial_policy_route"] != routes[int(errors[i, 0].argmin())]:
            raise RuntimeError("First-frame feature scorer disagrees with actual behavior router")
    path = directory / "features.npz"
    np.savez_compressed(path, reconstruction=errors, valid=valid, labels=labels)
    write_sha256_sidecar(path)
    x, v, y = torch.from_numpy(errors), torch.from_numpy(valid), torch.from_numpy(labels)
    validate_features(x, v)
    stats = {"split": split, "episode_count": len(rows), "score_seconds": seconds,
             "agent_decisions": sum(r["collected_agent_decisions"] for r in rows),
             "valid_nonterminal_transitions": int(valid[:, 1:].sum()),
             "ended_before_budget": sum(r["ended"] for r in rows),
             "batched_vs_production_first_score_max_abs_difference": first_score_difference,
             "features_sha256": sha256_file(path)}
    write_json_atomic(directory / "collection.json", stats)
    return (x, v, y), stats


def accuracy_row(predicted, first, labels, selected, k):
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (labels[selected], predicted[selected]), 1)
    n = int(matrix.sum())
    return {"sample_count": n, "correct_count": int(np.trace(matrix)),
            "accuracy": float(np.trace(matrix) / n) if n else None,
            "per_task_accuracy": [float(matrix[i, i] / row.sum()) if row.sum() else None for i, row in enumerate(matrix)],
            "confusion_matrix": matrix.tolist(),
            "wrong_first_corrected": int(((first != labels) & (predicted == labels) & selected).sum()),
            "correct_first_broken": int(((first == labels) & (predicted != labels) & selected).sum())}


def summarize_predictions(predictions, valid, labels, first, route_count):
    """Both same-cohort up-to-t and available-exact-prefix denominators are saved."""
    last = valid.sum(1) - 1
    rows = []
    for name, seed, prediction in predictions:
        for t in range(valid.shape[1]):
            available = valid[:, t]
            capped = prediction[np.arange(len(valid)), np.minimum(t, last)]
            for population, pred, selected in (
                ("all_episodes_up_to_frames", capped, np.ones(len(valid), bool)),
                ("exact_frames_available", prediction[:, t], available),
            ):
                rows.append({"router": name, "router_seed": seed, "observed_frame_budget": t + 1,
                             "population": population, "episodes_with_exact_frames": int(available.sum()),
                             **accuracy_row(pred, first, labels, selected, route_count)})
    return rows


@torch.no_grad()
def evaluate_routers(models, test, output):
    x, valid, y = test
    n, t, k = x.shape
    first = x[:, 0].argmin(-1).numpy()
    baseline = np.broadcast_to(first[:, None], (n, t)).copy()
    mean_prediction = (x.cumsum(1) / torch.arange(1, t + 1)[None, :, None]).argmin(-1).numpy()
    predictions = [("first_frame_mse", None, baseline), ("mean_prefix_mse", None, mean_prediction)]
    arrays = {"valid": valid.numpy(), "labels": y.numpy(), "first_frame_mse": baseline,
              "mean_prefix_mse": mean_prediction}
    for (kind, seed), model in models.items():
        logits = model(x)
        predicted = logits.argmax(-1).numpy()
        name = "temporal_gru" if kind == "gru" else "pointwise_mlp"
        predictions.append((name, seed, predicted))
        arrays[f"{name}_{seed}_logits"] = logits.numpy()
        if kind == "mlp":
            predictions.append(("learned_first_frame_mlp", seed,
                                np.broadcast_to(predicted[:, :1], (n, t)).copy()))
    path = output / "test_predictions.npz"
    np.savez_compressed(path, **arrays); write_sha256_sidecar(path)
    return summarize_predictions(predictions, valid.numpy(), y.numpy(), first, k)


def report(results):
    rows = results["tables"]
    names = ("first_frame_mse", "mean_prefix_mse", "learned_first_frame_mlp", "pointwise_mlp", "temporal_gru")
    frames = [f for f in (1, 2, 5, 9, 17) if f <= results["frame_count"]]
    lines = ["# CoinRun task-ID prefix accuracy — pilot", "",
             "Frozen boundary checkpoint; no switching/stopping, rewards or WM/AC updates.",
             "Main table retains every test episode, using its last available prediction if it ended early.",
             "Learned entries are means over router training seeds, not an ensemble or independent world-model seeds.", "",
             "| Router | " + " | ".join(f"up to {f} frames" for f in frames) + " |",
             "| --- | " + " | ".join("---:" for _ in frames) + " |"]
    for name in names:
        values = []
        for f in frames:
            selected = [r["accuracy"] for r in rows if r["router"] == name and
                        r["observed_frame_budget"] == f and r["population"] == "all_episodes_up_to_frames"]
            values.append(f"{np.mean(selected):.2%}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend(["", "Exact-length counts, all individual seeds, confusion matrices and correction/break counts are in results.json.",
                  "MLP uses the current RSSM-error vector: no explicit temporal aggregation, but the RSSM itself is recurrent.",
                  "This is held-out classification on fixed first-frame-policy trajectories, not closed-loop rerouting accuracy.",
                  "One world-model checkpoint and three visual variants do not establish a multi-checkpoint high-accuracy claim."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-verification", type=Path, required=True)
    # A new budget is explicit in the resolved config; unknown CLI keys are errors.
    for key in ("train_episodes_per_task", "validation_episodes_per_task", "test_episodes_per_task",
                "prefix_decisions", "epochs", "router_seed_count", "cpu_threads", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    checkpoint, output, verification = (rooted(args.pop(k)) for k in ("checkpoint", "output_dir", "upstream_verification"))
    cfg = ExperimentConfig(**args)
    plan = seed_plan(cfg)
    git = require_synced_training_git_state(ROOT)
    receipt = json.loads(verification.read_text()); verify_launch(receipt, git)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        wm, actors, config, routes, payload, digest = load_model(checkpoint, device)
        from clworldmodel.environments.coinrun import COINRUN_TASKS, CoinRunFactory, PROCGEN_COMMIT
        names = COINRUN_TASKS[:len(routes)]
        before = weight_digest(wm, actors)
        manifest = {
            "protocol": PROTOCOL, "classification": "pilot", "metric_schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "project_git": git,
            "upstream_verification": receipt, "resolved_config": asdict(cfg),
            "resolved_world_model_training_config": config.to_dict(),
            "arrow_base_commit": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
            "vendored_manifest_sha256": sha256_file(ROOT / "third_party/arrow/MANIFEST.sha256"),
            "checkpoint": {"path": str(checkpoint), "sha256": digest, "training_commit": payload["project_git_commit"],
                           "training_seed": payload["seed"], "completed_epochs": payload["completed_epochs"]},
            "task_names": names, "eligible_route_ids": routes, "split_environment_seeds": plan,
            "task_order": list(names), "task_boundary_schedule": "separate fixed-task episode prefixes, not continual updates",
            "environment_options": {name: CoinRunFactory(name).options for name in names},
            "procgen_commit": PROCGEN_COMMIT, "raw_observation_dtype": "uint8", "model_input_dtype": "float32/autocast bfloat16",
            "feature_dtype_device": "float32 CPU", "router_training_device": "CPU",
            "runtime": {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                        "cpu_count": os.cpu_count(), "cuda_build": torch.version.cuda,
                        "packages": {key: metadata.version(key) for key in ("torch", "numpy", "gymnasium", "gym3", "procgen")},
                        "constraints_sha256": sha256_file(ROOT / "requirements/d_autoroute_coinrun.constraints.txt"),
                        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
                        "gpu": str(torch.cuda.get_device_properties(device)) if device.type == "cuda" else None,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True) if device.type == "cuda" else None},
            "determinism": {"latent": "mode", "policy": "argmax", "cudnn_benchmark": False,
                            "cudnn_deterministic": True, "tf32": False, "strict_deterministic_algorithms": False,
                            "router_sampler": "owned seeded torch.Generator", "caveat": "same-runtime seeded; no cross-platform bitwise claim"},
            "budgets": {"max_agent_decisions": len(routes) * sum(map(len, plan.values())) * cfg.prefix_decisions,
                        "router_updates_per_model": cfg.epochs * ((len(routes) * cfg.train_episodes_per_task + cfg.router_batch_size - 1) // cfg.router_batch_size),
                        "router_model_count": 2 * cfg.router_seed_count, "world_model_updates": 0,
                        "actor_critic_updates": 0, "replay_writes": 0, "replay_capacity": 0, "replay_bytes": 0},
            "selection": "validation mean per-episode post-first-prefix accuracy; tie: prefix NLL; first best epoch retained",
            "isolation": "new train-only labels fit router; val only selects epoch; test collected after snapshots sealed; no old evaluation data reused",
            "behavior_policy": "existing first-frame route locked; frozen private actor argmax; labels never enter action selection",
            "terminal_handling": "exclude native autoreset frame; right padding is masked; all-cohort curves retain last real prediction",
            "rewards": "preserved only in raw collection records; never a routing input, objective or selection metric",
            "frozen_state_sha256_before": before,
        }
        write_json_atomic(output / "manifest.json", manifest)
        train, train_stats = collect_split(wm, actors, routes, names, cfg, "train", output)
        validation, validation_stats = collect_split(wm, actors, routes, names, cfg, "validation", output)
        models, training_records = train_routers(train, validation, cfg, output / "routers")
        seal = {f"{kind}:{seed}": weight_digest(model) for (kind, seed), model in models.items()}
        write_json_atomic(output / "pretest_seal.json", {"sealed_at_utc": datetime.now(timezone.utc).isoformat(),
                          "router_state_digests": seal, "test_environment_not_created_yet": True})
        test, test_stats = collect_split(wm, actors, routes, names, cfg, "test", output)
        tables = evaluate_routers(models, test, output)
        if seal != {f"{kind}:{seed}": weight_digest(model) for (kind, seed), model in models.items()}:
            raise RuntimeError("Router parameters changed after held-out test was opened")
        after = weight_digest(wm, actors)
        if before != after:
            raise RuntimeError("Frozen world model or Actor changed")
        result = {"protocol": PROTOCOL, "classification": "pilot", "complete": True,
                  "frame_count": cfg.prefix_decisions + 1, "tables": tables,
                  "collection": [train_stats, validation_stats, test_stats],
                  "actual_agent_decisions": sum(s["agent_decisions"] for s in (train_stats, validation_stats, test_stats)),
                  "router_updates": sum(r["router_updates"] for r in training_records),
                  "world_model_updates": 0, "actor_critic_updates": 0, "replay_writes": 0,
                  "router_test_state_unchanged": True, "frozen_state_sha256_after": after,
                  "elapsed_seconds": time.perf_counter() - started,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        write_json_atomic(output / "results.json", result)
        write_text_atomic(output / "SUMMARY.md", report(result))
        print(report(result), flush=True)
    except BaseException:
        write_json_atomic(output / "failure.json", {"complete": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
