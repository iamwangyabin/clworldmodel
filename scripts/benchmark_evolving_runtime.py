#!/usr/bin/env python3
"""Benchmark production-shaped Evolving-Core update loops without environments.

This benchmark performs gradient updates on deterministic synthetic replay.  It
therefore obeys the training provenance gate even though it collects no frames
and makes no learning-quality claim.  The same harness runs before and after
runtime-only changes so state hashes and scalar metrics can be compared.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
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

from ac import train_ac_from_wm  # noqa: E402
from git_provenance import require_synced_training_git_state  # noqa: E402
from replay import FifoReplay, LongTermReplay, MultiTypeReplay  # noqa: E402
import train  # noqa: E402
from smoke_evolving_atomic_rssm import (  # noqa: E402
    _config,
    _synthetic_batch,
    _world_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--wm-warmup", type=int, default=5)
    parser.add_argument("--wm-steps", type=int, default=50)
    parser.add_argument("--actor-warmup", type=int, default=5)
    parser.add_argument("--actor-steps", type=int, default=100)
    parser.add_argument("--compile-world-model-loss", action="store_true")
    return parser


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _task_replay(config) -> MultiTypeReplay:
    replay = MultiTypeReplay(
        FifoReplay(
            32,
            32,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        LongTermReplay(
            32,
            32,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        sampling_weights=(0.5, 0.5),
    )
    for task_id in (0, 1):
        replay.add(
            *_synthetic_batch(
                config,
                task_id=task_id,
                time=32,
                sequences=16,
            ),
            task_id=task_id,
        )
    return replay


def _actor_replay(config) -> MultiTypeReplay:
    replay = MultiTypeReplay(
        FifoReplay(
            4,
            128,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        LongTermReplay(
            4,
            128,
            config.action_space,
            "cpu",
            store_task_ids=True,
            observation_dtype="uint8",
        ),
        sampling_weights=(0.5, 0.5),
    )
    replay.add(
        *_synthetic_batch(config, task_id=0, time=4, sequences=128),
        task_id=0,
    )
    return replay


def _world_model_benchmark(
    config,
    device: torch.device,
    warmup: int,
    steps: int,
    *,
    compile_loss: bool,
) -> dict:
    world_model = _world_model(config, device)
    world_model.activate_task_expert(0)
    boundary_teacher = copy.deepcopy(world_model).eval()
    boundary_teacher.requires_grad_(False)
    if not world_model.initialize_task_expert(1, 0):
        raise RuntimeError("Task-1 private topology did not initialize")
    world_model.activate_task_expert(1)
    replay = _task_replay(config)
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
        get=lambda task_id: SimpleNamespace(ac=SimpleNamespace(actor=actor))
    )
    if compile_loss:
        torch._dynamo.config.cache_size_limit = 64
        world_model.compute_loss_and_trace = torch.compile(
            world_model.compute_loss_and_trace,
            dynamic=False,
            mode="reduce-overhead",
        )
    signature = inspect.signature(train._evolving_world_model_update)
    supports_deferred_diagnostics = "materialize_diagnostics" in signature.parameters

    def update():
        extra = (
            {"materialize_diagnostics": False}
            if supports_deferred_diagnostics
            else {}
        )
        return train._evolving_world_model_update(
            config=config,
            wm=world_model,
            boundary_teacher=boundary_teacher,
            actor_critic_bank=actor_bank,
            replay_buffer=replay,
            current_task_id=1,
            memory_task_id=0,
            sequence_length=32,
            shared_optimizer=shared_optimizer,
            private_optimizer=private_optimizer,
            route_optimizer=route_optimizer,
            **extra,
        )

    for _ in range(warmup):
        update()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = None
    for _ in range(steps):
        result = update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if result is None:
        raise RuntimeError("World-model benchmark needs at least one measured step")
    metrics, diagnostics, gradient_norm = result
    checked = {
        "current_loss": metrics["Loss/evolving_current_total"],
        "memory_loss": metrics["Memory/Loss/evolving_memory_total"],
        "gradient_norm": gradient_norm,
    }
    return {
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000.0 / steps,
        "updates_per_second": steps / elapsed,
        "compiled_loss": compile_loss,
        "deferred_diagnostics_fast_path": supports_deferred_diagnostics,
        "returned_diagnostic_count": len(diagnostics),
        "scalars": {
            name: float(value.detach().cpu()) for name, value in checked.items()
        },
        "world_model_state_sha256": _state_hash(world_model),
    }


def _actor_benchmark(config, device: torch.device, warmup: int, steps: int) -> dict:
    world_model = _world_model(config, device)
    world_model.activate_task_expert(0)
    replay = _actor_replay(config)
    actor_opt, _, _ = train_ac_from_wm(
        world_model,
        replay,
        steps=warmup,
        n_sync=config.ac_train_sync,
        dream_steps=config.ac_dream_steps,
        lr=config.ac_lr,
        task_id=0,
        entropy_scale=config.ac_entropy_scale,
        grad_clip=config.ac_grad_clip,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actor_opt, performance, metrics = train_ac_from_wm(
        world_model,
        replay,
        steps=steps,
        n_sync=config.ac_train_sync,
        dream_steps=config.ac_dream_steps,
        aco=actor_opt,
        lr=config.ac_lr,
        task_id=0,
        entropy_scale=config.ac_entropy_scale,
        grad_clip=config.ac_grad_clip,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000.0 / steps,
        "updates_per_second": steps / elapsed,
        "approx_performance": float(performance.detach().cpu()),
        "metrics": metrics,
        "actor_critic_state_sha256": _state_hash(actor_opt.ac),
    }


def main() -> int:
    args = _parser().parse_args()
    if min(args.wm_warmup, args.actor_warmup) < 0:
        raise ValueError("Warmup counts must be non-negative")
    if min(args.wm_steps, args.actor_steps) < 1:
        raise ValueError("Measured step counts must be positive")
    project_git = require_synced_training_git_state(ROOT)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Runtime benchmark requires CUDA")
    torch.cuda.set_device(device)
    _seed_everything(args.seed)
    config = _config()
    started_at = datetime.now(timezone.utc)
    torch.cuda.reset_peak_memory_stats(device)
    world_model = _world_model_benchmark(
        config,
        device,
        args.wm_warmup,
        args.wm_steps,
        compile_loss=args.compile_world_model_loss,
    )
    actor = _actor_benchmark(config, device, args.actor_warmup, args.actor_steps)
    completed_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "classification": "benchmark",
        "scientific_result": False,
        "synthetic_data": True,
        "environment_interaction": False,
        "project_git": project_git,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "fixed_shapes": {
            "world_model_time": config.mb_t_size,
            "world_model_current_sequences": config.current_batch_n,
            "world_model_memory_sequences": config.memory_batch_n,
            "actor_context_sequences": config.ac_train_sync,
            "actor_dream_steps": config.ac_dream_steps,
        },
        "world_model": world_model,
        "actor_critic": actor,
    }
    _write_json(output_dir / "benchmark.json", payload)
    _write_json(
        output_dir / "run_status.json",
        {"schema_version": 1, "complete": True, "return_code": 0},
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
