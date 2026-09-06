#!/usr/bin/env python3
"""Atari orchestration only; all learning calls execute pinned official code.

Public run entry: run_dream_rehearsal_official_atari.py. A verified launch
manifest is mandatory even when this executable is called directly.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dream_rehearsal_reference_support import (
    ROOT, load_reference_runtime, resolve_model_config, sha256, verify_reference_sources,
)
from git_provenance import require_synced_training_git_state
from launcher_support import write_json

sys.path.insert(0, str(ROOT / "src"))
from clworldmodel.reference.dream_rehearsal import (
    MEMORY_PAIR_METHOD, MemoryPairConfig, OfficialDreamRehearsalConfig, ReplayLibraries, run_phase_chunks,
)


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


@contextmanager
def isolated_evaluation_rng(torch, np, seed: int):
    """Evaluation draws never alter subsequent replay/model/actor randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class EpisodeArchive:
    """One append-only pixel mmap, avoiding an open file per episode/field.

    Inactive episode dicts are retained in-place so phase libraries keep their
    references. Active episodes stay appendable. Auxiliary arrays stay in RAM;
    full upstream NPZ archives contain the persistent copies of every field.
    """

    def __init__(self, path: Path, max_frames: int, image_shape: tuple[int, int, int]):
        import numpy as np

        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replay mmap: {path}")
        self.pixels = np.memmap(path, mode="w+", dtype=np.uint8,
                                shape=(max_frames, *image_shape))
        self.path = path
        self.cursor = 0
        self.spans: dict[str, tuple[int, int]] = {}

    def archive(self, episodes, active_ids: set[str]) -> None:
        import numpy as np

        for episode_id, episode in episodes.items():
            if episode_id in active_ids or episode_id in self.spans:
                continue
            frames = np.asarray(episode["image"])
            if frames.dtype != np.uint8 or frames.shape[1:] != self.pixels.shape[1:]:
                raise ValueError("Replay images must retain the adapter's uint8 shape")
            end = self.cursor + len(frames)
            if end > len(self.pixels):
                raise RuntimeError("Never-clear pixel archive capacity exceeded; refusing overwrite")
            view = self.pixels[self.cursor:end]
            view[:] = frames
            view.flags.writeable = False
            mapped = {k: (view if k == "image" else np.asarray(v)) for k, v in episode.items()}
            episode.clear()
            episode.update(mapped)
            self.spans[episode_id] = (self.cursor, end)
            self.cursor = end
        self.pixels.flush()

    def accounting(self) -> dict:
        stat = self.path.stat()
        return {
            "path": self.path.name, "capacity_frames": len(self.pixels),
            "used_frames_including_resets": self.cursor,
            "logical_file_bytes": stat.st_size, "allocated_file_bytes": stat.st_blocks * 512,
            "episode_spans": self.spans,
        }


def replay_accounting(episodes) -> dict[str, Any]:
    import numpy as np

    transitions = sum(len(e["reward"]) - 1 for e in episodes.values())
    tensor_bytes = sum(np.asarray(v).nbytes for e in episodes.values() for v in e.values())
    mapped_bytes = sum(v.nbytes for e in episodes.values() for v in e.values() if isinstance(v, np.memmap))
    return {
        "episodes": len(episodes), "collected_transitions_retained": transitions,
        "logical_tensor_bytes": tensor_bytes, "mapped_array_bytes": mapped_bytes,
        "python_and_filesystem_overhead_included": False,
        "ordinary_training_sampling": "shared_full_history", "dataset_size": 0,
        "phase_libraries_copy_observations": False,
    }


def episode_archive_accounting(directory: Path) -> dict[str, int]:
    """Report actual NPZ storage separately from live replay/mmap array bytes."""
    sizes = [path.stat() for path in directory.glob("*.npz")]
    return {
        "episode_files": len(sizes),
        "logical_file_bytes": sum(stat.st_size for stat in sizes),
        "allocated_file_bytes": sum(stat.st_blocks * 512 for stat in sizes),
    }


def runtime_info(config) -> dict:
    import torch

    requirements = ROOT / "requirements/dream_rehearsal_official.txt"
    packages = {}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = metadata.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Pinned reference environment required: {name}=={expected}, got {actual}")
        packages[name] = actual
    if config.device.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expose exactly one CUDA device for this reference run")
    info = {
        "python": sys.version, "os": platform.platform(), "cpu": platform.processor(),
        "cpu_count": os.cpu_count(), "packages": packages,
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "known_nondeterminism": "CUDA kernels are not forced deterministic; reference default preserved",
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["accelerator"] = {"name": p.name, "count": torch.cuda.device_count(), "memory_bytes": p.total_memory}
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                         check=True, text=True, capture_output=True).stdout.splitlines()
    return info


def run(output: Path) -> None:
    import numpy as np
    import torch

    launch = json.loads((output / "launch.json").read_text())
    subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
    state = require_synced_training_git_state(ROOT)
    if state != launch["project_git"] or verify_reference_sources() != launch["reference_sources"]:
        raise RuntimeError("Launch provenance changed before training")
    config_data = json.loads((output / "protocol_config.json").read_text())
    config_type = MemoryPairConfig if config_data.get("protocol") == MEMORY_PAIR_METHOD else OfficialDreamRehearsalConfig
    config = config_type.from_dict(config_data)
    memory_pair = isinstance(config, MemoryPairConfig)
    if config.as_dict() != launch["config"]:
        raise RuntimeError("Resolved protocol differs from launch manifest")
    if (output / "runtime.json").exists():
        raise FileExistsError("This is not a resumable runner; refusing to restart in an existing run")
    # mmap + upstream compressed episode archives coexist. This conservative
    # bound intentionally ignores compression; do not launch on a nearly full disk.
    required_disk = 2 * launch["replay"]["live_history_observation_bytes_lower_bound"] + 5 * 1024**3
    if shutil.disk_usage(output).free < required_disk:
        raise RuntimeError(f"Need at least {required_disk} free bytes for declared history storage")
    torch.set_num_threads(config.cpu_threads)
    runtime = runtime_info(config)
    D, tools, reference, parallel, wrappers = load_reference_runtime()
    from clworldmodel.reference.atari import make_atari

    model_config = resolve_model_config(config, output, tools_module=tools)
    write_json(output / "resolved_model_config.json", vars(model_config))
    write_json(output / "runtime.json", runtime)
    tools.set_seed_everywhere(config.seed)

    class CaptureLogger(tools.Logger):
        def __init__(self):
            super().__init__(output, 0)
            self.captured = {}

        def scalar(self, name, value):
            self.captured[name] = float(value)
            super().scalar(name, value)

        def video(self, name, value):
            # Reporting-only: raw evaluation episodes are archived, not videos.
            pass

    logger = CaptureLogger()
    dataset_factory = lambda eps: D.make_dataset(eps, model_config)
    if memory_pair:
        from clworldmodel.reference.episode_history import EpisodeHistory
        from dream_rehearsal_collection import RetainedHistoryCollector

        history = EpisodeHistory(
            output / "retained_history", total_decisions=config.projected_budgets()["agent_decisions"],
            capacity_transitions=config.history_capacity_transitions,
            block_length=config.retention_block_transitions, rng=random.Random(config.seed),
            dataset_factory=dataset_factory,
        )
        archive = None
    else:
        history = ReplayLibraries(dataset_factory)
        archive = EpisodeArchive(
            output / "observations.uint8.mmap",
            2 * config.projected_budgets()["agent_decisions"] + 2 * len(config.tasks),
            (config.image_size, config.image_size, 3),
        )
    train_dir = output / "train_eps"
    eval_handles, all_handles, train_adapters, eval_adapters, eval_caches = [], [], [], [], []
    agent = None
    total_rehearsal = 0
    eval_round = 0
    evaluations = []
    phase_records = []

    def wrap(base, seed):
        env = wrappers.OneHotAction(base)
        env._random.seed(seed)
        env.action_space.seed(seed)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        result = parallel.Damy(env)
        all_handles.append(result)
        return result

    def counters():
        return {
            "agent_decisions": sum(e.agent_decisions for e in train_adapters),
            "raw_environment_frames": sum(e.raw_frames for e in train_adapters),
            "evaluation_agent_decisions": sum(e.agent_decisions for e in eval_adapters),
            "evaluation_raw_frames": sum(e.raw_frames for e in eval_adapters),
            "world_model_updates": agent._update_count if agent is not None else 0,
            "base_actor_updates": agent._update_count if agent is not None else 0,
            "base_critic_updates": agent._update_count if agent is not None else 0,
            "rehearsal_actor_updates": total_rehearsal,
        }

    try:
        for phase_id, task in enumerate(config.tasks):
            train_seed = (config.seed + phase_id * 1000) % (2**32)
            eval_seed = (train_seed + 100) % (2**32)
            base = make_atari(task, config, train_seed)
            train_adapters.append(base)
            train_handle = wrap(base, train_seed)
            eval_base = make_atari(task, config, eval_seed)
            eval_adapters.append(eval_base)
            eval_handles.append(wrap(eval_base, eval_seed))
            eval_caches.append(OrderedDict())
            phase_start_keys = set(history.episodes)

            simulate_training = tools.simulate
            prefill_tools = tools
            if memory_pair:
                collector = RetainedHistoryCollector(train_handle, history, tools,
                                                      phase=phase_id, num_actions=model_config.num_actions)
                simulate_training = collector.simulate
                # No global/source monkey-patch. The author helper receives a
                # collection adapter; its random policy is unchanged.
                prefill_tools = SimpleNamespace(OneHotDist=tools.OneHotDist, simulate=simulate_training)
            reference.random_prefill(prefill_tools, model_config, [train_handle], history.episodes,
                                     train_dir, logger, config.prefill_decisions)
            if memory_pair:
                collector.online = True
            # Same shared dictionary, not a current-task filtered view.
            if agent is None:
                agent = D.Dreamer(train_handle.observation_space, train_handle.action_space,
                                  model_config, logger, history.ordinary_dataset()).to(config.device)
                agent.requires_grad_(False)
                write_json(output / "parameter_accounting.json", {
                    "unique_agent_parameters": sum(p.numel() for p in agent.parameters()),
                    "world_model_parameters": sum(p.numel() for p in agent._wm.parameters()),
                    "actor_parameters": sum(p.numel() for p in agent._task_behavior.actor.parameters()),
                    "critic_parameters": sum(p.numel() for p in agent._task_behavior.value.parameters()),
                    "one_shared_actor": True,
                })
            else:
                agent._dataset = history.ordinary_dataset()
            keys_before_online = set(history.episodes)
            simulation_state = None  # New task; NOT reset between 2000-step chunks.

            def train_chunk(decisions):
                nonlocal simulation_state
                before = base.agent_decisions
                simulation_state = simulate_training(
                    agent, [train_handle], history.episodes, train_dir, logger,
                    limit=0, steps=decisions, state=simulation_state,
                )
                if base.agent_decisions - before != decisions:
                    raise RuntimeError("Reference collector overshot the declared chunk")
                if archive is not None:
                    archive.archive(history.episodes, {train_handle.id})

            def rehearse(prior_id, updates):
                nonlocal total_rehearsal
                losses = []
                if memory_pair:
                    history.require_rehearsal_phase(prior_id)
                for _ in range(updates):
                    # THE official function, unmodified (including feats[-1]
                    # bootstrap, 0.3/+10, top25%, and the original actor optimizer).
                    loss = reference.tunnel_update(
                        agent, next(history.rehearsal_datasets[prior_id]),
                        topk_frac=config.top_fraction, cont_grading=True,
                    )
                    if not math.isfinite(loss):
                        raise FloatingPointError("Non-finite official rehearsal loss")
                    losses.append(loss)
                    total_rehearsal += 1
                append_json(output / "rehearsal.jsonl", {
                    "protocol": config.protocol,
                    "history_arm": config.history_arm if memory_pair else "full_reference",
                    "phase_id": phase_id, "replay_phase_id": prior_id,
                    "actor_bc_loss_mean": float(np.mean(losses)),
                    "actor_updates": updates, **counters(),
                })

            def evaluate(completed):
                nonlocal eval_round
                rng_seed = (config.seed + 1000000 + eval_round) % (2**32)
                with isolated_evaluation_rng(torch, np, rng_seed):
                    for j, handle in enumerate(eval_handles):
                        eb = eval_adapters[j]
                        index = len(eb.episode_returns)
                        with torch.no_grad():
                            tools.simulate(
                                lambda obs, done, state: agent(obs, done, state, training=False),
                                [handle], eval_caches[j], output / f"eval_eps_{j:02d}", logger,
                                is_eval=True, episodes=config.eval_episodes,
                            )
                        returns = eb.episode_returns[index:]
                        if len(returns) != config.eval_episodes:
                            raise RuntimeError("Evaluation did not finish the configured episode count")
                        row = {
                            "protocol": config.protocol,
                            "history_arm": config.history_arm if memory_pair else "full_reference",
                            "schema_version": 1, "phase_id": phase_id, "eval_round": eval_round,
                            "phase_online_agent_decisions": completed,
                            "task_index": j, "task_name": config.tasks[j],
                            "raw_episode_returns": returns,
                            "raw_return_mean": float(np.mean(returns)),
                            "raw_return_std": float(np.std(returns)),
                            "evaluation_rng_seed": rng_seed,
                            "environment_base_seed": (config.seed + j * 1000 + 100) % (2**32),
                            "environment_rng_protocol": "seed_once_then_advance",
                            "actor_policy": "argmax", "latent_policy": "upstream_stochastic",
                            "evaluation_transitions_enter_training": False,
                            **counters(),
                        }
                        append_json(output / "evaluation.jsonl", row)
                        evaluations.append(row)
                accounting = history.accounting() if memory_pair else replay_accounting(history.episodes)
                collected, retained = counters()["agent_decisions"], accounting["collected_transitions_retained"]
                if memory_pair and accounting["collected_transitions"] != collected:
                    raise RuntimeError("Collector/replay interaction counters differ")
                capacity = config.history_capacity_transitions if memory_pair else None
                if capacity is None and retained != collected:
                    raise RuntimeError("Full-history invariant failed: collected transitions were lost")
                if capacity is not None and retained > capacity:
                    raise RuntimeError("Bounded history exceeded the declared transition capacity")
                write_json(output / "replay_accounting.json", accounting)
                write_json(output / "progress.json", {
                    "phase_id": phase_id, "task_name": task, "eval_round": eval_round,
                    "phase_online_agent_decisions": completed, **counters(),
                })
                eval_round += 1

            run_phase_chunks(config, tuple(range(phase_id)), train=train_chunk,
                             rehearse=rehearse, evaluate=evaluate)
            if memory_pair:
                history.finish_phase(phase_id)
                history.flush()
                replay_storage = {"retained_history": history.accounting(include_index=True)}
            else:
                history.finish_phase(phase_id, keys_before_online)
                # Original reference archive path; never used by either pair arm.
                own_with_prefill = {k: v for k, v in history.episodes.items() if k not in phase_start_keys}
                tools.save_episodes(train_dir, own_with_prefill)
                archive.archive(history.episodes, set())
                replay_storage = {"pixel_mmap": archive.accounting()}
            write_json(output / "replay_archive.json", {
                **replay_storage,
                "training_npz": episode_archive_accounting(train_dir),
                "evaluation_npz_separate_from_replay": {
                    str(j): episode_archive_accounting(output / f"eval_eps_{j:02d}")
                    for j in range(phase_id + 1)
                },
            })
            checkpoint = output / f"analysis_phase_{phase_id:02d}.pt"
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({
                "artifact_kind": "analysis_snapshot", "resumable": False,
                "omitted": ["optimizer states", "RNG states", "environment state", "sampler state"],
                "agent_state_dict": agent.state_dict(), "phase_id": phase_id,
                "protocol": config.as_dict(), "counters": counters(),
            }, temporary)
            os.replace(temporary, checkpoint)
            checkpoint.with_suffix(".pt.sha256").write_text(f"{sha256(checkpoint)}  {checkpoint.name}\n")
            phase_records.append({"phase_id": phase_id, "task_name": task,
                                  "train_environment_seed": train_seed, **counters()})

        expected = config.projected_budgets()
        actual = counters()
        for field in ("agent_decisions", "world_model_updates"):
            if expected[field] != actual[field]:
                raise RuntimeError(f"Budget accounting mismatch: {field}: {actual[field]} vs {expected[field]}")
        if actual["rehearsal_actor_updates"] != expected["rehearsal_updates"]:
            raise RuntimeError("Rehearsal update count differs from the declared schedule")
        final = []
        for j, task in enumerate(config.tasks):
            rows = [r for r in evaluations if r["phase_id"] == len(config.tasks)-1 and r["task_index"] == j]
            final.append({
                "task_name": task, "last_checkpoint_raw_return_mean": rows[-1]["raw_return_mean"],
                "last_rounds_raw_return_mean": float(np.mean([r["raw_return_mean"] for r in rows[-config.final_average_rounds:]])),
                "averaged_eval_rounds": [r["eval_round"] for r in rows[-config.final_average_rounds:]],
            })
        write_json(output / "summary.json", {
            "schema_version": 1, "protocol": config.protocol,
            "history_arm": config.history_arm if memory_pair else "full_reference",
            "history_capacity_transitions": config.history_capacity_transitions if memory_pair else None,
            "classification": config.classification, "paper_reproduction_claim": False,
            "final_per_task": final, "phase_records": phase_records,
            "counters": actual, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        # Report close failures rather than suppressing them as a successful run.
        for handle in all_handles:
            handle.close()
        logger._writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.run_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
