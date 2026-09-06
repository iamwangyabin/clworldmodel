#!/usr/bin/env python3
"""Launch the unmodified official learning components on a named Atari protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dream_rehearsal_reference_support import ROOT, sha256, verify_reference_sources
from git_provenance import git_state, require_synced_training_git_state
from launcher_support import run_and_tee, write_json

sys.path.insert(0, str(ROOT / "src"))
from clworldmodel.reference.dream_rehearsal import (
    ARROW_TRANSITION_CAPACITY, MemoryPairConfig, OfficialDreamRehearsalConfig,
)


def main(*, config_type=OfficialDreamRehearsalConfig) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Strict protocol JSON; paths are repository-relative")
    parser.add_argument("--seed", type=int, help="Actual RNG seed; omitted means the config's seed")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--device", choices=("cuda:0", "cpu"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two-task, small-interaction wiring run; learning constants unchanged")
    parser.add_argument("--dry-run", action="store_true")
    if config_type is MemoryPairConfig:
        parser.add_argument("--history", choices=("full", "bounded"),
                            help="Only memory retention changes; bounded capacity is 524288 transitions")
    args = parser.parse_args()
    if args.config and args.smoke:
        parser.error("Use either --config or --smoke, not both")

    def root_path(path: Path) -> Path:
        path = path.expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    if args.config:
        data = json.loads(root_path(args.config).read_text())
    else:
        preset = config_type.smoke() if args.smoke else config_type()
        data = preset.as_dict()
    for name in ("seed", "cpu_threads", "device"):
        if getattr(args, name) is not None:
            data[name] = getattr(args, name)
    if config_type is MemoryPairConfig and args.history is not None:
        data["history_capacity_transitions"] = None if args.history == "full" else ARROW_TRANSITION_CAPACITY
    config = config_type.from_dict(data)
    name = (f"dream_rehearsal_memory_pair_{config.history_arm}" if isinstance(config, MemoryPairConfig)
            else "dream_rehearsal_official_atari")
    output = root_path(args.output_dir or Path("runs") / f"{name}_{config.classification}_seed{config.seed}")
    python = root_path(args.python)
    command = [str(python), str(ROOT / "scripts/train_dream_rehearsal_official_atari.py"),
               "--run-dir", str(output)]
    reference_sources = verify_reference_sources()
    if args.dry_run:
        project_git = git_state(ROOT)
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {output}")
        # A stale local tracking ref is not evidence that a commit is pushed.
        subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
        project_git = require_synced_training_git_state(ROOT)

    manifest = {
        "schema_version": 1, "method": config.protocol, "protocol": config.protocol,
        "classification": config.classification, "config": config.as_dict(),
        "project_git": project_git, "reference_sources": reference_sources,
        "dependency_constraints": {
            "path": "requirements/dream_rehearsal_official.txt",
            "sha256": sha256(ROOT / "requirements/dream_rehearsal_official.txt"),
            "isolated_environment_required": True,
        },
        "projected_budgets": config.projected_budgets(),
        "counter_schema": {
            "version": 1,
            "upstream_metrics_jsonl_step": "action_repeat * cumulative online_agent_decisions; excludes prefill and reset no-ops",
            "project_jsonl_counters": "explicit cumulative training/evaluation decisions, actual frames, and optimizer calls",
        },
        "replay": {
            "retention": "never_clear_shared_episode_history", "dataset_size": 0,
            "ordinary_training": "shared_full_history",
            "sampler": "unmodified_nm512_length_weighted_episode_sampler",
            "sampler_rng_seed": 0,
            "rehearsal_sampling": "each_prior_phase_own_online_episodes",
            "observation_dtype": "uint8", "storage_device": "cpu_mmap_inactive_episodes",
            "disk_storage": "one_append_only_pixel_mmap_plus_upstream_lossless_npz_episodes",
            "live_history_observation_bytes_lower_bound": (
                config.projected_budgets()["agent_decisions"] * config.image_size ** 2 * 3
            ),
            "lower_bound_excludes": ["reset observations", "actions/rewards/flags", "Python containers", "indexing"],
            "task_labels_in_model_inputs": False,
        },
        "claims": {
            "paper_minigrid_reproduction": False,
            "learning_source_byte_identical": True,
            "compute_matched_to_arrow": False,
            "storage_matched_to_arrow": False,
            "cause_of_old_failures_proven": False,
        },
        "declared_adaptations": [
            "Six Atari environments via Gymnasium/ALE, full 18 actions, RGB64, repeat4, raw rewards.",
            "Fixed per-task interaction cap including random prefill; no MiniGrid 0.6 all-good early stop.",
            "No phantom 541st epoch/task-0 revisit. A final short fragment never creates an extra rehearsal event.",
            "Unbounded dataset_size=0; stock NM512's 1M-transition default would evict Atari history.",
            "Inactive uint8 images use one append-only CPU mmap; sampler input parity is tested, no episodes are removed.",
            "Evaluation RNG is isolated from training; per-episode raw returns and actual frames are recorded.",
            "Model snapshots are analysis-only, not resumable; run directories cannot be overwritten or resumed.",
            "Author did not publish an NM512 commit; our explicit pin is not a verified author-environment pin.",
        ],
        "command": command, "output_dir": str(output),
    }
    if isinstance(config, MemoryPairConfig):
        common = config.as_dict()
        common.pop("history_capacity_transitions")
        algorithm_hash = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
        maximum = config.history_capacity_transitions or config.projected_budgets()["agent_decisions"]
        manifest["memory_comparison_contract"] = {
            "arm": config.history_arm, "only_arm_specific_config_key": "history_capacity_transitions",
            "shared_algorithm_and_schedule_sha256": algorithm_hash,
            "separate_models_or_trainers_per_arm": False,
            "retention_rng_seed": config.seed, "retention_rng": "independent Python Random instance",
        }
        manifest["replay"].update({
            "retention": "all_transition_blocks" if config.history_arm == "full" else "uniform_algorithm_r_transition_blocks",
            "transition_capacity": config.history_capacity_transitions,
            "retention_block_transitions": config.retention_block_transitions,
            "ordinary_training": "shared_retained_history",
            "rehearsal_sampling": "live_views_of_retained_prior_phase_online_data",
            "storage_device": "cpu_fixed_slot_mmaps",
            "disk_storage": "same_retention_store_only; no full training NPZ backup",
            "live_history_observation_bytes_lower_bound": maximum * config.image_size ** 2 * 3,
            "phase_libraries_can_retain_evicted_samples": False,
            "sampler_preserves_contiguous_original_episodes": True,
        })
        manifest["claims"]["sample_capacity_matched_to_arrow"] = config.history_arm == "bounded"
        manifest["declared_adaptations"] = [s for s in manifest["declared_adaptations"]
            if not s.startswith(("Unbounded dataset_size", "Inactive uint8 images"))]
        manifest["declared_adaptations"].extend([
            "Both arms use one streamed episode-preserving history store and one collection adapter; uncapped source-sampler and collector parity are tested.",
            "Only the history cap changes: retain all blocks or an unbiased reservoir of 1024 blocks of 512 real transitions.",
            "Context/reset observations are additional byte overhead, not additional stored transitions; all memory and current/reset context are accounted separately.",
            "No complete training NPZ archive or independent frozen rehearsal library can bypass the bounded cap.",
        ])
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(config.seed)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(config.cpu_threads)
    output.mkdir(parents=True)
    manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "launch.json", manifest)
    write_json(output / "protocol_config.json", config.as_dict())
    try:
        code = run_and_tee(command, cwd=ROOT, env=env, log_path=output / "train.log")
    except (OSError, KeyboardInterrupt) as exc:
        write_json(output / "run_status.json", {
            "complete": False, "error": str(exc), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise
    write_json(output / "run_status.json", {
        "complete": code == 0, "return_code": code,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if code:
        raise subprocess.CalledProcessError(code, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
