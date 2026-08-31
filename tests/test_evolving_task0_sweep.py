"""Contracts for the fixed-order Evolving-Core Task-0 hyperparameter sweep."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)
sys.path.insert(0, str(SCRIPTS))

try:
    import sortedcontainers  # noqa: F401
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - minimal hosts omit experiment deps.
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(VENDORED_ATARI))
    from config import Config
    import run_preemptible_evolving_task0_queue as preemptible_queue
    from run_evolving_atomic_rssm import _resolved_config
    from run_evolving_task0_sweep import (
        BASELINE_HPARAMETERS,
        DURATION_PROFILE_EPOCHS,
        ENV16_PROTOCOL,
        PROFILE_OVERRIDES,
        TASK_ORDER,
        _budget_manifest,
        _resolved_sweep_config,
        _training_command,
    )
    from select_evolving_task0_profile import _select


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EvolvingTask0SweepTests(unittest.TestCase):
    @staticmethod
    def _source() -> dict:
        path = (
            ROOT
            / "third_party"
            / "arrow"
            / "Configs"
            / "Atari configs"
            / "CL-task configs"
            / "Original Order"
            / (
                "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,"
                "ALE_Seaquest,ALE_Enduro-s0-arrow.json"
            )
        )
        # Match the launcher input exactly: published JSON may omit schema
        # defaults that a round-trip through Config.to_dict() would materialize.
        return json.loads(path.read_text(encoding="utf-8"))

    def test_each_profile_changes_exactly_one_preregistered_hyperparameter(self) -> None:
        fixed = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            task0_profile="fixed_v1",
        )
        for profile, override in PROFILE_OVERRIDES.items():
            with self.subTest(profile=profile):
                data = _resolved_sweep_config(self._source(), profile=profile)
                config = Config.from_dict(data)
                self.assertEqual(config.epochs, 90)
                self.assertEqual(config.evolving_task0_profile, profile)
                self.assertEqual(data["ac_lr"], config.ac_lr)
                self.assertEqual(
                    tuple(task.name for task in config.esc.env_configs), TASK_ORDER
                )
                changes = {
                    name: data[name]
                    for name in BASELINE_HPARAMETERS
                    if data[name] != fixed[name]
                }
                self.assertEqual(changes, override)

    def test_nondefault_profile_must_stop_at_first_boundary(self) -> None:
        data = _resolved_sweep_config(
            self._source(), profile="task0_shared_lr_3e4"
        )
        data["epochs"] = 270
        with self.assertRaisesRegex(ValueError, "stop exactly"):
            Config.from_dict(data)

    def test_duration_profiles_change_only_task0_acquisition_budget(self) -> None:
        for profile, duration in DURATION_PROFILE_EPOCHS.items():
            with self.subTest(profile=profile):
                data = _resolved_sweep_config(self._source(), profile=profile)
                config = Config.from_dict(data)
                self.assertEqual(config.epochs, duration)
                self.assertEqual(config.evolving_task0_profile, profile)
                self.assertNotIn("swap_sched", data["esc"]["kwargs"])
                self.assertEqual(
                    data["esc"]["kwargs"]["task_durations"],
                    [duration, 90, 90],
                )
                self.assertEqual(
                    {
                        name: data[name] for name in BASELINE_HPARAMETERS
                    },
                    BASELINE_HPARAMETERS,
                )

    def test_duration_profile_rejects_schedule_drift(self) -> None:
        data = _resolved_sweep_config(
            self._source(), profile="task0_epochs_180"
        )
        data["esc"]["kwargs"]["task_durations"][0] = 179
        with self.assertRaisesRegex(ValueError, "exact task durations"):
            Config.from_dict(data)

    def test_profile_rejects_undeclared_parameter_drift(self) -> None:
        data = _resolved_sweep_config(
            self._source(), profile="task0_shared_lr_1e4"
        )
        data["task_private_lr"] = 3e-4
        with self.assertRaisesRegex(ValueError, "fixed optimizer"):
            Config.from_dict(data)

    def test_sweep_budget_is_task0_only_and_explicit(self) -> None:
        config = _resolved_sweep_config(
            self._source(), profile="task0_private_lr_3e4"
        )
        budget = _budget_manifest(config)
        self.assertEqual(budget["raw_environment_frames"], 5_898_240)
        self.assertEqual(budget["online_world_model_updates"], 90_000)
        self.assertEqual(budget["boundary_consolidation_world_model_updates"], 1_000)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 91_000)
        self.assertEqual(budget["actor_critic_updates"], 72_000)
        self.assertEqual(budget["online_current_sequences"], 1_440_000)
        self.assertEqual(budget["online_memory_sequences"], 0)
        self.assertEqual(budget["consolidation_sequences"], 16_000)
        self.assertFalse(budget["heldout_final_evaluation_performed"])

    def test_env16_preserves_budget_with_explicit_trajectory_partition(self) -> None:
        data = _resolved_sweep_config(
            self._source(),
            profile="task0_private_lr_3e4",
            collection_envs=16,
        )
        config = Config.from_dict(data)
        budget = _budget_manifest(data)
        self.assertEqual(config.n_sync, 16)
        self.assertEqual(config.gen_seq_len, 1024)
        self.assertEqual(config.n_sync * config.gen_seq_len, 16_384)
        self.assertEqual(budget["raw_environment_frames"], 5_898_240)
        self.assertEqual(budget["online_world_model_updates"], 90_000)
        self.assertEqual(budget["actor_critic_updates"], 72_000)

    def test_env16_has_a_complete_standalone_fixed_control(self) -> None:
        data = _resolved_sweep_config(
            self._source(), profile="fixed_v1", collection_envs=16
        )
        config = Config.from_dict(data)
        self.assertEqual(config.epochs, 90)
        self.assertEqual(config.evolving_task0_profile, "fixed_v1")
        self.assertEqual(config.n_sync, 16)
        self.assertEqual(
            {name: data[name] for name in BASELINE_HPARAMETERS},
            BASELINE_HPARAMETERS,
        )
        with self.assertRaisesRegex(ValueError, "standalone fixed_v1"):
            _resolved_sweep_config(self._source(), profile="fixed_v1")

    def test_env16_does_not_redefine_the_duration_sweep(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen at four"):
            _resolved_sweep_config(
                self._source(), profile="task0_epochs_120", collection_envs=16
            )

    def test_preemptible_queue_distinguishes_owned_and_external_cuda_pids(self) -> None:
        gpu = {
            "memory_used_mib": 1000,
            "compute_pids": [{"pid": 11}, {"pid": 22}],
        }
        with patch.object(
            preemptible_queue,
            "_pid_process_group",
            side_effect=lambda pid: {11: 123, 22: 999}[pid],
        ):
            self.assertEqual(
                preemptible_queue._external_compute_pids(
                    gpu, owned_process_group=123
                ),
                [22],
            )
        self.assertFalse(
            preemptible_queue._gpu_is_idle(gpu, idle_memory_mib=64)
        )
        self.assertTrue(
            preemptible_queue._gpu_is_idle(
                {"memory_used_mib": 1, "compute_pids": []},
                idle_memory_mib=64,
            )
        )

    def test_duration_budget_scales_samples_and_updates_explicitly(self) -> None:
        config = _resolved_sweep_config(
            self._source(), profile="task0_epochs_180"
        )
        budget = _budget_manifest(config)
        self.assertEqual(budget["task_duration_epochs"], 180)
        self.assertEqual(budget["raw_environment_frames"], 11_796_480)
        self.assertEqual(budget["online_world_model_updates"], 180_000)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 181_000)
        self.assertEqual(budget["actor_critic_updates"], 144_000)
        self.assertEqual(budget["online_current_sequences"], 2_880_000)

    def test_sweep_command_never_requests_heldout_final(self) -> None:
        command = _training_command(
            python=Path("/env/bin/python"),
            config_path=Path("/run/config.json"),
            output_dir=Path("/run"),
            task_snapshot_dir=Path("/run/task_boundaries"),
            project_commit="a" * 40,
        )
        self.assertNotIn("--evaluate-final", command)
        self.assertIn("--fused-adam", command)
        self.assertNotIn("--compile-world-model", command)

    def test_selection_uses_preconsolidation_score_and_fixed_tie_break(self) -> None:
        scores = {
            "fixed_v1": 10.0,
            "task0_shared_lr_1e4": 20.0,
            "task0_shared_lr_3e4": 20.0,
            "task0_private_lr_3e4": 15.0,
            "task0_actor_lr_2e4": 5.0,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dirs = []
            for profile, score in scores.items():
                run_dir = root / profile
                validation_dir = run_dir / "evolving_core_consolidation"
                validation_dir.mkdir(parents=True)
                config = dict(BASELINE_HPARAMETERS)
                config.update(PROFILE_OVERRIDES.get(profile, {}))
                config.update(
                    {
                        "evolving_task0_profile": profile,
                        "epochs": 270 if profile == "fixed_v1" else 90,
                        "esc": {"kwargs": {"swap_sched": 90}},
                    }
                )
                if profile == "fixed_v1":
                    # The already-running legacy control predates explicit
                    # materialization of the schema-default Actor LR.
                    config.pop("ac_lr")
                (run_dir / "launch.json").write_text(
                    json.dumps(
                        {
                            "task_order": list(TASK_ORDER),
                            "seed_index": 0,
                            "project_git": {"commit": profile},
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "resolved_training_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                (validation_dir / "task_00_pre_validation.json").write_text(
                    json.dumps(
                        {
                            "validation": {
                                "raw_mean": [score],
                                "task_seeds": [12345],
                            },
                            "rollouts_per_task": 16,
                            "heldout_final_data_used": False,
                        }
                    ),
                    encoding="utf-8",
                )
                # A deliberately more favorable held-out value must be ignored.
                (run_dir / "final_evaluation.json").write_text(
                    json.dumps({"raw_mean": [1_000_000.0]}), encoding="utf-8"
                )
                if profile != "fixed_v1":
                    (run_dir / "run_status.json").write_text(
                        json.dumps({"complete": True}), encoding="utf-8"
                    )
                candidate_dirs.append(run_dir)

            selection = _select(candidate_dirs)

        self.assertEqual(
            selection["winner"]["profile"], "task0_shared_lr_3e4"
        )
        self.assertEqual(selection["winner"]["score"], 20.0)
        self.assertFalse(selection["heldout_final_data_read"])

    def test_duration_selection_prefers_shortest_near_best_budget(self) -> None:
        scores = {
            "fixed_v1": 100.0,
            "task0_epochs_120": 120.0,
            "task0_epochs_150": 124.0,
            "task0_epochs_180": 125.0,
            "task0_epochs_240": 123.0,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dirs = []
            for profile, score in scores.items():
                duration = DURATION_PROFILE_EPOCHS.get(profile, 90)
                run_dir = root / profile
                validation_dir = run_dir / "evolving_core_consolidation"
                validation_dir.mkdir(parents=True)
                schedule = (
                    {"swap_sched": 90}
                    if profile == "fixed_v1"
                    else {"task_durations": [duration, 90, 90]}
                )
                config = {
                    **BASELINE_HPARAMETERS,
                    "evolving_task0_profile": profile,
                    "epochs": 270 if profile == "fixed_v1" else duration,
                    "esc": {"kwargs": schedule},
                }
                (run_dir / "launch.json").write_text(
                    json.dumps(
                        {
                            "task_order": list(TASK_ORDER),
                            "seed_index": 0,
                            "project_git": {"commit": profile},
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "resolved_training_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                (validation_dir / "task_00_pre_validation.json").write_text(
                    json.dumps(
                        {
                            "validation": {
                                "raw_mean": [score],
                                "task_seeds": [12345],
                            },
                            "rollouts_per_task": 16,
                            "heldout_final_data_used": False,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "final_evaluation.json").write_text(
                    json.dumps({"raw_mean": [1_000_000.0]}), encoding="utf-8"
                )
                if profile != "fixed_v1":
                    (run_dir / "run_status.json").write_text(
                        json.dumps({"complete": True}), encoding="utf-8"
                    )
                candidate_dirs.append(run_dir)

            selection = _select(candidate_dirs, family="duration")

        self.assertEqual(selection["maximum_observed_score"], 125.0)
        self.assertEqual(selection["near_best_score_threshold"], 118.75)
        self.assertEqual(selection["winner"]["profile"], "task0_epochs_120")
        self.assertEqual(
            [row["task0_acquisition_epochs"] for row in selection["learning_curve"]],
            [90, 120, 150, 180, 240],
        )
        self.assertFalse(selection["heldout_final_data_read"])

    def test_env16_selection_requires_one_complete_matched_cohort(self) -> None:
        scores = {
            "fixed_v1": 10.0,
            "task0_shared_lr_1e4": 20.0,
            "task0_shared_lr_3e4": 30.0,
            "task0_private_lr_3e4": 15.0,
            "task0_actor_lr_2e4": 5.0,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dirs = []
            for profile, score in scores.items():
                run_dir = root / profile
                validation_dir = run_dir / "evolving_core_consolidation"
                validation_dir.mkdir(parents=True)
                config = {
                    **BASELINE_HPARAMETERS,
                    **PROFILE_OVERRIDES.get(profile, {}),
                    "evolving_task0_profile": profile,
                    "epochs": 90,
                    "n_sync": 16,
                    "gen_seq_len": 1024,
                    "esc": {"kwargs": {"swap_sched": 90}},
                }
                (run_dir / "launch.json").write_text(
                    json.dumps(
                        {
                            "protocol": ENV16_PROTOCOL,
                            "task_order": list(TASK_ORDER),
                            "seed_index": 0,
                            "project_git": {"commit": "a" * 40},
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "resolved_training_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                (run_dir / "run_status.json").write_text(
                    json.dumps({"complete": True}), encoding="utf-8"
                )
                (validation_dir / "task_00_pre_validation.json").write_text(
                    json.dumps(
                        {
                            "validation": {
                                "raw_mean": [score],
                                "task_seeds": [12345],
                            },
                            "rollouts_per_task": 16,
                            "heldout_final_data_used": False,
                        }
                    ),
                    encoding="utf-8",
                )
                candidate_dirs.append(run_dir)

            selection = _select(candidate_dirs)
            self.assertEqual(selection["protocol"], ENV16_PROTOCOL)
            self.assertEqual(selection["collection_envs"], 16)
            self.assertEqual(selection["winner"]["profile"], "task0_shared_lr_3e4")

            fixed_status = root / "fixed_v1" / "run_status.json"
            fixed_status.write_text(
                json.dumps({"complete": False}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                _select(candidate_dirs)
            fixed_status.write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )

            mixed_launch = root / "task0_actor_lr_2e4" / "launch.json"
            launch = json.loads(mixed_launch.read_text(encoding="utf-8"))
            launch["protocol"] = "legacy-control"
            mixed_launch.write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot mix protocols"):
                _select(candidate_dirs)


if __name__ == "__main__":
    unittest.main()
