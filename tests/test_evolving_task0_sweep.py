"""Contracts for the fixed-order Evolving-Core Task-0 hyperparameter sweep."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


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
    from run_evolving_atomic_rssm import _resolved_config
    from run_evolving_task0_sweep import (
        BASELINE_HPARAMETERS,
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
        return Config.from_file(path).to_dict()

    def test_each_profile_changes_exactly_one_preregistered_hyperparameter(self) -> None:
        fixed = _resolved_config(
            self._source(), task_order="mspacman-boxing-crazyclimber"
        )
        for profile, override in PROFILE_OVERRIDES.items():
            with self.subTest(profile=profile):
                data = _resolved_sweep_config(self._source(), profile=profile)
                config = Config.from_dict(data)
                self.assertEqual(config.epochs, 90)
                self.assertEqual(config.evolving_task0_profile, profile)
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
                    }
                )
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


if __name__ == "__main__":
    unittest.main()
