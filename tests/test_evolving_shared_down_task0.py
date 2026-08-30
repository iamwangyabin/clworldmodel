"""Contracts for the shared-frozen-down Evolving-Core Task-0 pilot."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


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
    from run_evolving_atomic_rssm import (
        DEFAULT_MECHANISM_PROFILE,
        SHARED_DOWN_PARAMETERIZATION,
        _mechanism_capacity_manifest,
        _resolved_config,
    )
    from run_evolving_shared_down_task0 import (
        TASK_ORDER,
        _budget_manifest,
        _resolved_pilot_config,
        _training_command,
    )


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EvolvingSharedDownTask0Tests(unittest.TestCase):
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

    def test_pilot_changes_only_parameterization_and_stop_epoch(self) -> None:
        dense = _resolved_config(self._source(), task_order="arrow-original-six")
        pilot = _resolved_pilot_config(self._source())
        differing = {key for key in dense if dense[key] != pilot[key]}

        self.assertEqual(
            differing, {"epochs", "task_mechanism_parameterization"}
        )
        config = Config.from_dict(pilot)
        self.assertEqual(config.epochs, 90)
        self.assertEqual(config.rssm_num_experts, 6)
        self.assertEqual(
            tuple(task.name for task in config.esc.env_configs), TASK_ORDER
        )
        self.assertEqual(
            config.task_mechanism_parameterization,
            SHARED_DOWN_PARAMETERIZATION,
        )
        self.assertEqual(
            (
                config.task_mechanism_recurrent_width,
                config.task_mechanism_representation_width,
                config.task_mechanism_transition_width,
            ),
            (512, 512, 256),
        )

    def test_capacity_ledger_counts_shared_basis_once(self) -> None:
        capacity = _mechanism_capacity_manifest(
            task_count=6,
            mechanism_profile=DEFAULT_MECHANISM_PROFILE,
            mechanism_parameterization=SHARED_DOWN_PARAMETERIZATION,
        )
        self.assertEqual(
            capacity["parameters_per_task"],
            {
                "recurrent": 264_704,
                "representation_posterior": 535_552,
                "transition_prior": 264_704,
                "total": 1_064_960,
            },
        )
        self.assertEqual(
            capacity["shared_frozen_down_parameters"],
            {
                "recurrent": 262_656,
                "representation_posterior": 2_359_808,
                "transition_prior": 131_328,
                "total": 2_753_792,
            },
        )
        self.assertEqual(capacity["private_mechanism_parameters"], 6_389_760)
        self.assertEqual(capacity["reuse_route_parameters"], 180)
        self.assertEqual(capacity["mechanism_and_route_parameters"], 9_143_732)

    def test_pilot_budget_and_command_exclude_heldout_final(self) -> None:
        config = _resolved_pilot_config(self._source())
        budget = _budget_manifest(config)
        self.assertEqual(budget["route_allocation_task_count"], 6)
        self.assertEqual(budget["tasks_trained"], 1)
        self.assertEqual(budget["raw_environment_frames"], 5_898_240)
        self.assertEqual(budget["online_world_model_updates"], 90_000)
        self.assertEqual(
            budget["boundary_consolidation_world_model_updates"], 1_000
        )
        self.assertEqual(budget["actor_critic_updates"], 72_000)
        self.assertFalse(budget["heldout_final_evaluation_performed"])

        command = _training_command(
            python=Path("/env/python"),
            config_path=Path("/run/config.json"),
            output_dir=Path("/run"),
            task_snapshot_dir=Path("/run/snapshots"),
            project_commit="a" * 40,
        )
        self.assertNotIn("--evaluate-final", command)
        self.assertIn("--profile-stages", command)

    def test_shared_down_is_not_a_baseline_or_compact_side_effect(self) -> None:
        invalid = self._source()
        invalid["task_mechanism_parameterization"] = SHARED_DOWN_PARAMETERIZATION
        with self.assertRaisesRegex(ValueError, "only for Evolving-Core"):
            Config.from_dict(invalid)

        with self.assertRaisesRegex(ValueError, "preserves matched_512"):
            _resolved_config(
                self._source(),
                task_order="arrow-original-six",
                mechanism_profile="compact_128_128_64",
                mechanism_parameterization=SHARED_DOWN_PARAMETERIZATION,
            )


if __name__ == "__main__":
    unittest.main()
