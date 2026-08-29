"""Launcher contracts for the from-scratch Evolving-Core protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


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
        ORIGINAL_SIX_MINIMUM_FREE_BYTES,
        ORIGINAL_SIX_TASK_PROTOCOL,
        PROTOCOL,
        TASK_ORDERS,
        _budget_manifest,
        _protocol_for_task_order,
        _resolved_config,
        _storage_preflight,
        _training_command,
    )


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EvolvingAtomicRssmLauncherTests(unittest.TestCase):
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

    def test_every_predeclared_order_builds_the_same_fixed_method(self) -> None:
        source = self._source()
        for order_name, expected_order in TASK_ORDERS.items():
            with self.subTest(order=order_name):
                data = _resolved_config(source, task_order=order_name)
                config = Config.from_dict(data)
                self.assertEqual(
                    tuple(task.name for task in config.esc.env_configs),
                    expected_order,
                )
                self.assertEqual(config.epochs, 90 * len(expected_order))
                self.assertEqual(config.rssm_num_experts, len(expected_order))
                self.assertTrue(config.uses_evolving_atomic_rssm)
                self.assertTrue(config.evolving_shared_core)
                self.assertTrue(config.uses_task_private_heads)
                self.assertFalse(config.uses_full_task_rssm_experts)
                self.assertEqual(
                    (config.current_batch_n, config.memory_batch_n), (12, 4)
                )
                self.assertEqual(config.boundary_consolidation_steps, 1000)

    def test_original_six_is_separately_named_and_preserves_arrow_order(self) -> None:
        expected = (
            "ALE/MsPacman-v5",
            "ALE/Boxing-v5",
            "ALE/CrazyClimber-v5",
            "ALE/Frostbite-v5",
            "ALE/Seaquest-v5",
            "ALE/Enduro-v5",
        )
        config = Config.from_dict(
            _resolved_config(self._source(), task_order="arrow-original-six")
        )

        self.assertEqual(
            tuple(task.name for task in config.esc.env_configs), expected
        )
        self.assertEqual(config.epochs, 540)
        self.assertEqual(config.rssm_num_experts, 6)
        self.assertEqual(config.evolving_checkpoint_retention, "latest_boundary")
        self.assertEqual(
            _protocol_for_task_order("arrow-original-six"),
            ORIGINAL_SIX_TASK_PROTOCOL,
        )
        self.assertEqual(
            _protocol_for_task_order("mspacman-boxing-crazyclimber"), PROTOCOL
        )

    def test_budget_ledger_separates_online_and_consolidation_compute(self) -> None:
        config = _resolved_config(
            self._source(), task_order="mspacman-boxing-crazyclimber"
        )
        budget = _budget_manifest(config)

        self.assertEqual(budget["online_world_model_updates"], 270_000)
        self.assertEqual(
            budget["boundary_consolidation_world_model_updates"], 3_000
        )
        self.assertEqual(budget["total_world_model_optimizer_steps"], 273_000)
        self.assertEqual(budget["online_current_sequences"], 3_600_000)
        self.assertEqual(budget["online_memory_sequences"], 720_000)
        self.assertEqual(budget["consolidation_sequences"], 48_000)
        self.assertTrue(budget["consolidation_is_extra_compute"])
        self.assertFalse(budget["evaluation_transitions_enter_replay"])

    def test_original_six_budget_is_explicit(self) -> None:
        config = _resolved_config(self._source(), task_order="arrow-original-six")
        budget = _budget_manifest(config)

        self.assertEqual(budget["task_count"], 6)
        self.assertEqual(budget["task_duration_epochs"], [90] * 6)
        self.assertEqual(budget["raw_environment_frames"], 35_389_440)
        self.assertEqual(budget["online_world_model_updates"], 540_000)
        self.assertEqual(
            budget["boundary_consolidation_world_model_updates"], 6_000
        )
        self.assertEqual(budget["total_world_model_optimizer_steps"], 546_000)
        self.assertEqual(budget["actor_critic_updates"], 432_000)
        self.assertEqual(budget["online_current_sequences"], 6_840_000)
        self.assertEqual(budget["online_memory_sequences"], 1_800_000)
        self.assertEqual(budget["consolidation_sequences"], 96_000)
        self.assertEqual(budget["checkpoint_retention"], "latest_boundary")
        self.assertEqual(
            budget["retained_boundary_replay_asset_bytes"], 6_442_450_944
        )
        self.assertEqual(
            budget["peak_boundary_replay_asset_bytes"], 12_884_901_888
        )
        self.assertEqual(
            budget["minimum_live_plus_peak_replay_observation_bytes"],
            19_327_352_832,
        )

    def test_original_six_storage_preflight_rejects_low_space(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            low_space = SimpleNamespace(
                total=100 * 1024**3,
                used=60 * 1024**3,
                free=40 * 1024**3,
            )
            with mock.patch(
                "run_evolving_atomic_rssm.shutil.disk_usage",
                return_value=low_space,
            ):
                with self.assertRaisesRegex(RuntimeError, "at least 48 GiB"):
                    _storage_preflight(
                        output_dir=root / "run",
                        replay_mmap_root=root / "replay",
                        task_order="arrow-original-six",
                    )

            enough_space = SimpleNamespace(
                total=100 * 1024**3,
                used=40 * 1024**3,
                free=60 * 1024**3,
            )
            with mock.patch(
                "run_evolving_atomic_rssm.shutil.disk_usage",
                return_value=enough_space,
            ):
                result = _storage_preflight(
                    output_dir=root / "run",
                    replay_mmap_root=root / "replay",
                    task_order="arrow-original-six",
                )
            self.assertEqual(
                result["required_output_free_bytes"],
                ORIGINAL_SIX_MINIMUM_FREE_BYTES,
            )

    def test_command_is_from_scratch_single_gpu_and_uncompiled(self) -> None:
        command = _training_command(
            python=Path("/env/bin/python"),
            config_path=Path("/run/config.json"),
            output_dir=Path("/run"),
            task_snapshot_dir=Path("/run/task_boundaries"),
            project_commit="a" * 40,
        )

        self.assertIn("--task-bank-snapshot-dir", command)
        self.assertIn("--evaluate-final", command)
        self.assertIn("--fused-adam", command)
        self.assertNotIn("--compile-world-model", command)
        self.assertNotIn("--init-task1-boundary-snapshot", command)
        self.assertNotIn("--init-analysis-snapshot", command)


if __name__ == "__main__":
    unittest.main()
