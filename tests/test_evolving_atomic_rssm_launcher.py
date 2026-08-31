"""Launcher contracts for the from-scratch Evolving-Core protocol."""

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
        SHARED_FASTKAN_STABLE_BEHAVIOR,
        TASK_ORDERS,
        _behavior_update_budget,
        _budget_manifest,
        _parameter_manifest,
        _resolved_config,
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
                self.assertEqual(config.epochs, 270)
                self.assertTrue(config.uses_evolving_atomic_rssm)
                self.assertTrue(config.evolving_shared_core)
                self.assertEqual(config.evolving_task0_profile, "fixed_v2")
                self.assertEqual(config.first_task_shared_core_lr, 3e-4)
                self.assertEqual(config.shared_core_lr, 1e-4)
                self.assertTrue(config.uses_task_private_heads)
                self.assertFalse(config.uses_full_task_rssm_experts)
                self.assertEqual(
                    (config.current_batch_n, config.memory_batch_n), (12, 4)
                )
                self.assertEqual(config.boundary_consolidation_steps, 1000)

    def test_fixed_v1_remains_available_without_redefining_it(self) -> None:
        v1_data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            task0_profile="fixed_v1",
        )
        v2_data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            task0_profile="fixed_v2",
        )
        config = Config.from_dict(v1_data)

        self.assertEqual(config.epochs, 270)
        self.assertEqual(config.evolving_task0_profile, "fixed_v1")
        self.assertEqual(config.first_task_shared_core_lr, 2e-4)
        self.assertEqual(config.shared_core_lr, 1e-4)
        self.assertEqual(
            {
                name: (v1_data[name], v2_data[name])
                for name in v1_data
                if v1_data[name] != v2_data[name]
            },
            {
                "evolving_task0_profile": ("fixed_v1", "fixed_v2"),
                "first_task_shared_core_lr": (2e-4, 3e-4),
            },
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

    def test_shared_down_fastkan_is_one_persistent_stable_behavior_pair(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
        )
        config = Config.from_dict(data)

        self.assertEqual(
            config.continual_method,
            "evolving_atomic_rssm_shared_fastkan_arrow",
        )
        self.assertTrue(config.uses_evolving_atomic_rssm)
        self.assertTrue(config.uses_shared_actor)
        self.assertTrue(config.uses_replay_rehearsed_shared_behavior)
        self.assertFalse(config.task_private_actor_critic)
        self.assertEqual(
            config.task_mechanism_parameterization,
            "shared_frozen_down_film",
        )
        self.assertEqual(config.actor_network, "fast_kan_ac_stable")
        self.assertEqual(config.fastkan_hidden_features, 53)
        self.assertEqual(config.ac_optimizer, "laprop")
        self.assertEqual(config.ac_lr, 4e-5)
        self.assertEqual(config.ac_replay_critic_loss_scale, 0.3)
        self.assertTrue(config.ac_use_slow_critic_targets)
        self.assertTrue(config.ac_corrected_imagination_bootstrap)
        self.assertEqual(
            config.evolving_shared_behavior_current_task_fraction, 0.75
        )
        self.assertFalse(config.shared_actor_imagination_distillation)

        data["evolving_shared_behavior_current_task_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "fixed optimizer, replay, interface"):
            Config.from_dict(data)

        data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
        )
        data["task_mechanism_parameterization"] = "dense_private"
        with self.assertRaisesRegex(ValueError, "requires mechanism parameterization"):
            Config.from_dict(data)

        legacy = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
        )
        legacy["task_mechanism_parameterization"] = "shared_frozen_down_film"
        with self.assertRaisesRegex(ValueError, "requires mechanism parameterization"):
            Config.from_dict(legacy)

    def test_shared_behavior_rehearsal_preserves_the_216k_update_budget(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
        )

        self.assertEqual(
            _behavior_update_budget(data),
            {"0": 99_000, "1": 63_000, "2": 54_000},
        )
        budget = _budget_manifest(data)
        self.assertEqual(budget["actor_critic_updates"], 216_000)
        self.assertEqual(
            sum(budget["actor_critic_updates_by_task_route"].values()),
            216_000,
        )
        self.assertFalse(
            budget["shared_behavior_rehearsal_adds_optimizer_steps"]
        )

    def test_shared_fastkan_parameter_ledger_combines_shared_down_and_behavior(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="mspacman-boxing-crazyclimber",
            behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
        )
        parameters = _parameter_manifest(data)

        self.assertEqual(parameters["world_model_parameters"], 42_675_539)
        self.assertEqual(
            parameters["mechanism_parameterization"],
            "shared_frozen_down_film",
        )
        self.assertEqual(parameters["shared_frozen_down_parameters"], 2_753_792)
        self.assertEqual(parameters["behavior_parameters"], 1_700_670)
        self.assertEqual(parameters["online_parameters"], 44_376_209)
        self.assertEqual(
            parameters["comparison_to_matched_world_model_private_mlp"][
                "difference"
            ],
            -3_444_213,
        )
        self.assertEqual(
            parameters["comparison_to_dense_evolving_v2_private_mlp"][
                "difference"
            ],
            -8_944_117,
        )
        self.assertEqual(
            parameters["comparison_to_arrow_50"]["difference"],
            23_162_395,
        )
        self.assertEqual(
            parameters["per_task_world_model_additions"],
            {"0": 1_099_200, "1": 9_661_841, "2": 9_661_853},
        )
        self.assertEqual(
            parameters["training_only_behavior_copies"][
                "peak_behavior_parameters_excluding_optimizer"
            ],
            3_401_340,
        )

    def test_shared_fastkan_rejects_relabeling_the_v1_world_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "inherits the fixed_v2"):
            _resolved_config(
                self._source(),
                task_order="mspacman-boxing-crazyclimber",
                task0_profile="fixed_v1",
                behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
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
