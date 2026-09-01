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
        COMPACT_MECHANISM_ORIGINAL_SIX_PROTOCOL,
        COMPACT_MECHANISM_PROFILE,
        DEFAULT_MECHANISM_PROFILE,
        ORIGINAL_SIX_MINIMUM_FREE_BYTES,
        ORIGINAL_SIX_TASK_PROTOCOL,
        PROTOCOL,
        SHARED_DISTILLED_HEADS_PROFILE,
        SHARED_DISTILLED_HEADS_ORIGINAL_SIX_PROTOCOL,
        SHARED_FASTKAN_STABLE_BEHAVIOR,
        TASK_ORDERS,
        _behavior_update_budget,
        _budget_manifest,
        _mechanism_capacity_manifest,
        _parameter_manifest,
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
                expected_task0_profile = (
                    "fixed_v1" if order_name == "arrow-original-six" else "fixed_v2"
                )
                self.assertEqual(
                    config.evolving_task0_profile, expected_task0_profile
                )
                self.assertEqual(
                    config.first_task_shared_core_lr,
                    2e-4 if order_name == "arrow-original-six" else 3e-4,
                )
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

    def test_compact_profile_changes_only_mechanism_capacity(self) -> None:
        matched_data = _resolved_config(
            self._source(), task_order="arrow-original-six"
        )
        compact_data = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            mechanism_profile=COMPACT_MECHANISM_PROFILE,
        )
        differing_keys = {
            key
            for key in matched_data
            if matched_data[key] != compact_data[key]
        }
        self.assertEqual(
            differing_keys,
            {
                "task_mechanism_capacity_profile",
                "task_mechanism_recurrent_width",
                "task_mechanism_representation_width",
                "task_mechanism_transition_width",
            },
        )
        compact = Config.from_dict(compact_data)
        self.assertEqual(
            compact.task_mechanism_capacity_profile,
            COMPACT_MECHANISM_PROFILE,
        )
        self.assertEqual(
            (
                compact.task_mechanism_recurrent_width,
                compact.task_mechanism_representation_width,
                compact.task_mechanism_transition_width,
            ),
            (128, 128, 64),
        )
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                mechanism_profile=COMPACT_MECHANISM_PROFILE,
            ),
            COMPACT_MECHANISM_ORIGINAL_SIX_PROTOCOL,
        )
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                mechanism_profile=DEFAULT_MECHANISM_PROFILE,
            ),
            ORIGINAL_SIX_TASK_PROTOCOL,
        )

    def test_compact_profile_has_exact_declared_parameter_reduction(self) -> None:
        compact = _mechanism_capacity_manifest(
            task_count=6,
            mechanism_profile=COMPACT_MECHANISM_PROFILE,
        )
        matched = _mechanism_capacity_manifest(
            task_count=6,
            mechanism_profile=DEFAULT_MECHANISM_PROFILE,
        )

        self.assertEqual(
            compact["parameters_per_task"],
            {
                "recurrent": 132_736,
                "representation_posterior": 731_264,
                "transition_prior": 100_416,
                "total": 964_416,
            },
        )
        self.assertEqual(compact["private_mechanism_parameters"], 5_786_496)
        self.assertEqual(compact["reuse_route_parameters"], 180)
        self.assertEqual(compact["mechanism_and_route_parameters"], 5_786_676)
        self.assertEqual(matched["mechanism_and_route_parameters"], 22_897_332)

    def test_compact_profile_rejects_shorter_curricula(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete ARROW original-six"):
            _resolved_config(
                self._source(),
                task_order="mspacman-boxing-crazyclimber",
                mechanism_profile=COMPACT_MECHANISM_PROFILE,
            )

    def test_compact_profile_is_not_a_baseline_side_effect(self) -> None:
        invalid = self._source()
        invalid.update(
            {
                "task_mechanism_capacity_profile": COMPACT_MECHANISM_PROFILE,
                "task_mechanism_recurrent_width": 128,
                "task_mechanism_representation_width": 128,
                "task_mechanism_transition_width": 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "only for Evolving-Core"):
            Config.from_dict(invalid)

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

    def test_shared_distilled_heads_change_only_prediction_head_ownership(self) -> None:
        dense = _resolved_config(
            self._source(), task_order="arrow-original-six"
        )
        shared = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
        )
        differing_keys = {
            key for key in dense if dense[key] != shared[key]
        }

        self.assertEqual(
            differing_keys,
            {
                "continual_method",
                "task_private_heads",
                "task_shared_prediction_heads",
                "shared_prediction_distill_scale",
            },
        )
        config = Config.from_dict(shared)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertTrue(config.task_private_actor_critic)
        self.assertEqual(config.actor_network, "mlp")
        self.assertEqual(config.task_mechanism_parameterization, "dense_private")
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            ),
            SHARED_DISTILLED_HEADS_ORIGINAL_SIX_PROTOCOL,
        )

    def test_shared_distilled_heads_have_exact_six_task_parameter_ledger(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
        )
        parameters = _parameter_manifest(data)

        self.assertEqual(parameters["prediction_head_topology"], "single_shared")
        self.assertEqual(parameters["world_model_parameters"], 42_601_625)
        self.assertEqual(parameters["behavior_parameters"], 10_289_766)
        self.assertEqual(parameters["online_parameters"], 52_891_391)
        self.assertEqual(parameters["fp32_parameter_bytes"], 211_565_564)
        self.assertEqual(
            parameters["per_task_world_model_additions"],
            {
                "0": 3_850_432,
                "1": 3_850_444,
                "2": 3_850_456,
                "3": 3_850_468,
                "4": 3_850_480,
                "5": 3_850_492,
            },
        )
        self.assertEqual(
            parameters["comparison_to_dense_evolving_v2_private_mlp"]["difference"],
            -42_813_145,
        )
        self.assertAlmostEqual(
            parameters["comparison_to_dense_evolving_v2_private_mlp"][
                "relative_difference"
            ],
            -0.4473470829010654,
        )
        self.assertEqual(
            parameters["training_only_prediction_head_teacher"]["parameters"],
            8_562_629,
        )

    def test_shared_distilled_heads_reject_shared_behavior_or_compressed_qfp(self) -> None:
        with self.assertRaisesRegex(ValueError, "private MLP Actor-Critic"):
            _resolved_config(
                self._source(),
                task_order="mspacman-boxing-crazyclimber",
                behavior_profile=SHARED_FASTKAN_STABLE_BEHAVIOR,
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            )

        with self.assertRaisesRegex(ValueError, "dense matched_512"):
            _resolved_config(
                self._source(),
                task_order="arrow-original-six",
                mechanism_profile=COMPACT_MECHANISM_PROFILE,
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
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
