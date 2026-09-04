"""Contracts for shared-base, adaptive task-residual Actor-Critic."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)
sys.path.insert(0, str(PROJECT_SRC))
sys.path.insert(0, str(SCRIPTS))

from run_evolving_atomic_rssm import (  # noqa: E402
    ADAPTIVE_QFP_AC_COMPRESSION_METHOD,
    ADAPTIVE_QFP_AC_COMPRESSION_PROTOCOL,
    ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR,
    SHARED_DISTILLED_HEADS_PROFILE,
    _budget_manifest,
    _parameter_manifest,
    _protocol_for_task_order,
    _resolved_config,
)
from summarize_continual_metrics import (  # noqa: E402
    _budget_signature,
    _policy,
    _resource_accounting,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - minimal hosts omit PyTorch.
    torch = None
    nn = None

if torch is not None:
    from clworldmodel.models.adaptive_behavior import (  # noqa: E402
        TaskRoutedResidualCategoricalHead,
    )

experiment_dependencies_available = False
if torch is not None:
    try:
        import gymnasium  # noqa: F401
        import sortedcontainers  # noqa: F401

        sys.path.insert(0, str(VENDORED_ATARI))
        from ac import ActorCritic, ActorCriticOpt  # noqa: E402
        from config import Config  # noqa: E402
        import train  # noqa: E402

        experiment_dependencies_available = True
    except ModuleNotFoundError:  # pragma: no cover - lightweight local torch env.
        pass


def _source_dict() -> dict:
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
    return json.loads(path.read_text(encoding="utf-8"))


class AdaptiveSharedActorCriticLauncherTests(unittest.TestCase):
    def test_new_method_is_separate_and_has_fixed_declared_budget(self) -> None:
        data = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            behavior_profile=ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR,
            adaptive_qfp_compression=True,
        )
        budget = _budget_manifest(data)
        parameters = _parameter_manifest(data)

        self.assertEqual(data["continual_method"], ADAPTIVE_QFP_AC_COMPRESSION_METHOD)
        self.assertFalse(data["task_private_actor_critic"])
        self.assertTrue(data["adaptive_behavior_residuals"])
        self.assertEqual(data["adaptive_behavior_hidden_features"], 512)
        self.assertEqual(data["adaptive_behavior_num_atoms"], 4)
        self.assertEqual(
            data["adaptive_behavior_width_fractions"],
            [0.75, 0.5, 0.25, 0.125],
        )
        self.assertEqual(budget["adaptive_compression_world_model_updates"], 6_000)
        self.assertEqual(budget["adaptive_behavior_compression_updates"], 6_000)
        self.assertEqual(
            budget["adaptive_behavior_compression_imagined_states"],
            1_536_000,
        )
        self.assertEqual(
            budget["adaptive_behavior_compression_validation_rollouts"], 480
        )
        comparison_budget = _budget_signature(
            data,
            [90] * 6,
            {
                "budgets": budget,
                "fifo_slots": 512,
                "ltdm_slots": 512,
                "sequence_length": 512,
            },
        )
        self.assertEqual(
            comparison_budget["adaptive_behavior_compression_updates"], 6_000
        )
        self.assertEqual(
            comparison_budget[
                "adaptive_behavior_compression_imagined_states"
            ],
            1_536_000,
        )
        self.assertEqual(
            comparison_budget["total_actor_critic_optimizer_steps"], 438_000
        )
        self.assertEqual(
            _policy(data, ROOT / "does_not_exist.json"),
            "deterministic_argmax_and_latent_mode",
        )
        self.assertEqual(parameters["behavior_parameters"], 12_036_591)
        self.assertEqual(
            parameters["adaptive_behavior_compression"][
                "minimum_final_behavior_parameters"
            ],
            3_039_855,
        )
        self.assertLess(
            parameters["adaptive_joint_compression"][
                "minimum_final_online_parameters"
            ],
            parameters["online_parameters"],
        )
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
                behavior_profile=ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR,
                adaptive_qfp_compression=True,
            ),
            ADAPTIVE_QFP_AC_COMPRESSION_PROTOCOL,
        )

    def test_original_d_method_remains_private_actor_critic(self) -> None:
        data = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
        )
        self.assertEqual(
            data["continual_method"],
            "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        )
        self.assertTrue(data["task_private_actor_critic"])
        self.assertFalse(data["adaptive_behavior_residuals"])

    def test_reporter_keeps_behavior_compression_separate_from_qfp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            behavior_dir = root / "adaptive_behavior_compression"
            behavior_dir.mkdir()
            artifact = {
                "artifact_kind": (
                    "evolving_core_return_gated_actor_critic_residual_compression"
                ),
                "completed_task_id": 0,
                "optimizer_updates": 1_000,
                "imagined_states": 256_000,
                "selected_width_fraction": 0.5,
                "selected_dense_fallback": False,
                "actor_critic_parameters_removed": 123,
                "selected_layout": {
                    "actor": [256, 512, 512, 512, 512, 512],
                    "critic": [256, 512, 512, 512, 512, 512],
                },
            }
            (behavior_dir / "task_00_boundary.json").write_text(
                json.dumps(artifact), encoding="utf-8"
            )

            accounting = _resource_accounting(root)

        behavior = accounting["adaptive_behavior_compression"]
        self.assertEqual(behavior["task_count"], 1)
        self.assertEqual(behavior["optimizer_updates"], 1_000)
        self.assertEqual(behavior["imagined_states"], 256_000)
        self.assertEqual(behavior["selected_width_fractions"], [0.5])
        self.assertEqual(behavior["actor_critic_parameters_removed"], 123)

    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_named_config_rejects_a_second_slow_critic(self) -> None:
        data = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            behavior_profile=ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR,
            adaptive_qfp_compression=True,
        )
        config = Config.from_dict(data)
        self.assertEqual(config.ac_slow_critic_regularizer, 0.0)
        self.assertFalse(config.ac_use_slow_critic_targets)

        with_slow_critic = copy.deepcopy(data)
        with_slow_critic["ac_slow_critic_regularizer"] = 1.0
        with self.assertRaisesRegex(ValueError, "one routed critic"):
            Config.from_dict(with_slow_critic)

        with_slow_targets = copy.deepcopy(data)
        with_slow_targets["ac_use_slow_critic_targets"] = True
        with self.assertRaisesRegex(ValueError, "one routed critic"):
            Config.from_dict(with_slow_targets)

        with_fastkan = copy.deepcopy(data)
        with_fastkan["actor_network"] = "fast_kan_ac"
        with self.assertRaisesRegex(ValueError, "persistent Dreamer MLP"):
            Config.from_dict(with_fastkan)


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class AdaptiveSharedActorCriticModuleTests(unittest.TestCase):
    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_shared_bases_are_rng_paired_with_private_mlp_control(self) -> None:
        torch.manual_seed(5)
        control = ActorCritic(10, 4)
        control_next = torch.randn(3)
        torch.manual_seed(5)
        adaptive = ActorCritic(
            10,
            4,
            adaptive_behavior_residuals=True,
            adaptive_behavior_num_tasks=3,
            adaptive_behavior_hidden_features=8,
            adaptive_behavior_residual_scale=0.1,
            adaptive_behavior_num_atoms=2,
            adaptive_behavior_reuse=True,
        )
        adaptive_next = torch.randn(3)

        for expected, actual in zip(
            control.actor.parameters(), adaptive.actor.base_logits.parameters()
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for expected, actual in zip(
            control.critic.parameters(), adaptive.critic.base_logits.parameters()
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(adaptive_next, control_next, atol=0, rtol=0)

    def test_zero_effect_residual_preserves_shared_base_then_routes_tasks(self) -> None:
        torch.manual_seed(7)
        base = nn.Linear(5, 3)
        head = TaskRoutedResidualCategoricalHead(
            base,
            in_features=5,
            out_features=3,
            num_tasks=3,
            hidden_features=8,
            residual_scale=0.1,
            num_atoms=2,
            reuse_enabled=True,
        )
        inputs = torch.randn(4, 5)
        expected = torch.log_softmax(base(inputs), dim=-1)
        for task_id in range(3):
            head.set_task_route(task_id)
            torch.testing.assert_close(head(inputs), expected, atol=0, rtol=0)

        with torch.no_grad():
            head.residual_bank.mechanism_for(1).up.bias.fill_(2.0)
        head.set_task_route(0)
        task0 = head(inputs)
        head.set_task_route(1)
        task1 = head(inputs)
        self.assertFalse(torch.equal(task0, task1))

    def test_compact_layout_round_trips_and_physically_removes_parameters(self) -> None:
        torch.manual_seed(11)
        head = TaskRoutedResidualCategoricalHead(
            nn.Linear(6, 4),
            in_features=6,
            out_features=4,
            num_tasks=3,
            hidden_features=8,
            residual_scale=0.1,
            num_atoms=2,
            reuse_enabled=True,
        )
        dense_parameters = sum(parameter.numel() for parameter in head.parameters())
        head.residual_bank.compress_task(0, hidden_features=4)
        compact_parameters = sum(parameter.numel() for parameter in head.parameters())
        self.assertLess(compact_parameters, dense_parameters)
        self.assertEqual(head.compression_layout(), [4, 8, 8])

        head.set_task_route(1)
        inputs = torch.randn(5, 6)
        expected = head(inputs)
        state = copy.deepcopy(head.state_dict())
        restored = TaskRoutedResidualCategoricalHead(
            nn.Linear(6, 4),
            in_features=6,
            out_features=4,
            num_tasks=3,
            hidden_features=8,
            residual_scale=0.1,
            num_atoms=2,
            reuse_enabled=True,
        )
        restored.load_state_dict(state, strict=True)
        self.assertEqual(restored.compression_layout(), [4, 8, 8])
        torch.testing.assert_close(restored(inputs), expected)

    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_actor_critic_routes_both_heads_and_resumes_compact_topology(self) -> None:
        torch.manual_seed(17)
        actor_critic = ActorCritic(
            10,
            4,
            adaptive_behavior_residuals=True,
            adaptive_behavior_num_tasks=3,
            adaptive_behavior_hidden_features=8,
            adaptive_behavior_residual_scale=0.1,
            adaptive_behavior_num_atoms=2,
            adaptive_behavior_reuse=True,
        )
        actor_critic.activate_training_task(0)
        actor_critic.actor.residual_bank.compress_task(0, hidden_features=4)
        actor_critic.critic.residual_bank.compress_task(0, hidden_features=4)
        actor_critic.set_task_route(1)
        inputs = torch.randn(3, 10)
        expected = actor_critic(inputs)
        source = ActorCriticOpt(
            actor_critic,
            torch.optim.Adam(actor_critic.parameters(), lr=1e-4),
        )
        state = train._actor_critic_opt_resumable_state_dict(source)

        restored_ac = ActorCritic(
            10,
            4,
            adaptive_behavior_residuals=True,
            adaptive_behavior_num_tasks=3,
            adaptive_behavior_hidden_features=8,
            adaptive_behavior_residual_scale=0.1,
            adaptive_behavior_num_atoms=2,
            adaptive_behavior_reuse=True,
        )
        restored = ActorCriticOpt(
            restored_ac,
            torch.optim.Adam(restored_ac.parameters(), lr=1e-4),
        )
        train._load_actor_critic_opt_resumable_state_dict(restored, state)

        self.assertEqual(
            restored.ac.adaptive_behavior_layout(),
            {"actor": [4, 8, 8], "critic": [4, 8, 8]},
        )
        # Route identity is scheduler state rather than a CUDA buffer; every
        # task-aware evaluation/training boundary sets it explicitly.
        restored.ac.set_task_route(1)
        actual = restored.ac(inputs)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
