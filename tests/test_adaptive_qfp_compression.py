"""Contracts for Dense-acquire, return-gated adaptive Q/F/P compression."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)
sys.path.insert(0, str(SCRIPTS))
from run_evolving_atomic_rssm import (  # noqa: E402
    ADAPTIVE_QFP_COMPRESSION_PROTOCOL,
    ADAPTIVE_QFP_NO_ATOM_REG_METHOD,
    ADAPTIVE_QFP_NO_ATOM_REG_PROTOCOL,
    SHARED_DISTILLED_HEADS_PROFILE,
    _budget_manifest,
    _parameter_manifest,
    _protocol_for_task_order,
    _resolved_config,
)
from summarize_continual_metrics import _budget_signature  # noqa: E402

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal hosts omit PyTorch.
    torch = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(SCRIPTS))
    from clworldmodel.continual import mechanism_output_distillation_losses
    from clworldmodel.models.mechanism_bank import (
        MechanismBank,
        ResidualMechanism,
    )

experiment_dependencies_available = False
if torch is not None:
    try:
        import gymnasium  # noqa: F401
        import sortedcontainers  # noqa: F401

        sys.path.insert(0, str(VENDORED_ATARI))
        from config import Config
        import train

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


class AdaptiveQfpLauncherStaticTests(unittest.TestCase):
    def test_named_launcher_and_exact_budget_are_dependency_free(self) -> None:
        data = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
        )
        budget = _budget_manifest(data)
        parameters = _parameter_manifest(data)

        self.assertEqual(
            data["continual_method"],
            "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        )
        self.assertEqual(data["task_mechanism_parameterization"], "adaptive_dense_width")
        self.assertEqual(
            data["adaptive_compression_width_fractions"],
            [0.75, 0.5, 0.25, 0.125],
        )
        self.assertEqual(budget["adaptive_compression_world_model_updates"], 6_000)
        self.assertEqual(budget["adaptive_compression_sequences"], 96_000)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 552_000)
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
            comparison_budget["adaptive_compression_world_model_updates"],
            6_000,
        )
        self.assertEqual(
            comparison_budget["total_world_model_optimizer_steps"], 552_000
        )
        self.assertEqual(parameters["online_parameters"], 52_897_535)
        self.assertEqual(
            parameters["adaptive_compression"]["minimum_final_online_parameters"],
            32_935_103,
        )
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
                adaptive_qfp_compression=True,
            ),
            ADAPTIVE_QFP_COMPRESSION_PROTOCOL,
        )

    def test_adaptive_launcher_rejects_missing_shared_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "shared distilled"):
            _resolved_config(
                _source_dict(),
                task_order="arrow-original-six",
                adaptive_qfp_compression=True,
            )

    def test_no_atom_output_regularization_is_a_named_isolated_ablation(self) -> None:
        data = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
            disable_atom_output_regularization=True,
        )

        self.assertEqual(data["continual_method"], ADAPTIVE_QFP_NO_ATOM_REG_METHOD)
        self.assertEqual(data["task_atom_output_regularization"], 0.0)
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
                adaptive_qfp_compression=True,
                disable_atom_output_regularization=True,
            ),
            ADAPTIVE_QFP_NO_ATOM_REG_PROTOCOL,
        )
        control = _resolved_config(
            _source_dict(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
        )
        changed = {
            key: (control[key], data[key])
            for key in data
            if control.get(key) != data[key]
        }
        self.assertEqual(
            changed,
            {
                "continual_method": (
                    "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
                    ADAPTIVE_QFP_NO_ATOM_REG_METHOD,
                ),
                "task_atom_output_regularization": (1e-4, 0.0),
            },
        )
        with self.assertRaisesRegex(ValueError, "requires adaptive"):
            _resolved_config(
                _source_dict(),
                task_order="arrow-original-six",
                disable_atom_output_regularization=True,
            )


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class AdaptiveQfpCompressionTests(unittest.TestCase):
    @staticmethod
    def _source() -> dict:
        return Config.from_dict(_source_dict()).to_dict()

    def test_structured_pruning_keeps_top_channels_inside_each_atom(self) -> None:
        source = ResidualMechanism(
            in_features=3,
            out_features=2,
            hidden_features=8,
            num_atoms=2,
        )
        with torch.no_grad():
            source.norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
            source.norm.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))
            source.down.weight.copy_(
                torch.arange(24, dtype=torch.float32).reshape(8, 3) + 1
            )
            source.down.bias.copy_(torch.arange(8, dtype=torch.float32))
            # Within each four-channel atom, later channels have larger scores.
            source.up.weight.copy_(
                torch.tensor(
                    [
                        [1, 2, 3, 4, 1, 2, 3, 4],
                        [1, 2, 3, 4, 1, 2, 3, 4],
                    ],
                    dtype=torch.float32,
                )
            )
            source.up.bias.copy_(torch.tensor([0.25, -0.5]))

        torch.manual_seed(1234)
        rng_before = torch.random.get_rng_state().clone()
        compact, selected = ResidualMechanism.structured_pruned_copy(
            source, hidden_features=4
        )

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(selected, [2, 3, 6, 7])
        self.assertEqual(compact.hidden_features, 4)
        self.assertEqual(compact.atom_width, 2)
        torch.testing.assert_close(compact.norm.weight, source.norm.weight)
        torch.testing.assert_close(compact.norm.bias, source.norm.bias)
        torch.testing.assert_close(compact.down.weight, source.down.weight[selected])
        torch.testing.assert_close(compact.down.bias, source.down.bias[selected])
        torch.testing.assert_close(compact.up.weight, source.up.weight[:, selected])
        torch.testing.assert_close(compact.up.bias, source.up.bias)
        self.assertLess(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in source.parameters()),
        )
        inputs = torch.randn(5, 3)
        atoms = compact.atom_outputs(inputs)
        self.assertEqual(atoms.shape, (5, 2, 2))
        torch.testing.assert_close(atoms.sum(-2), compact(inputs))

    def test_adaptive_bank_state_dict_rebuilds_compact_topology_before_load(self) -> None:
        torch.manual_seed(11)
        bank = MechanismBank(
            num_tasks=3,
            in_features=4,
            out_features=3,
            hidden_features=8,
            num_atoms=2,
            include_task0=True,
            reuse_enabled=True,
            parameterization="adaptive_dense_width",
        )
        report = bank.compress_task(0, hidden_features=4)
        self.assertEqual(report["old_hidden_features"], 8)
        self.assertEqual(report["new_hidden_features"], 4)
        self.assertEqual(bank.compression_layout(), [4, 8, 8])

        with torch.no_grad():
            bank.route_for(1).logits.fill_(0.7)
        inputs = torch.randn(6, 4)
        expected = bank(inputs, task_id=1)
        state = copy.deepcopy(bank.state_dict())

        restored = MechanismBank(
            num_tasks=3,
            in_features=4,
            out_features=3,
            hidden_features=8,
            num_atoms=2,
            include_task0=True,
            reuse_enabled=True,
            parameterization="adaptive_dense_width",
        )
        self.assertEqual(restored.compression_layout(), [8, 8, 8])
        restored.load_state_dict(state, strict=True)

        self.assertEqual(restored.compression_layout(), [4, 8, 8])
        torch.testing.assert_close(restored(inputs, task_id=1), expected)
        self.assertEqual(
            restored.parameter_report()["mechanism_hidden_features_per_task"],
            [4, 8, 8],
        )

    def test_mechanism_distillation_matches_three_qfp_outputs(self) -> None:
        student = {
            "current_atom_outputs": {
                name: torch.randn(3, 2, 5, requires_grad=True)
                for name in ("recurrent", "posterior", "prior")
            }
        }
        teacher = {
            "current_atom_outputs": {
                name: value.detach().clone()
                for name, value in student["current_atom_outputs"].items()
            }
        }

        losses = mechanism_output_distillation_losses(student, teacher)

        self.assertEqual(set(losses), {"recurrent", "posterior", "prior", "total"})
        for value in losses.values():
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        losses["total"].backward()
        self.assertTrue(
            all(
                value.grad is not None
                for value in student["current_atom_outputs"].values()
            )
        )

    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_return_gate_handles_positive_negative_and_improved_candidates(self) -> None:
        self.assertTrue(
            train._adaptive_compression_candidate_passes(
                teacher_return=100.0,
                candidate_return=95.0,
                maximum_relative_drop=0.05,
            )
        )
        self.assertFalse(
            train._adaptive_compression_candidate_passes(
                teacher_return=100.0,
                candidate_return=94.9,
                maximum_relative_drop=0.05,
            )
        )
        self.assertTrue(
            train._adaptive_compression_candidate_passes(
                teacher_return=-2.0,
                candidate_return=-1.0,
                maximum_relative_drop=0.05,
            )
        )
        self.assertFalse(
            train._adaptive_compression_candidate_passes(
                teacher_return=-2.0,
                candidate_return=-2.11,
                maximum_relative_drop=0.05,
            )
        )

    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_named_launcher_config_is_dense_acquire_and_shared_head_only(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
        )
        config = Config.from_dict(data)

        self.assertEqual(
            config.continual_method,
            "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
        )
        self.assertTrue(config.uses_adaptive_qfp_compression)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertTrue(config.task_private_actor_critic)
        self.assertEqual(
            config.task_mechanism_parameterization, "adaptive_dense_width"
        )
        self.assertEqual(
            (
                config.task_mechanism_recurrent_width,
                config.task_mechanism_representation_width,
                config.task_mechanism_transition_width,
            ),
            (512, 512, 256),
        )
        self.assertEqual(
            config.adaptive_compression_width_fractions,
            [0.75, 0.5, 0.25, 0.125],
        )
        self.assertEqual(config.adaptive_compression_steps_per_candidate, 250)
        self.assertEqual(config.adaptive_compression_rollouts, 16)
        self.assertEqual(config.adaptive_compression_max_return_drop, 0.05)
        self.assertEqual(config.adaptive_compression_qfp_distill_scale, 1.0)
        self.assertEqual(
            _protocol_for_task_order(
                "arrow-original-six",
                prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
                adaptive_qfp_compression=True,
            ),
            ADAPTIVE_QFP_COMPRESSION_PROTOCOL,
        )

        changed_budget = copy.deepcopy(data)
        changed_budget["adaptive_compression_steps_per_candidate"] = 249
        with self.assertRaisesRegex(ValueError, "fixed optimizer"):
            Config.from_dict(changed_budget)

        no_atom = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
            disable_atom_output_regularization=True,
        )
        no_atom_config = Config.from_dict(no_atom)
        self.assertEqual(
            no_atom_config.continual_method, ADAPTIVE_QFP_NO_ATOM_REG_METHOD
        )
        self.assertEqual(no_atom_config.task_atom_output_regularization, 0.0)
        invalid_no_atom = copy.deepcopy(no_atom)
        invalid_no_atom["task_atom_output_regularization"] = 1e-4
        with self.assertRaisesRegex(ValueError, "fixed optimizer"):
            Config.from_dict(invalid_no_atom)

        with self.assertRaisesRegex(ValueError, "shared distilled"):
            _resolved_config(
                self._source(),
                task_order="arrow-original-six",
                adaptive_qfp_compression=True,
            )

    @unittest.skipUnless(
        experiment_dependencies_available,
        "requires the pinned Atari experiment dependencies",
    )
    def test_compression_compute_and_parameter_bounds_are_explicit(self) -> None:
        data = _resolved_config(
            self._source(),
            task_order="arrow-original-six",
            prediction_head_profile=SHARED_DISTILLED_HEADS_PROFILE,
            adaptive_qfp_compression=True,
        )
        budget = _budget_manifest(data)
        parameters = _parameter_manifest(data)

        self.assertEqual(budget["adaptive_compression_world_model_updates"], 6_000)
        self.assertEqual(budget["adaptive_compression_sequences"], 96_000)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 552_000)
        self.assertTrue(budget["adaptive_compression_is_extra_compute"])
        self.assertEqual(
            parameters["adaptive_compression"]["dense_acquisition_widths"],
            [512, 512, 256],
        )
        self.assertEqual(
            parameters["adaptive_compression"]["minimum_candidate_widths"],
            [64, 64, 32],
        )
        self.assertLess(
            parameters["adaptive_compression"]["minimum_final_online_parameters"],
            parameters["online_parameters"],
        )


if __name__ == "__main__":
    unittest.main()
