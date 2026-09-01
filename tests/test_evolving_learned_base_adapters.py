"""Contracts for learned Task-0 bases and fixed-rank private adapters."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)

try:
    import sortedcontainers  # noqa: F401
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - minimal hosts omit experiment deps.
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.models.mechanism_bank import (
        LearnedBaseLowRankMechanism,
        MechanismBank,
    )
    from clworldmodel.models.prediction_adapters import ZeroEffectFeatureAdapter
    from config import Config
    from run_evolving_learned_base_adapters import (
        MECHANISM_LOW_RANK,
        METHOD_KEY,
        PREDICTION_ADAPTER_RANK,
        PROTOCOL,
        _mechanism_capacity_manifest,
        _parameter_manifest,
        _resolved_config,
    )
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class LearnedBaseAdapterTests(unittest.TestCase):
    @staticmethod
    def _published_config_data() -> dict:
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

    @staticmethod
    def _world_model() -> WorldModel:
        class WideEmbedder(nn.Module):
            output_size = 4096

            def __init__(self) -> None:
                super().__init__()
                self.offset = nn.Parameter(torch.zeros(self.output_size))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.offset.unsqueeze(0).expand(images.shape[0], -1)

        return WorldModel(
            3,
            (2, 3),
            4,
            8,
            cnn_depth=4,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="reconstruction",
            num_task_experts=3,
            full_task_experts=False,
            full_task_rssm_experts=False,
            task_private_heads=False,
            task_shared_prediction_heads=True,
            task_private_prediction_adapters=True,
            prediction_adapter_rank=4,
            prediction_adapter_residual_scale=0.1,
            freeze_shared_prediction_heads_after_task0=True,
            evolving_shared_core=True,
            task_projected_image_encoder=True,
            task_symmetric_image_projectors=True,
            task_projector_bottleneck_features=64,
            task_mechanism_bank=True,
            task_mechanism_reuse=False,
            task_mechanism_recurrent_width=8,
            task_mechanism_representation_width=8,
            task_mechanism_transition_width=8,
            task_mechanism_num_atoms=4,
            task_mechanism_parameterization="learned_task0_low_rank",
            task_mechanism_low_rank=4,
            task_symmetric_mechanisms=True,
            image_embedder=WideEmbedder(),
        )

    def test_learned_base_delta_is_zero_effect_and_non_duplicating(self) -> None:
        torch.manual_seed(7)
        bank = MechanismBank(
            num_tasks=3,
            in_features=8,
            out_features=6,
            hidden_features=8,
            residual_scale=0.1,
            reuse_enabled=False,
            num_atoms=4,
            include_task0=True,
            parameterization="learned_task0_low_rank",
            low_rank_rank=4,
        )
        base = bank.mechanism_for(0)
        later = bank.mechanism_for(1)
        self.assertIsInstance(later, LearnedBaseLowRankMechanism)
        self.assertIs(later.base_mechanism(), base)
        self.assertFalse(any("_base" in name for name in bank.state_dict()))

        inputs = torch.randn(5, 8)
        with torch.no_grad():
            base.up.weight.normal_()
            base.up.bias.normal_()
        correction, private_delta = bank.forward_with_current(inputs, 1)
        torch.testing.assert_close(correction, base(inputs), rtol=0, atol=0)
        torch.testing.assert_close(
            private_delta, torch.zeros_like(private_delta), rtol=0, atol=0
        )

        task0_before = bank(inputs, 0).detach().clone()
        with torch.no_grad():
            later.up_out.bias.fill_(0.5)
        self.assertFalse(torch.equal(bank(inputs, 1), task0_before))
        torch.testing.assert_close(bank(inputs, 0), task0_before, rtol=0, atol=0)

        base_ids = {id(parameter) for parameter in base.parameters()}
        later_ids = {id(parameter) for parameter in later.parameters()}
        self.assertFalse(base_ids & later_ids)
        bank.activate_task(1)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in later.parameters()))

        copied = copy.deepcopy(bank)
        self.assertIs(
            copied.mechanism_for(1).base_mechanism(), copied.mechanism_for(0)
        )

    def test_prediction_feature_adapter_is_exact_identity_at_install(self) -> None:
        adapter = ZeroEffectFeatureAdapter(12, 4, residual_scale=0.1)
        inputs = torch.randn(3, 12, requires_grad=True)

        torch.testing.assert_close(adapter(inputs), inputs, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 132)
        adapter(inputs).square().sum().backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.0)

    def test_world_model_ownership_freezes_base_heads_after_task0(self) -> None:
        wm = self._world_model()
        shared_ids = {
            id(parameter)
            for values in wm.shared_parameter_groups().values()
            for parameter in values
        }
        base_head_ids = {
            id(parameter)
            for module in (wm.decoder, wm.reward_fc, wm.continue_fc)
            for parameter in module.parameters()
        }
        task0_ids = {id(parameter) for parameter in wm.private_parameters(0)}
        task1_ids = {id(parameter) for parameter in wm.private_parameters(1)}
        adapter1_ids = {
            id(parameter)
            for head_name in ("observation", "reward", "continue")
            for parameter in wm.prediction_adapter_for(head_name, 1).parameters()
        }

        self.assertFalse(shared_ids & base_head_ids)
        self.assertTrue(base_head_ids <= task0_ids)
        self.assertTrue(adapter1_ids <= task1_ids)
        self.assertFalse(base_head_ids & task1_ids)
        self.assertEqual(wm.route_parameters(1), [])
        self.assertNotIn("observation_head", wm.shared_core_state_dict())

        self.assertTrue(wm.initialize_task_expert(1, 0))
        wm.activate_task_expert(0)
        self.assertTrue(all(parameter.requires_grad for parameter in base_head_ids_to_params(wm)))
        wm.activate_task_expert(1)
        self.assertTrue(all(not parameter.requires_grad for parameter in base_head_ids_to_params(wm)))
        self.assertTrue(
            all(
                parameter.requires_grad
                for head_name in ("observation", "reward", "continue")
                for parameter in wm.prediction_adapter_for(head_name, 1).parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for head_name in ("observation", "reward", "continue")
                for parameter in wm.prediction_adapter_for(head_name, 2).parameters()
            )
        )

        features = torch.randn(2, wm.zh_transform.out_features)
        torch.testing.assert_close(
            wm.predict_reward_symlog(features, 0),
            wm.predict_reward_symlog(features, 1),
            rtol=0,
            atol=0,
        )

    def test_named_config_and_exact_six_task_parameter_ledger(self) -> None:
        data = _resolved_config(self._published_config_data())
        config = Config.from_dict(data)

        self.assertEqual(config.continual_method, METHOD_KEY)
        self.assertTrue(config.uses_evolving_atomic_rssm)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertTrue(config.task_private_prediction_adapters)
        self.assertTrue(config.freeze_shared_prediction_heads_after_task0)
        self.assertFalse(config.task_mechanism_reuse)
        self.assertEqual(config.task_mechanism_low_rank, MECHANISM_LOW_RANK)
        self.assertEqual(config.prediction_adapter_rank, PREDICTION_ADAPTER_RANK)
        self.assertTrue(config.task_private_actor_critic)
        self.assertEqual(config.actor_network, "mlp")
        self.assertEqual(config.epochs, 540)
        self.assertIn("LowRank32QFP", PROTOCOL)

        mechanisms = _mechanism_capacity_manifest(6)
        self.assertEqual(mechanisms["task0_base_parameters"], 3_816_192)
        self.assertEqual(
            mechanisms["per_later_task_low_rank_delta"]["total"], 359_168
        )
        self.assertEqual(mechanisms["registered_dormant_route_parameters"], 180)

        parameters = _parameter_manifest(data)
        self.assertEqual(parameters["world_model_parameters"], 26_860_185)
        self.assertEqual(parameters["behavior_parameters"], 10_295_910)
        self.assertEqual(parameters["online_parameters"], 37_156_095)
        self.assertEqual(
            parameters["prediction_adapter_parameters_per_later_task"],
            308_736,
        )
        self.assertEqual(
            parameters["per_task_world_model_additions"],
            {
                "0": 3_850_432,
                "1": 702_156,
                "2": 702_168,
                "3": 702_180,
                "4": 702_192,
                "5": 702_204,
            },
        )
        self.assertEqual(
            parameters["comparison_to_dense_evolving_v2_private_mlp"][
                "difference"
            ],
            -58_554_585,
        )

    def test_named_config_rejects_rank_or_reuse_drift(self) -> None:
        rank_drift = _resolved_config(self._published_config_data())
        rank_drift["task_mechanism_low_rank"] = 16
        with self.assertRaisesRegex(ValueError, "fixes Q/F/P low-rank size to 32"):
            Config.from_dict(rank_drift)

        reuse_drift = _resolved_config(self._published_config_data())
        reuse_drift["task_mechanism_reuse"] = True
        with self.assertRaisesRegex(ValueError, "disables old-atom reuse"):
            Config.from_dict(reuse_drift)

        adapter_drift = _resolved_config(self._published_config_data())
        adapter_drift["prediction_adapter_rank"] = 16
        with self.assertRaisesRegex(ValueError, "fixed optimizer, replay, interface"):
            Config.from_dict(adapter_drift)


def base_head_ids_to_params(wm: WorldModel):
    return (
        parameter
        for module in (wm.decoder, wm.reward_fc, wm.continue_fc)
        for parameter in module.parameters()
    )


if __name__ == "__main__":
    unittest.main()
