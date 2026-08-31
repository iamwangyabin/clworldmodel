"""Focused contracts for the shared-base MB-RSSM continual method."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)

try:
    import torch
    import torch.nn as nn
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - minimal host environments omit torch.
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.models.mechanism_bank import MechanismBank
    from config import Config
    from rssm import Representation, Transition
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class CnnMechanismBankTests(unittest.TestCase):
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

    @classmethod
    def _method_config_data(cls, *, reuse: bool = True) -> dict:
        data = cls._published_config_data()
        data["esc"]["env_configs"] = data["esc"]["env_configs"][:3]
        data.update(
            {
                "continual_method": "cnn_mechanism_bank_arrow",
                "rssm_num_experts": 3,
                "dino_fullbank_current_task_fraction": 1.0,
                "observation_objective": "reconstruction",
                "observation_encoder": "cnn",
                "task_banked_image_encoder": False,
                "task_projected_image_encoder": True,
                "task_projector_bottleneck_features": 64,
                "task_lora_recurrent_rank": 0,
                "task_lora_representation_rank": 0,
                "task_lora_transition_rank": 0,
                "task_recurrent_output_adapter_features": 0,
                "task_mechanism_bank": True,
                "task_mechanism_reuse": reuse,
                "task_mechanism_recurrent_width": 512,
                "task_mechanism_representation_width": 512,
                "task_mechanism_transition_width": 256,
                "task_mechanism_residual_scale": 0.1,
                "compute_dtype": "bfloat16",
                "replay_observation_dtype": "uint8",
                "random_policy": "new",
                "actor_network": "mlp",
                "fresh_ac": False,
                "residual_correction": "none",
                "shared_core_mode": "task1_frozen_mechanism_bank",
            }
        )
        for replay_config in data["replay_buffers"]:
            replay_config["rb_device"] = "cpu"
        return data

    def test_config_accepts_reuse_and_capacity_matched_no_reuse(self) -> None:
        reuse = Config.from_dict(self._method_config_data(reuse=True))
        no_reuse = Config.from_dict(self._method_config_data(reuse=False))

        self.assertTrue(reuse.uses_full_task_experts)
        self.assertTrue(reuse.task_mechanism_bank)
        self.assertTrue(reuse.task_mechanism_reuse)
        self.assertFalse(no_reuse.task_mechanism_reuse)
        self.assertEqual(reuse.task_update_fraction, 1.0)

        invalid = self._method_config_data()
        invalid["task_lora_transition_rank"] = 1
        with self.assertRaisesRegex(ValueError, "disables every RSSM LoRA"):
            Config.from_dict(invalid)

        invalid = self._method_config_data()
        invalid["task_mechanism_transition_width"] = 128
        with self.assertRaisesRegex(ValueError, "fixes bank/recurrent"):
            Config.from_dict(invalid)

    def test_production_mechanism_parameter_budget_is_exact(self) -> None:
        banks = (
            MechanismBank(
                num_tasks=3,
                in_features=512,
                out_features=512,
                hidden_features=512,
            ),
            MechanismBank(
                num_tasks=3,
                in_features=4096 + 512,
                out_features=32 * 32,
                hidden_features=512,
            ),
            MechanismBank(
                num_tasks=3,
                in_features=512,
                out_features=32 * 32,
                hidden_features=256,
            ),
        )
        per_bank = [
            bank.parameter_report()["mechanism_parameters_per_later_task"][0]
            for bank in banks
        ]
        self.assertEqual(per_bank, [526_336, 2_894_336, 395_520])
        self.assertEqual(sum(per_bank), 3_816_192)
        self.assertEqual(
            sum(
                bank.parameter_report()["route_parameters_per_later_task"][1]
                for bank in banks
            ),
            3,
        )

    def test_zero_effect_and_route_gradient_contract(self) -> None:
        bank = MechanismBank(
            num_tasks=3,
            in_features=4,
            out_features=3,
            hidden_features=5,
        )
        inputs = torch.randn(7, 4)
        self.assertTrue(torch.equal(bank(inputs, 0), torch.zeros(7, 3)))
        self.assertTrue(torch.equal(bank(inputs, 1), torch.zeros(7, 3)))
        self.assertTrue(torch.equal(bank(inputs, 2), torch.zeros(7, 3)))

        with torch.no_grad():
            bank.mechanisms[0].up.bias.fill_(0.75)
        bank.activate_task(2)
        loss = bank(inputs, 2).sum()
        loss.backward()
        route_grad = bank.routes[1].logits.grad
        self.assertIsNotNone(route_grad)
        self.assertGreater(float(route_grad.abs().sum()), 0.0)
        self.assertFalse(
            any(parameter.requires_grad for parameter in bank.mechanisms[0].parameters())
        )

        no_reuse = MechanismBank(
            num_tasks=3,
            in_features=4,
            out_features=3,
            hidden_features=5,
            reuse_enabled=False,
        )
        no_reuse.activate_task(2)
        self.assertFalse(no_reuse.routes[1].logits.requires_grad)
        self.assertEqual(no_reuse.route_values(2), [0.0])

    def test_non_mechanism_distribution_refactor_preserves_outputs(self) -> None:
        torch.manual_seed(17)
        representation = Representation((2, 3), 7, 5, 8, 2, uniform=0.01)
        embedding = torch.randn(4, 7)
        hidden = torch.randn(4, 5)
        logits = representation.logits(embedding, hidden)
        expected_post = representation.inter_to_z_dist(logits)
        expected_post = (
            (1 - representation.uniform) * expected_post.exp()
            + representation.uniform / representation.ls[1]
        ).log()
        torch.testing.assert_close(
            representation(embedding, hidden), expected_post, rtol=0, atol=0
        )

        transition = Transition((2, 3), 5, 8, 2, uniform=0.01)
        expected_prior = transition.h_to_z_prior(hidden)
        expected_prior = (
            (1 - transition.uniform) * expected_prior.exp()
            + transition.uniform / transition.ls[1]
        ).log()
        torch.testing.assert_close(
            transition(hidden), expected_prior, rtol=0, atol=0
        )

    def test_task2_updates_only_new_mechanisms_routes_projector_and_heads(self) -> None:
        class WideEmbedder(nn.Module):
            output_size = 4096

            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                pooled = images.mean((-3, -2, -1), keepdim=False).unsqueeze(-1)
                return pooled.repeat(1, self.output_size) * self.scale

        torch.manual_seed(23)
        world_model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=4,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="reconstruction",
            num_task_experts=3,
            full_task_experts=True,
            task_projected_image_encoder=True,
            task_projector_bottleneck_features=64,
            task_mechanism_bank=True,
            task_mechanism_reuse=True,
            task_mechanism_recurrent_width=4,
            task_mechanism_representation_width=6,
            task_mechanism_transition_width=3,
            image_embedder=WideEmbedder(),
        )
        self.assertEqual(len(world_model.rssm.recurrent_experts), 0)
        self.assertEqual(len(world_model.rssm.representation_experts), 0)
        self.assertEqual(len(world_model.rssm.transition_experts), 0)

        self.assertTrue(world_model.initialize_task_expert(1, 0))
        observations = torch.rand(3, 2, 3, 64, 64)
        prev_z, prev_h = world_model.rssm.initial_state(2)
        prev_a = torch.nn.functional.one_hot(
            torch.tensor([0, 1]), num_classes=4
        ).float()
        reset = torch.zeros(2, 1)
        task0 = world_model.rssm(
            prev_z, prev_a, prev_h, observations[0], reset, stochastic=False, task_id=0
        )
        task1 = world_model.rssm(
            prev_z, prev_a, prev_h, observations[0], reset, stochastic=False, task_id=1
        )
        for base_value, new_value in zip(task0, task1):
            torch.testing.assert_close(base_value, new_value, rtol=0, atol=0)

        # Simulate nonzero Task-1 mechanisms so Task 2's zero gates receive signal.
        with torch.no_grad():
            world_model.rssm.recurrent_mechanism_bank.mechanisms[0].up.bias.fill_(0.1)
            world_model.rssm.representation_mechanism_bank.mechanisms[
                0
            ].up.bias.copy_(
                torch.linspace(-0.2, 0.2, steps=6)
            )
            world_model.rssm.transition_mechanism_bank.mechanisms[
                0
            ].up.bias.copy_(
                torch.linspace(0.2, -0.2, steps=6)
            )

        self.assertTrue(world_model.initialize_task_expert(2, 1))
        world_model.activate_task_expert(2)
        allowed_prefixes = (
            "rssm.image_projectors.1.",
            "rssm.recurrent_mechanism_bank.mechanisms.1.",
            "rssm.recurrent_mechanism_bank.routes.1.",
            "rssm.representation_mechanism_bank.mechanisms.1.",
            "rssm.representation_mechanism_bank.routes.1.",
            "rssm.transition_mechanism_bank.mechanisms.1.",
            "rssm.transition_mechanism_bank.routes.1.",
            "decoder_experts.1.",
            "reward_experts.1.",
            "continue_experts.1.",
        )
        trainable_names = {
            name for name, parameter in world_model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith(allowed_prefixes) for name in trainable_names),
            trainable_names,
        )

        world_model.zero_grad(set_to_none=True)
        prior_probe = world_model.rssm.prior(torch.randn(4, 5), task_id=2)
        prior_probe[..., 0].sum().backward()
        transition_route = (
            world_model.rssm.transition_mechanism_bank.routes[1].logits
        )
        self.assertIsNotNone(transition_route.grad)
        self.assertGreater(float(transition_route.grad.abs().sum()), 0.0)

        frozen = {
            name: value.detach().clone()
            for name, value in world_model.state_dict().items()
            if name.startswith(
                (
                    "rssm.image_embedder.",
                    "rssm.recurrent.",
                    "rssm.representation.",
                    "rssm.transition.",
                    "rssm.recurrent_mechanism_bank.mechanisms.0.",
                    "rssm.representation_mechanism_bank.mechanisms.0.",
                    "rssm.transition_mechanism_bank.mechanisms.0.",
                    "decoder.",
                    "decoder_experts.0.",
                    "reward_fc.",
                    "reward_experts.0.",
                    "continue_fc.",
                    "continue_experts.0.",
                )
            )
        }
        optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        rewards = torch.randn(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = world_model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            task_id=2,
        )
        loss.backward()
        route_parameters = (
            world_model.rssm.recurrent_mechanism_bank.routes[1].logits,
            world_model.rssm.representation_mechanism_bank.routes[1].logits,
            world_model.rssm.transition_mechanism_bank.routes[1].logits,
        )
        self.assertTrue(all(parameter.grad is not None for parameter in route_parameters))
        route_gradient_magnitudes = [
            float(parameter.grad.abs().sum()) for parameter in route_parameters
        ]
        self.assertTrue(
            all(magnitude > 0 for magnitude in route_gradient_magnitudes[:2]),
            route_gradient_magnitudes,
        )
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Loss/recon", metrics)
        for name, expected in frozen.items():
            torch.testing.assert_close(
                world_model.state_dict()[name], expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
