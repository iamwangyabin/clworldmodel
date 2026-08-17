"""Focused contracts for the KAN-Dreamer-aligned FastKAN behavior heads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
PROJECT_SRC = ROOT / "src"

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised on the GPU host
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from ac import ActorCritic
    from clworldmodel.models.fast_kan import (
        FastKANActor,
        FastKANCritic,
        FastKANLayer,
        FixedGaussianRBF,
    )
    from clworldmodel.optim import LaProp
    from config import Config


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class FastKANActorCriticTests(unittest.TestCase):
    def _published_config(self) -> Config:
        return Config.from_file(
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

    def _fastkan_config_data(self) -> dict:
        data = self._published_config().to_dict()
        data.update(
            {
                "actor_network": "fast_kan_ac",
                "ac_optimizer": "laprop",
                "ac_lr": 4e-5,
                "ac_fresh_lr": 4e-5,
                "ac_optimizer_eps": 1e-20,
                "ac_optimizer_warmup_steps": 1000,
                "ac_agc_clip": 0.3,
                "ac_grad_clip": 0.0,
                "ac_dream_steps": 15,
                "ac_discount": 1.0 - 1.0 / 333.0,
                "ac_persistent_return_norm": True,
                "ac_slow_critic_regularizer": 1.0,
            }
        )
        return data

    def test_rbf_centers_are_fixed_uniform_buffers(self) -> None:
        rbf = FixedGaussianRBF(grid_min=-2.0, grid_max=2.0, num_grids=8)
        self.assertEqual(dict(rbf.named_parameters()), {})
        self.assertIn("centers", dict(rbf.named_buffers()))
        torch.testing.assert_close(rbf.centers, torch.linspace(-2.0, 2.0, 8))
        self.assertAlmostEqual(rbf.bandwidth, 4.0 / 7.0)
        basis = rbf(rbf.centers)
        torch.testing.assert_close(torch.diagonal(basis), torch.ones(8))

    def test_layer_is_vectorized_and_both_branches_receive_gradients(self) -> None:
        torch.manual_seed(7)
        layer = FastKANLayer(5, 3, num_grids=8)
        inputs = torch.randn(4, 2, 5)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, (4, 2, 3))
        self.assertEqual(layer.basis_activations(inputs).shape, (4, 2, 5, 8))
        outputs.square().mean().backward()
        for name in ("rbf_weight", "base_weight", "norm.scale"):
            gradient = dict(layer.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)

    def test_actor_unimix_and_critic_zero_output_support_time_batch_axes(self) -> None:
        torch.manual_seed(11)
        actor = FastKANActor(10, 4, hidden_features=3, hidden_layers=3)
        critic = FastKANCritic(10, 255, hidden_features=3, hidden_layers=3)
        state = torch.randn(5, 2, 10)

        action_logs = actor(state)
        critic_logs = critic(state)
        self.assertEqual(action_logs.shape, (5, 2, 4))
        self.assertEqual(critic_logs.shape, (5, 2, 255))
        torch.testing.assert_close(
            torch.logsumexp(action_logs, -1), torch.zeros(5, 2), atol=1e-6, rtol=0
        )
        self.assertTrue(torch.all(action_logs.exp() >= 0.01 / 4))
        torch.testing.assert_close(
            critic_logs.exp(),
            torch.full_like(critic_logs, 1.0 / 255),
            atol=1e-7,
            rtol=1e-6,
        )

    def test_arrow_bridge_replaces_actor_and_critic(self) -> None:
        actor_critic = ActorCritic(16, 4, actor_network="fast_kan_ac")
        self.assertIsInstance(actor_critic.actor, FastKANActor)
        self.assertIsInstance(actor_critic.critic, FastKANCritic)
        action_logs, values = actor_critic(torch.randn(3, 2, 16))
        self.assertEqual(action_logs.shape, (3, 2, 4))
        self.assertEqual(values.shape, (3, 2, 1))

    def test_atari_parameter_accounting_is_explicitly_not_mlp_matched(self) -> None:
        actor_critic = ActorCritic(1536, 18, actor_network="fast_kan_ac")
        actor_parameters = sum(p.numel() for p in actor_critic.actor.parameters())
        critic_parameters = sum(p.numel() for p in actor_critic.critic.parameters())
        self.assertEqual(actor_parameters, 498_090)
        self.assertEqual(critic_parameters, 570_849)
        self.assertEqual(actor_parameters + critic_parameters, 1_068_939)
        self.assertLess(actor_parameters + critic_parameters, 1_714_961)

    def test_named_config_requires_paper_aligned_behavior_settings(self) -> None:
        config = Config.from_dict(self._fastkan_config_data())
        self.assertEqual(config.fastkan_hidden_features, 34)
        self.assertEqual(config.fastkan_grid_size, 8)
        self.assertEqual(config.ac_optimizer, "laprop")
        self.assertEqual(config.ac_dream_steps, 15)

        invalid = self._fastkan_config_data()
        invalid["fastkan_grid_size"] = 7
        with self.assertRaisesRegex(ValueError, "KAN-Dreamer-aligned"):
            Config.from_dict(invalid)

    def test_laprop_warmup_and_agc_state_match_the_documented_order(self) -> None:
        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        optimizer = LaProp(
            [parameter],
            lr=0.1,
            betas=(0.0, 0.5),
            agc_clip=0.3,
            warmup_steps=2,
        )
        parameter.grad = torch.tensor([6.0, 8.0])
        original = parameter.detach().clone()
        optimizer.step()

        torch.testing.assert_close(parameter, original)
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg_sq"],
            0.5 * torch.tensor([0.9, 1.2]).square(),
        )

        parameter.grad = torch.tensor([6.0, 8.0])
        optimizer.step()
        self.assertFalse(torch.equal(parameter, original))


if __name__ == "__main__":
    unittest.main()
