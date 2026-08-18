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
    from ac import ActorCritic, dream_rollout, replay_lambda_returns
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

    def _parameter_matched_fastkan_config_data(self) -> dict:
        data = self._fastkan_config_data()
        data.update(
            {
                "actor_network": "fast_kan_ac_param_matched",
                "fastkan_hidden_features": 53,
                "ac_replay_critic_loss_scale": 0.3,
            }
        )
        return data

    def _stable_fastkan_config_data(self) -> dict:
        data = self._parameter_matched_fastkan_config_data()
        data.update(
            {
                "actor_network": "fast_kan_ac_stable",
                "ac_use_slow_critic_targets": True,
                "ac_corrected_imagination_bootstrap": True,
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

    def test_width_53_fastkan_pair_matches_the_arrow_mlp_budget(self) -> None:
        actor_critic = ActorCritic(
            1536,
            18,
            actor_network="fast_kan_ac_param_matched",
            fastkan_hidden_features=53,
        )
        actor_parameters = sum(p.numel() for p in actor_critic.actor.parameters())
        critic_parameters = sum(p.numel() for p in actor_critic.critic.parameters())
        self.assertEqual(actor_parameters, 793_692)
        self.assertEqual(critic_parameters, 906_978)
        self.assertEqual(actor_parameters + critic_parameters, 1_700_670)
        self.assertEqual(actor_parameters + critic_parameters - 1_714_961, -14_291)

    def test_stable_bridge_keeps_both_parameter_matched_fastkan_heads(self) -> None:
        actor_critic = ActorCritic(
            1536,
            18,
            actor_network="fast_kan_ac_stable",
            fastkan_hidden_features=53,
        )
        self.assertIsInstance(actor_critic.actor, FastKANActor)
        self.assertIsInstance(actor_critic.critic, FastKANCritic)
        self.assertEqual(
            sum(parameter.numel() for parameter in actor_critic.parameters()),
            1_700_670,
        )

    def test_slow_value_baseline_decouples_actor_from_online_critic(self) -> None:
        torch.manual_seed(13)
        actor_critic = ActorCritic(
            6,
            3,
            actor_network="fast_kan_ac_stable",
            fastkan_hidden_features=4,
        )
        states = torch.randn(2, 1, 6)
        actions = torch.zeros(2, 1, 3)
        actions[..., 0] = 1.0
        returns = torch.full((2, 1, 1), 2.0)

        reinforce_zero, _, _ = actor_critic.compute_loss(
            states,
            actions,
            returns,
            torch.tensor(1.0),
            actor_baseline_values=torch.zeros_like(returns),
        )
        reinforce_one, _, _ = actor_critic.compute_loss(
            states,
            actions,
            returns,
            torch.tensor(1.0),
            actor_baseline_values=torch.ones_like(returns),
        )
        torch.testing.assert_close(reinforce_zero, 2.0 * reinforce_one)

    def test_corrected_rollout_bootstraps_the_post_transition_state(self) -> None:
        class DummyRssm:
            @staticmethod
            def initial_state(n_sync: int):
                return torch.zeros(n_sync, 1, 1), torch.zeros(n_sync, 1)

            def __call__(
                self,
                z,
                actions,
                h,
                images,
                resets,
                temperature=1.0,
            ):
                del resets, temperature
                if images is not None:
                    time, batch = actions.shape[:2]
                    context_z = torch.zeros(time, batch, 1, 1)
                    context_h = torch.arange(1, time + 1, dtype=torch.float32)
                    context_h = context_h.view(time, 1, 1).expand(time, batch, 1)
                    return None, context_z, context_h
                return None, z, h + 1.0

        class DummyWorldModel:
            rssm = DummyRssm()

            @staticmethod
            def zh_transform(z, h):
                del z
                return h

            @staticmethod
            def reward_fc(zh):
                return torch.zeros_like(zh)

            @staticmethod
            def continue_fc(zh):
                return torch.ones_like(zh)

        class DummyReplay:
            @staticmethod
            def minibatch(time, batch, mb_device):
                actions = torch.zeros(time, batch, 2, device=mb_device)
                images = torch.zeros(time, batch, 3, 64, 64, device=mb_device)
                rewards = torch.zeros(time, batch, 1, device=mb_device)
                continues = torch.ones(time, batch, 1, device=mb_device)
                resets = torch.zeros(time, batch, 1, device=mb_device)
                return actions, images, rewards, continues, resets

        class DummyActorCritic:
            def __call__(self, state):
                action_logs = torch.full(
                    (*state.shape[:-1], 2),
                    -0.6931471805599453,
                    device=state.device,
                )
                return action_logs, state[..., -1:]

        rollout_kwargs = {
            "wm": DummyWorldModel(),
            "ac": DummyActorCritic(),
            "data": DummyReplay(),
            "n_sync": 1,
            "n_steps": 1,
            "discount": 0.5,
        }
        legacy_returns = dream_rollout(**rollout_kwargs)[3]
        corrected_returns = dream_rollout(
            **rollout_kwargs,
            corrected_terminal_bootstrap=True,
        )[3]

        torch.testing.assert_close(legacy_returns, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(corrected_returns, torch.tensor([[[2.5]]]))

    def test_replay_lambda_returns_use_same_index_rewards_and_stop_at_terminals(self) -> None:
        rewards = torch.tensor([[[1.0]], [[2.0]], [[4.0]], [[8.0]]])
        continues = torch.tensor([[[1.0]], [[0.0]], [[1.0]], [[1.0]]])
        bootstrap = torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
        targets = replay_lambda_returns(
            rewards,
            continues,
            bootstrap,
            discount=0.5,
            lam=0.5,
        )
        expected = torch.tensor([[[6.5]], [[2.0]], [[24.0]]])
        torch.testing.assert_close(targets, expected)

    def test_parameter_matched_config_requires_replay_value_loss(self) -> None:
        config = Config.from_dict(self._parameter_matched_fastkan_config_data())
        self.assertEqual(config.fastkan_hidden_features, 53)
        self.assertEqual(config.ac_replay_critic_loss_scale, 0.3)

        invalid = self._parameter_matched_fastkan_config_data()
        invalid["ac_replay_critic_loss_scale"] = 0.0
        with self.assertRaisesRegex(ValueError, "FastKAN settings"):
            Config.from_dict(invalid)

    def test_stable_config_requires_slow_targets_and_correct_terminal_bootstrap(self) -> None:
        config = Config.from_dict(self._stable_fastkan_config_data())
        self.assertTrue(config.ac_use_slow_critic_targets)
        self.assertTrue(config.ac_corrected_imagination_bootstrap)

        invalid = self._stable_fastkan_config_data()
        invalid["ac_corrected_imagination_bootstrap"] = False
        with self.assertRaisesRegex(ValueError, "FastKAN settings"):
            Config.from_dict(invalid)

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
