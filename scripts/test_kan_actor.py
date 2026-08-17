"""Focused contracts for fixed-grid and trainable-anchor ReLU-KAN actors."""

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
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from ac import ActorCritic
    from clworldmodel.models.relu_kan import (
        AdaptiveReLUKANActor,
        BoundedReLUKANActor,
        FixedGridReLUKANLayer,
        ReLUKANActor,
        TrainableAnchorReLUKANLayer,
    )
    from config import Config


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class KANActorTests(unittest.TestCase):
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

    def test_basis_matches_fixed_relu_kan_formula_fixture(self) -> None:
        layer = FixedGridReLUKANLayer(
            1, 1, grid_size=5, spline_order=3
        ).double()
        basis = layer.basis_activations(torch.tensor([[0.0]], dtype=torch.float64))
        expected = torch.tensor(
            [0.5625, 1.0, 0.5625, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        torch.testing.assert_close(basis[0, 0], expected, rtol=1e-6, atol=1e-7)

    def test_trainable_anchors_initialize_to_the_fixed_grid_formula(self) -> None:
        fixed = FixedGridReLUKANLayer(2, 3, grid_size=5, spline_order=3).double()
        adaptive = TrainableAnchorReLUKANLayer(
            2, 3, grid_size=5, spline_order=3
        ).double()
        inputs = torch.tensor([[0.0, 0.2], [0.6, 1.0]], dtype=torch.float64)

        torch.testing.assert_close(
            adaptive.basis_activations(inputs),
            fixed.basis_activations(inputs),
            rtol=2e-6,
            atol=2e-7,
        )
        self.assertEqual(adaptive.anchor_start.shape, (2, 8))
        self.assertEqual(adaptive.anchor_raw_width.shape, (2, 8))
        torch.testing.assert_close(
            adaptive.anchor_widths(),
            adaptive.initial_anchor_width,
            rtol=2e-6,
            atol=2e-7,
        )

    def test_trainable_anchor_layer_receives_anchor_gradients_and_orders_supports(self) -> None:
        layer = TrainableAnchorReLUKANLayer(2, 2, grid_size=3, spline_order=1)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.zero_()
        inputs = torch.tensor([[0.15, 0.3], [0.5, 0.7], [0.8, 0.9]])

        layer(inputs).square().mean().backward()

        for name in ("anchor_start", "anchor_raw_width", "weight"):
            gradient = dict(layer.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)
        self.assertTrue(torch.all(layer.anchor_ends() > layer.anchor_start))

    def test_adaptive_actor_starts_with_the_bounded_actor_function(self) -> None:
        torch.manual_seed(23)
        bounded = BoundedReLUKANActor(
            10,
            3,
            recurrent_features=4,
            hidden_features=2,
            grid_size=3,
            spline_order=1,
        )
        torch.manual_seed(23)
        adaptive = AdaptiveReLUKANActor(
            10,
            3,
            recurrent_features=4,
            hidden_features=2,
            grid_size=3,
            spline_order=1,
        )
        state = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8, -0.2, 0.3, 0.9]],
            dtype=torch.float32,
        )

        for layer_index in (0, 1):
            torch.testing.assert_close(
                adaptive.network.layers[layer_index].weight,
                bounded.network.layers[layer_index].weight,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                adaptive.network.layers[layer_index].bias,
                bounded.network.layers[layer_index].bias,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            adaptive(state), bounded(state), rtol=2e-6, atol=2e-7
        )

    def test_actor_supports_time_batch_axes_and_actor_gradients(self) -> None:
        torch.manual_seed(5)
        actor = ReLUKANActor(
            10,
            3,
            recurrent_features=4,
            hidden_features=2,
            grid_size=3,
            spline_order=1,
        )
        discrete = torch.nn.functional.one_hot(
            torch.randint(0, 6, (4, 2)), num_classes=6
        ).float()
        recurrent = 2 * torch.rand(4, 2, 4) - 1
        state = torch.cat((discrete, recurrent), dim=-1)

        layer_inputs = []
        hook = actor.network.layers[0].register_forward_pre_hook(
            lambda _module, args: layer_inputs.append(args[0].detach())
        )
        action_logs = actor(state)
        hook.remove()
        self.assertEqual(action_logs.shape, (4, 2, 3))
        torch.testing.assert_close(layer_inputs[0][..., :6], discrete)
        torch.testing.assert_close(
            layer_inputs[0][..., 6:], 0.5 * (recurrent + 1.0)
        )
        torch.testing.assert_close(
            torch.logsumexp(action_logs, dim=-1), torch.zeros(4, 2), atol=1e-6, rtol=0
        )

        (-action_logs[..., 0].mean()).backward()
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in actor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0)
        parameter_names = {name for name, _ in actor.named_parameters()}
        buffer_names = {name for name, _ in actor.named_buffers()}
        self.assertFalse(any("phase_" in name for name in parameter_names))
        self.assertIn("network.layers.0.phase_low", buffer_names)
        self.assertIn("network.layers.0.phase_high", buffer_names)

    def test_full_actor_matches_original_trainable_parameter_budget(self) -> None:
        kan_actor = ReLUKANActor(
            1536,
            18,
            recurrent_features=512,
            hidden_features=64,
            grid_size=5,
            spline_order=3,
        )
        mlp_actor = ActorCritic(1536, 18, actor_network="mlp").actor

        kan_parameters = sum(parameter.numel() for parameter in kan_actor.parameters())
        mlp_parameters = sum(parameter.numel() for parameter in mlp_actor.parameters())
        self.assertEqual(kan_parameters, 795_730)
        self.assertEqual(mlp_parameters, 797_202)
        self.assertLess(abs(kan_parameters - mlp_parameters) / mlp_parameters, 0.002)

    def test_bounded_actor_keeps_every_second_layer_input_on_the_fixed_grid(self) -> None:
        torch.manual_seed(23)
        actor = BoundedReLUKANActor(
            10,
            3,
            recurrent_features=4,
            hidden_features=2,
            grid_size=3,
            spline_order=1,
        )
        discrete = torch.nn.functional.one_hot(
            torch.randint(0, 6, (4, 2)), num_classes=6
        ).float()
        recurrent = 2 * torch.rand(4, 2, 4) - 1
        state = torch.cat((discrete, recurrent), dim=-1)

        second_layer_inputs = []
        hook = actor.network.layers[1].register_forward_pre_hook(
            lambda _module, args: second_layer_inputs.append(args[0].detach())
        )
        action_logs = actor(state)
        hook.remove()

        hidden = second_layer_inputs[0]
        self.assertTrue(torch.all(hidden > 0.0))
        self.assertTrue(torch.all(hidden < 1.0))
        active_basis = actor.network.layers[1].basis_activations(hidden).ne(0).any(-1)
        self.assertTrue(torch.all(active_basis))
        self.assertEqual(action_logs.shape, (4, 2, 3))

        (-action_logs[..., 0].mean()).backward()
        for name in (
            "network.layers.0.weight",
            "network.layers.1.weight",
            "hidden_adapter.0.weight",
        ):
            gradient = dict(actor.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)

    def test_bounded_actor_remains_parameter_matched_to_the_mlp(self) -> None:
        bounded_actor = BoundedReLUKANActor(
            1536,
            18,
            recurrent_features=512,
            hidden_features=64,
            grid_size=5,
            spline_order=3,
        )
        bounded_parameters = sum(
            parameter.numel() for parameter in bounded_actor.parameters()
        )
        self.assertEqual(bounded_parameters, 795_858)
        self.assertLess(abs(bounded_parameters - 797_202) / 797_202, 0.002)

    def test_adaptive_actor_keeps_the_bounded_interface_and_learns_anchors(self) -> None:
        torch.manual_seed(41)
        actor = AdaptiveReLUKANActor(
            10,
            3,
            recurrent_features=4,
            hidden_features=2,
            grid_size=3,
            spline_order=1,
        )
        discrete = torch.nn.functional.one_hot(
            torch.randint(0, 6, (4, 2)), num_classes=6
        ).float()
        recurrent = 2 * torch.rand(4, 2, 4) - 1
        state = torch.cat((discrete, recurrent), dim=-1)

        second_layer_inputs = []
        hook = actor.network.layers[1].register_forward_pre_hook(
            lambda _module, args: second_layer_inputs.append(args[0].detach())
        )
        action_logs = actor(state)
        hook.remove()

        hidden = second_layer_inputs[0]
        self.assertTrue(torch.all(hidden > 0.0))
        self.assertTrue(torch.all(hidden < 1.0))
        self.assertTrue(
            torch.all(actor.network.layers[1].basis_activations(hidden).ne(0).any(-1))
        )
        self.assertEqual(action_logs.shape, (4, 2, 3))

        (-action_logs[..., 0].mean()).backward()
        parameters = dict(actor.named_parameters())
        for name in (
            "network.layers.0.anchor_start",
            "network.layers.0.anchor_raw_width",
            "network.layers.1.anchor_start",
            "network.layers.1.anchor_raw_width",
            "hidden_adapter.0.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)

    def test_adaptive_actor_records_the_intentional_anchor_parameter_increase(self) -> None:
        adaptive_actor = AdaptiveReLUKANActor(
            1536,
            18,
            recurrent_features=512,
            hidden_features=64,
            grid_size=5,
            spline_order=3,
        )
        adaptive_parameters = sum(
            parameter.numel() for parameter in adaptive_actor.parameters()
        )
        self.assertEqual(adaptive_parameters, 821_458)
        self.assertEqual(adaptive_parameters - 795_858, 25_600)

    def test_explicit_mlp_option_preserves_default_actor_initialization(self) -> None:
        torch.manual_seed(17)
        default = ActorCritic(16, 4)
        torch.manual_seed(17)
        explicit = ActorCritic(16, 4, actor_network="mlp")

        for name, default_value in default.state_dict().items():
            torch.testing.assert_close(
                default_value, explicit.state_dict()[name], rtol=0, atol=0
            )

    def test_config_defaults_to_mlp_and_validates_matched_kan(self) -> None:
        published = self._published_config()
        self.assertEqual(published.actor_network, "mlp")

        kan_data = published.to_dict()
        kan_data["actor_network"] = "relu_kan"
        kan = Config.from_dict(kan_data)
        self.assertEqual(kan.actor_kan_hidden_features, 64)
        self.assertEqual(kan.actor_kan_grid_size + kan.actor_kan_spline_order, 8)
        self.assertEqual(
            (kan.actor_kan_input_min, kan.actor_kan_input_max), (0.0, 1.0)
        )

        kan_data["actor_network"] = "relu_kan_bounded"
        bounded = Config.from_dict(kan_data)
        self.assertEqual(bounded.actor_network, "relu_kan_bounded")

        kan_data["actor_kan_hidden_features"] = 63
        with self.assertRaisesRegex(ValueError, "coefficient matching"):
            Config.from_dict(kan_data)

        kan_data["actor_kan_hidden_features"] = 64
        kan_data["actor_kan_input_min"] = -1.0
        with self.assertRaisesRegex(ValueError, r"\[0, 1\] grid"):
            Config.from_dict(kan_data)

        kan_data["actor_kan_input_min"] = 0.0
        kan_data["actor_kan_trainable_grid"] = True
        with self.assertRaisesRegex(ValueError, "Only relu_kan_adaptive"):
            Config.from_dict(kan_data)

        kan_data["actor_network"] = "relu_kan_adaptive"
        adaptive = Config.from_dict(kan_data)
        self.assertTrue(adaptive.actor_kan_trainable_grid)

        kan_data["actor_kan_trainable_grid"] = False
        with self.assertRaisesRegex(ValueError, "requires actor_kan_trainable_grid=True"):
            Config.from_dict(kan_data)

    def test_actor_critic_changes_only_actor_architecture(self) -> None:
        actor_critic = ActorCritic(
            16,
            4,
            actor_network="relu_kan",
            h_dim=6,
            kan_hidden_features=2,
            kan_grid_size=3,
            kan_spline_order=1,
        )
        self.assertIsInstance(actor_critic.actor, ReLUKANActor)
        self.assertIsInstance(actor_critic.critic, nn.Sequential)
        self.assertIsInstance(actor_critic.critic[-1], nn.LogSoftmax)
        self.assertEqual(actor_critic.critic[-2].out_features, 255)

        bounded_actor_critic = ActorCritic(
            16,
            4,
            actor_network="relu_kan_bounded",
            h_dim=6,
            kan_hidden_features=2,
            kan_grid_size=3,
            kan_spline_order=1,
        )
        self.assertIsInstance(bounded_actor_critic.actor, BoundedReLUKANActor)
        self.assertIsInstance(bounded_actor_critic.critic, nn.Sequential)

        adaptive_actor_critic = ActorCritic(
            16,
            4,
            actor_network="relu_kan_adaptive",
            h_dim=6,
            kan_hidden_features=2,
            kan_grid_size=3,
            kan_spline_order=1,
        )
        self.assertIsInstance(adaptive_actor_critic.actor, AdaptiveReLUKANActor)
        self.assertIsInstance(adaptive_actor_critic.critic, nn.Sequential)


if __name__ == "__main__":
    unittest.main()
