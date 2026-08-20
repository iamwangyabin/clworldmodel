"""Focused contracts for the fixed-capacity KARROW Frozen-Core method."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import torch.nn as nn
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from ac import ActorCritic, ResidualCategoricalHead
    from clworldmodel.models.frozen_dinov3 import FrozenDinoV3Encoder
    from clworldmodel.models.residual_corrections import (
        LocalRBFKANCore,
        ParameterMatchedMLPCore,
        ResidualCorrection,
        soft_basis_support_overlap,
    )
    from clworldmodel.replay import ArrowFrozenFeatureCache
    from config import Config
    from replay import FifoReplay, LongTermReplay, MultiTypeReplay
    from rssm import Recurrent
    from wm import WorldModel, batch_standardized_smooth_l1


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class KarrowMethodTests(unittest.TestCase):
    @staticmethod
    def _published_config() -> Config:
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

    @classmethod
    def _karrow_config_data(cls, residual: str = "kan") -> dict:
        data = cls._published_config().to_dict()
        data.update(
            {
                "observation_objective": "dinov3_next_feature",
                "observation_encoder": "dinov3_vits16",
                "dinov3_model_path": "/tmp/frozen-dinov3-vits16",
                "actor_network": "mlp",
                "fresh_ac": False,
                "residual_correction": residual,
                "shared_core_mode": (
                    "freeze_after_first_task" if residual != "none" else "trainable"
                ),
            }
        )
        return data

    def test_config_accepts_three_arms_and_rejects_protocol_confounders(self) -> None:
        for residual in ("none", "mlp", "kan"):
            with self.subTest(residual=residual):
                config = Config.from_dict(self._karrow_config_data(residual))
                self.assertEqual(config.residual_correction, residual)
                self.assertEqual(config.observation_encoder, "dinov3_vits16")

        invalid = self._karrow_config_data("kan")
        invalid["shared_core_mode"] = "trainable"
        with self.assertRaisesRegex(ValueError, "shared_core_mode"):
            Config.from_dict(invalid)

        invalid = self._karrow_config_data()
        invalid["algorithm"] = "dv3"
        with self.assertRaisesRegex(ValueError, "ARROW mixed replay"):
            Config.from_dict(invalid)

        invalid = self._karrow_config_data()
        invalid["fresh_ac"] = 10
        with self.assertRaisesRegex(ValueError, "persistent actor-critic"):
            Config.from_dict(invalid)

    def test_config_isolates_spatial_patch_v2_from_completed_cls_v1(self) -> None:
        spatial = self._karrow_config_data("kan")
        spatial.update(
            {
                "observation_objective": "dinov3_posterior_feature",
                "dinov3_feature_mode": "patch_grid",
                "dinov3_patch_pool_size": 4,
                "dinov3_patch_feature_dim": 64,
                "dinov3_patch_projection": "task1_pca",
                "dinov3_patch_projection_frames": 512,
                "dinov3_feature_loss_kind": "batch_standardized_smooth_l1",
                "dinov3_feature_std_floor": 0.05,
            }
        )
        config = Config.from_dict(spatial)
        self.assertEqual(config.observation_objective, "dinov3_posterior_feature")
        self.assertEqual(config.dinov3_feature_mode, "patch_grid")
        self.assertEqual(config.dinov3_patch_feature_dim, 64)
        self.assertEqual(config.dinov3_patch_projection, "task1_pca")

        invalid = spatial.copy()
        invalid["dinov3_feature_mode"] = "cls"
        with self.assertRaisesRegex(ValueError, "spatial v2"):
            Config.from_dict(invalid)

        invalid = spatial.copy()
        invalid["dinov3_patch_feature_dim"] = 384
        with self.assertRaisesRegex(ValueError, "64 dimensions"):
            Config.from_dict(invalid)

        invalid = self._karrow_config_data("kan")
        invalid["dinov3_feature_mode"] = "patch_grid"
        with self.assertRaisesRegex(ValueError, "v1"):
            Config.from_dict(invalid)

    def test_kan_and_mlp_cores_are_exactly_parameter_matched(self) -> None:
        kan_core = LocalRBFKANCore(64, num_grids=8)
        mlp_core = ParameterMatchedMLPCore(64, num_grids=8)
        self.assertEqual(sum(p.numel() for p in kan_core.parameters()), 32_768)
        self.assertEqual(
            sum(p.numel() for p in kan_core.parameters()),
            sum(p.numel() for p in mlp_core.parameters()),
        )

        for in_features, out_features in ((512, 512), (512, 18), (512, 255)):
            kan = ResidualCorrection(in_features, out_features, kind="kan")
            mlp = ResidualCorrection(in_features, out_features, kind="mlp")
            self.assertEqual(
                sum(p.numel() for p in kan.parameters()),
                sum(p.numel() for p in mlp.parameters()),
            )
            inputs = torch.randn(3, in_features)
            torch.testing.assert_close(kan(inputs), torch.zeros(3, out_features))
            torch.testing.assert_close(mlp(inputs), torch.zeros(3, out_features))

    def test_fixed_basis_support_overlap_separates_distant_inputs(self) -> None:
        core = LocalRBFKANCore(4)
        first = core.basis_activations(torch.full((8, 4), -2.0))
        same = core.basis_activations(torch.full((8, 4), -2.0))
        distant = core.basis_activations(torch.full((8, 4), 2.0))
        torch.testing.assert_close(
            soft_basis_support_overlap(first, same), torch.tensor(1.0)
        )
        self.assertLess(soft_basis_support_overlap(first, distant).item(), 0.2)

    def test_zero_init_dynamics_residual_preserves_standard_grucell(self) -> None:
        kwargs = {
            "ls": (2, 3),
            "a_dim": 4,
            "h_dim": 5,
            "mlp_features": 8,
            "mlp_layers": 2,
        }
        torch.manual_seed(17)
        baseline = Recurrent(**kwargs)
        torch.manual_seed(17)
        karrow = Recurrent(**kwargs, residual_correction="kan")

        self.assertIsInstance(baseline.rnn, nn.GRUCell)
        self.assertIsInstance(karrow.rnn, nn.GRUCell)
        for name, parameter in baseline.named_parameters():
            torch.testing.assert_close(parameter, dict(karrow.named_parameters())[name])
        z = torch.randn(3, 2, 3)
        actions = torch.randn(3, 4)
        hidden = torch.randn(3, 5)
        torch.testing.assert_close(baseline(z, actions, hidden), karrow(z, actions, hidden))

    def test_zero_init_actor_critic_residuals_preserve_mlp_outputs(self) -> None:
        torch.manual_seed(23)
        baseline = ActorCritic(16, 4)
        torch.manual_seed(23)
        karrow = ActorCritic(16, 4, residual_correction="kan")
        self.assertIsInstance(karrow.actor, ResidualCategoricalHead)
        self.assertIsInstance(karrow.critic, ResidualCategoricalHead)
        state = torch.randn(5, 2, 16)
        torch.testing.assert_close(baseline.actor(state), karrow.actor(state))
        torch.testing.assert_close(baseline.critic(state), karrow.critic(state))

    def test_freezing_keeps_only_residual_adapters_trainable(self) -> None:
        class FakeEmbedder(nn.Module):
            output_size = 6

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((2, 3)).repeat(1, 2)

        world_model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="dinov3_next_feature",
            observation_encoder="dinov3_vits16",
            residual_correction="kan",
            image_embedder=FakeEmbedder(),
        )
        world_model.freeze_shared_core()

        for module in (
            world_model.rssm.image_embedder,
            world_model.rssm.recurrent.za_fcs,
            world_model.rssm.recurrent.rnn,
            world_model.rssm.representation.eh_to_inter,
            world_model.rssm.transition.h_to_z_prior,
            world_model.feature_predictor,
            world_model.reward_fc,
            world_model.continue_fc,
        ):
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        for module in (
            world_model.rssm.recurrent.residual,
            world_model.rssm.representation.residual,
            world_model.rssm.transition.residual,
            world_model.feature_predictor_residual,
            world_model.reward_residual,
            world_model.continue_residual,
        ):
            self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))

        actor_critic = ActorCritic(11, 4, residual_correction="kan")
        actor_critic.freeze_shared_core()
        for head in (actor_critic.actor, actor_critic.critic):
            self.assertTrue(all(not parameter.requires_grad for parameter in head.trunk.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in head.base_head.parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in head.residual.parameters()))

    def test_frozen_encoder_resizes_chunks_and_never_builds_gradients(self) -> None:
        class FakeBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(hidden_size=384)
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.batch_sizes: list[int] = []

            def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
                self.batch_sizes.append(pixel_values.shape[0])
                pooled = pixel_values.mean((1, 2, 3), keepdim=False).unsqueeze(-1)
                cls = (pooled * self.scale).expand(-1, 384)
                return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))

        backbone = FakeBackbone()
        encoder = FrozenDinoV3Encoder(
            None,
            input_size=256,
            max_batch_size=2,
            backbone=backbone,
        )
        encoder.train()
        output = encoder(torch.rand(5, 3, 64, 64, requires_grad=True))
        self.assertEqual(output.shape, (5, 384))
        self.assertEqual(backbone.batch_sizes, [2, 2, 1])
        self.assertFalse(output.requires_grad)
        self.assertFalse(encoder.training)
        self.assertFalse(backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_frozen_encoder_uses_pooled_patch_tokens_not_cls_or_registers(self) -> None:
        class FakeBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(
                    hidden_size=384,
                    patch_size=16,
                    num_register_tokens=2,
                )
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
                batch = pixel_values.shape[0]
                cls = torch.full((batch, 1, 384), -100.0, device=pixel_values.device)
                registers = torch.full(
                    (batch, 2, 384), -200.0, device=pixel_values.device
                )
                patch_ids = torch.arange(
                    256, device=pixel_values.device, dtype=pixel_values.dtype
                ).view(1, 256, 1)
                patches = (patch_ids * self.scale).expand(batch, -1, 384)
                return SimpleNamespace(
                    last_hidden_state=torch.cat((cls, registers, patches), dim=1)
                )

        encoder = FrozenDinoV3Encoder(
            None,
            input_size=256,
            feature_mode="patch_grid",
            patch_pool_size=4,
            patch_feature_dim=64,
            patch_projection="task1_pca",
            backbone=FakeBackbone(),
        )
        images = torch.rand(2, 3, 64, 64)
        raw_features = encoder.extract_patch_features(images)
        self.assertEqual(raw_features.shape, (2, 4, 4, 384))
        self.assertTrue(encoder.requires_projection_fit)
        with self.assertRaisesRegex(RuntimeError, "Fit the Task-1"):
            encoder(images)

        torch.manual_seed(41)
        calibration = torch.randn(8, 4, 4, 384)
        metadata = encoder.fit_patch_projection(calibration)
        self.assertEqual(metadata["calibration_patch_samples"], 128)
        self.assertFalse(encoder.requires_projection_fit)

        output = encoder(images)
        self.assertEqual(output.shape, (2, 4 * 4 * 64))
        grid = output.view(2, 4, 4, 64)
        source = torch.arange(256, dtype=grid.dtype).view(1, 1, 16, 16)
        expected = torch.nn.functional.adaptive_avg_pool2d(source, (4, 4))[0, 0]
        torch.testing.assert_close(raw_features[0, :, :, 0], expected)
        torch.testing.assert_close(
            grid, encoder.project_patch_features(raw_features),
        )
        gram = encoder.patch_projection.T @ encoder.patch_projection
        torch.testing.assert_close(gram, torch.eye(64), atol=2e-5, rtol=2e-5)
        self.assertFalse(encoder.patch_projection.requires_grad)

        restored = FrozenDinoV3Encoder(
            None,
            input_size=256,
            feature_mode="patch_grid",
            patch_pool_size=4,
            patch_feature_dim=64,
            patch_projection="task1_pca",
            backbone=FakeBackbone(),
        )
        restored.load_state_dict(encoder.state_dict())
        self.assertFalse(restored.requires_projection_fit)
        torch.testing.assert_close(restored(images), output)

    def test_standardized_feature_loss_rejects_constant_shortcut(self) -> None:
        targets = torch.tensor(
            [
                [[-2.0, 1.0], [0.0, 3.0]],
                [[2.0, 5.0], [4.0, 7.0]],
            ],
            requires_grad=True,
        )
        perfect, constant = batch_standardized_smooth_l1(
            targets.detach().clone(), targets, std_floor=0.05
        )
        torch.testing.assert_close(perfect, torch.zeros_like(perfect))
        self.assertGreater(constant.mean().item(), 0.1)

        predictions = torch.zeros_like(targets, requires_grad=True)
        losses, constant = batch_standardized_smooth_l1(
            predictions, targets, std_floor=0.05
        )
        torch.testing.assert_close(losses, constant)
        losses.mean().backward()
        self.assertIsNone(targets.grad)
        self.assertGreater(predictions.grad.abs().sum().item(), 0)

    def test_feature_objective_is_decoder_free_and_stops_target_gradient(self) -> None:
        class FakeEmbedder(nn.Module):
            output_size = 6

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((2, 3)).repeat(1, 2)

        torch.manual_seed(31)
        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="dinov3_next_feature",
            observation_encoder="dinov3_vits16",
            residual_correction="kan",
            image_embedder=FakeEmbedder(),
        )
        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        observations = torch.zeros(3, 2, 3, 64, 64)
        features = torch.randn(3, 2, 6, requires_grad=True)
        rewards = torch.randn(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)
        resets[1, 1] = 1

        loss, metrics = model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            observation_features=features,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(hasattr(model, "decoder"))
        self.assertIn("Loss/dinov3_feature", metrics)
        self.assertIsNone(features.grad)
        self.assertGreater(model.feature_predictor.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(
            metrics["Metric/dinov3_feature_valid_fraction"], torch.tensor(0.5)
        )

    def test_spatial_feature_objective_reconstructs_every_posterior_state(self) -> None:
        class FakeEmbedder(nn.Module):
            output_size = 6

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((2, 3)).repeat(1, 2)

        torch.manual_seed(37)
        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="dinov3_posterior_feature",
            observation_encoder="dinov3_vits16",
            dinov3_feature_loss_kind="batch_standardized_smooth_l1",
            dinov3_feature_std_floor=0.05,
            residual_correction="kan",
            image_embedder=FakeEmbedder(),
        )
        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        observations = torch.zeros(3, 2, 3, 64, 64)
        features = torch.randn(3, 2, 6, requires_grad=True)
        rewards = torch.randn(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)
        resets[1, 1] = 1

        loss, metrics = model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            observation_features=features,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(features.grad)
        self.assertGreater(model.feature_predictor.weight.grad.abs().sum().item(), 0)
        self.assertIn("Metric/dinov3_constant_feature_loss", metrics)
        self.assertIn("Metric/dinov3_model_to_constant_ratio", metrics)
        torch.testing.assert_close(
            metrics["Metric/dinov3_feature_valid_fraction"], torch.tensor(1.0)
        )

    def test_feature_cache_follows_arrow_write_and_sample_metadata(self) -> None:
        replay = MultiTypeReplay(
            FifoReplay(2, 2, 3, "cpu"),
            LongTermReplay(2, 2, 3, "cpu"),
        )
        cache = ArrowFrozenFeatureCache(replay, 3, dtype=torch.float16)
        actions = torch.zeros(2, 2, 3)
        observations = torch.zeros(2, 2, 3, 64, 64)
        rewards = torch.zeros(2, 2, 1)
        continues = torch.ones(2, 2, 1)
        resets = torch.zeros(2, 2, 1)
        features = torch.tensor(
            [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
             [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]]
        )
        np.random.seed(7)
        write_slots = replay.add(actions, observations, rewards, continues, resets)
        cache.record(write_slots, features)
        self.assertEqual(cache.storage_bytes, 48)

        random.seed(11)
        np.random.seed(13)
        sampled = cache.minibatch(2, 2, "cpu")
        sampled_features = sampled[2]
        self.assertEqual(sampled_features.shape, (2, 2, 3))
        for sequence in sampled_features.swapaxes(0, 1):
            self.assertTrue(
                torch.equal(sequence, features[:, 0])
                or torch.equal(sequence, features[:, 1])
            )

if __name__ == "__main__":
    unittest.main()
