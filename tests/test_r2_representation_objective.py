"""Focused parity and gradient contracts for the opt-in R2 objective."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
PROJECT_SRC = ROOT / "src"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.models.r2 import barlow_twins_loss
    from config import Config
    from rssm import Rssm
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class R2RepresentationObjectiveTests(unittest.TestCase):
    def test_published_config_defaults_to_reconstruction_and_validates_r2(self) -> None:
        config_path = (
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
        published = Config.from_file(config_path)
        self.assertEqual(published.observation_objective, "reconstruction")

        r2_data = published.to_dict()
        r2_data["observation_objective"] = "r2"
        r2 = Config.from_dict(r2_data)
        self.assertEqual(r2.observation_objective, "r2")
        self.assertEqual(r2.r2_barlow_loss_scale, 0.05)
        self.assertEqual(r2.r2_redundancy_scale, 5e-4)

        r2_data["r2_normalization_eps"] = 0
        with self.assertRaisesRegex(ValueError, "r2_normalization_eps"):
            Config.from_dict(r2_data)

    def test_barlow_loss_matches_fixed_official_formula_fixture(self) -> None:
        projected = torch.tensor(
            [[1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 4.0], [4.0, 3.0, 2.0]],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 3.0, 0.0], [4.0, 2.0, 3.0], [2.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

        loss, invariance, redundancy = barlow_twins_loss(projected, target)

        self.assertAlmostEqual(invariance.item(), 3.6661276450556723, places=12)
        self.assertAlmostEqual(redundancy.item(), 0.681428554593361, places=12)
        self.assertAlmostEqual(loss.item(), 3.666468359332969, places=12)

    def test_barlow_target_is_stop_gradient(self) -> None:
        projected = torch.randn(8, 5, requires_grad=True)
        target = torch.randn(8, 5, requires_grad=True)

        loss, _, _ = barlow_twins_loss(projected, target)
        loss.backward()

        self.assertIsNotNone(projected.grad)
        self.assertGreater(projected.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_precomputed_embeddings_preserve_rssm_posterior_semantics(self) -> None:
        torch.manual_seed(7)
        rssm = Rssm(
            img_channels=3,
            ls=(2, 3),
            a_dim=4,
            h_dim=5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
        ).eval()
        observations = torch.rand(3, 2, 3, 64, 64)
        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        resets = torch.zeros(3, 2, 1)
        initial_z, initial_h = rssm.initial_state(2)

        embeddings = rssm.embed_observations(observations)
        for stochastic in (False, True):
            with self.subTest(stochastic=stochastic):
                torch.manual_seed(101)
                direct = rssm(
                    initial_z,
                    actions,
                    initial_h,
                    observations,
                    resets,
                    stochastic=stochastic,
                )
                torch.manual_seed(101)
                precomputed = rssm.observe_embeddings(
                    initial_z,
                    actions,
                    initial_h,
                    embeddings,
                    resets,
                    stochastic=stochastic,
                )

                for direct_tensor, precomputed_tensor in zip(direct, precomputed):
                    torch.testing.assert_close(
                        direct_tensor, precomputed_tensor, rtol=0, atol=0
                    )

    def test_r2_world_model_replaces_decoder_with_bias_free_projector(self) -> None:
        reconstruction = WorldModel(
            3, (2, 3), 4, 5, cnn_depth=2, mlp_features=8, mlp_layers=2
        )
        r2 = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="r2",
        )

        self.assertTrue(hasattr(reconstruction, "decoder"))
        self.assertFalse(hasattr(reconstruction, "r2_projector"))
        self.assertFalse(hasattr(r2, "decoder"))
        self.assertTrue(hasattr(r2, "r2_projector"))
        self.assertIsNone(r2.r2_projector.bias)
        self.assertEqual(
            tuple(r2.r2_projector.weight.shape),
            (r2.rssm.image_embedder.output_size, 2 * 3 + 5),
        )
        self.assertLess(
            sum(parameter.numel() for parameter in r2.parameters()),
            sum(parameter.numel() for parameter in reconstruction.parameters()),
        )

    def test_r2_world_model_loss_is_finite_and_updates_student_path(self) -> None:
        torch.manual_seed(11)
        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="r2",
        )
        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        observations = torch.rand(3, 2, 3, 64, 64)
        rewards = torch.randn(3, 2, 1)
        continues = torch.tensor(
            [[[1.0], [0.0]], [[1.0], [1.0]], [[0.0], [1.0]]]
        )
        resets = torch.zeros(3, 2, 1)

        loss, metrics = model.compute_loss(
            actions, observations, rewards, continues, resets
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Loss/r2_barlow", metrics)
        self.assertIn("Loss/r2_barlow_scaled", metrics)
        torch.testing.assert_close(
            metrics["Loss/r2_barlow_scaled"], 0.05 * metrics["Loss/r2_barlow"]
        )
        self.assertNotIn("Loss/recon", metrics)
        self.assertNotIn("Metric/low_kl_recon_loss", metrics)
        self.assertGreater(model.r2_projector.weight.grad.abs().sum().item(), 0)
        encoder_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.rssm.image_embedder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0)


if __name__ == "__main__":
    unittest.main()
