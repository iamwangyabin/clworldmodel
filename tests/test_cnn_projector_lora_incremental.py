"""Focused contracts for snapshot-seeded CNN projector/RSSM-LoRA training."""

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
    from config import Config
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class CnnProjectorLoraIncrementalTests(unittest.TestCase):
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
    def _method_config_data(cls) -> dict:
        data = cls._published_config_data()
        data["esc"]["env_configs"] = data["esc"]["env_configs"][:3]
        data.update(
            {
                "continual_method": "cnn_projector_lora_arrow",
                "rssm_num_experts": 3,
                "dino_fullbank_current_task_fraction": 1.0,
                "observation_objective": "reconstruction",
                "observation_encoder": "cnn",
                "task_banked_image_encoder": False,
                "task_projected_image_encoder": True,
                "task_projector_bottleneck_features": 64,
                "task_lora_recurrent_rank": 128,
                "task_lora_representation_rank": 128,
                "task_lora_transition_rank": 32,
                "compute_dtype": "bfloat16",
                "replay_observation_dtype": "uint8",
                "random_policy": "new",
                "actor_network": "mlp",
                "fresh_ac": False,
                "residual_correction": "none",
                "shared_core_mode": "task1_frozen_projector_lora",
            }
        )
        for replay_config in data["replay_buffers"]:
            replay_config["rb_device"] = "cpu"
        return data

    def test_config_is_explicit_and_rejects_one_shared_lora(self) -> None:
        config = Config.from_dict(self._method_config_data())
        self.assertTrue(config.uses_full_task_experts)
        self.assertTrue(config.task_projected_image_encoder)
        self.assertFalse(config.task_banked_image_encoder)
        self.assertEqual(
            (
                config.task_lora_recurrent_rank,
                config.task_lora_representation_rank,
                config.task_lora_transition_rank,
            ),
            (128, 128, 32),
        )

        invalid = self._method_config_data()
        invalid["task_lora_recurrent_rank"] = 32
        with self.assertRaisesRegex(ValueError, "fixes recurrent/representation"):
            Config.from_dict(invalid)

    def test_task2_updates_only_projector_lora_and_private_heads(self) -> None:
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
            num_task_experts=2,
            full_task_experts=True,
            task_projected_image_encoder=True,
            task_projector_bottleneck_features=64,
            task_lora_recurrent_rank=2,
            task_lora_representation_rank=2,
            task_lora_transition_rank=2,
            image_embedder=WideEmbedder(),
        )
        optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        world_model.activate_task_expert(0)
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        self.assertIs(
            world_model.rssm.recurrent_for(1)
            .rnn.parametrizations.weight_ih.original,
            world_model.rssm.recurrent.rnn.weight_ih,
        )
        self.assertIs(
            world_model.rssm.representation_for(1)
            .eh_to_inter[0]
            .parametrizations.weight.original,
            world_model.rssm.representation.eh_to_inter[0].weight,
        )
        self.assertIs(
            world_model.rssm.transition_for(1)
            .h_to_z_prior[0]
            .parametrizations.weight.original,
            world_model.rssm.transition.h_to_z_prior[0].weight,
        )

        observations = torch.rand(3, 2, 3, 64, 64)
        task1_features = world_model.rssm.embed_observations(
            observations, task_id=0
        )
        task2_features = world_model.rssm.embed_observations(
            observations, task_id=1
        )
        torch.testing.assert_close(task2_features, task1_features, rtol=0, atol=0)

        world_model.activate_task_expert(1)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in world_model.rssm.image_embedder.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in world_model.rssm.image_projectors[0].parameters()
            )
        )
        lora_parameters = [
            parameter
            for module in (
                world_model.rssm.recurrent_for(1),
                world_model.rssm.representation_for(1),
                world_model.rssm.transition_for(1),
            )
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and not name.endswith(".original")
        ]
        self.assertTrue(lora_parameters)
        self.assertFalse(
            any(
                parameter.requires_grad
                for module in (
                    world_model.rssm.recurrent_for(1),
                    world_model.rssm.representation_for(1),
                    world_model.rssm.transition_for(1),
                )
                for name, parameter in module.named_parameters()
                if name.endswith(".original")
            )
        )

        frozen_task1 = {
            name: value.detach().clone()
            for name, value in world_model.state_dict().items()
            if name.startswith(
                (
                    "rssm.image_embedder.",
                    "rssm.recurrent.",
                    "rssm.representation.",
                    "rssm.transition.",
                    "decoder.",
                    "reward_fc.",
                    "continue_fc.",
                )
            )
        }
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
            task_id=1,
        )
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Loss/recon", metrics)
        self.assertTrue(any(parameter.grad is not None for parameter in lora_parameters))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in world_model.rssm.image_projectors[0].parameters()
            )
        )
        for name, expected in frozen_task1.items():
            torch.testing.assert_close(
                world_model.state_dict()[name], expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
