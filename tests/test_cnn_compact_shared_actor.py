"""Focused contracts for compact RSSM routes and shared-actor rehearsal."""

from __future__ import annotations

import copy
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
    from ac import actor_policy_kl, dream_frozen_actor_policy
    from config import Config
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class CnnCompactSharedActorTests(unittest.TestCase):
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
                "continual_method": "cnn_compact_shared_actor_arrow",
                "rssm_num_experts": 3,
                "dino_fullbank_current_task_fraction": 1.0,
                "observation_objective": "reconstruction",
                "observation_encoder": "cnn",
                "task_banked_image_encoder": False,
                "task_projected_image_encoder": True,
                "task_projector_bottleneck_features": 64,
                "task_lora_recurrent_rank": 0,
                "task_lora_representation_rank": 32,
                "task_lora_transition_rank": 32,
                "task_recurrent_output_adapter_features": 32,
                "shared_actor_imagination_distillation": True,
                "shared_actor_distill_scale": 1.0,
                "shared_actor_distill_interval": 4,
                "shared_actor_distill_n_sync": 128,
                "shared_actor_distill_burnin_steps": 16,
                "shared_actor_distill_steps": 16,
                "compute_dtype": "bfloat16",
                "replay_observation_dtype": "uint8",
                "random_policy": "new",
                "actor_network": "mlp",
                "fresh_ac": False,
                "residual_correction": "none",
                "shared_core_mode": "task1_frozen_projector_compact_rssm",
            }
        )
        for replay_config in data["replay_buffers"]:
            replay_config["rb_device"] = "cpu"
        return data

    def test_config_fixes_compact_route_and_shared_actor_contract(self) -> None:
        config = Config.from_dict(self._method_config_data())
        self.assertTrue(config.uses_task_experts)
        self.assertTrue(config.uses_full_task_experts)
        self.assertTrue(config.uses_shared_actor)
        self.assertEqual(
            (
                config.task_lora_recurrent_rank,
                config.task_lora_representation_rank,
                config.task_lora_transition_rank,
                config.task_recurrent_output_adapter_features,
            ),
            (0, 32, 32, 32),
        )

        invalid = self._method_config_data()
        invalid["task_lora_representation_rank"] = 64
        with self.assertRaisesRegex(ValueError, "compact recurrent/representation"):
            Config.from_dict(invalid)

        invalid = self._method_config_data()
        invalid["shared_actor_imagination_distillation"] = False
        with self.assertRaisesRegex(ValueError, "imagination distillation"):
            Config.from_dict(invalid)

    @staticmethod
    def _tiny_world_model() -> "WorldModel":
        class WideEmbedder(nn.Module):
            output_size = 4096

            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                pooled = images.mean((-3, -2, -1), keepdim=False).unsqueeze(-1)
                return pooled.repeat(1, self.output_size) * self.scale

        return WorldModel(
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
            task_lora_recurrent_rank=0,
            task_lora_representation_rank=2,
            task_lora_transition_rank=2,
            task_recurrent_output_adapter_features=3,
            image_embedder=WideEmbedder(),
        )

    def test_task2_recurrent_route_shares_base_and_only_adapts_output(self) -> None:
        torch.manual_seed(19)
        world_model = self._tiny_world_model()
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        route = world_model.rssm.recurrent_for(1)

        self.assertIs(route.base, world_model.rssm.recurrent)
        self.assertFalse(
            any(
                key.startswith("rssm.recurrent_experts.0.base")
                for key in world_model.state_dict()
            )
        )
        adapter_parameters = sum(
            parameter.numel() for parameter in route.adapter.parameters()
        )
        self.assertLess(adapter_parameters, 100)

        z = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 0]]), num_classes=3
        ).float()
        action = torch.nn.functional.one_hot(torch.tensor([1, 3]), 4).float()
        hidden = torch.randn(2, 5)
        torch.testing.assert_close(
            route(z, action, hidden),
            world_model.rssm.recurrent(z, action, hidden),
            rtol=0,
            atol=0,
        )

        world_model.activate_task_expert(1)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in world_model.rssm.recurrent.parameters()
            )
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in route.adapter.parameters())
        )
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in world_model.rssm.representation_for(1).parameters()
            )
        )

    def test_frozen_world_model_imagination_supplies_actor_kl_targets(self) -> None:
        torch.manual_seed(29)
        world_model = self._tiny_world_model()
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        state_features = 2 * 3 + 5
        teacher = nn.Sequential(
            nn.Linear(state_features, 7),
            nn.SiLU(),
            nn.Linear(7, 4),
            nn.LogSoftmax(-1),
        )
        student = copy.deepcopy(teacher)
        states, teacher_logs = dream_frozen_actor_policy(
            world_model,
            teacher,
            task_id=1,
            n_sync=3,
            burnin_steps=2,
            dream_steps=4,
        )
        self.assertEqual(states.shape, (4, 3, state_features))
        self.assertEqual(teacher_logs.shape, (4, 3, 4))
        self.assertFalse(states.requires_grad)
        self.assertFalse(teacher_logs.requires_grad)
        torch.testing.assert_close(
            actor_policy_kl(student, states, teacher_logs),
            torch.zeros(()),
            rtol=0,
            atol=1e-6,
        )

        with torch.no_grad():
            student[2].bias.add_(torch.tensor([0.0, 1.0, -1.0, 0.5]))
        self.assertGreater(
            float(actor_policy_kl(student, states, teacher_logs).item()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
