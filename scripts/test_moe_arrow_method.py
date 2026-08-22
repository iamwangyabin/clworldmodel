"""Focused CPU contracts for the task-aware MoE-ARROW method."""

from __future__ import annotations

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
    from clworldmodel.continual import ActorCriticBank, allocate_task_updates
    from clworldmodel.models.frozen_dinov3 import FrozenDinoV3Encoder
    from ac import train_ac_from_wm
    from config import Config
    from replay import FifoReplay, MultiTypeReplay
    from rssm import Rssm
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class MoeArrowMethodTests(unittest.TestCase):
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
        data.update(
            {
                "continual_method": "moe_arrow",
                "rssm_num_experts": len(data["esc"]["env_configs"]),
                "moe_arrow_current_task_fraction": 0.5,
                "observation_objective": "dinov3_next_feature",
                "observation_encoder": "dinov3_vits16",
                "dinov3_model_path": "/tmp/frozen-dinov3-vits16",
                "dinov3_feature_mode": "patch_grid",
                "dinov3_patch_pool_size": 4,
                "dinov3_patch_feature_dim": 64,
                "dinov3_patch_projection": "fixed_orthogonal",
                "dinov3_patch_projection_frames": 0,
                "dinov3_patch_projection_seed": 0,
                "dinov3_feature_loss_kind": "cosine",
                "actor_network": "mlp",
                "fresh_ac": False,
                "residual_correction": "none",
                "shared_core_mode": "trainable",
            }
        )
        return data

    @classmethod
    def _fullbank_config_data(cls) -> dict:
        data = cls._method_config_data()
        data.update(
            {
                "continual_method": "dino_fullbank_arrow",
                "dino_fullbank_current_task_fraction": 1.0,
                "observation_objective": "dinov3_posterior_feature",
                "dinov3_feature_loss_kind": "batch_standardized_smooth_l1",
                "random_policy": "new",
                "shared_core_mode": "task_isolated",
            }
        )
        return data

    @classmethod
    def _patchbank_config_data(cls) -> dict:
        data = cls._fullbank_config_data()
        data.update(
            {
                "continual_method": "dino_patchbank_arrow",
                "observation_objective": "reconstruction",
                "dinov3_patch_pool_size": 16,
                "dinov3_patch_feature_dim": 384,
                "dinov3_patch_projection": "none",
                "dinov3_feature_loss_kind": "cosine",
            }
        )
        return data

    def test_config_requires_the_complete_named_task_aware_protocol(self) -> None:
        config = Config.from_dict(self._method_config_data())
        self.assertEqual(config.continual_method, "moe_arrow")
        self.assertEqual(config.rssm_num_experts, 6)

        invalid = self._method_config_data()
        invalid["rssm_num_experts"] = 2
        with self.assertRaisesRegex(ValueError, "one RSSM expert per scheduled task"):
            Config.from_dict(invalid)

        invalid = self._method_config_data()
        invalid["residual_correction"] = "kan"
        invalid["shared_core_mode"] = "freeze_after_first_task"
        with self.assertRaisesRegex(ValueError, "does not use residual corrections"):
            Config.from_dict(invalid)

        invalid = self._method_config_data()
        invalid["dinov3_patch_projection"] = "task1_pca"
        invalid["dinov3_patch_projection_frames"] = 512
        with self.assertRaisesRegex(ValueError, "task-independent fixed projection"):
            Config.from_dict(invalid)

    def test_fullbank_config_is_a_separate_fixed_protocol(self) -> None:
        config = Config.from_dict(self._fullbank_config_data())
        self.assertEqual(config.continual_method, "dino_fullbank_arrow")
        self.assertTrue(config.uses_task_experts)
        self.assertTrue(config.uses_full_task_experts)
        self.assertEqual(config.task_update_fraction, 1.0)

        invalid = self._fullbank_config_data()
        invalid["dino_fullbank_current_task_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "all updates to the current task"):
            Config.from_dict(invalid)

        invalid = self._fullbank_config_data()
        invalid["observation_objective"] = "dinov3_next_feature"
        invalid["dinov3_feature_loss_kind"] = "cosine"
        with self.assertRaisesRegex(ValueError, "posterior DINOv3 features"):
            Config.from_dict(invalid)

        invalid = self._fullbank_config_data()
        invalid["random_policy"] = "first"
        with self.assertRaisesRegex(ValueError, "random collection for each new task"):
            Config.from_dict(invalid)

        invalid = self._fullbank_config_data()
        invalid["shared_core_mode"] = "trainable"
        with self.assertRaisesRegex(ValueError, "task_isolated"):
            Config.from_dict(invalid)

    def test_patchbank_config_retains_all_patches_and_pixel_reconstruction(self) -> None:
        config = Config.from_dict(self._patchbank_config_data())
        self.assertEqual(config.continual_method, "dino_patchbank_arrow")
        self.assertEqual(config.observation_objective, "reconstruction")
        self.assertEqual(config.dinov3_patch_pool_size, 16)
        self.assertEqual(config.dinov3_patch_feature_dim, 384)
        self.assertEqual(config.dinov3_patch_projection, "none")
        self.assertTrue(config.uses_full_task_experts)

        invalid = self._patchbank_config_data()
        invalid["dinov3_patch_pool_size"] = 4
        with self.assertRaisesRegex(ValueError, "complete 16x16 patch grid"):
            Config.from_dict(invalid)

        invalid = self._patchbank_config_data()
        invalid["dinov3_patch_projection"] = "fixed_orthogonal"
        invalid["dinov3_patch_feature_dim"] = 64
        with self.assertRaisesRegex(ValueError, "all 384 patch channels"):
            Config.from_dict(invalid)

        invalid = self._patchbank_config_data()
        invalid["observation_objective"] = "dinov3_posterior_feature"
        invalid["dinov3_patch_pool_size"] = 4
        invalid["dinov3_patch_feature_dim"] = 64
        invalid["dinov3_patch_projection"] = "fixed_orthogonal"
        invalid["dinov3_feature_loss_kind"] = "batch_standardized_smooth_l1"
        with self.assertRaisesRegex(ValueError, "pixel reconstruction"):
            Config.from_dict(invalid)

    def test_fifo_and_mixed_replay_filter_homogeneous_task_minibatches(self) -> None:
        def values(marker: float, sequences: int = 2):
            actions = torch.full((3, sequences, 4), marker)
            observations = torch.full((3, sequences, 3, 64, 64), marker)
            rewards = torch.full((3, sequences, 1), marker)
            continues = torch.ones(3, sequences, 1)
            resets = torch.zeros(3, sequences, 1)
            return actions, observations, rewards, continues, resets

        first = FifoReplay(3, 4, 4, "cpu", store_task_ids=True)
        second = FifoReplay(3, 4, 4, "cpu", store_task_ids=True)
        replay = MultiTypeReplay(first, second, sampling_weights=(0.5, 0.5))
        replay.add(*values(10.0), task_id=0)
        replay.add(*values(20.0), task_id=1)

        np.random.seed(3)
        sample = replay.minibatch_with_metadata(2, 32, "cpu", task_id=1)
        actions, _, rewards, _, _, replay_index, _, sequence_indices = sample
        source = replay.replays[replay_index]
        sampled_task_ids = source.task_ids[sequence_indices]
        self.assertTrue(torch.equal(sampled_task_ids, torch.ones_like(sampled_task_ids)))
        self.assertTrue(torch.equal(actions, torch.full_like(actions, 20.0)))
        self.assertTrue(torch.equal(rewards, torch.full_like(rewards, 20.0)))
        self.assertEqual(replay.available_task_ids(), (0, 1))

        with self.assertRaisesRegex(ValueError, "task 7"):
            replay.minibatch(2, 1, "cpu", task_id=7)
        with self.assertRaisesRegex(ValueError, "requires task_id"):
            first.add(*values(30.0))
        with self.assertRaisesRegex(TypeError, "integer"):
            first.add(*values(30.0), task_id=[1, 1])

        baseline = FifoReplay(3, 2, 4, "cpu")
        self.assertIsNone(baseline.task_ids)
        baseline.add(*values(30.0))
        with self.assertRaisesRegex(ValueError, "without task-id storage"):
            baseline.add(*values(40.0), task_id=0)

    def test_hard_routing_updates_only_the_selected_dynamics_expert(self) -> None:
        class Embedder(nn.Module):
            output_size = 7

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((-2, -1))[:, :1].expand(-1, self.output_size)

        torch.manual_seed(5)
        rssm = Rssm(
            3,
            (2, 3),
            4,
            5,
            4,
            8,
            2,
            image_embedder=Embedder(),
            num_task_experts=2,
        )
        self.assertIs(rssm.representation_for(0), rssm.representation_for(1))
        self.assertEqual(len(rssm.representation_experts), 0)
        with torch.no_grad():
            rssm.recurrent_experts[0].rnn.bias_ih[0].add_(0.5)

        previous_z = torch.zeros(3, 2, 3)
        previous_action = torch.zeros(3, 4)
        previous_action[:, 0] = 1
        previous_hidden = torch.zeros(3, 5)
        reset = torch.zeros(3, 1)

        _, _, hidden_zero = rssm(
            previous_z,
            previous_action,
            previous_hidden,
            None,
            reset,
            stochastic=False,
            task_id=0,
        )
        _, _, hidden_one = rssm(
            previous_z,
            previous_action,
            previous_hidden,
            None,
            reset,
            stochastic=False,
            task_id=1,
        )
        self.assertFalse(torch.allclose(hidden_zero, hidden_one))

        rssm.zero_grad(set_to_none=True)
        hidden_one.square().sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in rssm.recurrent.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in rssm.recurrent_experts[0].parameters()
            )
        )

        with self.assertRaisesRegex(ValueError, "task_id"):
            rssm(
                previous_z,
                previous_action,
                previous_hidden,
                None,
                reset,
                task_id=torch.tensor([0, 1, 0]),
            )

    def test_update_allocation_preserves_the_total_budget(self) -> None:
        allocation = allocate_task_updates(
            800,
            current_task_id=2,
            available_task_ids=(0, 1, 2),
            current_task_fraction=0.5,
        )
        self.assertEqual(sum(allocation.values()), 800)
        self.assertEqual(allocation[2], 400)
        self.assertEqual(allocation[0], 200)
        self.assertEqual(allocation[1], 200)

        first_task = allocate_task_updates(
            800,
            current_task_id=0,
            available_task_ids=(0,),
            current_task_fraction=0.5,
        )
        self.assertEqual(first_task, {0: 800})

        current_only = allocate_task_updates(
            800,
            current_task_id=2,
            available_task_ids=(0, 1, 2),
            current_task_fraction=1.0,
        )
        self.assertEqual(current_only[2], 800)
        self.assertEqual(current_only[0], 0)
        self.assertEqual(current_only[1], 0)

    def test_fixed_patch_projection_is_task_independent_and_rng_isolated(self) -> None:
        class Backbone(nn.Module):
            config = SimpleNamespace(
                hidden_size=384,
                patch_size=16,
                num_register_tokens=0,
            )

            def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
                batch = pixel_values.shape[0]
                tokens = torch.arange(
                    (1 + 16 * 16) * 384,
                    dtype=torch.float32,
                    device=pixel_values.device,
                ).reshape(1, 1 + 16 * 16, 384)
                return SimpleNamespace(last_hidden_state=tokens.expand(batch, -1, -1))

        torch.manual_seed(17)
        expected_next_random = torch.rand(4)
        torch.manual_seed(17)
        first = FrozenDinoV3Encoder(
            None,
            feature_mode="patch_grid",
            patch_pool_size=4,
            patch_feature_dim=64,
            patch_projection="fixed_orthogonal",
            patch_projection_seed=9,
            backbone=Backbone(),
        )
        actual_next_random = torch.rand(4)
        torch.testing.assert_close(actual_next_random, expected_next_random)

        second = FrozenDinoV3Encoder(
            None,
            feature_mode="patch_grid",
            patch_pool_size=4,
            patch_feature_dim=64,
            patch_projection="fixed_orthogonal",
            patch_projection_seed=9,
            backbone=Backbone(),
        )
        torch.testing.assert_close(first.patch_projection, second.patch_projection)
        torch.testing.assert_close(
            first.patch_projection.T @ first.patch_projection,
            torch.eye(64),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertFalse(first.requires_projection_fit)
        self.assertEqual(first(torch.zeros(2, 3, 64, 64)).shape, (2, 1024))

    def test_full_patch_encoder_preserves_every_spatial_token_coordinate(self) -> None:
        class Backbone(nn.Module):
            config = SimpleNamespace(
                hidden_size=384,
                patch_size=16,
                num_register_tokens=0,
            )

            def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
                batch = pixel_values.shape[0]
                patch_tokens = torch.arange(
                    16 * 16 * 384,
                    dtype=torch.float32,
                    device=pixel_values.device,
                ).reshape(1, 16 * 16, 384)
                cls = torch.full(
                    (1, 1, 384),
                    -1.0,
                    dtype=torch.float32,
                    device=pixel_values.device,
                )
                tokens = torch.cat((cls, patch_tokens), dim=1)
                return SimpleNamespace(last_hidden_state=tokens.expand(batch, -1, -1))

        encoder = FrozenDinoV3Encoder(
            None,
            feature_mode="patch_grid",
            patch_pool_size=16,
            patch_feature_dim=384,
            patch_projection="none",
            backbone=Backbone(),
        )
        encoded = encoder(torch.zeros(2, 3, 64, 64))
        self.assertEqual(encoder.output_size, 16 * 16 * 384)
        self.assertEqual(encoded.shape, (2, 16 * 16 * 384))
        torch.testing.assert_close(
            encoded[0], torch.arange(16 * 16 * 384, dtype=torch.float32)
        )

    def test_actor_bank_warm_starts_weights_but_not_optimizer_objects(self) -> None:
        def factory(task_id: int):
            actor_critic = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            return SimpleNamespace(
                ac=actor_critic,
                opt=optimizer,
                slow_critic=None,
                return_scale_ema=torch.tensor(float(task_id)),
                return_mean_ema=None,
            )

        bank = ActorCriticBank()
        source = bank.ensure(0, factory)
        with torch.no_grad():
            source.ac.weight.fill_(3.0)
        target = bank.ensure(1, factory, warm_start_from=0)

        torch.testing.assert_close(target.ac.weight, source.ac.weight)
        self.assertIsNot(target.ac, source.ac)
        self.assertIsNot(target.opt, source.opt)
        self.assertEqual(target.opt.state, {})
        self.assertEqual(target.return_scale_ema.item(), 0.0)

    def test_actor_bank_can_create_a_fresh_task_and_freeze_old_actors(self) -> None:
        def factory(task_id: int):
            actor_critic = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            return SimpleNamespace(
                ac=actor_critic,
                opt=optimizer,
                slow_critic=None,
                return_scale_ema=torch.tensor(float(task_id)),
                return_mean_ema=None,
            )

        torch.manual_seed(13)
        bank = ActorCriticBank()
        source = bank.ensure(0, factory)
        with torch.no_grad():
            source.ac.weight.fill_(3.0)
        target = bank.ensure(1, factory)

        self.assertFalse(torch.equal(target.ac.weight, source.ac.weight))
        bank.activate(1)
        self.assertFalse(any(parameter.requires_grad for parameter in source.ac.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in target.ac.parameters()))

    def test_fullbank_routes_and_trains_the_complete_selected_world_model(self) -> None:
        class Embedder(nn.Module):
            output_size = 6

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((-2, -1)).repeat(1, 2)

        torch.manual_seed(19)
        world_model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=4,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="dinov3_posterior_feature",
            dinov3_feature_loss_kind="batch_standardized_smooth_l1",
            num_task_experts=2,
            full_task_experts=True,
            image_embedder=Embedder(),
        )
        optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

        world_model.activate_task_expert(0)
        self.assertIsNot(
            world_model.rssm.representation_for(0),
            world_model.rssm.representation_for(1),
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in world_model.rssm.representation.parameters())
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in world_model.rssm.representation_experts[0].parameters()
            )
        )
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        torch.testing.assert_close(
            world_model.feature_predictor_for(1).weight,
            world_model.feature_predictor_for(0).weight,
        )

        world_model.activate_task_expert(1)
        self.assertFalse(
            any(parameter.requires_grad for parameter in world_model.rssm.representation.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in world_model.rssm.representation_experts[0].parameters()
            )
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in world_model.feature_predictor.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in world_model.feature_predictor_experts[0].parameters()
            )
        )
        frozen_representation = {
            name: value.detach().clone()
            for name, value in world_model.rssm.representation.state_dict().items()
        }
        selected_feature_weight = (
            world_model.feature_predictor_experts[0].weight.detach().clone()
        )

        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        observations = torch.zeros(3, 2, 3, 64, 64)
        features = torch.randn(3, 2, 6, requires_grad=True)
        rewards = torch.randn(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)

        loss, metrics = world_model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            observation_features=features,
            task_id=1,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Metric/dinov3_constant_feature_loss", metrics)
        self.assertIsNone(features.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in world_model.rssm.representation.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in world_model.rssm.representation_experts[0].parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in world_model.feature_predictor.parameters())
        )
        self.assertGreater(
            world_model.feature_predictor_experts[0].weight.grad.abs().sum().item(),
            0,
        )
        optimizer.step()
        for name, value in world_model.rssm.representation.state_dict().items():
            torch.testing.assert_close(value, frozen_representation[name], rtol=0, atol=0)
        self.assertFalse(
            torch.equal(
                world_model.feature_predictor_experts[0].weight,
                selected_feature_weight,
            )
        )

    def test_patchbank_routes_pixel_decoders_and_precomputed_full_features(self) -> None:
        class Embedder(nn.Module):
            output_size = 12

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean((-2, -1)).repeat(1, 4)

        torch.manual_seed(29)
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
            image_embedder=Embedder(),
        )
        self.assertIsNot(world_model.decoder_for(0), world_model.decoder_for(1))
        world_model.activate_task_expert(0)
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        for source, target in zip(
            world_model.decoder_for(0).parameters(),
            world_model.decoder_for(1).parameters(),
        ):
            torch.testing.assert_close(source, target)

        world_model.activate_task_expert(1)
        self.assertFalse(any(p.requires_grad for p in world_model.decoder_for(0).parameters()))
        self.assertTrue(all(p.requires_grad for p in world_model.decoder_for(1).parameters()))

        actions = torch.nn.functional.one_hot(
            torch.tensor([[0, 1], [2, 3], [1, 0]]), num_classes=4
        ).float()
        observations = torch.rand(3, 2, 3, 64, 64)
        features = torch.randn(3, 2, Embedder.output_size, requires_grad=True)
        rewards = torch.randn(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)

        loss, metrics = world_model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            observation_features=features,
            task_id=1,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Loss/recon", metrics)
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in world_model.decoder_for(0).parameters()))
        self.assertTrue(any(p.grad is not None for p in world_model.decoder_for(1).parameters()))
        self.assertTrue(
            all(
                p.grad is None
                for p in world_model.rssm.representation_for(0).parameters()
            )
        )
        self.assertTrue(
            any(
                p.grad is not None
                for p in world_model.rssm.representation_for(1).parameters()
            )
        )

    def test_tiny_task_routed_world_model_and_actor_update(self) -> None:
        class Embedder(nn.Module):
            output_size = 7

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                pooled = images.mean((-2, -1))
                return torch.cat((pooled, pooled[:, :1].expand(-1, 4)), dim=-1)

        torch.manual_seed(23)
        world_model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=4,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="dinov3_next_feature",
            dinov3_feature_loss_kind="cosine",
            num_task_experts=2,
            image_embedder=Embedder(),
        )
        self.assertTrue(world_model.initialize_task_expert(1, 0))

        replay = FifoReplay(4, 2, 4, "cpu", store_task_ids=True)
        action_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        actions = torch.nn.functional.one_hot(action_indices, 4).float()
        observations = torch.rand(4, 2, 3, 64, 64)
        rewards = torch.randn(4, 2, 1)
        continues = torch.ones(4, 2, 1)
        resets = torch.zeros(4, 2, 1)
        replay.add(
            actions,
            observations,
            rewards,
            continues,
            resets,
            task_id=1,
        )

        loss, metrics = world_model.compute_loss(
            actions,
            observations,
            rewards,
            continues,
            resets,
            task_id=1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("Loss/dinov3_feature", metrics)
        loss.backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in world_model.rssm.recurrent.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in world_model.rssm.recurrent_experts[0].parameters()
            )
        )

        actor_opt, performance, actor_metrics = train_ac_from_wm(
            world_model,
            replay,
            steps=1,
            n_sync=2,
            dream_steps=2,
            lr=1e-4,
            task_id=1,
        )
        self.assertTrue(torch.isfinite(performance))
        self.assertTrue(actor_opt.ac.training)
        self.assertTrue(np.isfinite(actor_metrics["total_loss"]))


if __name__ == "__main__":
    unittest.main()
