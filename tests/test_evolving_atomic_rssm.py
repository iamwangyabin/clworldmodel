"""Focused contracts for the Evolving-Core Atomic RSSM protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
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
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.continual import project_component_gradients
    from clworldmodel.continual.evolving_core import (
        _gradient_in_parameter_layout,
    )
    from config import Config
    from replay import FifoReplay, LongTermReplay, MultiTypeReplay
    import train
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EvolvingAtomicRssmTests(unittest.TestCase):
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
                "continual_method": "evolving_atomic_rssm_arrow",
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
                "task_mechanism_reuse": True,
                "task_mechanism_capacity_profile": "matched_512",
                "task_mechanism_recurrent_width": 512,
                "task_mechanism_representation_width": 512,
                "task_mechanism_transition_width": 256,
                "task_mechanism_residual_scale": 0.1,
                "task_mechanism_num_atoms": 4,
                "task_mechanism_reuse_probe_epochs": 0,
                "task_mechanism_route_lr_scale": 1.0,
                "task_mechanism_consolidation_batches": 8,
                "task_mechanism_min_contribution": 0.01,
                "task_mechanism_max_validation_drop": 0.05,
                "compute_dtype": "bfloat16",
                "replay_observation_dtype": "uint8",
                "random_policy": "new",
                "actor_network": "mlp",
                "fresh_ac": False,
                "evaluation_seed_protocol": "fixed_validation_heldout_final",
                "residual_correction": "none",
                "shared_core_mode": "evolving_replay_protected",
                "evolving_task0_profile": "fixed_v2",
                "evolving_shared_core": True,
                "first_task_shared_core_lr": 3e-4,
                "shared_core_lr": 1e-4,
                "task_private_lr": 2e-4,
                "task_route_lr": 1e-3,
                "current_batch_n": 12,
                "memory_batch_n": 4,
                "memory_loss_scale": 1.0,
                "interface_q_scale": 0.1,
                "interface_h_scale": 0.05,
                "interface_actor_scale": 0.05,
                "component_gradient_projection": True,
                "task_atom_output_regularization": 1e-4,
                "boundary_consolidation_steps": 1000,
                "boundary_consolidation_lr": 2e-5,
                "boundary_max_return_drop": 0.05,
                "task_private_heads": True,
                "task_private_actor_critic": True,
                "task_atomic_routes": True,
                "full_task_rssm_experts": False,
            }
        )
        for replay_config in data["replay_buffers"]:
            replay_config["rb_device"] = "cpu"
        return data

    @staticmethod
    def _world_model(
        mechanism_parameterization: str = "dense_private",
    ) -> WorldModel:
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
            task_private_heads=True,
            evolving_shared_core=True,
            task_projected_image_encoder=True,
            task_symmetric_image_projectors=True,
            task_projector_bottleneck_features=64,
            task_mechanism_bank=True,
            task_mechanism_reuse=True,
            task_mechanism_recurrent_width=8,
            task_mechanism_representation_width=8,
            task_mechanism_transition_width=8,
            task_mechanism_num_atoms=4,
            task_mechanism_parameterization=mechanism_parameterization,
            task_symmetric_mechanisms=True,
            image_embedder=WideEmbedder(),
        )

    @staticmethod
    def _replay_batch(time: int, sequences: int, task_id: int) -> tuple:
        action_ids = torch.full((time, sequences), task_id % 4)
        actions = torch.nn.functional.one_hot(action_ids, 4).float()
        observations = torch.full(
            (time, sequences, 3, 64, 64), task_id / 10.0
        )
        rewards = torch.full((time, sequences, 1), float(task_id))
        continues = torch.ones(time, sequences, 1)
        resets = torch.zeros(time, sequences, 1)
        return actions, observations, rewards, continues, resets

    def test_named_config_separates_private_heads_from_full_rssm_copies(self) -> None:
        config = Config.from_dict(self._method_config_data())

        self.assertTrue(config.uses_task_experts)
        self.assertFalse(config.uses_full_task_experts)
        self.assertTrue(config.uses_task_private_heads)
        self.assertFalse(config.uses_full_task_rssm_experts)
        self.assertTrue(config.evolving_shared_core)
        self.assertEqual(
            config.evolving_checkpoint_retention, "all_boundaries"
        )

        invalid = self._method_config_data()
        invalid["task_private_heads"] = False
        with self.assertRaisesRegex(ValueError, "fixed optimizer, replay, interface"):
            Config.from_dict(invalid)

        rolling = self._method_config_data()
        rolling["evolving_checkpoint_retention"] = "latest_boundary"
        self.assertEqual(
            Config.from_dict(rolling).evolving_checkpoint_retention,
            "latest_boundary",
        )

        invalid_retention = self._method_config_data()
        invalid_retention["evolving_checkpoint_retention"] = "unknown"
        with self.assertRaisesRegex(ValueError, "checkpoint retention"):
            Config.from_dict(invalid_retention)

    def test_task0_has_zero_effect_projector_atoms_and_private_heads(self) -> None:
        torch.manual_seed(7)
        wm = self._world_model()

        self.assertEqual(len(wm.rssm.image_projectors), 3)
        self.assertIs(
            wm.rssm.image_projector_for(0), wm.rssm.image_projectors[0]
        )
        for bank in (
            wm.rssm.recurrent_mechanism_bank,
            wm.rssm.representation_mechanism_bank,
            wm.rssm.transition_mechanism_bank,
        ):
            self.assertTrue(bank.include_task0)
            self.assertIsNotNone(bank.mechanism_for(0))

        features = torch.randn(3, 4096)
        torch.testing.assert_close(
            wm.rssm.image_projector_for(0)(features), features, rtol=0, atol=0
        )
        hidden = torch.randn(3, 8)
        correction, current = (
            wm.rssm.recurrent_mechanism_bank.forward_with_current(hidden, 0)
        )
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        torch.testing.assert_close(current, torch.zeros_like(current))
        self.assertIsNot(wm.decoder_for(0), wm.decoder_for(1))
        self.assertIsNot(
            wm._head_for(wm.reward_fc, wm.reward_experts, 0),
            wm._head_for(wm.reward_fc, wm.reward_experts, 1),
        )

    def test_shared_down_world_model_keeps_basis_out_of_optimizer_ownership(
        self,
    ) -> None:
        wm = self._world_model("shared_frozen_down_film")
        banks = (
            wm.rssm.recurrent_mechanism_bank,
            wm.rssm.representation_mechanism_bank,
            wm.rssm.transition_mechanism_bank,
        )
        shared_down_ids = {
            id(parameter)
            for bank in banks
            for parameter in bank.shared_down.parameters()
        }
        self.assertEqual(
            wm.rssm.task_mechanism_parameterization,
            "shared_frozen_down_film",
        )
        self.assertTrue(shared_down_ids)
        self.assertTrue(
            all(
                id(parameter) not in shared_down_ids
                for task_id in range(3)
                for parameter in (
                    *wm.private_parameters(task_id),
                    *wm.route_parameters(task_id),
                )
            )
        )
        self.assertTrue(
            all(
                id(parameter) not in shared_down_ids
                for parameters in wm.shared_parameter_groups().values()
                for parameter in parameters
            )
        )

        wm.activate_task_expert(0)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for bank in banks
                for parameter in bank.shared_down.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in wm.private_parameters(0)
            )
        )

        copied = copy.deepcopy(wm)
        for bank in (
            copied.rssm.recurrent_mechanism_bank,
            copied.rssm.representation_mechanism_bank,
            copied.rssm.transition_mechanism_bank,
        ):
            for mechanism in bank.mechanisms:
                self.assertIs(mechanism.down_projection(), bank.shared_down)

    def test_task_activation_keeps_shared_core_and_only_current_private_plastic(self) -> None:
        wm = self._world_model()
        self.assertTrue(wm.initialize_task_expert(1, 0))
        self.assertTrue(wm.initialize_task_expert(2, 1))
        wm.activate_task_expert(2)

        shared = {
            id(parameter)
            for values in wm.shared_parameter_groups().values()
            for parameter in values
        }
        task2 = {
            id(parameter)
            for parameter in (*wm.private_parameters(2), *wm.route_parameters(2))
        }
        old = {
            id(parameter)
            for task_id in (0, 1)
            for parameter in (
                *wm.private_parameters(task_id),
                *wm.route_parameters(task_id),
            )
        }
        self.assertFalse(shared & task2)
        self.assertFalse((shared | task2) & old)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in wm.shared_parameter_groups()["encoder"]
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in (*wm.private_parameters(2), *wm.route_parameters(2))
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad and parameter.grad is None
                for task_id in (0, 1)
                for parameter in (*wm.private_parameters(task_id), *wm.route_parameters(task_id))
            )
        )

    def test_component_projection_removes_only_conflicting_current_direction(self) -> None:
        current = (torch.tensor([1.0, -2.0]), torch.tensor([0.5]))
        memory = (torch.tensor([-2.0, 1.0]), torch.tensor([-0.5]))

        combined, diagnostic = project_component_gradients(
            current, memory, memory_scale=0.0
        )

        projected_dot = sum((left * right).sum() for left, right in zip(combined, memory))
        self.assertTrue(diagnostic.conflicted)
        self.assertGreaterEqual(float(projected_dot), -1e-6)

    def test_assigned_gradient_uses_parameter_stride_for_fused_adam(self) -> None:
        parameter = nn.Parameter(torch.zeros(4, 4, 4, 4))
        channels_last = torch.arange(
            parameter.numel(), dtype=parameter.dtype
        ).reshape_as(parameter).contiguous(memory_format=torch.channels_last)
        self.assertNotEqual(channels_last.stride(), parameter.stride())

        assigned = _gradient_in_parameter_layout(parameter, channels_last)

        self.assertEqual(assigned.stride(), parameter.stride())
        torch.testing.assert_close(assigned, channels_last)

    def test_task_indexed_ltdm_sampling_uses_cached_pure_slots(self) -> None:
        fifo = FifoReplay(2, 4, 4, "cpu", store_task_ids=True)
        ltdm = LongTermReplay(2, 4, 4, "cpu", store_task_ids=True)
        replay = MultiTypeReplay(fifo, ltdm, sampling_weights=(0.5, 0.5))
        for task_id in (0, 1):
            np.random.seed(10 + task_id)
            replay.add(*self._replay_batch(2, 2, task_id), task_id=task_id)

        self.assertEqual(set(ltdm.valid_slots_by_task), {0, 1})
        batch = replay.minibatch_for_task(
            1, sequence_length=2, sequences=8, source="ltdm", mb_device="cpu"
        )
        self.assertEqual(batch.task_ids.unique().tolist(), [1])
        torch.testing.assert_close(batch[2], torch.ones_like(batch[2]))

    def test_shared_adam_state_survives_task_activation(self) -> None:
        wm = self._world_model()
        shared = [
            parameter
            for values in wm.shared_parameter_groups().values()
            for parameter in values
        ]
        optimizer = torch.optim.Adam(shared, lr=2e-4)
        loss = sum(parameter.square().sum() for parameter in shared)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        parameter = shared[0]
        before = int(optimizer.state[parameter]["step"].item())

        wm.initialize_task_expert(1, 0)
        wm.activate_task_expert(1)
        optimizer.param_groups[0]["lr"] = 1e-4
        loss = sum(parameter.square().sum() for parameter in shared)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        after = int(optimizer.state[parameter]["step"].item())

        self.assertEqual(after, before + 1)

    def test_complete_checkpoint_round_trip_restores_training_state(self) -> None:
        class FakeActorBank:
            def __init__(self) -> None:
                self.value = torch.tensor([3.0])
                self.restored = False

            def resumable_state_dict(self) -> dict:
                return {
                    "schema_version": 1,
                    "resumable": True,
                    "value": self.value.clone(),
                }

            def load_resumable_state_dict(self, state, _factory) -> None:
                self.value = state["value"].clone()
                self.restored = True

        config = Config.from_dict(self._method_config_data())
        world_model = nn.Linear(2, 2)
        teacher = copy.deepcopy(world_model)
        shared_optimizer = torch.optim.Adam([world_model.weight], lr=1e-4)
        private_optimizers = {
            0: torch.optim.Adam([world_model.bias], lr=2e-4)
        }
        for optimizer, loss in (
            (shared_optimizer, world_model.weight.square().sum()),
            (private_optimizers[0], world_model.bias.square().sum()),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        fifo = FifoReplay(2, 4, 4, "cpu", store_task_ids=True)
        ltdm = LongTermReplay(2, 4, 4, "cpu", store_task_ids=True)
        replay = MultiTypeReplay(fifo, ltdm, sampling_weights=(0.5, 0.5))
        np.random.seed(41)
        replay.add(*self._replay_batch(2, 2, 0), task_id=0)
        expected_world_model = copy.deepcopy(world_model.state_dict())
        expected_rewards = ltdm.rews.clone()
        actor_bank = FakeActorBank()
        schedule = SimpleNamespace(_step=4)
        generators = [np.random.default_rng(seed) for seed in range(4)]

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "boundary.pt"
            train._save_evolving_resumable_checkpoint(
                checkpoint,
                config=config,
                wm=world_model,
                boundary_teacher=teacher,
                shared_optimizer=shared_optimizer,
                private_optimizers=private_optimizers,
                route_optimizers={},
                actor_critic_bank=actor_bank,
                replay_buffer=replay,
                environment_schedule=schedule,
                epoch=4,
                current_task_id=0,
                world_model_updates=17,
                actor_critic_updates=11,
                total_env_steps=101,
                task_update_rng=generators[0],
                collection_environment_seed_rng=generators[1],
                validation_environment_seed_rng=generators[2],
                final_environment_seed_rng=generators[3],
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                set(payload),
                {
                    "schema_version",
                    "artifact_kind",
                    "resumable",
                    "config",
                    "world_model",
                    "boundary_teacher",
                    "optimizers",
                    "replay",
                    "rng",
                    "schedule",
                    "counters",
                    "replay_checkpoint_semantics",
                },
            )
            self.assertIn("actor_critic_bank", payload["optimizers"])
            self.assertTrue(checkpoint.with_suffix(".pt.sha256").is_file())

            with torch.no_grad():
                world_model.weight.add_(10)
                teacher.weight.sub_(10)
                ltdm.rews.zero_()
            actor_bank.value.zero_()
            counters = train._restore_evolving_resumable_checkpoint(
                checkpoint,
                config=config,
                wm=world_model,
                boundary_teacher=teacher,
                shared_optimizer=shared_optimizer,
                private_optimizers=private_optimizers,
                route_optimizers={},
                actor_critic_bank=actor_bank,
                actor_critic_factory=lambda _task_id: None,
                replay_buffer=replay,
                environment_schedule=schedule,
                task_update_rng=generators[0],
                collection_environment_seed_rng=generators[1],
                validation_environment_seed_rng=generators[2],
                final_environment_seed_rng=generators[3],
            )

        for name, expected in expected_world_model.items():
            torch.testing.assert_close(world_model.state_dict()[name], expected)
        torch.testing.assert_close(ltdm.rews, expected_rewards)
        self.assertTrue(actor_bank.restored)
        self.assertEqual(schedule._step, 5)
        self.assertEqual(counters["completed_epochs"], 5)
        self.assertEqual(counters["world_model_updates"], 17)
        self.assertEqual(counters["actor_critic_updates"], 11)
        self.assertEqual(counters["raw_environment_frames"], 101)

    def test_mmap_checkpoint_asset_is_not_mutated_after_restore(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = FifoReplay(
                2,
                2,
                4,
                "cpu",
                store_task_ids=True,
                observation_storage_path=root / "source.mmap",
                observation_dtype="uint8",
            )
            source.add(*self._replay_batch(2, 2, 0), task_id=0)
            checkpoint = root / "task_00_pre_consolidation.pt"
            state = train._snapshot_checkpoint_replay_mmaps(
                source.state_dict(), checkpoint_path=checkpoint
            )
            observations = state["observations"]
            asset = Path(observations["path"])
            digest_before = hashlib.sha256(asset.read_bytes()).hexdigest()

            target = FifoReplay(
                2,
                2,
                4,
                "cpu",
                store_task_ids=True,
                observation_storage_path=root / "working.mmap",
                observation_dtype="uint8",
            )
            target.load_state_dict(state)
            target.obss.fill_(255)
            digest_after = hashlib.sha256(asset.read_bytes()).hexdigest()

            self.assertEqual(digest_after, digest_before)
            self.assertNotEqual(target.observation_storage_path, asset)

    def test_latest_boundary_retention_prunes_only_after_new_pair_is_durable(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            old_asset = checkpoint_dir / "task_00_replay_assets"
            old_asset.mkdir()
            (old_asset / "observations.mmap").write_bytes(b"old-replay")
            for task_id in (0, 1):
                for phase in ("pre_consolidation", "post_consolidation"):
                    path = checkpoint_dir / f"task_{task_id:02d}_{phase}.pt"
                    path.write_bytes(f"task-{task_id}-{phase}".encode())
                    path.with_suffix(".pt.sha256").write_text(
                        "fixture checksum\n", encoding="utf-8"
                    )

            artifact = train._apply_evolving_checkpoint_retention(
                checkpoint_dir,
                completed_task_id=1,
                retention="latest_boundary",
            )

            self.assertFalse(old_asset.exists())
            for phase in ("pre_consolidation", "post_consolidation"):
                self.assertFalse(
                    (checkpoint_dir / f"task_00_{phase}.pt").exists()
                )
                current = checkpoint_dir / f"task_01_{phase}.pt"
                self.assertTrue(current.is_file())
                self.assertTrue(current.with_suffix(".pt.sha256").is_file())
            self.assertEqual(artifact["retention"], "latest_boundary")
            self.assertTrue((checkpoint_dir / "retention.json").is_file())

    def test_latest_boundary_retention_never_prunes_for_incomplete_new_pair(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            old = checkpoint_dir / "task_00_pre_consolidation.pt"
            old.write_bytes(b"old")
            old.with_suffix(".pt.sha256").write_text("checksum\n", encoding="utf-8")
            current = checkpoint_dir / "task_01_pre_consolidation.pt"
            current.write_bytes(b"current")
            current.with_suffix(".pt.sha256").write_text(
                "checksum\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(FileNotFoundError, "complete current"):
                train._apply_evolving_checkpoint_retention(
                    checkpoint_dir,
                    completed_task_id=1,
                    retention="latest_boundary",
                )

            self.assertTrue(old.is_file())
            self.assertTrue(old.with_suffix(".pt.sha256").is_file())

    def test_consolidation_evaluation_error_restores_core_and_adam(self) -> None:
        class TinyWorldModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = nn.Parameter(torch.tensor([1.0]))
                self.activated_task = None

            def shared_core_state_dict(self):
                return {"core": self.core.detach().clone()}

            def load_shared_core_state_dict(self, state) -> None:
                with torch.no_grad():
                    self.core.copy_(state["core"])

            def activate_shared_only(self) -> None:
                self.core.requires_grad_(True)

            def activate_task_expert(self, task_id: int) -> None:
                self.activated_task = task_id

            def shared_parameter_groups(self):
                return {"encoder": [self.core]}

            def compute_loss(self, *_batch, task_id: int):
                return self.core.square().sum(), {}

        world_model = TinyWorldModel()
        optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        config = SimpleNamespace(
            uses_evolving_atomic_rssm=True,
            boundary_consolidation_steps=2,
            boundary_consolidation_lr=0.1,
            boundary_max_return_drop=0.05,
            mb_t_size=1,
            mb_n_size=1,
            compute_dtype="float32",
            esc=SimpleNamespace(
                env_configs=[SimpleNamespace(rew_scale=1.0)]
            ),
        )
        replay = SimpleNamespace(
            minibatch_for_task=lambda *_args, **_kwargs: (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
        )
        actor_bank = SimpleNamespace(get=lambda _task_id: object())
        writer = SimpleNamespace(add_scalar=lambda *_args, **_kwargs: None)
        before = world_model.core.detach().clone()
        optimizer_before = copy.deepcopy(optimizer.state_dict())

        with TemporaryDirectory() as directory, mock.patch.object(
            train,
            "_evaluate_policy_tasks",
            side_effect=[([1.0], [0.0]), RuntimeError("evaluation failed")],
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                train._consolidate_evolving_shared_core(
                    config=config,
                    wm=world_model,
                    shared_optimizer=optimizer,
                    replay_buffer=replay,
                    actor_critic_bank=actor_bank,
                    completed_task_id=0,
                    eval_funcs=[object()],
                    validation_task_seeds=[7],
                    epoch=0,
                    global_step=3,
                    log_dir=Path(directory),
                    writer=writer,
                )
            pre_validation = json.loads(
                (
                    Path(directory)
                    / "evolving_core_consolidation"
                    / "task_00_pre_validation.json"
                ).read_text(encoding="utf-8")
            )

        torch.testing.assert_close(world_model.core, before)
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertEqual(world_model.activated_task, 0)
        self.assertEqual(
            pre_validation["artifact_kind"],
            "evolving_core_pre_consolidation_validation",
        )
        self.assertEqual(pre_validation["validation"]["raw_mean"], [1.0])
        self.assertEqual(pre_validation["consolidation_updates_completed"], 0)
        self.assertFalse(pre_validation["heldout_final_data_used"])


if __name__ == "__main__":
    unittest.main()
