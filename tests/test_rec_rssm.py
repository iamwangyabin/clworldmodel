"""Focused contracts for lossless REC-RSSM atom routing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
    from clworldmodel.models.mechanism_bank import (
        MechanismBank,
        ResidualMechanism,
        ReuseRoute,
    )
    from config import Config
    from rssm import Rssm
    import train as arrow_train
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class RecRssmTests(unittest.TestCase):
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
                "continual_method": "rec_rssm_arrow",
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
                "task_mechanism_recurrent_width": 512,
                "task_mechanism_representation_width": 512,
                "task_mechanism_transition_width": 256,
                "task_mechanism_residual_scale": 0.1,
                "task_mechanism_num_atoms": 4,
                "task_mechanism_reuse_probe_epochs": 1,
                "task_mechanism_route_lr_scale": 5.0,
                "task_mechanism_consolidation_batches": 8,
                "task_mechanism_min_contribution": 0.01,
                "task_mechanism_max_validation_drop": 0.05,
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

    def test_named_config_fixes_atom_probe_and_consolidation_contract(self) -> None:
        config = Config.from_dict(self._method_config_data())
        self.assertEqual(config.task_mechanism_num_atoms, 4)
        self.assertEqual(config.task_mechanism_reuse_probe_epochs, 1)
        self.assertEqual(config.task_mechanism_route_lr_scale, 5.0)
        self.assertEqual(config.task_mechanism_consolidation_batches, 8)

        invalid = self._method_config_data()
        invalid["task_mechanism_reuse"] = False
        with self.assertRaisesRegex(ValueError, "requires atom reuse"):
            Config.from_dict(invalid)

        invalid = self._method_config_data()
        invalid["task_mechanism_num_atoms"] = 2
        with self.assertRaisesRegex(ValueError, "fixes atom/probe"):
            Config.from_dict(invalid)

    def test_lossless_atom_sum_preserves_full_mechanism(self) -> None:
        torch.manual_seed(7)
        mechanism = ResidualMechanism(
            in_features=9,
            out_features=7,
            hidden_features=12,
            residual_scale=0.17,
            num_atoms=4,
        ).double()
        with torch.no_grad():
            mechanism.up.weight.normal_()
            mechanism.up.bias.normal_()
        inputs = torch.randn(2, 3, 5, 9, dtype=torch.float64)

        full = mechanism(inputs)
        atoms = mechanism.atom_outputs(inputs)

        self.assertEqual(atoms.shape, (2, 3, 5, 4, 7))
        torch.testing.assert_close(
            atoms.sum(dim=-2), full, rtol=1e-12, atol=1e-12
        )

    def test_scalar_gate_checkpoint_migrates_to_four_identical_atom_gates(self) -> None:
        legacy = ReuseRoute(num_old_mechanisms=2, num_atoms=1)
        with torch.no_grad():
            legacy.logits.copy_(torch.tensor([[0.25], [-0.5]]))
            legacy.hard_mask.copy_(torch.tensor([[1.0], [0.0]]))

        atom_route = ReuseRoute(num_old_mechanisms=2, num_atoms=4)
        atom_route.load_state_dict(legacy.state_dict(), strict=True)

        expected_logits = legacy.logits.expand(-1, 4)
        expected_mask = legacy.hard_mask.expand(-1, 4)
        torch.testing.assert_close(atom_route.logits, expected_logits)
        torch.testing.assert_close(atom_route.hard_mask, expected_mask)
        torch.testing.assert_close(
            atom_route.validated_shared_mask, torch.zeros(2, 4)
        )
        expected_gates = (
            torch.tanh(legacy.logits) * legacy.hard_mask
        ).expand(-1, 4)
        torch.testing.assert_close(atom_route(torch.zeros(1)), expected_gates)

        raw_legacy = {"logits": torch.tensor([0.1, -0.2])}
        atom_route.load_state_dict(raw_legacy, strict=True)
        torch.testing.assert_close(
            atom_route.logits,
            raw_legacy["logits"].unsqueeze(-1).expand(-1, 4),
        )
        torch.testing.assert_close(atom_route.hard_mask, torch.ones(2, 4))
        torch.testing.assert_close(
            atom_route.validated_shared_mask, torch.zeros(2, 4)
        )

    def test_probe_and_expand_have_disjoint_gradient_ownership(self) -> None:
        torch.manual_seed(11)
        bank = MechanismBank(
            num_tasks=3,
            in_features=6,
            out_features=5,
            hidden_features=8,
            num_atoms=4,
        )
        with torch.no_grad():
            bank.mechanisms[0].up.weight.normal_()
            bank.mechanisms[0].up.bias.normal_()
        inputs = torch.randn(7, 6)

        bank.activate_task(2, phase="reuse_probe")
        self.assertTrue(bank.routes[1].logits.requires_grad)
        self.assertFalse(
            any(parameter.requires_grad for parameter in bank.mechanisms[0].parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in bank.mechanisms[1].parameters())
        )
        bank(inputs, 2).sum().backward()
        self.assertGreater(float(bank.routes[1].logits.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in bank.mechanisms[0].parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in bank.mechanisms[1].parameters())
        )

        bank.zero_grad(set_to_none=True)
        bank.activate_task(2, phase="full")
        self.assertTrue(
            all(parameter.requires_grad for parameter in bank.mechanisms[1].parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in bank.mechanisms[0].parameters())
        )
        bank(inputs, 2).sum().backward()
        self.assertIsNotNone(bank.mechanisms[1].up.weight.grad)
        self.assertIsNotNone(bank.routes[1].logits.grad)

    def test_world_model_probe_keeps_projector_heads_and_routes_plastic(self) -> None:
        class WideEmbedder(nn.Module):
            output_size = 4096

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.new_zeros((images.shape[0], self.output_size))

        world_model = WorldModel(
            3,
            (2, 3),
            4,
            8,
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
            task_mechanism_recurrent_width=8,
            task_mechanism_representation_width=8,
            task_mechanism_transition_width=8,
            task_mechanism_num_atoms=4,
            image_embedder=WideEmbedder(),
        )
        self.assertTrue(world_model.initialize_task_expert(1, 0))
        self.assertTrue(world_model.initialize_task_expert(2, 1))

        world_model.activate_task_expert(2, mechanism_phase="reuse_probe")
        trainable = {
            name for name, parameter in world_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(name.startswith("rssm.image_projectors.1.") for name in trainable)
        )
        self.assertTrue(
            any(name.startswith("decoder_experts.1.") for name in trainable)
        )
        self.assertTrue(
            any(name.startswith("reward_experts.1.") for name in trainable)
        )
        self.assertTrue(
            any(name.startswith("continue_experts.1.") for name in trainable)
        )
        self.assertTrue(
            any(
                name.startswith("rssm.recurrent_mechanism_bank.routes.1.")
                for name in trainable
            )
        )
        self.assertFalse(
            any("mechanism_bank.mechanisms.1." in name for name in trainable)
        )
        self.assertFalse(
            any("mechanism_bank.mechanisms.0." in name for name in trainable)
        )

        world_model.activate_task_expert(2, mechanism_phase="full")
        expanded = {
            name for name, parameter in world_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any("mechanism_bank.mechanisms.1." in name for name in expanded)
        )
        self.assertFalse(
            any("mechanism_bank.mechanisms.0." in name for name in expanded)
        )

    def test_optimizer_keeps_future_mechanisms_and_uses_five_x_route_lr(self) -> None:
        class DummyWorldModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = nn.Linear(3, 2)
                self.rssm = nn.Module()
                self.rssm.task_mechanism_bank_enabled = True
                for name in ("recurrent", "representation", "transition"):
                    setattr(
                        self.rssm,
                        f"{name}_mechanism_bank",
                        MechanismBank(
                            num_tasks=3,
                            in_features=4,
                            out_features=4,
                            hidden_features=8,
                            num_atoms=4,
                        ),
                    )

        world_model = DummyWorldModel()
        world_model.rssm.recurrent_mechanism_bank.mechanisms[1].requires_grad_(
            False
        )
        world_model.rssm.recurrent_mechanism_bank.routes[1].requires_grad_(False)
        groups = arrow_train._rec_optimizer_parameter_groups(
            world_model, wm_lr=2e-4, route_lr_scale=5.0
        )

        self.assertEqual([group["lr"] for group in groups], [2e-4, 1e-3])
        normal_ids = {id(parameter) for parameter in groups[0]["params"]}
        route_ids = {id(parameter) for parameter in groups[1]["params"]}
        all_ids = {id(parameter) for parameter in world_model.parameters()}
        self.assertFalse(normal_ids & route_ids)
        self.assertEqual(normal_ids | route_ids, all_ids)
        self.assertEqual(
            sum(parameter.numel() for parameter in groups[1]["params"]), 12
        )
        self.assertIn(
            id(world_model.rssm.recurrent_mechanism_bank.mechanisms[1].up.weight),
            normal_ids,
        )
        self.assertIn(
            id(world_model.rssm.recurrent_mechanism_bank.routes[1].logits),
            route_ids,
        )

    def test_hard_mask_controls_long_term_atom_identity(self) -> None:
        bank = MechanismBank(
            num_tasks=3,
            in_features=4,
            out_features=3,
            hidden_features=8,
            num_atoms=4,
        )
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        bank.apply_consolidated_mask(2, mask)
        bank.apply_validated_shared_mask(2, mask)
        manifest = bank.route_manifest(2)

        atom_users = {
            atom["atom_index"]: atom["users"]
            for atom in manifest["atoms"]
            if atom["owner_task"] == 1
        }
        self.assertEqual(atom_users, {0: [1, 2], 1: [1], 2: [1, 2], 3: [1]})

    def test_boxing_boundary_records_empty_route_masks_without_evaluation(self) -> None:
        world_model = nn.Module()
        world_model.rssm = nn.Module()
        world_model.rssm.task_mechanism_bank_enabled = True
        for name in ("recurrent", "representation", "transition"):
            setattr(
                world_model.rssm,
                f"{name}_mechanism_bank",
                MechanismBank(
                    num_tasks=3,
                    in_features=4,
                    out_features=4,
                    hidden_features=8,
                    num_atoms=4,
                ),
            )
        config = SimpleNamespace(
            continual_method="rec_rssm_arrow",
            task_mechanism_num_atoms=4,
            task_mechanism_consolidation_batches=8,
            task_mechanism_min_contribution=0.01,
            task_mechanism_max_validation_drop=0.05,
            esc=SimpleNamespace(
                env_configs=(
                    SimpleNamespace(name="MsPacman", rew_scale=1.0),
                    SimpleNamespace(name="Boxing", rew_scale=1.0),
                )
            ),
        )
        writer = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.object(
            arrow_train, "_evaluate_rec_route"
        ) as evaluate_route:
            artifact = arrow_train._consolidate_rec_routes(
                config=config,
                wm=world_model,
                aco=object(),
                replay_buffer=object(),
                completed_task_id=1,
                eval_env_fns=object(),
                validation_seed=123,
                epoch=179,
                global_step=90_000,
                log_dir=Path(temporary_dir),
                writer=writer,
            )

        evaluate_route.assert_not_called()
        self.assertEqual(artifact["candidate_count"], 0)
        self.assertIsNone(artifact["validation"])
        self.assertFalse(artifact["rollback"])
        self.assertEqual(
            artifact["accepted_masks"],
            {"recurrent": [], "posterior": [], "prior": []},
        )
        self.assertEqual(
            artifact["accepted_shared_masks"],
            {"recurrent": [], "posterior": [], "prior": []},
        )

    def test_boundary_consolidation_prunes_only_weak_atoms_and_can_rollback(self) -> None:
        class DummyWorldModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rssm = nn.Module()
                self.rssm.task_mechanism_bank_enabled = True
                for name in ("recurrent", "representation", "transition"):
                    bank = MechanismBank(
                        num_tasks=3,
                        in_features=4,
                        out_features=4,
                        hidden_features=8,
                        num_atoms=4,
                    )
                    with torch.no_grad():
                        bank.mechanisms[0].up.bias.fill_(1.0)
                        bank.routes[1].logits.fill_(1.0)
                    setattr(self.rssm, f"{name}_mechanism_bank", bank)

        class DummyReplay:
            @staticmethod
            def minibatch(*_args, **_kwargs):
                return tuple(torch.zeros(1) for _ in range(5))

        class DummyWriter:
            def __init__(self) -> None:
                self.values = []

            def add_scalar(self, *values) -> None:
                self.values.append(values)

        config = SimpleNamespace(
            continual_method="rec_rssm_arrow",
            task_mechanism_num_atoms=4,
            task_mechanism_consolidation_batches=2,
            task_mechanism_min_contribution=0.01,
            task_mechanism_max_validation_drop=0.05,
            mb_t_size=2,
            mb_n_size=1,
            esc=SimpleNamespace(
                env_configs=(
                    SimpleNamespace(name="MsPacman", rew_scale=1.0),
                    SimpleNamespace(name="Boxing", rew_scale=1.0),
                    SimpleNamespace(name="CrazyClimber", rew_scale=1.0),
                )
            ),
        )

        def fake_loss(*, wm, task_id, **_kwargs) -> float:
            for bank in arrow_train._rec_mechanism_banks(wm).values():
                bank(torch.ones(3, 4), task_id)
            disabled = []
            for bank_name, bank in arrow_train._rec_mechanism_banks(wm).items():
                mask = bank.routes[task_id - 1].hard_mask
                for old_index, atom_index in (mask == 0).nonzero().tolist():
                    disabled.append((bank_name, old_index, atom_index))
            if not disabled:
                return 1.0
            return 2.0 if disabled[0][2] == 0 else 0.5

        for pruned_mean, expected_mask, expected_shared_mask, expected_rollback in (
            (96.0, [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], False),
            (94.0, [[1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0]], True),
        ):
            with self.subTest(pruned_mean=pruned_mean):
                world_model = DummyWorldModel()
                immutable_before = {
                    name: tensor.detach().clone()
                    for name, tensor in world_model.state_dict().items()
                    if not name.endswith(("hard_mask", "validated_shared_mask"))
                }
                writer = DummyWriter()
                with tempfile.TemporaryDirectory() as temporary_dir:
                    with mock.patch.object(
                        arrow_train, "_rec_loss_over_batches", side_effect=fake_loss
                    ), mock.patch.object(
                        arrow_train,
                        "_evaluate_rec_route",
                        side_effect=[(100.0, 1.0), (pruned_mean, 1.5)],
                    ):
                        artifact = arrow_train._consolidate_rec_routes(
                            config=config,
                            wm=world_model,
                            aco=object(),
                            replay_buffer=DummyReplay(),
                            completed_task_id=2,
                            eval_env_fns=object(),
                            validation_seed=123,
                            epoch=269,
                            global_step=135_000,
                            log_dir=Path(temporary_dir),
                            writer=writer,
                        )
                    self.assertTrue(
                        (
                            Path(temporary_dir)
                            / "rec_rssm_consolidation"
                            / "task_02_boundary.json"
                        ).is_file()
                    )
                self.assertEqual(artifact["rollback"], expected_rollback)
                self.assertEqual(artifact["candidate_count"], 12)
                for bank in arrow_train._rec_mechanism_banks(world_model).values():
                    self.assertEqual(
                        bank.routes[1].hard_mask.detach().cpu().tolist(),
                        expected_mask,
                    )
                    self.assertEqual(
                        bank.routes[1]
                        .validated_shared_mask.detach()
                        .cpu()
                        .tolist(),
                        expected_shared_mask,
                    )
                self.assertEqual(
                    artifact["accepted_shared_masks"]["recurrent"],
                    expected_shared_mask,
                )
                for name, expected in immutable_before.items():
                    torch.testing.assert_close(
                        world_model.state_dict()[name], expected, rtol=0, atol=0
                    )

    def test_whole_gate_to_atoms_preserves_recurrent_posterior_and_prior(self) -> None:
        class WideEmbedder(nn.Module):
            output_size = 4096

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.new_zeros((images.shape[0], self.output_size))

        def build(num_atoms: int) -> Rssm:
            return Rssm(
                img_channels=3,
                ls=(2, 3),
                a_dim=4,
                h_dim=8,
                cnn_depth=4,
                mlp_features=8,
                mlp_layers=2,
                observation_encoder="cnn",
                num_task_experts=3,
                full_task_experts=True,
                task_projected_image_encoder=True,
                task_projector_bottleneck_features=64,
                task_mechanism_bank=True,
                task_mechanism_reuse=True,
                task_mechanism_recurrent_width=8,
                task_mechanism_representation_width=8,
                task_mechanism_transition_width=8,
                task_mechanism_num_atoms=num_atoms,
                image_embedder=WideEmbedder(),
            ).double()

        torch.manual_seed(19)
        whole = build(num_atoms=1)
        with torch.no_grad():
            for bank in (
                whole.recurrent_mechanism_bank,
                whole.representation_mechanism_bank,
                whole.transition_mechanism_bank,
            ):
                for mechanism in bank.mechanisms:
                    mechanism.up.weight.normal_()
                    mechanism.up.bias.normal_()
                bank.routes[1].logits.fill_(0.35)

        torch.manual_seed(29)
        atoms = build(num_atoms=4)
        atoms.load_state_dict(whole.state_dict(), strict=True)

        batch = 5
        prev_z = torch.nn.functional.one_hot(
            torch.randint(0, 3, (batch, 2)), num_classes=3
        ).double()
        prev_a = torch.nn.functional.one_hot(
            torch.randint(0, 4, (batch,)), num_classes=4
        ).double()
        prev_h = torch.randn(batch, 8, dtype=torch.float64)
        embedding = torch.randn(batch, 4096, dtype=torch.float64)

        whole_hidden = whole.recurrent_step(prev_z, prev_a, prev_h, task_id=2)
        atom_hidden = atoms.recurrent_step(prev_z, prev_a, prev_h, task_id=2)
        whole_posterior = whole.posterior_step(embedding, whole_hidden, task_id=2)
        atom_posterior = atoms.posterior_step(embedding, atom_hidden, task_id=2)
        whole_prior = whole.prior(whole_hidden, task_id=2)
        atom_prior = atoms.prior(atom_hidden, task_id=2)

        torch.testing.assert_close(atom_hidden, whole_hidden, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            atom_posterior, whole_posterior, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(atom_prior, whole_prior, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
