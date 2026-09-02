from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import ale_py  # noqa: F401
    import sortedcontainers  # noqa: F401
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal hosts omit experiment deps.
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.models.mechanism_bank import (
        AtomicLowRankResidualMechanism,
        MechanismBank,
        ResidualMechanism,
    )
    from config import Config
    from run_evolving_atomic_lora_shared_heads import (
        MECHANISM_LOW_RANK,
        METHOD_KEY,
        PROTOCOL,
        _materialize_task0_boundary_snapshot,
        _mechanism_capacity_manifest,
        _parameter_manifest,
        _resolved_config,
    )
    from smoke_evolving_atomic_rssm import (
        ATOMIC_LORA_SHARED_HEADS_PROFILE,
        _config as _smoke_config,
        _world_model as _smoke_world_model,
    )
    import train


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class AtomicLowRankMechanismTest(unittest.TestCase):
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

    def test_task0_is_dense_and_later_tasks_are_zero_effect_low_rank_atoms(self) -> None:
        bank = MechanismBank(
            num_tasks=3,
            in_features=12,
            out_features=8,
            hidden_features=8,
            residual_scale=0.1,
            reuse_enabled=True,
            num_atoms=2,
            include_task0=True,
            parameterization="dense_task0_low_rank_atoms",
            low_rank_rank=4,
        )

        self.assertIsInstance(bank.mechanism_for(0), ResidualMechanism)
        self.assertIsInstance(
            bank.mechanism_for(1), AtomicLowRankResidualMechanism
        )
        self.assertIsInstance(
            bank.mechanism_for(2), AtomicLowRankResidualMechanism
        )
        report = bank.parameter_report()
        self.assertEqual(report["low_rank_rank"], 4)
        self.assertEqual(report["low_rank_atom_width"], 2)

        inputs = torch.randn(5, 12)
        correction, current = bank.forward_with_current(inputs, 1)
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        torch.testing.assert_close(current, torch.zeros_like(current))

        mechanism = bank.mechanism_for(1)
        assert isinstance(mechanism, AtomicLowRankResidualMechanism)
        with torch.no_grad():
            mechanism.up_out.weight.fill_(0.25)
        current = mechanism(inputs)
        atoms = mechanism.atom_outputs(inputs)
        self.assertEqual(tuple(atoms.shape), (5, 2, 8))
        torch.testing.assert_close(atoms.sum(dim=-2), current)

    def test_low_rank_atoms_can_reuse_dense_task0_atoms_without_duplicating_base(self) -> None:
        bank = MechanismBank(
            num_tasks=2,
            in_features=8,
            out_features=4,
            hidden_features=8,
            reuse_enabled=True,
            num_atoms=2,
            include_task0=True,
            parameterization="dense_task0_low_rank_atoms",
            low_rank_rank=4,
        )
        task0 = bank.mechanism_for(0)
        assert isinstance(task0, ResidualMechanism)
        with torch.no_grad():
            task0.up.weight.fill_(0.2)
            route = bank.route_for(1)
            assert route is not None and route.logits is not None
            route.logits.fill_(0.5)

        inputs = torch.randn(3, 8)
        correction, current = bank.forward_with_current(inputs, 1)
        torch.testing.assert_close(current, torch.zeros_like(current))
        expected = (
            task0.atom_outputs(inputs)
            * torch.tanh(torch.tensor(0.5)).reshape(1, 1, 1)
        ).sum(dim=-2)
        torch.testing.assert_close(correction, expected)

    def test_rank_must_partition_evenly_across_atoms(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            MechanismBank(
                num_tasks=2,
                in_features=8,
                out_features=4,
                hidden_features=8,
                reuse_enabled=True,
                num_atoms=4,
                include_task0=True,
                parameterization="dense_task0_low_rank_atoms",
                low_rank_rank=6,
            )

    def test_named_config_and_exact_parameter_ledger(self) -> None:
        data = _resolved_config(self._published_config_data())
        config = Config.from_dict(data)

        self.assertEqual(config.continual_method, METHOD_KEY)
        self.assertTrue(config.uses_evolving_atomic_rssm)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertTrue(config.task_mechanism_reuse)
        self.assertFalse(config.task_private_prediction_adapters)
        self.assertFalse(config.freeze_shared_prediction_heads_after_task0)
        self.assertEqual(config.task_mechanism_low_rank, MECHANISM_LOW_RANK)
        self.assertEqual(config.actor_network, "mlp")
        self.assertEqual(config.epochs, 540)
        self.assertIn("Task0BoundaryBootstrap", PROTOCOL)

        mechanisms = _mechanism_capacity_manifest(6)
        self.assertEqual(mechanisms["task0_dense_parameters"], 3_816_192)
        self.assertEqual(
            mechanisms["per_later_task_atomic_low_rank"]["total"], 1_391_360
        )
        self.assertEqual(mechanisms["route_parameters"], 180)

        parameters = _parameter_manifest(data)
        self.assertEqual(parameters["world_model_parameters"], 30_477_465)
        self.assertEqual(parameters["behavior_parameters"], 10_295_910)
        self.assertEqual(parameters["online_parameters"], 40_773_375)
        self.assertEqual(
            parameters["per_task_world_model_additions"],
            {
                "0": 3_850_432,
                "1": 1_425_612,
                "2": 1_425_624,
                "3": 1_425_636,
                "4": 1_425_648,
                "5": 1_425_660,
            },
        )
        self.assertEqual(
            parameters["comparison_to_dense_shared_heads_private_mlp"][
                "difference"
            ],
            -12_124_160,
        )
        self.assertEqual(
            parameters["comparison_to_failed_learned_base_rank32_pilot"][
                "difference"
            ],
            3_617_280,
        )

        smoke = _smoke_config(method_profile=ATOMIC_LORA_SHARED_HEADS_PROFILE)
        self.assertEqual(smoke.continual_method, METHOD_KEY)
        self.assertEqual(smoke.task_mechanism_low_rank, 128)

    def test_named_config_rejects_rank_reuse_or_prediction_adapter_drift(self) -> None:
        rank = _resolved_config(self._published_config_data())
        rank["task_mechanism_low_rank"] = 64
        with self.assertRaisesRegex(ValueError, "fixes Q/F/P rank to 128"):
            Config.from_dict(rank)

        reuse = _resolved_config(self._published_config_data())
        reuse["task_mechanism_reuse"] = False
        with self.assertRaisesRegex(ValueError, "requires old-atom reuse"):
            Config.from_dict(reuse)

        adapters = _resolved_config(self._published_config_data())
        adapters["task_private_prediction_adapters"] = True
        adapters["prediction_adapter_rank"] = 32
        adapters["freeze_shared_prediction_heads_after_task0"] = True
        with self.assertRaisesRegex(ValueError, "fixed optimizer"):
            Config.from_dict(adapters)

    def test_task0_boundary_snapshot_is_materialized_with_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "source" / "task_boundary_snapshots"
            source_directory.mkdir(parents=True)
            source = (
                source_directory
                / "boundary_01_task_00_completed_0090.pt"
            )
            source.write_bytes(b"immutable-task0-boundary")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            source.with_suffix(".pt.sha256").write_text(
                f"{digest}  {source.name}\n", encoding="ascii"
            )

            record = _materialize_task0_boundary_snapshot(
                source, root / "target"
            )
            copied = Path(str(record["path"]))
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.assertEqual(record["sha256"], digest)
            self.assertTrue(copied.with_suffix(".pt.sha256").is_file())

    def test_task0_seed_accepts_stateless_identity_observation_adapter(self) -> None:
        config = _smoke_config(method_profile=ATOMIC_LORA_SHARED_HEADS_PROFILE)
        world_model = _smoke_world_model(config, torch.device("cpu"))
        self.assertEqual(
            sum(parameter.numel() for parameter in world_model.parameters()),
            30_477_465,
        )
        self.assertFalse(
            any(
                name.startswith(
                    ("rssm.observation_adapter.", "zh_transform.")
                )
                for name in world_model.state_dict()
            )
        )
        report = train._seed_atomic_lora_task0_world_model(
            world_model, world_model.state_dict()
        )
        self.assertGreater(report["selected_parameter_count"], 0)
        actor_critic = train.build_actor_critic_opt(
            world_model,
            lr=config.ac_lr,
            **train._actor_critic_constructor_kwargs(config),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in actor_critic.ac.parameters()),
            1_715_985,
        )

    def test_task0_transition_accepts_only_declared_config_delta(self) -> None:
        target = Config.from_dict(_resolved_config(self._published_config_data()))
        source = copy.deepcopy(target.to_dict())
        source.update(
            {
                "continual_method": (
                    "evolving_atomic_rssm_learned_base_adapters_arrow"
                ),
                "task_mechanism_reuse": False,
                "task_mechanism_parameterization": "learned_task0_low_rank",
                "task_mechanism_low_rank": 32,
                "task_private_prediction_adapters": True,
                "prediction_adapter_rank": 32,
                "freeze_shared_prediction_heads_after_task0": True,
            }
        )
        counters = {
            "raw_environment_frames": 5_898_240,
            "world_model_updates": 91_000,
            "actor_critic_updates": 72_000,
        }
        payload = {
            "schema_version": 1,
            "artifact_kind": "evolving_core_atomic_rssm_resumable_checkpoint",
            "resumable": True,
            "config": source,
            "world_model": {},
            "boundary_teacher": {},
            "optimizers": {},
            "replay": {},
            "rng": {},
            "schedule": {
                "environment_step": 90,
                "epoch": 89,
                "completed_epochs": 90,
                "current_task_id": 0,
            },
            "counters": counters,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task0.pt"
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".pt.sha256").write_text(
                f"{digest}  {path.name}\n", encoding="ascii"
            )
            loaded, metadata = train._load_evolving_task0_transition_checkpoint(
                path, config=target
            )
            self.assertEqual(loaded["counters"], counters)
            self.assertIn("not an equivalent resume", metadata["scientific_scope"])

            changed_source = copy.deepcopy(source)
            changed_source["memory_batch_n"] = 8
            payload["config"] = changed_source
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".pt.sha256").write_text(
                f"{digest}  {path.name}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "outside the declared"):
                train._load_evolving_task0_transition_checkpoint(path, config=target)


if __name__ == "__main__":
    unittest.main()
