"""Deterministic recovery contracts; no simulator interaction or training run."""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from test_d_autoroute import new_config
from test_fastkan_autoroute import vendor_available
from d_autoroute_resume import inspect_resume, sha256, stage_resume_prefix
from launcher_support import run_and_tee


class ResumeLineageTests(unittest.TestCase):
    def fixture(self, root):
        config = new_config()
        source = root / "failed"
        checkpoint = source / "evolving_core_checkpoints/task_00_post_consolidation.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mock checkpoint, no tensors or training")
        checkpoint.with_suffix(".pt.sha256").write_text(sha256(checkpoint) + "\n")
        (source / "launch.json").write_text(json.dumps({"protocol": "fixture", "seed": config["seed"],
                                                       "project_git": {"commit": "a" * 40}}))
        (source / "resolved_training_config.json").write_text(json.dumps(config))
        (source / "run_status.json").write_text(json.dumps({"complete": False, "return_code": 1}))
        (source / "train.log").write_text("Starting Epoch  89\n[stage-time] epoch=89 total=1s\nStarting Epoch  90\n[stage-time] epoch=90 total=1s\nStarting Epoch  91\nTraceback failed suffix\n")
        for folder, names in {"task_routing": ["collection_epoch_0089_batch_00.json", "periodic_epoch_0090.json"],
                              "adaptive_qfp_compression": ["task_00_boundary.json", "task_01_failure.json"]}.items():
            (source / folder).mkdir()
            for name in names:
                (source / folder / name).write_text("{}")
        return config, source, checkpoint

    def test_resume_preserves_failed_suffix_and_copies_only_durable_prefix(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, source, path = self.fixture(root)
            original_hash = sha256(source / "train.log")
            lineage = inspect_resume(path, config, "fixture")
            self.assertEqual(lineage["remaining_epochs"], 450)
            self.assertEqual(lineage["discarded_completed_online_epochs"], 1)
            target = root / "continued"
            target.mkdir()
            prefix = stage_resume_prefix(target, lineage)
            self.assertNotIn("Starting Epoch  90", prefix.read_text())
            self.assertIn("Traceback", (target / "resume_parent/train.log").read_text())
            self.assertFalse((target / "task_routing/periodic_epoch_0090.json").exists())
            self.assertTrue((target / "adaptive_qfp_compression/task_00_boundary.json").is_file())
            self.assertFalse((target / "adaptive_qfp_compression/task_01_failure.json").exists())
            self.assertEqual(sha256(source / "train.log"), original_hash)
            with redirect_stdout(io.StringIO()):
                code = run_and_tee([sys.executable, "-c", "print('new attempt')"], cwd=root,
                                   env=os.environ.copy(), log_path=target / "train.log", prefix_log_path=prefix)
            self.assertEqual(code, 0)
            self.assertIn("Inherited prefix ends", (target / "train.log").read_text())
            self.assertNotIn("Traceback", (target / "train.log").read_text())

    def test_rejects_changed_protocol_seed_budget_corruption_or_nonfailed_parent(self):
        with TemporaryDirectory() as directory:
            config, source, path = self.fixture(Path(directory))
            for changed in ({**config, "seed": 42}, {**config, "epochs": 541}):
                with self.assertRaisesRegex(ValueError, "change"):
                    inspect_resume(path, changed, "fixture")
            with self.assertRaises(ValueError):
                inspect_resume(path.with_name("task_00_pre_consolidation.pt"), config, "fixture")
            (source / "run_status.json").write_text('{"complete": true, "return_code": 0}')
            with self.assertRaisesRegex(ValueError, "confirmed failed"):
                inspect_resume(path, config, "fixture")
            (source / "run_status.json").write_text('{"complete": false, "return_code": 1}')
            path.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_resume(path, config, "fixture")

    def test_entrypoint_forwards_only_explicit_resume_option(self):
        import run_evolving_atomic_rssm_d_autoroute as entry
        with mock.patch.object(entry, "_launch", return_value=0) as launch:
            entry.main(["--resume-from", "runs/failed/evolving_core_checkpoints/task_00_post_consolidation.pt", "--dry-run"])
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--resume-from") + 1],
                         str(entry.ROOT / "runs/failed/evolving_core_checkpoints/task_00_post_consolidation.pt"))


@unittest.skipUnless(vendor_available, "requires tensor runtime")
class ResumeCheckpointTests(unittest.TestCase):
    def test_compact_post_boundary_restore_retains_rng_actor_optimizer_and_schedule(self):
        from test_fastkan_autoroute import torch, Config, np, train
        from test_evolving_atomic_rssm import EvolvingAtomicRssmTests
        from clworldmodel.continual import ActorCriticBank
        config = Config.from_dict(new_config())
        wm = EvolvingAtomicRssmTests._world_model("adaptive_dense_width", shared_prediction_heads=True)
        dense = copy.deepcopy(wm)
        train._structured_adaptive_qfp_candidate(wm=wm, dense_teacher=dense, task_id=0, fraction=.5)
        teacher = copy.deepcopy(wm)
        factory = lambda _: train.build_actor_critic_opt(wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config))
        bank = ActorCriticBank()
        bank.ensure(0, factory)
        bank.activate(0)
        optimizer = torch.optim.Adam([next(wm.parameters())], lr=config.shared_core_lr)
        parameter = optimizer.param_groups[0]["params"][0]
        optimizer.state[parameter] = {"step": torch.tensor(7.), "exp_avg": torch.full_like(parameter, .1),
                                      "exp_avg_sq": torch.full_like(parameter, .2)}
        expected_bank = copy.deepcopy(bank.resumable_state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        generators = [np.random.default_rng(i) for i in range(4)]
        replay = mock.Mock()
        replay.state_dict.return_value = {"fixture": "no simulator transitions"}
        schedule = SimpleNamespace(_step=89)
        common = dict(config=config, wm=wm, boundary_teacher=teacher, shared_optimizer=optimizer,
                      private_optimizers={}, route_optimizers={}, actor_critic_bank=bank,
                      replay_buffer=replay, environment_schedule=schedule, task_update_rng=generators[0],
                      collection_environment_seed_rng=generators[1], validation_environment_seed_rng=generators[2],
                      final_environment_seed_rng=generators[3])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task_00_post_consolidation.pt"
            train._save_evolving_resumable_checkpoint(path, **common, epoch=89, current_task_id=0,
                world_model_updates=92000, actor_critic_updates=72000,
                total_env_steps=90 * config.n_sync * config.gen_seq_len * config.env_repeat)
            expected_draws = [g.integers(1000000) for g in generators]
            wm.load_state_dict(dense.state_dict())
            bank.get(0).ac.actor.requires_grad_(False)
            with torch.no_grad():
                next(bank.get(0).ac.actor.parameters()).add_(1.)
            optimizer.state[parameter]["exp_avg"].zero_()
            restored = train._restore_evolving_resumable_checkpoint(path, **common,
                actor_critic_factory=factory, require_post_boundary=True)
            self.assertEqual(restored["completed_epochs"], 90)
            self.assertEqual(restored["world_model_updates"], 92000)
            self.assertEqual(schedule._step, 90)
            self.assertEqual([g.integers(1000000) for g in generators], expected_draws)
            torch.testing.assert_close(bank.resumable_state_dict()["tasks"], expected_bank["tasks"])
            torch.testing.assert_close(optimizer.state_dict(), expected_optimizer)
            self.assertEqual(wm.rssm.adaptive_compression_layout(), teacher.rssm.adaptive_compression_layout())
            payload = torch.load(path, weights_only=False)
            for mutate in (lambda p: p["schedule"].update(environment_step=91),
                           lambda p: p["counters"].update(actor_critic_updates=0),
                           lambda p: p["optimizers"]["private_by_task"].update({"0": {}})):
                invalid = copy.deepcopy(payload)
                mutate(invalid)
                with self.assertRaises(ValueError):
                    train._validate_d_autoroute_continuation(path, invalid, config)
            with self.assertRaisesRegex(ValueError, "post-consolidation"):
                train._validate_d_autoroute_continuation(path.with_name("task_00_pre_consolidation.pt"), payload, config)
