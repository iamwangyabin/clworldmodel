"""CoinRun D-AutoRoute contracts; fixed tensors/mocks, no simulator or updates."""
from __future__ import annotations

import copy
import io
import json
import pickle
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from test_d_autoroute import new_config as atari_config
from test_fastkan_autoroute import Config, np, torch, train, trajectory
from run_evolving_atomic_rssm import (
    ROOT, SEEDS, D_AUTOROUTE_COINRUN_PROTOCOL, _resolved_config,
    _coinrun_source_config, _budget_manifest, _parameter_manifest,
)
from clworldmodel.environments.coinrun import CoinRunFactory, CoinRunEnv, COINRUN_TASKS
from clworldmodel.evaluation.metrics import raw_retention_metrics


def new_config(seed=0):
    _, source = _coinrun_source_config(seed)
    return _resolved_config(source, task_order="arrow-original-six", benchmark="procgen_coinrun",
                            prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
                            behavior_profile="private_mlp_autoroute")


class CoinRunProtocolTests(unittest.TestCase):
    def test_fixed_five_seeds_541_epochs_and_no_atari_parameter_leak(self):
        original = atari_config()
        for index, seed in enumerate(SEEDS):
            data = new_config(index)
            config = Config.from_dict(data)
            self.assertEqual(config.to_dict(), Config.from_dict(config.to_dict()).to_dict())
            self.assertEqual(data["seed"], seed)
            self.assertEqual(data["epochs"], 541)
            self.assertEqual(data["action_space"], 15)
            self.assertEqual(data["env_repeat"], 1)
            self.assertEqual(tuple(t["name"] for t in data["esc"]["env_configs"]), COINRUN_TASKS)
            self.assertTrue(config.task_private_actor_critic)
            self.assertTrue(config.uses_reconstruction_task_inference)
            for key in ("first_task_shared_core_lr", "shared_core_lr", "task_private_lr", "ac_lr",
                        "adaptive_compression_steps_per_candidate", "memory_batch_n"):
                self.assertEqual(data[key], original[key], key)
            budget = _budget_manifest(data)
            self.assertEqual(budget["online_world_model_updates"], 541000)
            self.assertEqual(budget["total_world_model_optimizer_steps"], 553000)
            self.assertEqual(budget["actor_critic_updates"], 432800)
            self.assertEqual(budget["actor_critic_updates_by_task_route"]["0"], 72800)
            self.assertEqual(sum(budget["actor_critic_updates_by_task_route"].values()), 432800)
            self.assertEqual(budget["online_current_sequences"] + budget["online_memory_sequences"], 541000 * 16)
            self.assertEqual(budget["adaptive_compression_validation_rollouts"], 1680)
            self.assertEqual(budget["raw_environment_frames"], 541 * 4 * (4096 - 1))
            self.assertEqual(_parameter_manifest(data)["online_parameters"], 52886765)
        self.assertEqual(original, atari_config())

    def test_config_fails_closed_for_wrong_adapter_order_actions_duration_or_method(self):
        data = new_config()
        for key, value in (("epochs", 540), ("action_space", 18), ("env_repeat", 4),
                           ("benchmark", "unknown"), ("continual_method", "none")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({**data, key: value})
        bad = copy.deepcopy(data)
        bad["esc"]["env_configs"][0]["adapter"] = "atari"
        with self.assertRaises(ValueError):
            Config.from_dict(bad)
        with self.assertRaises(TypeError):
            Config.from_dict({**data, "ignored_key": True})

    def test_revisit_keeps_all_routes_old_task_protection_and_no_seventh_boundary(self):
        config = Config.from_dict(new_config())
        for epoch, current, seen, old in ((0, 0, 1, ()), (90, 1, 2, (0,)),
                                          (539, 5, 6, (0,1,2,3,4)),
                                          (540, 0, 6, (1,2,3,4,5))):
            self.assertEqual(train._evolving_schedule_routes(config, epoch), (current, seen, old))
        self.assertIsNone(train._task_boundary_metadata(config, 540))
        schedule = config.get_env_schedule()
        schedule._step = 540
        self.assertEqual(schedule.current_task_index(), 0)
        self.assertFalse(schedule.is_new_env())

    def test_entrypoint_composes_shared_launcher_and_reports_raw_metrics(self):
        from run_evolving_atomic_rssm_d_autoroute_coinrun import main
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--dry-run", "--seed", "4"]), 0)
        launch, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(launch["protocol"], D_AUTOROUTE_COINRUN_PROTOCOL)
        self.assertEqual(launch["task_order"], list(COINRUN_TASKS))
        self.assertEqual(launch["metric_reporting"]["schema"], "raw-retention-v1")
        self.assertFalse(launch["task_identity_exposed_during_action_selection"])


class FakeNative:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.index = 0
        self.closed = False
    def observe(self):
        return (np.array([10. if self.index == 2 else 0.], dtype=np.float32),
                {"rgb": np.full((1,64,64,3), self.index, np.uint8)},
                np.array([self.index in (0,2)]))
    def act(self, actions):
        self.index += 1
    def close(self):
        self.closed = True


class CoinRunAdapterTests(unittest.TestCase):
    def test_factory_pickle_and_native_flags_match_published_variants(self):
        for name in COINRUN_TASKS:
            factory = CoinRunFactory(name)
            self.assertEqual(pickle.loads(pickle.dumps(factory)), factory)
            flags = factory.options
            self.assertEqual(flags["use_backgrounds"], "+NB" not in name)
            self.assertEqual(flags["restrict_themes"], "+RT" in name)
            self.assertEqual(flags["use_generated_assets"], "+GA" in name)
            self.assertEqual(flags["use_monochrome_assets"], "+MA" in name)
            self.assertEqual(flags["center_agent"], "+CA" not in name)
            self.assertEqual(factory.action_count, 15)
            self.assertEqual(factory.dummy_previous_action, 4)
        with self.assertRaises(ValueError):
            CoinRunFactory("CoinRun+unknown")
        with self.assertRaises(ValueError):
            CoinRunFactory("CoinRun").prepare(4, 1)

    def test_lazy_seeded_constructor_same_step_reset_and_exact_episode_reseed(self):
        with mock.patch("clworldmodel.environments.coinrun._native_environment", side_effect=FakeNative) as build:
            env = CoinRunEnv(CoinRunFactory("CoinRun"))
            build.assert_not_called()
            with self.assertRaises(ValueError):
                env.reset()
            obs, info = env.reset(seed=2**32 - 1)
            first = env._native
            self.assertEqual(build.call_args.kwargs["rand_seed"], 2**31 - 1)
            self.assertEqual(build.call_args.kwargs["num_threads"], 0)
            self.assertEqual(obs.shape, (64,64,3))
            self.assertEqual(obs.dtype, np.uint8)
            _, reward, term, trunc, _ = env.step(4)
            self.assertFalse(term)
            reset_obs, reward, term, trunc, _ = env.step(7)
            self.assertEqual(reward, 10.)
            self.assertTrue(term)
            self.assertFalse(trunc)
            obs, _ = env.reset()
            np.testing.assert_array_equal(obs, reset_obs)
            self.assertEqual(first.index, 2)  # SameStep must not act/reset again.
            self.assertEqual(build.call_count, 1)
            env.reset(seed=19)
            self.assertTrue(first.closed)
            self.assertEqual(build.call_count, 2)
            with self.assertRaises(ValueError):
                env.step(15)
            env.close()
            self.assertIsNone(env._native)

    def test_collector_15_actions_and_dummy_reset_action_without_atari_preprocessing(self):
        factory = CoinRunFactory("CoinRun")
        with mock.patch("clworldmodel.environments.coinrun._native_environment", side_effect=FakeNative), \
             mock.patch.object(trajectory, "AsyncVectorEnv", side_effect=lambda fns, **kw: __import__("gymnasium").vector.SyncVectorEnv(fns, **kw)), \
             mock.patch.object(trajectory, "AtariPreprocessing", side_effect=AssertionError("must not preprocess Procgen")):
            diagnostic = {}
            batch = trajectory.generate_trajectories(8, 2, env_fns=[factory]*2, env_repeat=1,
                                                      seed=21, eligible_route_ids=(0,), routing_diagnostics=diagnostic)
        acts, obs, rewards, conts, resets = batch
        self.assertEqual(acts.shape, (8, 15))
        self.assertEqual(obs.shape, (8,3,64,64))
        self.assertTrue(torch.all(acts.argmax(-1)[resets[:,0].bool()] == 4))
        self.assertEqual(diagnostic["environment_agent_decisions"], 6)
        reshaped = trajectory.reinterpret_nt_to_t_n(*batch, 2, 4)
        self.assertEqual(reshaped[0].shape, (2,4,15))


class RawMetricTests(unittest.TestCase):
    def test_full_report_keeps_540_and_541_separate_without_normalization(self):
        from summarize_continual_metrics import build_raw_run_report
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = new_config()
            (root / "resolved_training_config.json").write_text(json.dumps(config))
            (root / "launch.json").write_text(json.dumps({
                "method": "D-AutoRoute", "protocol": D_AUTOROUTE_COINRUN_PROTOCOL,
                "classification": "pilot", "seed_index": 0, "project_git": {"commit": "fixture"},
            }))
            lines = []
            for epoch in (90,180,270,360,450,540):
                lines.extend([f"Eval for epoch: {epoch}", "Eval means: [8,8,8,8,8,8]", "Eval stds: [0,0,0,0,0,0]"])
            (root / "train.log").write_text("\n".join(lines))
            (root / "final_evaluation.json").write_text(json.dumps({
                "evaluation_after_completed_epochs": 541, "seed_cohort": "heldout_final",
                "tasks": [{"raw_return_mean": 7., "raw_return_std": 0.,
                           "scaled_return_mean": 7., "scaled_return_std": 0.} for _ in range(6)],
            }))
            report = build_raw_run_report(root)
            self.assertIsNone(report["normalization"])
            self.assertEqual(report["metrics"]["mean_raw_forgetting"], 1.)
            self.assertEqual(report["first_pass_before_revisit"]["mean_raw_forgetting"], 0.)
            self.assertEqual(report["evaluation_checkpoints"][-1]["completed_epochs"], 541)
            self.assertFalse(report["evaluation_protocol"]["cohorts_paired"])


    def test_raw_retention_includes_final_revisit_and_preserves_negative_transfer(self):
        result = raw_retention_metrics([[8.,1.], [6.,9.], [7.,8.]], [0,1], 2)
        self.assertEqual(result["final_average_raw_return"], 7.5)
        self.assertEqual(result["per_task_raw_forgetting"], [1.,1.])
        self.assertEqual(result["mean_raw_forgetting"], 1.)
        self.assertEqual(result["backward_transfer_raw"], -1.)
        with self.assertRaises(ValueError):
            raw_retention_metrics([[1.,2.]], [0,1], 0)


if __name__ == "__main__":
    unittest.main()
