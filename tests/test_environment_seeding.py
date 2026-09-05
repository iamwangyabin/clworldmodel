"""Contracts for isolated, deterministic Atari environment seed streams."""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import numpy as np
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    np = None
    torch = None

if torch is not None:
    sys.path.insert(0, str(VENDORED_ATARI))
    import generate_trajectory
    import train


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EnvironmentSeedingTests(unittest.TestCase):
    def test_collection_and_evaluation_seed_streams_are_stable_and_disjoint(self) -> None:
        collect_a, validate_a, final_a = train._environment_seed_streams(123456789)
        collect_b, validate_b, final_b = train._environment_seed_streams(123456789)

        collect_values_a = [train._next_environment_seed(collect_a) for _ in range(4)]
        collect_values_b = [train._next_environment_seed(collect_b) for _ in range(4)]
        validate_values_a = [train._next_environment_seed(validate_a) for _ in range(4)]
        validate_values_b = [train._next_environment_seed(validate_b) for _ in range(4)]
        final_values_a = [train._next_environment_seed(final_a) for _ in range(4)]
        final_values_b = [train._next_environment_seed(final_b) for _ in range(4)]

        self.assertEqual(collect_values_a, collect_values_b)
        self.assertEqual(validate_values_a, validate_values_b)
        self.assertEqual(final_values_a, final_values_b)
        cohorts = {
            tuple(collect_values_a),
            tuple(validate_values_a),
            tuple(final_values_a),
        }
        self.assertEqual(len(cohorts), 3)

    def test_independent_task_seed_offset_selects_global_cohort_slot(self) -> None:
        _, validation_full, final_full = train._environment_seed_streams(123456789)
        validation_values = [
            train._next_environment_seed(validation_full) for _ in range(6)
        ]
        final_values = [train._next_environment_seed(final_full) for _ in range(6)]

        _, validation_independent, final_independent = train._environment_seed_streams(
            123456789
        )
        for _ in range(3):
            train._next_environment_seed(validation_independent)
            train._next_environment_seed(final_independent)

        self.assertEqual(
            train._next_environment_seed(validation_independent),
            validation_values[3],
        )
        self.assertEqual(
            train._next_environment_seed(final_independent),
            final_values[3],
        )

    def test_task_cosine_actor_schedule_restarts_per_task_without_eval_input(
        self,
    ) -> None:
        config = SimpleNamespace(
            esc=SimpleNamespace(kwargs={"swap_sched": 90}),
            ac_schedule="task_cosine_decay",
            ac_lr=1e-4,
            ac_entropy_scale=3e-4,
            ac_decay_start_task_epoch=40,
            ac_decay_end_task_epoch=90,
            ac_final_lr=2.5e-5,
            ac_final_entropy_scale=5e-5,
        )

        self.assertEqual(
            train._actor_critic_schedule_values(config, 39),
            (1e-4, 3e-4, 40),
        )
        midpoint_lr, midpoint_entropy, task_epoch = (
            train._actor_critic_schedule_values(config, 64)
        )
        self.assertEqual(task_epoch, 65)
        self.assertAlmostEqual(midpoint_lr, 6.25e-5)
        self.assertAlmostEqual(midpoint_entropy, 1.75e-4)
        self.assertEqual(
            train._actor_critic_schedule_values(config, 89),
            (2.5e-5, 5e-5, 90),
        )
        self.assertEqual(
            train._actor_critic_schedule_values(config, 90),
            (1e-4, 3e-4, 1),
        )

    def test_variable_task_durations_drive_schedule_boundaries_and_actor_age(
        self,
    ) -> None:
        templates = [lambda: index for index in range(3)]
        schedule = generate_trajectory.SequentialEnvironments(
            1,
            templates,
            task_durations=[90, 120, 120],
        )
        self.assertEqual(schedule.current_task_index(), 0)
        self.assertTrue(schedule.is_new_env())
        for _ in range(89):
            schedule.step()
        self.assertEqual(schedule.current_task_index(), 0)
        self.assertFalse(schedule.is_new_env())
        schedule.step()
        self.assertEqual(schedule.current_task_index(), 1)
        self.assertTrue(schedule.is_new_env())
        for _ in range(120):
            schedule.step()
        self.assertEqual(schedule.current_task_index(), 2)
        self.assertTrue(schedule.is_new_env())

        config = SimpleNamespace(
            esc=SimpleNamespace(
                kwargs={"task_durations": [90, 120, 120]},
                env_configs=[object(), object(), object()],
            ),
            ac_schedule="task_cosine_decay",
            ac_lr=2e-4,
            ac_entropy_scale=3e-4,
            ac_decay_start_task_epoch=60,
            ac_decay_end_task_epoch=120,
            ac_final_lr=5e-5,
            ac_final_entropy_scale=3e-4,
        )
        self.assertEqual(
            train._sequential_task_position(config, 90),
            (1, 1),
        )
        self.assertEqual(
            train._sequential_task_position(config, 209),
            (1, 120),
        )
        self.assertEqual(
            train._sequential_task_position(config, 210),
            (2, 1),
        )
        self.assertEqual(
            train._actor_critic_schedule_values(config, 209),
            (5e-5, 3e-4, 120),
        )
        self.assertEqual(
            train._actor_critic_schedule_values(config, 210),
            (2e-4, 3e-4, 1),
        )

        config.esc.env_schedule_type = (
            generate_trajectory.SequentialEnvironments
        )
        config.esc.env_configs = [
            SimpleNamespace(name=f"task-{index}", rew_scale=1.0)
            for index in range(3)
        ]
        self.assertEqual(
            train._task_boundary_metadata(config, 209),
            {
                "boundary_index": 2,
                "task_index": 1,
                "task_name": "task-1",
                "task_reward_scale": 1.0,
            },
        )
        self.assertIsNone(train._task_boundary_metadata(config, 210))
        self.assertEqual(
            train._task_boundary_metadata(config, 329)["task_index"],
            2,
        )

    def test_task_bank_evaluation_snapshot_is_atomic_and_non_resumable(self) -> None:
        class FakeConfig:
            algorithm = "arrow"
            seed = 7
            uses_shared_actor = False

            @staticmethod
            def to_dict():
                return {"algorithm": "arrow", "seed": 7}

        world_model = torch.nn.Linear(2, 3)
        actor_state = {"weight": torch.arange(4, dtype=torch.float32)}
        actor_bank = SimpleNamespace(
            inference_state_dict=lambda: {
                "schema_version": 1,
                "artifact_kind": "test_actor_bank",
                "resumable": False,
                "tasks": {"0": actor_state},
            }
        )
        with TemporaryDirectory() as temporary:
            path = train._save_task_bank_evaluation_snapshot(
                Path(temporary),
                config=FakeConfig(),
                wm=world_model,
                actor_critic_bank=actor_bank,
                completed_epochs=50,
                world_model_updates=50_000,
                actor_critic_updates=40_000,
                total_env_steps=1_000_000,
                task_seeds=[11],
                scaled_means=[20.0],
                scaled_stds=[4.0],
                raw_means=[2_000.0],
                raw_stds=[400.0],
                cohort="periodic_validation",
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

            self.assertFalse(payload["resumable"])
            self.assertEqual(payload["completed_epochs"], 50)
            self.assertEqual(payload["task_base_seeds"], [11])
            self.assertEqual(payload["evaluation"][0]["raw_return_mean"], 2_000.0)
            self.assertIn("optimizers", payload["omitted_state"])
            self.assertTrue(path.with_suffix(".pt.sha256").is_file())
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_task_boundary_snapshot_preserves_complete_bank_and_commit(self) -> None:
        class FakeConfig:
            algorithm = "arrow"
            seed = 7
            ac_train_steps = 800
            uses_shared_actor = False

            @staticmethod
            def to_dict():
                return {
                    "algorithm": "arrow",
                    "seed": 7,
                    "ac_train_steps": 800,
                }

        world_model = torch.nn.Linear(2, 3)
        completed_actor = SimpleNamespace(ac=torch.nn.Linear(3, 2))
        actor_state = {
            name: value.detach().cpu()
            for name, value in completed_actor.ac.state_dict().items()
        }
        actor_bank = SimpleNamespace(
            get=lambda task_id: completed_actor if task_id == 0 else None,
            inference_state_dict=lambda: {
                "schema_version": 1,
                "artifact_kind": "test_actor_bank",
                "resumable": False,
                "tasks": {"0": actor_state},
            },
        )
        task_metadata = {
            "boundary_index": 1,
            "task_index": 0,
            "task_name": "ALE/MsPacman-v5",
            "task_reward_scale": 0.05,
        }
        project_commit = "a" * 40

        with TemporaryDirectory() as temporary:
            snapshot_dir = Path(temporary)
            path = train._save_task_bank_boundary_snapshot(
                snapshot_dir,
                config=FakeConfig(),
                wm=world_model,
                actor_critic_bank=actor_bank,
                epoch=89,
                world_model_updates=90_000,
                total_env_steps=1_800_000,
                task_metadata=task_metadata,
                project_git_commit=project_commit,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            index = json.loads(
                (snapshot_dir / "index.json").read_text(encoding="utf-8")
            )

            self.assertEqual(payload["project_git_commit"], project_commit)
            self.assertEqual(payload["completed_epochs"], 90)
            self.assertEqual(payload["world_model_updates"], 90_000)
            self.assertEqual(payload["actor_critic_updates"], 72_000)
            self.assertEqual(payload["completed_task"], task_metadata)
            self.assertTrue(
                payload["saved_after_final_task_update_before_schedule_advance"]
            )
            self.assertFalse(payload["resumable"])
            self.assertEqual(
                set(payload["world_model_state_dict"]),
                set(world_model.state_dict()),
            )
            self.assertEqual(
                set(payload["actor_critic_bank_state_dict"]["tasks"]), {"0"}
            )
            self.assertEqual(index["project_git_commit"], project_commit)
            self.assertEqual(len(index["snapshots"]), 1)
            self.assertEqual(index["snapshots"][0]["path"], path.name)
            self.assertTrue(path.with_suffix(".pt.sha256").is_file())
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            self.assertFalse((snapshot_dir / "index.json.tmp").exists())

            with self.assertRaises(FileExistsError):
                train._save_task_bank_boundary_snapshot(
                    snapshot_dir,
                    config=FakeConfig(),
                    wm=world_model,
                    actor_critic_bank=actor_bank,
                    epoch=89,
                    world_model_updates=90_000,
                    total_env_steps=1_800_000,
                    task_metadata=task_metadata,
                    project_git_commit=project_commit,
                )

    def test_worker_reset_and_action_seeds_are_stable_and_disjoint(self) -> None:
        resets_a, actions_a = generate_trajectory._environment_worker_seeds(17, 4)
        resets_b, actions_b = generate_trajectory._environment_worker_seeds(17, 4)

        self.assertEqual(resets_a, resets_b)
        self.assertEqual(actions_a, actions_b)
        self.assertEqual(len(set(resets_a + actions_a)), 8)

    def test_seed_reaches_vector_reset_and_each_action_space(self) -> None:
        class FakeVectorEnv:
            reset_seed = None
            selected_autoreset_mode = None
            closed = False
            single_action_space = generate_trajectory.gym.spaces.Discrete(18)

            def __init__(self, factories, *, autoreset_mode) -> None:
                self.factories = factories
                type(self).selected_autoreset_mode = autoreset_mode

            def reset(self, *, seed=None):
                type(self).reset_seed = seed
                return np.zeros((2, 64, 64, 3), dtype=np.uint8), {}

            def close(self):
                type(self).closed = True

        expected_resets, expected_actions = (
            generate_trajectory._environment_worker_seeds(29, 2)
        )
        with mock.patch.object(generate_trajectory, "AsyncVectorEnv", FakeVectorEnv):
            generate_trajectory.generate_trajectories(
                2,
                2,
                env_fns=[lambda: None, lambda: None],
                seed=29,
            )
        self.assertEqual(FakeVectorEnv.reset_seed, expected_resets)
        self.assertEqual(
            FakeVectorEnv.selected_autoreset_mode,
            generate_trajectory.AutoresetMode.NEXT_STEP,
        )
        self.assertTrue(FakeVectorEnv.closed)

        class FakeActionSpace:
            seed_value = None

            def seed(self, seed) -> None:
                self.seed_value = seed

        class FakeEnv:
            action_space = FakeActionSpace()
            unwrapped = SimpleNamespace()

        fake_env = FakeEnv()
        with mock.patch.object(
            generate_trajectory,
            "AtariPreprocessing",
            side_effect=lambda env, **_kwargs: env,
        ):
            result = generate_trajectory._make_visual_env(
                lambda: fake_env, env_repeat=4, action_seed=expected_actions[0]
            )
        self.assertIs(result, fake_env)
        self.assertEqual(fake_env.action_space.seed_value, expected_actions[0])

    def test_evaluation_scope_restores_parent_rng_states(self) -> None:
        def seed_all() -> None:
            random.seed(31)
            np.random.seed(31)
            torch.manual_seed(31)

        seed_all()
        with train._preserve_training_rng_state():
            random.random()
            np.random.random()
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device="cuda")

        actual = (random.random(), np.random.random(), torch.rand(3))
        actual_cuda = torch.rand(3, device="cuda") if torch.cuda.is_available() else None

        seed_all()
        expected = (random.random(), np.random.random(), torch.rand(3))
        expected_cuda = (
            torch.rand(3, device="cuda") if torch.cuda.is_available() else None
        )

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        if actual_cuda is not None:
            torch.testing.assert_close(actual_cuda, expected_cuda, rtol=0, atol=0)

    def test_task_agnostic_evaluation_can_be_declared_deterministic(self) -> None:
        config = SimpleNamespace(
            n_sync=1,
            env_repeat=1,
            action_space=7,
            uses_task_experts=False,
            deterministic_evaluation=True,
        )
        with mock.patch.object(train, "evaluate", return_value=(0.5, 0.1)) as evaluate:
            means, stds = train._evaluate_policy_tasks(
                config,
                wm=object(),
                aco=None,
                eval_funcs=[[lambda: None]],
                task_seeds=[17],
            )

        self.assertEqual(means, [0.5])
        self.assertEqual(stds, [0.1])
        self.assertTrue(evaluate.call_args.kwargs["deterministic_policy"])
        self.assertNotIn("task_id", evaluate.call_args.kwargs)

    def test_raw_taskwise_evaluation_checkpoint_has_explicit_counters(self) -> None:
        with TemporaryDirectory() as temporary:
            path = train._write_evaluation_metrics(
                Path(temporary),
                checkpoint_kind="periodic_validation",
                collection_epoch_index=10,
                completed_update_epochs=10,
                training_environment_decisions=20_000,
                world_model_updates=1_159,
                actor_critic_updates=490,
                task_seeds=[17],
                task_configs=[SimpleNamespace(name="task-0", rew_scale=1.0)],
                scaled_means=[0.75],
                scaled_stds=[0.1],
                raw_means=[0.75],
                raw_stds=[0.1],
                deterministic_policy=True,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "raw-taskwise-evaluation-v1")
        self.assertEqual(
            payload["counters"],
            {
                "training_environment_decisions": 20_000,
                "world_model_updates": 1_159,
                "actor_critic_updates": 490,
            },
        )
        self.assertEqual(payload["tasks"][0]["raw_return_mean"], 0.75)
        self.assertEqual(
            payload["policy"], "deterministic_argmax_and_latent_mode"
        )


if __name__ == "__main__":
    unittest.main()
