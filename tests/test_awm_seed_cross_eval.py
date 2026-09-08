# SPDX-License-Identifier: Apache-2.0
"""No environment interaction: snapshot gate and unchanged evaluator delegation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import awm_seed_cross_eval as audit


class CrossSeedDiagnosticTests(unittest.TestCase):
    def test_report_uses_matching_epoch_and_excludes_final_or_later_task_scores(self):
        from report_awm_seed_audit import periodic_task0, chart
        log = ("Starting Epoch  20\nEval raw means: [12.5, 3]\n"
               "Starting Epoch  100\nEval raw means: [99]\nFinal eval raw means: [100]")
        self.assertEqual(periodic_task0(log), {20: 12.5})
        self.assertIn("polyline", chart([{20: 12.5}, {20: 11.}]))

    def test_reject_other_protocol_progress_and_seed(self):
        payload = {
            "artifact_kind": "task_bank_boundary_inference_snapshot",
            "schema_version": 1, "resumable": False,
            "saved_after_final_task_update_before_schedule_advance": True,
            "project_git_commit": audit.TRAINING_COMMIT, "completed_epochs": 90,
            "epoch": 89, "world_model_updates": 92000, "actor_critic_updates": 72000,
            "seed": audit.TRAINING_SEEDS[0],
            "completed_task": {"boundary_index": 1, "task_index": 0,
                               "task_name": "ALE/MsPacman-v5", "task_reward_scale": .05},
            "inference_routing": {"mode": "two_frame_probability_reconstruction",
                                  "protocol_version": 4, "eligible_route_ids": [0],
                                  "task_identity_input": False},
            "config": {"seed": audit.TRAINING_SEEDS[0], "n_sync": 4,
                       "env_repeat": 4, "actor_network": "mlp", "task_route_inference_version": 4,
                       "evaluation_episode_count_mode": "legacy",
                       "evaluation_seed_protocol": "fixed_validation_heldout_final"},
        }
        audit.validate_snapshot(payload, 0)
        for key, value in (("completed_epochs", 80), ("seed", 1337),
                           ("project_git_commit", "a" * 40), ("resumable", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_snapshot({**payload, key: value}, 0)
        modified = copy.deepcopy(payload)
        modified["inference_routing"]["protocol_version"] = 3
        with self.assertRaises(ValueError):
            audit.validate_snapshot(modified, 0)

    def test_observation_wrapper_passes_through_original_tensors_and_arguments(self):
        tensors = (torch.zeros(8, 18), None, torch.ones(8, 1),
                   torch.ones(8, 1), torch.zeros(8, 1))
        generate = Mock(return_value=tensors)
        module = SimpleNamespace(generate_trajectories=generate)
        calls = []

        def evaluate(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            calls.append((args, kwargs))
            self.assertIs(module.generate_trajectories("unchanged", no_images=True), tensors)
            kwargs["diagnostics"]["completed_episodes"] = 17
            return 100., 5.

        module.evaluate = evaluate
        config = SimpleNamespace(n_sync=4, env_repeat=4,
                                 task_route_inference="two_frame_probability_reconstruction",
                                 get_env_schedule=lambda: SimpleNamespace(eval_funcs=lambda: [["env"]]))
        with tempfile.TemporaryDirectory() as directory:
            result = audit.evaluate_cohort(module, "wm", "actor", config,
                                           audit.VALIDATION_SEEDS[0], Path(directory))
            saved = torch.load(Path(directory) / "legacy_evaluation_tensors.pt", weights_only=True)
            torch.testing.assert_close(saved, tensors)
        self.assertIs(module.generate_trajectories, generate)
        generate.assert_called_once_with("unchanged", no_images=True)
        self.assertEqual(calls[0][0], (4,))
        kwargs = calls[0][1]
        self.assertEqual(kwargs["eligible_route_ids"], (0,))
        self.assertNotIn("task_id", kwargs)
        self.assertTrue(kwargs["deterministic_policy"])
        self.assertEqual(kwargs["n_rollouts"], 16)
        self.assertEqual(result["raw_return_mean"], 2000.)
        self.assertEqual(result["diagnostics"]["completed_episodes"], 17)
        self.assertEqual(result["returned_tensor_samples"], 8)


if __name__ == "__main__":
    unittest.main()
