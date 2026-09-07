# SPDX-License-Identifier: Apache-2.0
"""Diagnostic scoring contracts: fixed tensors only, no simulator or updates."""
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("scripts", "src", "third_party/arrow/Code/ARROW_and_DV3/Atari"):
    sys.path.insert(0, str(ROOT / folder))
import probe_coinrun_sequence_routing as probe


class ActionFixture:
    """Prior predicts action; posterior sees image; route 1 inverts actions."""
    def __init__(self):
        self.calls = []

    def initial_state(self, batch):
        return torch.zeros(batch, 1, 2), torch.zeros(batch, 1)

    def posterior_step(self, embedding, hidden, task_id):
        pixel = embedding.flatten(1).mean(-1)
        return torch.stack((1 - pixel, pixel), -1)[:, None] * 20

    def image_embedder_for(self, route):
        return lambda x: x

    def adapt_observation_embeddings(self, x, task_id):
        return x

    def __call__(self, z, action, h, x, reset, *, task_id, stochastic):
        from rssm import straight_through_one_hot
        self.calls.append((task_id, x is None, h.clone(), reset.clone()))
        h = h * (1 - reset) + 1
        logits = (action.flip(-1) if task_id else action)[:, None] * 20 if x is None else self.posterior_step(x, h, task_id)
        logits, z = straight_through_one_hot(logits, stochastic=False)
        return logits, z, h


class ProbeTests(unittest.TestCase):
    def test_checkpoint_actor_reconstruction_uses_production_config_mapping(self):
        from test_d_autoroute_coinrun import new_config
        from config import Config
        from ac import build_actor_critic_opt
        from train import _actor_critic_constructor_kwargs
        import smoke_evolving_atomic_rssm
        from artifact_io import write_sha256_sidecar
        config = Config.from_dict(new_config())
        wm = torch.nn.Linear(1, 1)
        wm.ls, wm.h_dim, wm.a_dim = (1, 2), 1, 15
        ac = build_actor_critic_opt(wm, lr=config.ac_lr, **_actor_critic_constructor_kwargs(config)).ac
        payload = {"artifact_kind": "task_bank_boundary_inference_snapshot", "config": config.to_dict(),
                   "inference_routing": {"eligible_route_ids": [0, 1]}, "completed_task": {"task_index": 1},
                   "world_model_state_dict": wm.state_dict(),
                   "actor_critic_bank_state_dict": {"tasks": {"0": ac.state_dict(), "1": ac.state_dict()}}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(payload, path); write_sha256_sidecar(path)
            with mock.patch.object(smoke_evolving_atomic_rssm, "_world_model", return_value=wm):
                loaded, actors, _, routes, _, _ = probe.load_model(path, torch.device("cpu"))
        self.assertEqual(routes, (0, 1))
        self.assertTrue(all(not p.requires_grad for m in (loaded, actors) for p in m.parameters()))
        for key, value in ac.actor.state_dict().items():
            torch.testing.assert_close(value, actors.actors["1"].state_dict()[key])

    def test_cuda_initialized_before_allocator_reset_and_failure_recorded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            argv = ["probe", "--checkpoint", str(root / "weights.pt"), "--output-dir", str(root / "output"),
                    "--upstream-verification", str(receipt)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(probe, "verify_launch"), \
                    mock.patch.object(probe, "require_synced_training_git_state", return_value={}), \
                    mock.patch.object(torch.cuda, "set_device") as initialize, \
                    mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda _: initialize.assert_called_once()), \
                    mock.patch.object(probe, "load_model", side_effect=RuntimeError("preflight sentinel")):
                with self.assertRaisesRegex(RuntimeError, "preflight sentinel"):
                    probe.main()
            self.assertIn("preflight sentinel", (root / "output/failure.json").read_text())

    def test_prediction_precedes_observation_and_candidate_states_are_private(self):
        rssm = ActionFixture()
        model = SimpleNamespace(rssm=rssm, decoder_for=lambda route: lambda zh: zh[:, 1].reshape(-1, 1, 1, 1))
        x = torch.tensor([0., 1., 0., 1.]).reshape(1, 4, 1, 1, 1)
        a = torch.nn.functional.one_hot(torch.tensor([[1, 0, 1]]), 2).float()
        scores = probe.score_history(model, x, a, (0, 1), dummy_previous_action=0)
        torch.testing.assert_close(scores["reconstruction"], torch.zeros(1, 4, 2))
        torch.testing.assert_close(scores["prediction"][0, 1:, 0], torch.zeros(3))
        torch.testing.assert_close(scores["prediction"][0, 1:, 1], torch.ones(3))
        self.assertEqual(len(rssm.calls), 8)  # one recurrent call per route/frame
        self.assertEqual([float(c[2].item()) for c in rssm.calls], [0, 1, 2, 3] * 2)
        self.assertEqual([c[1] for c in rssm.calls], [False, True, True, True] * 2)
        torch.testing.assert_close(scores["posterior_prior_kl"][:, 0], torch.zeros(1, 2))
        self.assertTrue(bool((scores["posterior_prior_kl"][0, 1:, 1] > 0).all()))
        changed = probe.score_history(model, x, a.flip(-1), (0, 1), dummy_previous_action=0)
        torch.testing.assert_close(changed["prediction"][0, 1:, 0], torch.ones(3))
        reconstruction_only = probe.score_history(model, x, a, (0, 1), dummy_previous_action=0,
                                                  reconstruction_only=True)
        torch.testing.assert_close(reconstruction_only["reconstruction"], scores["reconstruction"], rtol=0, atol=0)
        self.assertTrue(all(not c[1] for c in rssm.calls[-8:]))

    def test_terminal_autoreset_excluded_and_shuffle_keeps_valid_action_multiset(self):
        observations, actions, rewards = [np.array([7])], [], []
        self.assertTrue(probe.append_transition(observations, actions, rewards, np.array([8]), 2, 0, False))
        self.assertFalse(probe.append_transition(observations, actions, rewards, np.array([99]), 3, 10, True))
        self.assertEqual([x.item() for x in observations], [7, 8])
        self.assertEqual(actions, [2])
        self.assertEqual(rewards, [0, 10])
        a = np.array([[0, 1, 2, 4], [0, 4, 4, 4]])
        valid = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], bool)
        result = probe.shuffle_actions(a, valid, 8)
        np.testing.assert_array_equal(result, probe.shuffle_actions(a, valid, 8))
        np.testing.assert_array_equal(np.sort(result[0, :3]), [0, 1, 2])
        np.testing.assert_array_equal(result[1], a[1])
        self.assertEqual(result[0, 3], 4)

    def test_matched_prefix_denominators_corrections_and_broken_routes(self):
        r = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]], float)
        p = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]], float)
        episodes = [{"cohort": "random", "task_index_for_audit_only": k} for k in (0, 1, 0)]
        cfg = probe.ProbeConfig(prefix_decisions=1, cohorts=("random",))
        tables = probe.summarize({"reconstruction": r, "prediction": p}, {"prediction": p},
                                 np.array([[1, 1], [1, 1], [1, 0]], bool), episodes, (0, 1), cfg)
        c = next(t for t in tables if t["router"] == "C_action_conditioned_prediction")
        self.assertEqual((c["sample_count"], c["short_prefixes_excluded"]), (2, 1))
        self.assertEqual((c["wrong_first_frame_corrected"], c["correct_first_frame_broken"]), (1, 1))
        self.assertEqual(c["accuracy"], .5)

    def test_frozen_fingerprint_covers_scalar_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        before = probe.weight_digest(model)
        model.num_batches_tracked.add_(1)
        self.assertNotEqual(before, probe.weight_digest(model))
        with self.assertRaises(ValueError):
            probe.ProbeConfig(episodes_per_task=0)


if __name__ == "__main__":
    unittest.main()
