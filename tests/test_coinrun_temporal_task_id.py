# SPDX-License-Identifier: Apache-2.0
"""Fixed-input router contracts. No environment construction or optimizer steps."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_coinrun_temporal_task_id import (
    ErrorRouter, ExperimentConfig, atomic_router_snapshot, fit_input_calibration,
    prefix_loss, seed_plan, selection_metrics, summarize_predictions, validate_features,
)


class TemporalTaskIDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.errors = torch.rand(3, 5, 3)
        self.valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        self.labels = torch.tensor([0, 1, 2])
        self.calibration = fit_input_calibration(self.errors, self.valid, self.labels)

    def test_config_and_seed_groups_are_fixed_disjoint_and_task_independent(self):
        cfg = ExperimentConfig()
        plan = seed_plan(cfg)
        self.assertEqual(plan, seed_plan(cfg))
        self.assertEqual([len(s) for s in plan.values()], [256, 128, 512])
        self.assertEqual(len(set(sum(plan.values(), []))), 896)
        with self.assertRaises(ValueError): ExperimentConfig(epochs=0)
        with self.assertRaises(TypeError): ExperimentConfig(unknown_setting=1)

    def test_normalization_ignores_padding_and_first_temperature_preserves_choice(self):
        changed = self.errors.clone(); changed[~self.valid] = 1e6
        other = fit_input_calibration(changed, self.valid, self.labels)
        for a, b in zip(self.calibration[:2], other[:2]): torch.testing.assert_close(a, b)
        self.assertEqual(self.calibration[2], other[2])
        torch.testing.assert_close((-self.errors[:, 0] / self.calibration[2]).argmax(-1), self.errors[:, 0].argmin(-1))

    def test_gru_first_frame_exact_and_prefix_causal_after_nonzero_correction(self):
        model = ErrorRouter("gru", *self.calibration, 8)
        with torch.no_grad(): model.output.weight.fill_(0.07); model.output.bias.copy_(torch.tensor([0., .1, -.2]))
        first = -self.errors[:, 0] / self.calibration[2]
        torch.testing.assert_close(model(self.errors)[:, 0], first, rtol=0, atol=0)
        full = model(self.errors)
        short = model(self.errors[:, :3])
        torch.testing.assert_close(full[:, :3], short, rtol=1e-6, atol=1e-6)
        changed = self.errors.clone(); changed[:, 3:] += 100
        torch.testing.assert_close(full[:, :3], model(changed)[:, :3], rtol=0, atol=0)

    def test_mlp_has_no_explicit_history_and_zero_head_matches_per_frame_argmin(self):
        model = ErrorRouter("mlp", *self.calibration, 8)
        torch.testing.assert_close(model(self.errors).argmax(-1), self.errors.argmin(-1))
        with torch.no_grad(): model.output.weight.fill_(0.05)
        full = model(self.errors)
        torch.testing.assert_close(full[:, 3:4], model(self.errors[:, 3:4]), rtol=0, atol=0)

    def test_prefix_loss_has_equal_episode_weights_and_zero_padding_gradient(self):
        logits = torch.randn(3, 5, 3, requires_grad=True)
        loss = prefix_loss(logits, self.labels, self.valid)
        expected = torch.stack([F.cross_entropy(logits[i, self.valid[i]], self.labels[i].expand(int(self.valid[i].sum()))) for i in range(3)]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()  # No optimizer step or parameter update.
        torch.testing.assert_close(logits.grad[~self.valid], torch.zeros_like(logits.grad[~self.valid]))
        self.assertTrue(bool((logits.grad[self.valid] != 0).any()))

    def test_features_reject_invalid_padding_and_nonfinite_values(self):
        validate_features(self.errors, self.valid)
        bad = self.valid.clone(); bad[2, 4] = True
        with self.assertRaises(ValueError): validate_features(self.errors, bad)
        with self.assertRaises(ValueError): validate_features(self.errors.double(), self.valid)
        bad_x = self.errors.clone(); bad_x[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): validate_features(bad_x, self.valid)

    def test_ended_episodes_are_retained_without_reading_padding(self):
        valid = np.array([[1, 1, 1], [1, 1, 0]], bool)
        first, labels = np.array([0, 0]), np.array([0, 1])
        pred = np.array([[0, 0, 1], [0, 1, 0]])
        rows = summarize_predictions([("test", 3, pred)], valid, labels, first, 2)
        all_rows = next(r for r in rows if r["observed_frame_budget"] == 3 and r["population"] == "all_episodes_up_to_frames")
        exact = next(r for r in rows if r["observed_frame_budget"] == 3 and r["population"] == "exact_frames_available")
        self.assertEqual((all_rows["sample_count"], all_rows["accuracy"]), (2, .5))
        self.assertEqual((all_rows["wrong_first_corrected"], all_rows["correct_first_broken"]), (1, 1))
        self.assertEqual((exact["sample_count"], exact["accuracy"]), (1, 0))

    def test_validation_forward_and_inference_snapshot_roundtrip_do_not_update_parameters(self):
        model = ErrorRouter("gru", *self.calibration, 8)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        metrics = selection_metrics(model, self.errors, self.valid, self.labels)
        for k, v in model.state_dict().items(): torch.testing.assert_close(before[k], v, rtol=0, atol=0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            atomic_router_snapshot(path, model, ExperimentConfig(), 4, 3, metrics)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored = ErrorRouter("gru", *self.calibration, 8)
            restored.load_state_dict(payload["state_dict"])
            torch.testing.assert_close(restored(self.errors), model(self.errors), rtol=0, atol=0)
            self.assertEqual(payload["artifact_kind"], "router_inference_only_not_resumable")
            self.assertTrue(path.with_suffix(".pt.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
