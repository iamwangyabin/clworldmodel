"""Ownership, fixed-budget and inference-state checks for the new controls."""
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"))
sys.path.insert(0, str(ROOT / "scripts"))
from config import Config
from wm import WorldModel
from ac import build_actor_critic_opt, train_ac_from_wm
from replay import FifoReplay, LongTermReplay, MultiTypeReplay
from run_capacity_control_atari import budgets, resolved_config
from clworldmodel.reference.capacity_controls import (
    CONTROLS, build_capacity_model, capacity_world_model_update,
)


class FixedEncoder(torch.nn.Module):
    output_size = 4096

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(4096))

    def forward(self, images):
        return self.offset.unsqueeze(0).expand(len(images), -1)


def small_model(control):
    return build_capacity_model(
        WorldModel, 3, (2, 3), 4, 8, cnn_depth=4, mlp_features=8,
        control=control, num_tasks=2, image_embedder=FixedEncoder(),
        task_mechanism_recurrent_width=8, task_mechanism_representation_width=8,
        task_mechanism_transition_width=8,
    )


def small_replay():
    buffers = (FifoReplay(16, 4, 4, "cpu", store_task_ids=True),
               LongTermReplay(16, 4, 4, "cpu", store_task_ids=True))
    replay = MultiTypeReplay(*buffers, sampling_weights=(0.5, 0.5))
    for task in (0, 1):
        actions = torch.nn.functional.one_hot(torch.zeros(16, 2, dtype=torch.long), 4).float()
        replay.add(actions, torch.rand(16, 2, 3, 64, 64), torch.zeros(16, 2, 1),
                   torch.ones(16, 2, 1), torch.zeros(16, 2, 1), task_id=task)
    return replay


class CapacityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_scope_and_unknown_keys(self):
        for control in CONTROLS:
            config = Config.from_dict(resolved_config(control))
            self.assertTrue(config.uses_capacity_control)
            self.assertTrue(config.uses_task_labelled_replay)
            self.assertFalse(config.uses_evolving_atomic_rssm)
            self.assertFalse(config.uses_adaptive_qfp_compression)
            self.assertEqual((config.epochs, config.mb_n_size), (540, 16))
            self.assertEqual(config.to_dict()["capacity_control"], control)
        bad = resolved_config("shared")
        bad["unknown_field"] = True
        with self.assertRaises(TypeError):
            Config.from_dict(bad)
        bad.pop("unknown_field")
        bad["continual_method"] = "none"
        with self.assertRaises(ValueError):
            Config.from_dict(bad)

    def test_smoke_changes_budget_not_width_or_minibatches(self):
        for control in CONTROLS:
            full, smoke = resolved_config(control), resolved_config(control, smoke=True)
            Config.from_dict(smoke)
            self.assertEqual(budgets(full)["world_model_updates"], 540000)
            self.assertEqual(budgets(full)["replay_tensor_bytes_without_task_metadata"], 6486491136)
            self.assertEqual(budgets(smoke)["world_model_updates"], 4)
            for key in ("mb_t_size", "mb_n_size", "ac_train_sync", "gru_units", "mlp_features"):
                self.assertEqual(smoke[key], full[key])

    def test_shared_loss_matches_underlying_model(self):
        model = small_model("shared")
        batch = small_replay().minibatch(4, 2, mb_device="cpu")
        torch.manual_seed(7)
        expected = model.models[0].compute_loss(*batch)[0]
        torch.manual_seed(7)
        actual = model.compute_loss(*batch, task_id=1)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.compute_loss(*batch)

    def test_fullbank_independence_and_frozen_shared_ownership(self):
        bank = small_model("fullbank")
        p0, p1 = next(bank.models[0].parameters()), next(bank.models[1].parameters())
        self.assertNotEqual(p0.data_ptr(), p1.data_ptr())
        before = p1.detach().clone()
        with torch.no_grad():
            p0.add_(1)
        torch.testing.assert_close(p1, before)
        frozen = small_model("frozen")
        frozen.initialize_task_expert(1, 0)
        frozen.activate_task_expert(1)
        shared = frozen.models[0].shared_parameter_groups()
        self.assertTrue(shared)
        self.assertTrue(all(not p.requires_grad for group in shared.values() for p in group))
        independent = small_model("independent")
        self.assertFalse(independent.models[0].rssm.task_mechanism_reuse)

    @patch("torch.cuda.is_available", return_value=False)
    def test_two_task_update_actor_and_inference_roundtrip(self, _cuda_unavailable):
        # Explicit CPU test: Adam must not query a CUDA capture stream merely
        # because the virtualized runtime advertises GPUs. GPU smoke is separate.
        for control in CONTROLS:
            with self.subTest(control=control):
                torch.manual_seed(123)
                model = small_model(control)
                replay = small_replay()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                config = SimpleNamespace(mb_n_size=4, current_batch_n=3, memory_batch_n=1,
                                         mb_t_size=4, memory_loss_scale=1.0, compute_dtype="float32")
                for task in (0, 1):
                    if task:
                        model.initialize_task_expert(task, task - 1)
                    model.activate_task_expert(task)
                    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                    metrics, norm = capacity_world_model_update(config, model, replay, optimizer,
                                                                 task, np.random.default_rng(5))
                    self.assertTrue(torch.isfinite(norm))
                    for name, parameter in model.named_parameters():
                        if name in frozen:
                            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
                actor = build_actor_critic_opt(model, lr=1e-4)
                result, _, _ = train_ac_from_wm(model, replay, 1, 2, aco=actor,
                                                task_id=1, dream_steps=2)
                self.assertIs(result, actor)
                stream = io.BytesIO()
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream)
                stream.seek(0)
                state = torch.load(stream, map_location="cpu", weights_only=False)
                restored = small_model(control)
                restored.load_state_dict(state["model"])
                restored.activate_task_expert(1)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
                restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
                restored_optimizer.load_state_dict(state["optimizer"])
                self.assertEqual(len(optimizer.state), len(restored_optimizer.state))


if __name__ == "__main__":
    unittest.main()
