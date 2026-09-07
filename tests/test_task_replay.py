"""Focused CPU contracts for the retained task-policy and replay components."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import torch.nn as nn
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.continual import ActorCriticBank, allocate_task_updates
    from ac import ActorCritic, ActorCriticTrainingStep, train_ac_from_wm
    from config import Config
    from replay import FifoReplay, MultiTypeReplay
    from rssm import Representation, Rssm
    from wm import WorldModel


@unittest.skipIf(torch is None, "requires PyTorch and Atari imports")
class TaskReplayTests(unittest.TestCase):
    def test_fifo_and_mixed_replay_filter_homogeneous_task_minibatches(self) -> None:
        def values(marker: float, sequences: int = 2):
            actions = torch.full((3, sequences, 4), marker)
            observations = torch.full((3, sequences, 3, 64, 64), marker)
            rewards = torch.full((3, sequences, 1), marker)
            continues = torch.ones(3, sequences, 1)
            resets = torch.zeros(3, sequences, 1)
            return actions, observations, rewards, continues, resets

        first = FifoReplay(3, 4, 4, "cpu", store_task_ids=True)
        second = FifoReplay(3, 4, 4, "cpu", store_task_ids=True)
        replay = MultiTypeReplay(first, second, sampling_weights=(0.5, 0.5))
        replay.add(*values(10.0), task_id=0)
        replay.add(*values(20.0), task_id=1)

        np.random.seed(3)
        sample = replay.minibatch_with_metadata(2, 32, "cpu", task_id=1)
        actions, _, rewards, _, _, replay_index, _, sequence_indices = sample
        source = replay.replays[replay_index]
        sampled_task_ids = source.task_ids[sequence_indices]
        self.assertTrue(torch.equal(sampled_task_ids, torch.ones_like(sampled_task_ids)))
        self.assertTrue(torch.equal(actions, torch.full_like(actions, 20.0)))
        self.assertTrue(torch.equal(rewards, torch.full_like(rewards, 20.0)))
        self.assertEqual(replay.available_task_ids(), (0, 1))

        with self.assertRaisesRegex(ValueError, "task 7"):
            replay.minibatch(2, 1, "cpu", task_id=7)
        with self.assertRaisesRegex(ValueError, "requires task_id"):
            first.add(*values(30.0))
        with self.assertRaisesRegex(TypeError, "integer"):
            first.add(*values(30.0), task_id=[1, 1])

        baseline = FifoReplay(3, 2, 4, "cpu")
        self.assertIsNone(baseline.task_ids)
        baseline.add(*values(30.0))
        with self.assertRaisesRegex(ValueError, "without task-id storage"):
            baseline.add(*values(40.0), task_id=0)

    def test_update_allocation_preserves_the_total_budget(self) -> None:
        allocation = allocate_task_updates(
            800,
            current_task_id=2,
            available_task_ids=(0, 1, 2),
            current_task_fraction=0.5,
        )
        self.assertEqual(sum(allocation.values()), 800)
        self.assertEqual(allocation[2], 400)
        self.assertEqual(allocation[0], 200)
        self.assertEqual(allocation[1], 200)

        first_task = allocate_task_updates(
            800,
            current_task_id=0,
            available_task_ids=(0,),
            current_task_fraction=0.5,
        )
        self.assertEqual(first_task, {0: 800})

        current_only = allocate_task_updates(
            800,
            current_task_id=2,
            available_task_ids=(0, 1, 2),
            current_task_fraction=1.0,
        )
        self.assertEqual(current_only[2], 800)
        self.assertEqual(current_only[0], 0)
        self.assertEqual(current_only[1], 0)

    def test_actor_bank_warm_starts_weights_but_not_optimizer_objects(self) -> None:
        def factory(task_id: int):
            actor_critic = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            return SimpleNamespace(
                ac=actor_critic,
                opt=optimizer,
                slow_critic=None,
                return_scale_ema=torch.tensor(float(task_id)),
                return_mean_ema=None,
            )

        bank = ActorCriticBank()
        source = bank.ensure(0, factory)
        with torch.no_grad():
            source.ac.weight.fill_(3.0)
        target = bank.ensure(1, factory, warm_start_from=0)

        torch.testing.assert_close(target.ac.weight, source.ac.weight)
        self.assertIsNot(target.ac, source.ac)
        self.assertIsNot(target.opt, source.opt)
        self.assertEqual(target.opt.state, {})
        self.assertEqual(target.return_scale_ema.item(), 0.0)

    def test_actor_bank_can_create_a_fresh_task_and_freeze_old_actors(self) -> None:
        def factory(task_id: int):
            actor_critic = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            return SimpleNamespace(
                ac=actor_critic,
                opt=optimizer,
                slow_critic=None,
                return_scale_ema=torch.tensor(float(task_id)),
                return_mean_ema=None,
            )

        torch.manual_seed(13)
        bank = ActorCriticBank()
        source = bank.ensure(0, factory)
        with torch.no_grad():
            source.ac.weight.fill_(3.0)
        target = bank.ensure(1, factory)

        self.assertFalse(torch.equal(target.ac.weight, source.ac.weight))
        bank.activate(1)
        self.assertFalse(any(parameter.requires_grad for parameter in source.ac.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in target.ac.parameters()))
