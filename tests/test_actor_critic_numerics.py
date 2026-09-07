"""Focused contracts for the KAN-Dreamer-aligned FastKAN behavior heads."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
PROJECT_SRC = ROOT / "src"

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised on the GPU host
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from ac import (
        ActorCritic,
        ActorCriticOpt,
        dream_rollout,
        replay_lambda_returns,
        train_ac_from_wm,
    )

    from clworldmodel.optim import LaProp
    from config import Config


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class ActorCriticNumericsTests(unittest.TestCase):




























    def test_corrected_rollout_bootstraps_the_post_transition_state(self) -> None:
        class DummyRssm:
            @staticmethod
            def initial_state(n_sync: int):
                return torch.zeros(n_sync, 1, 1), torch.zeros(n_sync, 1)

            def __call__(
                self,
                z,
                actions,
                h,
                images,
                resets,
                temperature=1.0,
            ):
                del resets, temperature
                if images is not None:
                    time, batch = actions.shape[:2]
                    context_z = torch.zeros(time, batch, 1, 1)
                    context_h = torch.arange(1, time + 1, dtype=torch.float32)
                    context_h = context_h.view(time, 1, 1).expand(time, batch, 1)
                    return None, context_z, context_h
                return None, z, h + 1.0

        class DummyWorldModel:
            rssm = DummyRssm()

            @staticmethod
            def zh_transform(z, h):
                del z
                return h

            @staticmethod
            def reward_fc(zh):
                return torch.zeros_like(zh)

            @staticmethod
            def continue_fc(zh):
                return torch.ones_like(zh)

        class DummyReplay:
            @staticmethod
            def minibatch(time, batch, mb_device):
                actions = torch.zeros(time, batch, 2, device=mb_device)
                images = torch.zeros(time, batch, 3, 64, 64, device=mb_device)
                rewards = torch.zeros(time, batch, 1, device=mb_device)
                continues = torch.ones(time, batch, 1, device=mb_device)
                resets = torch.zeros(time, batch, 1, device=mb_device)
                return actions, images, rewards, continues, resets

        class DummyActorCritic:
            def __call__(self, state):
                action_logs = torch.full(
                    (*state.shape[:-1], 2),
                    -0.6931471805599453,
                    device=state.device,
                )
                return action_logs, state[..., -1:]

        rollout_kwargs = {
            "wm": DummyWorldModel(),
            "ac": DummyActorCritic(),
            "data": DummyReplay(),
            "n_sync": 1,
            "n_steps": 1,
            "discount": 0.5,
        }
        legacy_returns = dream_rollout(**rollout_kwargs)[3]
        corrected_returns = dream_rollout(
            **rollout_kwargs,
            corrected_terminal_bootstrap=True,
        )[3]

        torch.testing.assert_close(legacy_returns, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(corrected_returns, torch.tensor([[[2.5]]]))

    def test_replay_lambda_returns_use_same_index_rewards_and_stop_at_terminals(self) -> None:
        rewards = torch.tensor([[[1.0]], [[2.0]], [[4.0]], [[8.0]]])
        continues = torch.tensor([[[1.0]], [[0.0]], [[1.0]], [[1.0]]])
        bootstrap = torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
        targets = replay_lambda_returns(
            rewards,
            continues,
            bootstrap,
            discount=0.5,
            lam=0.5,
        )
        expected = torch.tensor([[[6.5]], [[2.0]], [[24.0]]])
        torch.testing.assert_close(targets, expected)







    def test_laprop_warmup_and_agc_state_match_the_documented_order(self) -> None:
        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        optimizer = LaProp(
            [parameter],
            lr=0.1,
            betas=(0.0, 0.5),
            agc_clip=0.3,
            warmup_steps=2,
        )
        parameter.grad = torch.tensor([6.0, 8.0])
        original = parameter.detach().clone()
        optimizer.step()

        torch.testing.assert_close(parameter, original)
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg_sq"],
            0.5 * torch.tensor([0.9, 1.2]).square(),
        )

        parameter.grad = torch.tensor([6.0, 8.0])
        optimizer.step()
        self.assertFalse(torch.equal(parameter, original))


if __name__ == "__main__":
    unittest.main()
