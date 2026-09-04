"""Focused tests for the bounded Dream Rehearsal port."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in experiment envs
    torch = None

if torch is not None:
    from clworldmodel.continual.dream_rehearsal import (
        DreamRehearsalConfig,
        crossed_rehearsal_intervals,
        realized_first_scores,
        rehearsal_update_allocation,
        selected_behavior_cloning_loss,
        top_fraction_indices,
    )


@unittest.skipIf(torch is None, "requires PyTorch")
class BoundedDreamRehearsalTests(unittest.TestCase):
    def test_paper_constants_are_explicit_and_validated(self) -> None:
        config = DreamRehearsalConfig()

        self.assertEqual(config.interval_agent_decisions, 2_000)
        self.assertEqual(config.updates_per_prior_task, 50)
        self.assertEqual(config.batch_sequences, 4)
        self.assertEqual(config.context_steps, 16)
        self.assertEqual(config.horizon, 15)
        self.assertEqual(config.top_fraction, 0.25)
        self.assertEqual(config.realized_threshold, 0.3)
        self.assertEqual(config.realized_bonus, 10.0)

        with self.assertRaisesRegex(ValueError, "top fraction"):
            DreamRehearsalConfig(top_fraction=0.0)

    def test_interval_accounting_preserves_fractional_remainders(self) -> None:
        self.assertEqual(crossed_rehearsal_intervals(0, 1_999, 2_000), 0)
        self.assertEqual(crossed_rehearsal_intervals(1_999, 2_001, 2_000), 1)
        self.assertEqual(crossed_rehearsal_intervals(2_001, 6_050, 2_000), 2)
        self.assertEqual(
            rehearsal_update_allocation(2, [2, 0, 1], 50),
            {0: 100, 1: 100, 2: 100},
        )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            crossed_rehearsal_intervals(0.0, 2_000, 2_000)

    def test_reach_weighting_suppresses_post_terminal_reward_and_value(self) -> None:
        rewards = torch.tensor(
            [
                [[0.0], [0.0]],
                [[1.0], [0.0]],
                [[100.0], [0.0]],
            ]
        )
        continues = torch.tensor(
            [
                [[1.0], [1.0]],
                [[0.0], [1.0]],
                [[1.0], [1.0]],
            ]
        )
        bootstrap = torch.tensor([[100.0], [2.0]])

        scores, realized, survived = realized_first_scores(
            rewards,
            continues,
            bootstrap,
            discount=1.0,
        )

        torch.testing.assert_close(realized[:, 0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(survived[:, 0], torch.tensor([0.0, 1.0]))
        # The terminated success gets its realized bonus but neither the
        # post-terminal reward nor the optimistic bootstrap.
        torch.testing.assert_close(scores[:, 0], torch.tensor([11.0, 2.0]))

    def test_realized_success_outranks_a_larger_value_promise(self) -> None:
        rewards = torch.tensor([[[0.31], [0.0]]])
        continues = torch.ones_like(rewards)
        bootstrap = torch.tensor([[0.0], [9.0]])

        scores, _, _ = realized_first_scores(
            rewards,
            continues,
            bootstrap,
            discount=1.0,
        )

        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertEqual(top_fraction_indices(scores, 0.25).tolist(), [0])

    def test_cloning_loss_updates_only_selected_trajectories(self) -> None:
        raw_logits = torch.tensor(
            [
                [[2.0, -2.0], [-2.0, 2.0], [1.0, -1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0], [2.0, -2.0], [-2.0, 2.0]],
            ],
            requires_grad=True,
        )
        log_probs = torch.log_softmax(raw_logits, dim=-1)
        actions = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            ]
        )
        selected = torch.tensor([1, 3], dtype=torch.long)

        loss = selected_behavior_cloning_loss(log_probs, actions, selected)
        loss.backward()

        self.assertGreater(float(raw_logits.grad[:, selected].abs().sum()), 0.0)
        self.assertEqual(float(raw_logits.grad[:, [0, 2]].abs().sum()), 0.0)

    def test_top_fraction_uses_floor_with_at_least_one(self) -> None:
        scores = torch.arange(7, dtype=torch.float32)
        chosen = top_fraction_indices(scores, 0.25)

        self.assertEqual(chosen.numel(), 1)
        self.assertEqual(chosen.item(), 6)


if __name__ == "__main__":
    unittest.main()
