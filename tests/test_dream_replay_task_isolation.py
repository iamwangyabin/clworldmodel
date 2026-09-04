"""Replay task metadata must not become a model input."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
PROJECT_SRC = ROOT / "src"

try:
    import torch
    import gymnasium  # noqa: F401
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - experiment environment coverage
    torch = None
else:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(ATARI))
    from ac import dream_rollout


@unittest.skipIf(torch is None, "requires the pinned Atari experiment environment")
class DreamReplayTaskIsolationTests(unittest.TestCase):
    def test_current_task_filters_replay_without_routing_shared_networks(self) -> None:
        class ReplaySpy:
            def __init__(self) -> None:
                self.task_ids: list[int | None] = []

            def minibatch(self, t, n, mb_device="cpu", task_id=None):
                self.task_ids.append(task_id)
                return (
                    torch.zeros(t, n, 2, device=mb_device),
                    torch.zeros(t, n, 3, 64, 64, device=mb_device),
                    torch.zeros(t, n, 1, device=mb_device),
                    torch.ones(t, n, 1, device=mb_device),
                    torch.zeros(t, n, 1, device=mb_device),
                )

        class RssmSpy:
            def __init__(self) -> None:
                self.task_ids: list[int | None] = []

            def initial_state(self, n):
                return torch.zeros(n, 1, 2), torch.zeros(n, 3)

            def __call__(self, z, actions, h, observations, resets, **kwargs):
                self.task_ids.append(kwargs.get("task_id"))
                if actions.ndim == 3:
                    t, n = actions.shape[:2]
                    return (
                        torch.zeros(t, n, 1, 2),
                        torch.zeros(t, n, 1, 2),
                        torch.zeros(t, n, 3),
                    )
                return torch.zeros_like(z), torch.zeros_like(z), torch.zeros_like(h)

        class WorldModelSpy:
            compute_dtype = "float32"

            def __init__(self) -> None:
                self.rssm = RssmSpy()
                self.prediction_task_ids: list[int | None] = []

            def zh_transform(self, z, h):
                return torch.cat((z.flatten(-2), h), dim=-1)

            def predict_reward_symlog(self, state, task_id):
                self.prediction_task_ids.append(task_id)
                return torch.zeros(*state.shape[:-1], 1)

            def predict_continue(self, state, task_id):
                self.prediction_task_ids.append(task_id)
                return torch.ones(*state.shape[:-1], 1)

        class ActorCriticSpy:
            def __call__(self, state):
                return torch.zeros(*state.shape[:-1], 2), torch.zeros(
                    *state.shape[:-1], 1
                )

        replay = ReplaySpy()
        wm = WorldModelSpy()
        dream_rollout(
            wm,
            ActorCriticSpy(),
            replay,
            n_sync=2,
            n_steps=1,
            n_ctx_frames=2,
            replay_task_id=3,
            task_id=None,
        )

        self.assertEqual(replay.task_ids, [3])
        self.assertEqual(wm.rssm.task_ids, [None, None])
        self.assertEqual(wm.prediction_task_ids, [None, None])


if __name__ == "__main__":
    unittest.main()
