"""Focused contracts for the native R2-Dreamer plus ARROW replay route."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
sys.path.insert(0, str(PROJECT_SRC))
sys.path.insert(0, str(ROOT))

from clworldmodel.r2dreamer.config import R2DreamerConfig

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None

if torch is not None:
    from clworldmodel.r2dreamer.agent import R2DreamerAgent, R2UpdateResult
    from clworldmodel.replay.arrow_r2_adapter import ArrowR2ReplayAdapter


def _tiny_config() -> R2DreamerConfig:
    return R2DreamerConfig(
        image_height=16,
        image_width=16,
        action_dim=3,
        batch_size=4,
        batch_length=4,
        stoch=2,
        deter=8,
        hidden=8,
        discrete=2,
        depth=1,
        units=8,
        encoder_mults=(1, 1),
        encoder_minres=4,
        rssm_blocks=1,
        reward_bins=5,
        warmup_updates=1,
        imagination_horizon=2,
        device="cpu",
        amp=False,
    )


class R2DreamerConfigTests(unittest.TestCase):
    def test_size12m_geometry_matches_upstream_profile(self) -> None:
        config = R2DreamerConfig()
        self.assertEqual(config.embedding_dim, 1024)
        self.assertEqual(config.feature_dim, 2560)
        self.assertEqual(config.sample_count, 1024)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.batch_length, 64)
        self.assertEqual(config.deter, 2048)
        self.assertEqual(config.discrete, 16)

    def test_rank_mismatched_objective_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "S=512 < E=1024"):
            R2DreamerConfig(batch_length=32)

    def test_unknown_config_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown R2-Dreamer"):
            R2DreamerConfig.from_dict({"batch_length": 64, "unknown": 1})


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class AtariEnvironmentRegistrationTests(unittest.TestCase):
    def test_vector_worker_registers_ale_before_making_environment(self) -> None:
        from scripts import train_r2dreamer_arrow_atari as trainer

        sentinel = object()

        class CapturingVectorEnvironment:
            def __init__(self, env_fns, *, autoreset_mode) -> None:
                self.env_fns = env_fns
                self.autoreset_mode = autoreset_mode

        with (
            patch.object(trainer, "AsyncVectorEnv", CapturingVectorEnvironment),
            patch.object(trainer, "AtariPreprocessing", side_effect=lambda env, **_: env),
            patch.object(trainer.gym, "register_envs") as register_envs,
        ):
            environment = trainer._make_vector_environment([lambda: sentinel], action_repeat=4)
            self.assertIs(environment.env_fns[0](), sentinel)

        register_envs.assert_called_once_with(trainer.ale_py)


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class AtariReplayPackingTests(unittest.TestCase):
    def test_worker_streams_preserve_temporal_order_in_arrow_slots(self) -> None:
        from scripts import train_r2dreamer_arrow_atari as trainer

        streams = torch.arange(8).reshape(4, 2, 1)
        packed = trainer._pack_worker_streams_for_arrow(
            streams,
            sequence_length=2,
            sequence_count=4,
        )

        self.assertEqual(tuple(packed.shape), (2, 4, 1))
        torch.testing.assert_close(
            packed.squeeze(-1),
            torch.tensor([[0, 4, 1, 5], [2, 6, 3, 7]]),
        )


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class R2DreamerArrowIntegrationTests(unittest.TestCase):
    def test_arrow_adapter_shifts_action_once_and_refreshes_latent_sidecar(self) -> None:
        config = _tiny_config()

        class SubReplay:
            t = 5
            n = 4

            def __init__(self) -> None:
                self.acts = torch.zeros(self.t, self.n, 3)
                self.obss = torch.zeros(self.t, self.n, 3, 16, 16)
                self.rews = torch.zeros(self.t, self.n, 1)
                self.conts = torch.ones(self.t, self.n, 1)
                self.resets = torch.zeros(self.t, self.n, 1)

        class Replay:
            n_valid = 4

            def __init__(self) -> None:
                self.replays = (SubReplay(),)

            def add(self, actions, observations, rewards, continues, resets):
                sub = self.replays[0]
                sub.acts.copy_(actions)
                sub.obss.copy_(observations)
                sub.rews.copy_(rewards)
                sub.conts.copy_(continues)
                sub.resets.copy_(resets)
                return ([0, 1, 2, 3],)

            def minibatch_with_metadata(self, mb_t: int, mb_n: int, mb_device: str):
                self.request = (mb_t, mb_n, mb_device)
                sub = self.replays[0]
                return (
                    sub.acts.to(mb_device),
                    sub.obss.to(mb_device),
                    sub.rews.to(mb_device),
                    sub.conts.to(mb_device),
                    sub.resets.to(mb_device),
                    0,
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                )

        replay = Replay()
        adapter = ArrowR2ReplayAdapter(replay, config)
        actions = torch.arange(5 * 4 * 3).reshape(5, 4, 3).float()
        observations = torch.arange(5 * 4 * 3 * 16 * 16).reshape(5, 4, 3, 16, 16).float() / 255.0
        rewards = torch.arange(5 * 4).reshape(5, 4, 1).float()
        continues = torch.ones(5, 4, 1)
        resets = torch.zeros(5, 4, 1)
        resets[1, 0, 0] = 1
        is_last = torch.zeros(5, 4, 1, dtype=torch.bool)
        is_last[2, 1, 0] = True
        stoch = torch.arange(5 * 4 * 2 * 2).reshape(5, 4, 2, 2).float()
        deter = torch.arange(5 * 4 * 8).reshape(5, 4, 8).float()
        adapter.add(actions, observations, rewards, continues, resets, is_last, stoch, deter)
        sample = adapter.sample()
        batch = sample.batch

        self.assertEqual(replay.request, (5, 4, "cpu"))
        self.assertEqual(tuple(batch.images.shape), (4, 4, 16, 16, 3))
        self.assertEqual(tuple(batch.actions.shape), (4, 4, 3))
        torch.testing.assert_close(batch.actions[0, 0], torch.tensor([0.0, 1.0, 2.0]))
        torch.testing.assert_close(batch.initial_stoch[0], stoch[0, 0])
        torch.testing.assert_close(batch.initial_deter[3], deter[0, 3])
        self.assertTrue(batch.is_first[0, 0])
        self.assertTrue(batch.is_last[1, 1])

        update = R2UpdateResult(
            metrics={},
            posterior_stoch=torch.full((4, 4, 2, 2), 7.0),
            posterior_deter=torch.full((4, 4, 8), 9.0),
        )
        adapter.update_latent_states(sample.reference, update)
        self.assertEqual(adapter._stoch_states[0][1, 0, 0, 0].item(), 7.0)
        self.assertEqual(adapter._deter_states[0][4, 3, 0].item(), 9.0)
        accounting = adapter.storage_accounting()
        self.assertGreater(accounting["r2_latent_state_storage_bytes"], 0)
        self.assertGreater(accounting["r2_transition_metadata_storage_bytes"], 0)

    def test_native_agent_is_decoder_free_and_updates_projector(self) -> None:
        torch.manual_seed(7)
        config = _tiny_config()
        agent = R2DreamerAgent(config)
        self.assertFalse(hasattr(agent, "decoder"))
        self.assertEqual(agent.embedding_dim, 16)
        self.assertEqual(tuple(agent.projector.w.weight.shape), (16, 12))

        class SubReplay:
            t = 5
            n = 4

            def __init__(self) -> None:
                self.acts = torch.nn.functional.one_hot(
                    torch.randint(0, 3, (self.t, self.n)), num_classes=3
                ).float()
                self.obss = torch.rand(self.t, self.n, 3, 16, 16)
                self.rews = torch.randn(self.t, self.n, 1)
                self.conts = torch.ones(self.t, self.n, 1)
                self.resets = torch.zeros(self.t, self.n, 1)

        class Replay:
            n_valid = 4

            def __init__(self) -> None:
                self.replays = (SubReplay(),)

            def add(self, actions, observations, rewards, continues, resets):
                return ([0, 1, 2, 3],)

            def minibatch_with_metadata(self, mb_t, mb_n, mb_device):
                sub = self.replays[0]
                return (
                    sub.acts.to(mb_device),
                    sub.obss.to(mb_device),
                    sub.rews.to(mb_device),
                    sub.conts.to(mb_device),
                    sub.resets.to(mb_device),
                    0,
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                )

        replay = Replay()
        adapter = ArrowR2ReplayAdapter(replay, config)
        sub = replay.replays[0]
        adapter.add(
            sub.acts,
            sub.obss,
            sub.rews,
            sub.conts,
            sub.resets,
            torch.zeros(5, 4, 1, dtype=torch.bool),
            torch.zeros(5, 4, 2, 2),
            torch.zeros(5, 4, 8),
        )
        batch = adapter.sample().batch
        before = agent.projector.w.weight.detach().clone()
        update = agent.update_batch(batch)
        metrics = update.metrics

        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertFalse(torch.equal(before, agent.projector.w.weight.detach()))
        self.assertGreater(metrics["metric/grad_norm"], 0)


if __name__ == "__main__":
    unittest.main()
