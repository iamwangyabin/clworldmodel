"""Synthetic replay storage contracts; no ROMs, collection run or updates."""

from collections import Counter
from itertools import product
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clworldmodel.reference.episode_history import EpisodeHistory


def row(n, *, first=False, last=False):
    return {
        "image": np.full((4, 4, 3), n, np.uint8),
        "is_first": np.array(first), "is_last": np.array(last),
        "is_terminal": np.array(last), "reward": np.float32(n),
        "discount": np.float32(not last), "action": np.eye(3, dtype=np.float32)[n % 3],
        "logprob": np.float32(-n / 10),
    }


class Choices:
    def __init__(self, values):
        self.values = iter(values)

    def randrange(self, n):
        value = next(self.values)
        assert 0 <= value < n
        return value


class EpisodeHistoryTests(unittest.TestCase):
    def make(self, root, capacity=None, total=20, block=2, rng=None):
        return EpisodeHistory(root, total_decisions=total, capacity_transitions=capacity,
                              block_length=block, rng=rng or random.Random(12),
                              dataset_factory=lambda eps: eps)

    def test_uncapped_blocks_reconstruct_original_episode_without_boundary_resets(self):
        with TemporaryDirectory() as td:
            history = self.make(Path(td) / "history")
            initial = history.context_row(row(0, first=True))
            history.begin_episode("ep", initial, phase=0, online=True)
            expected = [initial]
            for n in range(1, 8):
                expected.append(row(n, last=n == 7))
                history.add_transition("ep", expected[-1])
                self.assertEqual(len(history.episodes), 1)
                for key, values in history.episodes["ep"].items():
                    np.testing.assert_array_equal(values[:], np.stack([r[key] for r in expected]))
            self.assertEqual(history.accounting()["collected_transitions_retained"], 7)
            self.assertEqual(history.accounting()["stored_rows_including_context"], 11)

    def test_capacity_is_never_exceeded_and_gaps_never_become_fake_transitions(self):
        with TemporaryDirectory() as td:
            h = self.make(Path(td) / "h", capacity=4, total=8, rng=Choices([2, 0]))
            h.begin_episode("ep", row(0, first=True), phase=0, online=True)
            stale = None
            for n in range(1, 9):
                h.add_transition("ep", row(n))
                self.assertLessEqual(h.accounting()["collected_transitions_retained"], 4)
                if n == 2:
                    stale = h.episodes["ep"]["image"]
            self.assertEqual(h.slot_blocks, [3, 1])
            self.assertEqual(set(h.episodes), {"ep/retained-3", "ep/retained-7"})
            for first in (3, 7):
                ep = h.episodes[f"ep/retained-{first}"]
                np.testing.assert_array_equal(ep["reward"][:], [0, first, first + 1])
                np.testing.assert_array_equal(ep["is_first"][:], [True, False, False])
                self.assertEqual(ep["image"][0][0, 0, 0], first - 1)
            with self.assertRaisesRegex(RuntimeError, "evicted"):
                stale[:]
            self.assertEqual(h.accounting()["collected_transitions_retained"], 4)
            self.assertEqual(h.accounting()["logical_allocated_tensor_bytes"], sum(a.nbytes for a in h.arrays.values()))
            self.assertFalse(list((Path(td) / "h").glob("*.npz")))

    def test_retention_is_uniform_algorithm_r_by_exhaustive_random_choices(self):
        counts = Counter()
        with TemporaryDirectory() as td:
            for k, choices in enumerate(product(range(3), range(4))):
                h = self.make(Path(td) / str(k), capacity=2, total=4, block=1, rng=Choices(choices))
                h.begin_episode("ep", row(0, first=True), phase=0, online=True)
                for n in range(1, 5):
                    h.add_transition("ep", row(n))
                counts[tuple(sorted(h.slot_blocks))] += 1
            self.assertEqual(len(counts), 6)
            self.assertEqual(set(counts.values()), {2})

    def test_phase_libraries_are_live_views_and_have_no_evicted_history_backup(self):
        with TemporaryDirectory() as td:
            h = self.make(Path(td) / "h", capacity=4, total=6, rng=Choices([0]))
            for ep in ("old0", "old1"):
                h.begin_episode(ep, row(0, first=True), phase=0, online=True)
                h.add_transition(ep, row(1))
                h.add_transition(ep, row(2, last=True))
            h.finish_phase(0)
            library = h.rehearsal_datasets[0]
            self.assertIs(library, h.phase_episodes[0])
            self.assertEqual(set(library), {"old0", "old1"})
            h.begin_episode("current", row(0, first=True), phase=1, online=True)
            h.add_transition("current", row(3))
            self.assertEqual(set(library), {"old1"})
            self.assertNotIn("old0", h._catalog)
            self.assertIs(library["old1"], h.episodes["old1"])
            self.assertEqual(set(h.ordinary_dataset()), {"old1", "current"})
            self.assertTrue(all("task_id" not in ep for ep in h.episodes.values()))

    def test_prefill_is_in_ordinary_replay_but_not_the_online_rehearsal_library(self):
        with TemporaryDirectory() as td:
            h = self.make(Path(td) / "h")
            for ep, online in (("prefill", False), ("online0", True), ("online1", True)):
                h.begin_episode(ep, row(0, first=True), phase=0, online=online)
                h.add_transition(ep, row(1, last=True))
            h.finish_phase(0)
            self.assertEqual(set(h.ordinary_dataset()), {"prefill", "online0", "online1"})
            self.assertEqual(set(h.rehearsal_datasets[0]), {"online0", "online1"})

    def test_capped_and_full_are_identical_before_capacity_and_rng_is_isolated(self):
        before = random.getstate()
        with TemporaryDirectory() as td:
            a = self.make(Path(td) / "full", total=6)
            b = self.make(Path(td) / "bounded", total=6, capacity=6)
            for ep in ("a", "b"):
                for h in (a, b):
                    h.begin_episode(ep, row(0, first=True), phase=0, online=True)
                for n in range(1, 4):
                    for h in (a, b):
                        h.add_transition(ep, row(n, last=n == 3))
                    self.assertEqual(list(a.episodes), list(b.episodes))
                    for key in a.episodes:
                        for name in a.episodes[key]:
                            np.testing.assert_array_equal(a.episodes[key][name][:], b.episodes[key][name][:])
            self.assertEqual(a.accounting()["collected_transitions_retained"], 6)
            self.assertEqual(b.accounting()["collected_transitions_retained"], 6)
        self.assertEqual(random.getstate(), before)

    def test_invalid_capacity_budget_and_overwrite_fail(self):
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "integral"):
                self.make(Path(td) / "invalid", capacity=3)
            h = self.make(Path(td) / "h", total=1)
            h.begin_episode("ep", row(0, first=True), phase=0, online=True)
            h.add_transition("ep", row(1))
            with self.assertRaisesRegex(RuntimeError, "budget"):
                h.add_transition("ep", row(2))
            with self.assertRaises(FileExistsError):
                self.make(Path(td) / "h")


if __name__ == "__main__":
    unittest.main()
