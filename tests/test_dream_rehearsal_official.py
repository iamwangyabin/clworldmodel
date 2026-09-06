"""Contracts for the official-code Atari baseline (no ROMs or training runs)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.reference.dream_rehearsal import (
    OfficialDreamRehearsalConfig,
    ReplayLibraries,
    phase_chunks,
    run_phase_chunks,
)


class OfficialDreamRehearsalScheduleTests(unittest.TestCase):
    def test_unknown_keys_and_nonreference_algorithm_options_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            OfficialDreamRehearsalConfig.from_dict({"current_task_only": True})
        with self.assertRaisesRegex(ValueError, "official"):
            OfficialDreamRehearsalConfig.from_dict({"rehearsal_updates": 400})
        with self.assertRaisesRegex(ValueError, "official"):
            OfficialDreamRehearsalConfig.from_dict({"batch_size": 4})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            OfficialDreamRehearsalConfig.from_dict({"seed": True})
        with self.assertRaisesRegex(ValueError, "smaller than"):
            OfficialDreamRehearsalConfig.from_dict({"seed": 2**32})
        with self.assertRaisesRegex(ValueError, "numeric"):
            OfficialDreamRehearsalConfig.from_dict({"sticky_probability": False})

    def test_exact_chunks_and_no_extra_steps_or_tail_rehearsal(self):
        cfg = OfficialDreamRehearsalConfig()
        chunks = list(phase_chunks(cfg))
        self.assertEqual(len(chunks), 737)
        self.assertEqual([c.agent_decisions for c in chunks[:-1]], [2000] * 736)
        self.assertEqual(chunks[-1].agent_decisions, 60)
        self.assertFalse(chunks[-1].rehearsal_due)
        self.assertEqual(sum(c.agent_decisions for c in chunks) + cfg.prefill_decisions,
                         cfg.task_total_decisions)

    def test_rehearsal_interleaves_instead_of_bursting_at_16384(self):
        cfg = OfficialDreamRehearsalConfig.smoke()
        events = []
        run_phase_chunks(
            cfg, (0, 1),
            train=lambda n: events.append(("train", n)),
            rehearse=lambda j, n: events.append(("rehearse", j, n)),
            evaluate=lambda completed: events.append(("eval", completed)),
        )
        self.assertEqual(events, [
            ("eval", 0), ("train", 2000),
            ("rehearse", 0, 50), ("rehearse", 1, 50), ("eval", 2000),
            ("train", 500), ("eval", 2500),
        ])

    def test_ordinary_dataset_remains_shared_and_libraries_are_phase_local(self):
        seen = []
        def factory(eps):
            seen.append(eps)
            return iter([eps])
        libraries = ReplayLibraries(factory)
        libraries.episodes["prefill0"] = {"reward": [0, 1]}
        before = set(libraries.episodes)
        libraries.episodes["old0"] = {"reward": [0, 1]}
        libraries.episodes["old1"] = {"reward": [0, 2]}
        libraries.finish_phase(0, before)
        libraries.episodes["current"] = {"reward": [0, 3]}
        dataset = libraries.ordinary_dataset()
        self.assertIs(next(dataset), libraries.episodes)
        self.assertEqual(set(seen[-1]), {"prefill0", "old0", "old1", "current"})
        self.assertEqual(set(libraries.phase_episodes[0]), {"old0", "old1"})
        self.assertIs(libraries.phase_episodes[0]["old0"], libraries.episodes["old0"])
        self.assertNotIn("task_id", libraries.phase_episodes[0]["old0"])

    def test_single_episode_phase_fails_instead_of_silently_skipping_rehearsal(self):
        libraries = ReplayLibraries(iter)
        libraries.episodes["one"] = {"reward": [0, 1]}
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            libraries.finish_phase(0, set())


class OfficialDreamRehearsalLaunchTests(unittest.TestCase):
    def test_cli_overrides_only_explicit_values(self):
        with TemporaryDirectory() as td:
            config = Path(td) / "protocol.json"
            config.write_text(json.dumps({"seed": 77, "cpu_threads": 3, "device": "cpu"}))
            command = [sys.executable, str(ROOT / "scripts/run_dream_rehearsal_official_atari.py"),
                       "--config", str(config), "--dry-run"]
            result = subprocess.run(command, cwd=td, check=True, capture_output=True, text=True)
            first = json.loads(result.stdout)["config"]
            self.assertEqual((first["seed"], first["cpu_threads"], first["device"]), (77, 3, "cpu"))
            result = subprocess.run(command + ["--seed", "78"], cwd=td, check=True, capture_output=True, text=True)
            second = json.loads(result.stdout)["config"]
            self.assertEqual((second["seed"], second["cpu_threads"], second["device"]), (78, 3, "cpu"))

    def test_dry_run_is_side_effect_free_and_reports_reference_not_arrow(self):
        with TemporaryDirectory() as td:
            output = Path(td) / "run"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_dream_rehearsal_official_atari.py"),
                "--dry-run", "--output-dir", str(output),
            ], cwd=td, check=True, text=True, capture_output=True)
            manifest = json.loads(result.stdout)
            self.assertFalse(output.exists())
            self.assertEqual(manifest["method"], "Dream-Rehearsal-OfficialCode-v1-Atari")
            self.assertFalse(manifest["claims"]["paper_minigrid_reproduction"])
            self.assertFalse(manifest["claims"]["compute_matched_to_arrow"])
            self.assertEqual(manifest["replay"]["ordinary_training"], "shared_full_history")
            self.assertEqual(manifest["config"]["batch_size"], 16)
            self.assertEqual(manifest["config"]["batch_length"], 64)
            self.assertEqual(manifest["projected_budgets"]["rehearsal_updates"], 552000)
            self.assertEqual(manifest["projected_budgets"]["agent_decisions"], 8847360)
            self.assertTrue(manifest["reference_sources"])

    def test_smoke_does_not_silently_change_batch_or_grading(self):
        cfg = OfficialDreamRehearsalConfig.smoke()
        self.assertEqual((cfg.batch_size, cfg.batch_length), (16, 64))
        self.assertEqual(cfg.realized_threshold, 0.3)
        self.assertEqual(cfg.realized_bonus, 10.0)
        self.assertEqual(cfg.rehearsal_updates, 50)
        self.assertEqual(cfg.classification, "smoke")


if __name__ == "__main__":
    unittest.main()
