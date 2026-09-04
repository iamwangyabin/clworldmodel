"""Typed-config contracts for never-clear Dream Rehearsal."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
SOURCE_CONFIG = (
    ROOT
    / "third_party"
    / "arrow"
    / "Configs"
    / "Atari configs"
    / "CL-task configs"
    / "Original Order"
    / "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,ALE_Seaquest,ALE_Enduro-s0-dv3.json"
)

try:
    import torch  # noqa: F401
    import cv2  # noqa: F401
    import gymnasium  # noqa: F401
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - experiment environment coverage
    Config = None
else:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(VENDORED_ATARI))
    from config import Config

    spec = importlib.util.spec_from_file_location(
        "never_clear_dream_rehearsal_launcher",
        SCRIPTS / "run_never_clear_dream_rehearsal_atari.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(launcher)


@unittest.skipIf(Config is None, "requires the pinned Atari experiment environment")
class NeverClearDreamRehearsalConfigTests(unittest.TestCase):
    @staticmethod
    def _data() -> dict:
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        return launcher._resolved_config(source, epochs=2)

    def test_config_requires_full_history_fifo_and_shared_task_agnostic_model(self) -> None:
        config = Config.from_dict(self._data())

        self.assertTrue(config.uses_dream_rehearsal)
        self.assertTrue(config.uses_never_clear_dream_rehearsal)
        self.assertFalse(config.uses_bounded_dream_rehearsal)
        self.assertTrue(config.uses_task_labelled_replay)
        self.assertFalse(config.uses_task_experts)
        self.assertEqual(config.sac_dv3_data_n_max, 64)
        self.assertEqual(config.replay_buffers[0].rb_type.__name__, "FifoReplay")
        self.assertEqual(config.replay_observation_dtype, "uint8")

    def test_wrong_capacity_or_reservoir_cannot_claim_never_clear(self) -> None:
        short = self._data()
        short["sac_dv3_data_n_max"] -= 1
        with self.assertRaisesRegex(ValueError, "one slot for every"):
            Config.from_dict(short)

        reservoir = self._data()
        reservoir["replay_buffers"] = [
            {"rb_type": "LongTermReplay", "rb_device": "cpu"}
        ]
        with self.assertRaisesRegex(ValueError, "FifoReplay"):
            Config.from_dict(reservoir)


if __name__ == "__main__":
    unittest.main()
