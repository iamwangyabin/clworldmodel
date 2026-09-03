"""Typed-config contracts for the vendored Dream Rehearsal integration."""

from __future__ import annotations

import copy
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
        "bounded_dream_rehearsal_launcher",
        SCRIPTS / "run_bounded_dream_rehearsal_atari.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(launcher)


@unittest.skipIf(Config is None, "requires the pinned Atari experiment environment")
class BoundedDreamRehearsalConfigTests(unittest.TestCase):
    @staticmethod
    def _data() -> dict:
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        return launcher._resolved_config(source, replay_slots=8)

    def test_typed_config_accepts_only_bounded_ltdm_task_metadata_path(self) -> None:
        config = Config.from_dict(self._data())

        self.assertTrue(config.uses_bounded_dream_rehearsal)
        self.assertTrue(config.uses_task_labelled_replay)
        self.assertFalse(config.uses_task_experts)
        self.assertEqual(config.algorithm, "dv3")
        self.assertEqual(config.sac_dv3_data_n_max, 8)
        self.assertEqual(config.replay_observation_dtype, "uint8")
        self.assertEqual(config.replay_buffers[0].rb_type.__name__, "LongTermReplay")
        self.assertEqual(config.replay_buffers[0].rb_device, "cpu")

    def test_fifo_and_task_aware_world_model_are_rejected(self) -> None:
        fifo = self._data()
        fifo["replay_buffers"] = [{"rb_type": "FifoReplay", "rb_device": "cpu"}]
        with self.assertRaisesRegex(ValueError, "LongTermReplay"):
            Config.from_dict(fifo)

        experts = self._data()
        experts["rssm_num_experts"] = 6
        with self.assertRaisesRegex(ValueError, "task-aware continual_method"):
            Config.from_dict(experts)

    def test_non_method_cannot_smuggle_a_rehearsal_schedule(self) -> None:
        data = copy.deepcopy(self._data())
        data["continual_method"] = "none"
        data["replay_observation_dtype"] = "float32"
        data["dream_rehearsal_horizon"] = 7

        with self.assertRaisesRegex(ValueError, "Dream-rehearsal settings require"):
            Config.from_dict(data)


if __name__ == "__main__":
    unittest.main()
