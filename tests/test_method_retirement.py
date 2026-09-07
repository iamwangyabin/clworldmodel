"""Retired choices fail closed; retained MLP behavior matches the prior source."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"))


class MethodRetirementTests(unittest.TestCase):
    def test_launchers_reject_retired_choices(self):
        from run_arrow_ar50_atari import _parser
        from run_evolving_atomic_rssm import _parser as evolving_parser

        for parser, args in (
            (_parser(), ["--actor-network", "fast_kan_ac_stable"]),
            (_parser(), ["--task-duration-epochs", "90"]),
            (evolving_parser(), ["--behavior-profile", "shared_fastkan_autoroute"]),
        ):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_fastkan_implementation_is_not_exported_or_importable(self):
        import clworldmodel.models as models

        self.assertEqual(models.__all__, ["SpatialFeatureProjector"])
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("clworldmodel.models.fast_kan")

    def test_config_and_actor_reject_retired_paths(self):
        from ac import ActorCritic
        from config import Config
        from run_arrow_ar50_atari import _config_path
        from run_evolving_atomic_rssm import _resolved_config

        baseline = json.loads(_config_path("original", 0).read_text())
        d = _resolved_config(baseline, task_order="arrow-original-six")
        for data in (
            {**baseline, "actor_network": "fast_kan_ac_stable"},
            {**d, "continual_method": "evolving_atomic_rssm_adaptive_compression_shared_heads_fastkan_autoroute_arrow"},
        ):
            with self.assertRaises(ValueError):
                Config.from_dict(data)
        with self.assertRaises(TypeError):
            Config.from_dict({**baseline, "fastkan_hidden_features": 53})
        with self.assertRaises(ValueError):
            ActorCritic(14, 4, actor_network="fast_kan_ac_stable")

    def test_mlp_initialization_and_outputs_match_pre_retirement(self):
        import torch
        from ac import ActorCritic

        expected = json.loads((ROOT / "tests/fixtures/retained_mlp_behavior_parity.json").read_text())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(expected["provenance"]["seed"])
            model = ActorCritic(14, 4)
            logs, values = model(torch.linspace(-1, 1, 28).reshape(2, 14))
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().contiguous().numpy().tobytes())
        self.assertEqual(digest.hexdigest(), expected["initial_state_sha256"])
        torch.testing.assert_close(logs, torch.tensor(expected["log_probs"]), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(values, torch.tensor(expected["values"]), rtol=1e-6, atol=1e-7)
