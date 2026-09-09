# SPDX-License-Identifier: Apache-2.0
"""Explicit seed launch metadata/config parity; no environment interaction."""
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import run_arrow_ar50_atari as baseline
import run_evolving_atomic_rssm as launcher
import run_evolving_atomic_rssm_d_autoroute as entry


class ExplicitSeedTests(unittest.TestCase):
    def dry_run(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(entry.main(["--dry-run", *args]), 0)
        manifest, _ = json.JSONDecoder().raw_decode(output.getvalue())
        return manifest

    def test_explicit_seed_changes_only_config_seed_and_records_no_preset_index(self):
        configs = []
        original = launcher._budget_manifest

        def capture(config):
            configs.append(config.copy())
            return original(config)

        with mock.patch.object(launcher, "_budget_manifest", side_effect=capture):
            preset = self.dry_run("--seed", "0")
            explicit = self.dry_run("--seed-value", "20260909")
        self.assertEqual(preset["seed_index"], 0)
        self.assertIsNone(explicit["seed_index"])
        self.assertEqual(explicit["seed"], 20260909)
        differences = {k for k in configs[0].keys() | configs[1].keys()
                       if configs[0].get(k) != configs[1].get(k)}
        self.assertEqual(differences, {"seed"})
        self.assertEqual(configs[1]["seed"], explicit["seed"])
        self.assertEqual(preset["protocol"], explicit["protocol"])
        self.assertIn("seed20260909", explicit["output_dir"])
        self.assertEqual(baseline.SEEDS, [123456789, 1337, 31337, 42, 987654321])
        self.assertEqual(self.dry_run("--seed", "4")["seed"], 987654321)
        self.assertEqual(self.dry_run("--seed-value", "0")["seed"], 0)

    def test_invalid_or_conflicting_seeds_are_rejected_by_both_entrypoints(self):
        for module in (entry, launcher):
            for args in (["--seed-value", "-1"], ["--seed-value", str(2**32)],
                         ["--seed-value", "bad"], ["--seed", "5"],
                         ["--seed", "0", "--seed-value", "42"]):
                with self.subTest(module=module.__name__, args=args):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        module._parser().parse_args(args)


if __name__ == "__main__":
    unittest.main()
