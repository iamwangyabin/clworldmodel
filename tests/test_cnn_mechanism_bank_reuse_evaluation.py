"""Focused checks for the MB-RSSM learned-reuse diagnostic."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal host environment.
    torch = None

from evaluate_cnn_mechanism_bank_reuse import (  # noqa: E402
    CONDITIONS,
    _categorical_js,
    _categorical_kl,
    _disabled_banks,
    _gate_condition,
    _parse_args,
    _route_values,
)

if torch is not None:
    from clworldmodel.models.mechanism_bank import MechanismBank


class MechanismBankReuseEvaluationTests(unittest.TestCase):
    def test_every_condition_disables_only_the_named_routes(self) -> None:
        self.assertEqual(_disabled_banks("full_reuse"), frozenset())
        self.assertEqual(
            _disabled_banks("no_reuse"),
            frozenset(("recurrent", "posterior", "prior")),
        )
        self.assertEqual(
            _disabled_banks("no_recurrent_reuse"), frozenset(("recurrent",))
        )
        self.assertEqual(
            _disabled_banks("no_posterior_reuse"), frozenset(("posterior",))
        )
        self.assertEqual(
            _disabled_banks("no_prior_reuse"), frozenset(("prior",))
        )
        with self.assertRaisesRegex(ValueError, "Unknown reuse condition"):
            _disabled_banks("unknown")

    def test_default_pilot_fixes_all_five_conditions_and_sixteen_rollouts(
        self,
    ) -> None:
        argv = [
            "evaluate_cnn_mechanism_bank_reuse.py",
            "--epoch270-checkpoint",
            "epoch270.pt",
            "--epoch260-checkpoint",
            "epoch260.pt",
            "--output-dir",
            "out",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = _parse_args()

        self.assertEqual(tuple(args.conditions), CONDITIONS)
        self.assertEqual(args.n_rollouts, 16)
        self.assertEqual(args.classification, "pilot")
        self.assertEqual(args.imagination_horizon, 15)

    def test_smoke_can_narrow_conditions_and_budget(self) -> None:
        argv = [
            "evaluate_cnn_mechanism_bank_reuse.py",
            "--epoch270-checkpoint",
            "epoch270.pt",
            "--epoch260-checkpoint",
            "epoch260.pt",
            "--output-dir",
            "out",
            "--classification",
            "smoke",
            "--n-rollouts",
            "1",
            "--diagnostic-decisions",
            "64",
            "--conditions",
            "full_reuse",
            "no_reuse",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = _parse_args()

        self.assertEqual(args.conditions, ["full_reuse", "no_reuse"])
        self.assertEqual(args.n_rollouts, 1)
        self.assertEqual(args.diagnostic_decisions, 64)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class MechanismBankReuseTensorTests(unittest.TestCase):
    @staticmethod
    def _world_model():
        class Rssm:
            recurrent_mechanism_bank = MechanismBank(
                num_tasks=3, in_features=4, out_features=4, hidden_features=3
            )
            representation_mechanism_bank = MechanismBank(
                num_tasks=3, in_features=4, out_features=4, hidden_features=3
            )
            transition_mechanism_bank = MechanismBank(
                num_tasks=3, in_features=4, out_features=4, hidden_features=3
            )

        class WorldModel:
            rssm = Rssm()

        model = WorldModel()
        with torch.no_grad():
            for index, bank in enumerate(
                (
                    model.rssm.recurrent_mechanism_bank,
                    model.rssm.representation_mechanism_bank,
                    model.rssm.transition_mechanism_bank,
                ),
                start=1,
            ):
                bank.routes[1].logits.fill_(0.1 * index)
        return model

    def test_gate_condition_is_exact_and_restores_every_route(self) -> None:
        model = self._world_model()
        original = _route_values(model, 2)

        with _gate_condition(torch, model, 2, "no_posterior_reuse"):
            changed = _route_values(model, 2)
            self.assertEqual(changed["posterior"], [0.0])
            self.assertEqual(changed["recurrent"], original["recurrent"])
            self.assertEqual(changed["prior"], original["prior"])

        self.assertEqual(_route_values(model, 2), original)

    def test_categorical_metrics_match_hand_computation(self) -> None:
        p = torch.tensor([[[math.log(0.5), math.log(0.5)]]])
        q = torch.tensor([[[math.log(0.75), math.log(0.25)]]])
        kl = _categorical_kl(torch, p, q)
        expected_kl = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
        self.assertAlmostEqual(float(kl), expected_kl, places=6)

        midpoint = (p.exp() + q.exp()) / 2
        expected_js = 0.5 * (
            (p.exp() * (p - midpoint.log())).sum()
            + (q.exp() * (q - midpoint.log())).sum()
        )
        self.assertAlmostEqual(
            float(_categorical_js(torch, p, q)), float(expected_js), places=6
        )


if __name__ == "__main__":
    unittest.main()
