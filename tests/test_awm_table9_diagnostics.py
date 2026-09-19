from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_awm_table9_diagnostics.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_awm_table9_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reuse_disabled_restores_all_bank_flags() -> None:
    banks = [SimpleNamespace(reuse_enabled=value) for value in (True, False, True)]
    model = SimpleNamespace(
        rssm=SimpleNamespace(
            recurrent_mechanism_bank=banks[0],
            representation_mechanism_bank=banks[1],
            transition_mechanism_bank=banks[2],
        )
    )

    with MODULE.reuse_disabled(model):
        assert [bank.reuse_enabled for bank in banks] == [False, False, False]

    assert [bank.reuse_enabled for bank in banks] == [True, False, True]


def test_paired_differences_preserve_task_pairing() -> None:
    left = [{"task_index": 0, "task_name": "A", "raw_return_mean": 9.0}]
    right = [{"task_index": 0, "task_name": "A", "raw_return_mean": 4.0}]

    assert MODULE.paired_differences(left, right, name="gap") == [
        {"task_index": 0, "task_name": "A", "gap": 5.0}
    ]
