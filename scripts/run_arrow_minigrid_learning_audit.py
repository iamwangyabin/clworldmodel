#!/usr/bin/env python3
"""Isolate the autoreset mismatch on ARROW's first MiniGrid task.

This is a named diagnostic, never a replacement for the recorded formal v1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from minigrid_campaign_support import ROOT, SEEDS, TASKS, launch_formal, replay_accounting
from run_arrow_minigrid_formal import SOURCE_CONFIG, _load_resolved_config


def audit_config(seed_index: int, nominal_rows: int, variant: str) -> dict:
    if nominal_rows < 10_000 or nominal_rows % 1000:
        raise ValueError("nominal_rows must be a multiple of 1000 and at least 10000")
    if variant not in {"legacy_next_step", "same_step"}:
        raise ValueError("Unknown autoreset diagnostic variant")
    config = _load_resolved_config(seed_index)
    epochs = (nominal_rows - 10_000) // 1000 + 1
    config["esc"]["env_configs"] = config["esc"]["env_configs"][:1]
    config["esc"]["kwargs"] = {"task_durations": [epochs]}
    config.update(
        epochs=epochs,
        collection_autoreset_mode=variant,
        learning_diagnostics=True,
    )
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-index", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument("--variant", choices=("legacy_next_step", "same_step"), default="same_step")
    parser.add_argument("--nominal-rows", type=int, default=750_000)
    parser.add_argument("--evidence-level", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    args = parser.parse_args(argv)
    config = audit_config(args.seed_index, args.nominal_rows, args.variant)
    slots = 2 * config["data_n_max"]
    replay = replay_accounting(
        config, total_slots=slots, fifo_slots=slots // 2,
        reservoir_slots=slots // 2, selection_probability={"fifo": 0.5, "reservoir": 0.5},
    )
    return launch_formal(
        args=args, config=config, source_config=SOURCE_CONFIG, method="ARROW-50",
        output_stem=f"arrow_50_doorkey_autoreset_audit_v1_{args.variant}_{args.nominal_rows}",
        replay=replay,
        method_semantics={
            "world_model": "unchanged ARROW-vendored DreamerV3-style model",
            "retention": "unchanged ARROW-50 FIFO/LTDM",
            "behavioral_change_from_v1": "explicit vector autoreset mode only",
            "autoreset_mode": args.variant,
            "added_diagnostics": "raw collection rewards/actions, sampled positive rewards, reward-model accuracy, imagined rewards",
            "known_remaining_deviations": [
                "legacy terminal-observation omission and previous-row reward assignment",
                "timeouts are still continuation terminals",
                "four-frame imagination context and historical actor-critic targets",
                "deterministic argmax and latent-mode evaluation",
            ],
        },
        command_options=("--arrow-replay-ratio", "50-50"),
        protocol="ARROW-50-MiniGrid-DoorKey-AutoresetAudit-v1",
        evidence_level=args.evidence_level,
        claim_scope="first-task learning diagnostic; not a paper reproduction or method ranking",
        task_names=TASKS[:1], task_epochs=(config["epochs"],),
        task_interactions=(args.nominal_rows,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
