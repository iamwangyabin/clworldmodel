#!/usr/bin/env python3
"""Launch one formal DreamerV3 reservoir seed on the MiniGrid order."""

from __future__ import annotations

from minigrid_campaign_support import (
    CONTINUAL_DREAMER_SOURCE_COMMIT,
    ROOT,
    _load_config,
    assert_configs_match_outside_replay,
    launch_formal,
    parse_args,
    replay_accounting,
)
from run_arrow_minigrid_formal import _load_resolved_config as load_arrow_config


SOURCE_CONFIG = ROOT / "configs" / "minigrid" / "dv3_rs_formal_v1.json"


def _load_resolved_config(seed_index: int) -> dict:
    config = _load_config(SOURCE_CONFIG, seed_index)
    reference = load_arrow_config(seed_index)
    assert_configs_match_outside_replay(config, reference)
    errors = []
    if config.get("algorithm") != "dv3":
        errors.append("algorithm must be dv3")
    replay_types = [item["rb_type"] for item in config["replay_buffers"]]
    if replay_types != ["LongTermReplay"]:
        errors.append("DV3-RS requires exactly one LongTermReplay")
    expected_slots = 2 * int(reference["data_n_max"])
    if config.get("sac_dv3_data_n_max") != expected_slots:
        errors.append(
            f"DV3-RS requires {expected_slots} slots to match ARROW total capacity"
        )
    if errors:
        raise RuntimeError("Invalid formal DV3-RS MiniGrid config: " + "; ".join(errors))
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Run one formal DV3-RS MiniGrid comparison seed", argv)
    config = _load_resolved_config(args.seed_index)
    total_slots = int(config["sac_dv3_data_n_max"])
    replay = replay_accounting(
        config,
        total_slots=total_slots,
        fifo_slots=0,
        reservoir_slots=total_slots,
        selection_probability={"fifo": 0.0, "reservoir": 1.0},
    )
    return launch_formal(
        args=args,
        config=config,
        source_config=SOURCE_CONFIG,
        method="DV3-RS",
        output_stem="dv3_rs_minigrid_3task_formal_v1",
        replay=replay,
        method_semantics={
            "role": "DreamerV3 mechanism port of Continual-Dreamer",
            "world_model": "vendored DreamerV3",
            "continual_dreamer_source_commit": CONTINUAL_DREAMER_SOURCE_COMMIT,
            "retention": "iid-random-key top-k uniform reservoir",
            "sampling": "uniform over retained fixed-length trajectories",
            "plan2explore": False,
            "not_a_paper_reproduction": True,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
