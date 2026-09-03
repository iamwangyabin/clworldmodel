#!/usr/bin/env python3
"""Launch one formal ARROW-50 seed on the Continual-Dreamer MiniGrid order."""

from __future__ import annotations

from minigrid_campaign_support import (
    ROOT,
    _load_config,
    launch_formal,
    parse_args,
    replay_accounting,
)


SOURCE_CONFIG = ROOT / "configs" / "minigrid" / "arrow_50_formal_v1.json"


def _load_resolved_config(seed_index: int) -> dict:
    config = _load_config(SOURCE_CONFIG, seed_index)
    expected = {"algorithm": "arrow", "arrow_replay_capacity_ratio": "50-50"}
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    replay_types = [item["rb_type"] for item in config["replay_buffers"]]
    if replay_types != ["FifoReplay", "LongTermReplay"]:
        mismatches["replay_buffers"] = (
            replay_types,
            ["FifoReplay", "LongTermReplay"],
        )
    if mismatches:
        raise RuntimeError(f"Invalid formal ARROW-50 MiniGrid config: {mismatches}")
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Run one formal ARROW-50 MiniGrid comparison seed", argv)
    config = _load_resolved_config(args.seed_index)
    total_slots = 2 * int(config["data_n_max"])
    fifo_slots = total_slots // 2
    replay = replay_accounting(
        config,
        total_slots=total_slots,
        fifo_slots=fifo_slots,
        reservoir_slots=total_slots - fifo_slots,
        selection_probability={"fifo": 0.5, "reservoir": 0.5},
    )
    return launch_formal(
        args=args,
        config=config,
        source_config=SOURCE_CONFIG,
        method="ARROW-50",
        output_stem="arrow_50_minigrid_3task_formal_v1",
        replay=replay,
        method_semantics={
            "world_model": "vendored DreamerV3",
            "retention": "equal-capacity FIFO plus random-key top-k reservoir",
            "sampling": "whole minibatch from FIFO or reservoir with equal probability",
        },
        command_options=("--arrow-replay-ratio", "50-50"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
