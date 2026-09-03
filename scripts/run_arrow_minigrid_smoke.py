#!/usr/bin/env python3
"""Launch the three-task ARROW-50 MiniGrid smoke on one CUDA device."""

from __future__ import annotations

from minigrid_smoke_support import (
    ROOT,
    launch_smoke,
    load_common_config,
    parse_args,
    replay_accounting,
)


SOURCE_CONFIG = ROOT / "configs" / "minigrid" / "arrow_50_smoke.json"


def _load_resolved_config(seed: int) -> dict:
    config = load_common_config(SOURCE_CONFIG, seed)
    expected = {
        "algorithm": "arrow",
        "arrow_replay_capacity_ratio": "50-50",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Invalid ARROW MiniGrid smoke config: {mismatches}")
    replay_types = [item["rb_type"] for item in config["replay_buffers"]]
    if replay_types != ["FifoReplay", "LongTermReplay"]:
        raise RuntimeError(f"MiniGrid smoke is not ARROW FIFO/LTDM: {replay_types}")
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Run the one-GPU ARROW-50 three-task MiniGrid smoke protocol", argv
    )
    config = _load_resolved_config(args.seed)
    total_slots = 2 * int(config["data_n_max"])
    fifo_slots = total_slots // 2
    replay = replay_accounting(
        config,
        total_slots=total_slots,
        fifo_slots=fifo_slots,
        reservoir_slots=total_slots - fifo_slots,
        selection_probability={"fifo": 0.5, "reservoir": 0.5},
    )
    return launch_smoke(
        args=args,
        config=config,
        source_config=SOURCE_CONFIG,
        method="ARROW-50",
        protocol="ARROW-50-MiniGrid-3Task-Smoke-v1",
        default_output_stem="arrow_minigrid_3task_smoke",
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
