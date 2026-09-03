#!/usr/bin/env python3
"""Run the DreamerV3 port of Continual-Dreamer's reservoir mechanism."""

from __future__ import annotations

from minigrid_smoke_support import (
    CONTINUAL_DREAMER_SOURCE_COMMIT,
    ROOT,
    assert_configs_match_outside_replay,
    launch_smoke,
    load_common_config,
    parse_args,
    replay_accounting,
)


SOURCE_CONFIG = ROOT / "configs" / "minigrid" / "dv3_rs_smoke.json"
ARROW_CONFIG = ROOT / "configs" / "minigrid" / "arrow_50_smoke.json"


def _load_resolved_config(seed: int) -> dict:
    config = load_common_config(SOURCE_CONFIG, seed)
    reference = load_common_config(ARROW_CONFIG, seed)
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
    if config["replay_buffers"][0].get("rb_device") != "cuda":
        errors.append("DV3-RS smoke requires CUDA replay storage")
    if errors:
        raise RuntimeError("Invalid DV3-RS MiniGrid smoke config: " + "; ".join(errors))
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(
        "Run the one-GPU DV3-RS three-task MiniGrid mechanism-reproduction smoke",
        argv,
    )
    config = _load_resolved_config(args.seed)
    total_slots = int(config["sac_dv3_data_n_max"])
    replay = replay_accounting(
        config,
        total_slots=total_slots,
        fifo_slots=0,
        reservoir_slots=total_slots,
        selection_probability={"fifo": 0.0, "reservoir": 1.0},
    )
    return launch_smoke(
        args=args,
        config=config,
        source_config=SOURCE_CONFIG,
        method="DV3-RS",
        protocol="DV3-RS-MiniGrid-3Task-Smoke-v1",
        default_output_stem="dv3_rs_minigrid_3task_smoke",
        replay=replay,
        method_semantics={
            "role": "DreamerV3 mechanism port of Continual-Dreamer",
            "world_model": "vendored DreamerV3",
            "continual_dreamer_source": {
                "repository": "https://github.com/skezle/continual-dreamer",
                "commit": CONTINUAL_DREAMER_SOURCE_COMMIT,
            },
            "retention": "iid-random-key top-k uniform reservoir",
            "sampling": "uniform over retained fixed-length trajectories",
            "plan2explore": False,
            "not_a_paper_reproduction": True,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
