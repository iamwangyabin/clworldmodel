#!/usr/bin/env python3
"""Launch full-patch DINOv3 DreamerV3 with task-isolated ARROW banks."""

from run_moe_arrow_atari import DINO_PATCHBANK_VARIANT, main


if __name__ == "__main__":
    raise SystemExit(main(DINO_PATCHBANK_VARIANT))
