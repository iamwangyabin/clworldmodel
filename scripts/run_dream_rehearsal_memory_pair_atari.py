#!/usr/bin/env python3
"""Same Dream Rehearsal; choose only full history or ARROW-sized random history."""

from run_dream_rehearsal_official_atari import main
from clworldmodel.reference.dream_rehearsal import MemoryPairConfig


if __name__ == "__main__":
    raise SystemExit(main(config_type=MemoryPairConfig))
