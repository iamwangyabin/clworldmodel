#!/usr/bin/env python3
"""Launch KARROW spatial-patch v2 while retaining the v1 launcher default."""

from run_karrow_ar50_atari import main


if __name__ == "__main__":
    raise SystemExit(main(default_visual_version="v2"))
