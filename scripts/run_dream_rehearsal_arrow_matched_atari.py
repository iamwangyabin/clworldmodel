#!/usr/bin/env python3
"""Run the full-history/bounded Dream Rehearsal pair on ARROW budgets."""

from run_bounded_dream_rehearsal_atari import main


if __name__ == "__main__":
    raise SystemExit(main(arrow_matched=True))
