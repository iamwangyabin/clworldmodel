#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CoinRun D-AutoRoute: six tasks plus the published final task-0 revisit.

Composes the same launcher/trainer/model as Atari; no benchmark-specific fork.
"""
from run_evolving_atomic_rssm_d_autoroute import main as _main


def main(argv=None):
    return _main(argv, benchmark="procgen_coinrun")


if __name__ == "__main__":
    raise SystemExit(main())
