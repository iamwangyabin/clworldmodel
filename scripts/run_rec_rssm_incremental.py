#!/usr/bin/env python3
"""Launch Task-2/3 REC-RSSM from a frozen Task-1 boundary snapshot."""

from run_cnn_mechanism_bank_incremental import main


if __name__ == "__main__":
    raise SystemExit(main(default_method_profile="rec-rssm"))
