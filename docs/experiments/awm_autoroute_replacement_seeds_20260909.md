# AWM-AutoRoute: two fresh replacement seeds (2026-09-09)

The user requested stopping current Atari S2 (31337) and S3 (42) after reviewing
partial returns, preserving S0, and starting two new seeds from scratch.
S2 stopped at 313 completed epochs and S3 at 113, with incomplete next epochs;
preserve their metrics, checkpoints and consumed-work records. They must not
be automatically resumed or omitted from the pilot history, including the
earlier user-stopped S1. This is outcome-informed replacement, not a new
unbiased predeclared cohort.

Before observing new outcomes, choose preset S4 (987654321) on 4090-2 and
explicit seed 20260909 on 3090 (campaign label S5, **not** baseline preset
index 5). The five ARROW baseline seeds remain unchanged. Both runs use the
sole maintained AWM-AutoRoute two-frame router, protocol v4, with the original
six tasks, 540 epochs, 552000 WM and 432000 AC updates and eight CPU threads.
No initialization/RNG bug fix, model, optimizer, evaluation or training-budget
change is mixed into this request.

The only launcher addition is mutually exclusive `--seed-value`, a uint32
seed used in the resolved config and manifest. Explicit values use the
verified original S0 config as a template, override only its resolved seed,
record `seed_index: null`, and use an unambiguous default output name.
Legacy `--seed 0..4` behavior is preserved. Tests verify these invariants.

Run each command from a clean, pushed and fetched upstream-synchronized commit
after target-host tests and CUDA smoke:

```sh
"$PYTHON" scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 4 \
  --cpu-threads 8 --classification pilot --python "$PYTHON" --output-dir "$NEW_S4_DIR"
"$PYTHON" scripts/run_evolving_atomic_rssm_d_autoroute.py --seed-value 20260909 \
  --cpu-threads 8 --classification pilot --python "$PYTHON" --output-dir "$NEW_S5_DIR"
```

Use independent new run directories, no resume arguments or inherited weights,
Replay, RNG or counters. Verify that the complete resolved config differs from
the earlier pilots only in seed. Save exact code/runtime/hardware provenance,
startup checks and actual acceptance records. Deployment is not established by
this document alone.

The user approved lossless archival of stopped S2 Replay to satisfy the
unchanged 48 GiB storage preflight. Verify each member and preserve two
checksum-verified compressed copies (local controller and 3090) before removing
only redundant original arrays/temporary archive from 4090-2. Preserve models,
logs and scores; record where to retrieve and restore Replay before any later
checkpoint use. Do not reduce the disk preflight or erase unrelated data.
