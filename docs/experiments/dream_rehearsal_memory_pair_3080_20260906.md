# Dream Rehearsal memory-only pair: 3080 launch plan (2026-09-06)

The user authorized the new memory-only comparison on **3080 only**. Other
servers, old results and existing ARROW environments are not modified.

Use `Dream-Rehearsal-MemoryPair-v1-Atari`, actual seed **123456789**, one
visible GPU and the same eight CPU threads for both arms. Run sequentially:

1. Full-history target-GPU smoke.
2. Bounded-history target-GPU smoke.
3. Full-history six-task pilot, only if both smoke commands succeed.
4. Bounded-history six-task pilot, only if the full pilot succeeds.

Any failed command stops the queue. Do not resume from old analysis snapshots,
overwrite a result directory, reduce the batch to fit, or move to another host.
Only history capacity differs between the two arms. This is a single-seed pilot,
not a validated paper reproduction or guaranteed Atari score.

The launch uses a separate branch/check-out containing only the Dream Rehearsal
integration over its recorded repository parent. The unrelated in-progress
retained-method retirement is not included or reverted. Each actual launcher
fetches GitHub, requires a clean upstream-synced commit and records its full
commit, fixed source hashes, resolved protocol, environment and budgets.

An isolated Python environment uses `requirements/dream_rehearsal_official.txt`;
its complete package freeze is saved in every run's runtime record. The original
ARROW environment is not upgraded. Source hash checks, CPU reference/storage
fixtures and the one-capacity-only dry-run comparison pass before deployment.
GPU smoke and pilot outcomes must be read from the timestamped run artifacts;
this plan is not evidence that a run has completed.

See `../protocols/dream_rehearsal_memory_pair_v1_atari.md` and decision 0056.

## First launch attempt: rejected before environment interaction

The 2026-09-06 13:19:57 +08:00 full-history smoke attempt at commit
`a617ebc7e93370f04f7d7b47529622a0dee16848` failed the pinned-runtime guard:
the launcher resolved the venv Python symlink to the base ARROW interpreter,
which imported torch 2.3.0+cu118 instead of the installed isolated 2.4.1+cu121.
The guard ran before constructing environments, collecting transitions or
updating parameters. The queue stopped; no pilot started. Preserve its launch,
status and error log rather than overwriting the failed attempt.

The fix makes the selected Python path absolute without dereferencing symlinks.
A regression test covers the common original-code launcher and both history
arms. No learning source, budget or protocol setting changes. The retry must
use the newly pushed fix commit and a fresh output directory.
