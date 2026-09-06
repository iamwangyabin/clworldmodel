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
