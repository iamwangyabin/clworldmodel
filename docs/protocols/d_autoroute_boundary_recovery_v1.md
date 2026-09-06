# D-AutoRoute mixed-precision boundary recovery v1

This is a runtime repair/continuation policy for the existing Atari 540-epoch
and CoinRun 541-epoch pilots, not a new seed, learning rule, or shortened protocol.
The failed source attempts and their original Git revisions remain immutable.

## Failure and repair invariant

At the first multi-task boundary, old and new RSSM routes can emit BF16 and FP32
states under autocast. Indexed assignment into a buffer allocated from the first
route raises a dtype error. Both latent and recurrent tensors now concatenate
all groups (normal PyTorch type promotion) before restoring worker order. Do not
cast an FP32 route into whichever dtype the first route happened to emit.
Homogeneous route batches preserve their original state dtype. Actor logits
already promote to FP32. Scores, eligible routes, locks, private MLP actors,
training labels, replay, optimization and evaluation budgets are unchanged.

The fixed/mock regression must fail on the old scatter for both route orders.
The CUDA recovery smoke must observe genuinely heterogeneous native route
outputs, check deterministic assembled values, then follow stochastic recurrent
states in train and eval modes. A smoke that happens to select only route 0
cannot validate this bug. These checks do not establish routing accuracy.

## Continuation contract

The independent Atari and CoinRun entry points accept `--resume-from` pointing
to a **full post-consolidation training checkpoint**. Only failed attempts with
identical resolved protocol/config/seed are accepted. Legacy missing default-
Atari metadata is normalized; no budget or hyperparameter differences are
permitted. Inference snapshots and pre-consolidation checkpoints are rejected:
the latter still owe boundary consolidation/compression and cannot skip it.

The trainer validates checksum, schema, config, acquisition boundary/counters,
private actor eligibility and retired private-WM optimizer ownership. It restores
compact WM/teacher state, shared Adam, full private AC/targets/optimizers/EMAs,
immutable replay assets into independent working mmaps, RNG and schedule. CPU
checkpoint loading keeps RNG byte tensors on CPU and avoids staging replay on
CUDA. The next task uses the ordinary initialization and update path. Environments
start with fresh resets at the durable boundary, as recorded by schema v1; this
is not bitwise mid-episode recovery. Pending epochs, not the full budget, execute.

A new output directory records both source and recovery commits. It contains a
hashed inherited log prefix and pre-boundary raw records; the reporting log
explicitly joins that prefix with the new attempt. The complete failed parent
log and manifests remain separately available. Old inference weights/indexes
retain their source commit and are never relabelled as recovery-commit weights.
New snapshots have their own index. Prefix clipping excludes failed/rolled-back
post-boundary metrics rather than selecting results by return. Every prefix
artifact records its source path/hash.

## Failed work and automatic recovery

All ten campaign attempts have a 90-completed-epoch durable boundary. CoinRun
seed index 3 completed online epoch 90 before failing in collection at epoch 91;
its 1,000 WM and 800 AC updates after the boundary are discarded work. The other
nine completed no further online epoch, but their task-1 collection and partial
evaluation still consumed compute. Incomplete collection/evaluation counters
may be unavailable; never report those costs as zero. Preserve failed suffixes,
wall time, smoke costs and rollback overhead separately from logical budgets.
Atari still ends at 540; CoinRun still executes the final task-0 revisit at 541.

Monitor recovery is bounded: restart only confirmed stopped attempts, using a
validated clean/pushed revision and checksummed compatible boundary. Prevent
duplicate seeds/processes; keep the assigned GPUs and two-run/GPU cap. Permit at
most two transient retries per incident, with backoff and a new lineage record.
A deterministic code error is repaired and tested, not retried indefinitely.
Unknown protocol/config/checkpoint incompatibility or resource conflicts require
attention. Notify on repair/restart, failure, completion or a needed decision,
not every unchanged poll. Initial startup without a durable checkpoint and
pre-consolidation-only recovery are outside this automatic continuation path.

## Validation commands

```bash
PYTHONPATH=src:tests:scripts python -m unittest test_d_autoroute test_d_autoroute_coinrun test_d_autoroute_resume test_fastkan_autoroute test_evolving_atomic_rssm test_adaptive_qfp_compression test_uint8_replay test_mixed_precision test_script_support
# After clean pushed provenance and upstream verification, on the assigned GPU:
python scripts/smoke_d_autoroute_resume.py --checkpoint /path/to/failed/evolving_core_checkpoints/task_00_post_consolidation.pt --output-dir /path/to/new-smoke
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 0 --cpu-threads 8 --resume-from /path/to/failed/evolving_core_checkpoints/task_00_post_consolidation.pt --output-dir /path/to/new-atari-attempt
# CoinRun uses the same options on run_evolving_atomic_rssm_d_autoroute_coinrun.py.
```
