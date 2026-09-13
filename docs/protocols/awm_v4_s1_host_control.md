# AWM-AutoRoute v4: S1 first-task host control

User authorized on 2026-09-08: run an additional S1 (1337) on **4090-1 GPU 0**
in parallel, without stopping either original v4 run. This is a **pilot host
control**, not tuned hyperparameters or an extra independent training seed.
Protocol suffix: `-FirstTask90-HostControl-v1` on the existing v4 protocol.

## Matched prefix

The resolved config is byte-for-byte identical to the original S1's config,
including its 540-epoch, six-task schedule and all six allocated expert slots.
Keep the original Actor/RSSM initialization, CPU/CUDA RNG behavior (including
the previously audited CPU-only Actor RNG fork), all seeds, eight CPU threads,
BF16 compute/FP32 parameters, TF32, fused Adam, replay, learning rates,
exploration, sampling, resets, task order and evaluation semantics unchanged.
Do not incorporate S0 weights or inherit any checkpoint, Replay or optimizer.

The only trainer change is a **default-off termination option**, checked after
the complete first task boundary and snapshots. Stop before any task-1
interaction, initialization or updates. Existing full launches do not take the
branch. No config validation is weakened and no alternate router is added.

Run exactly 90 online epochs, 90,000 online WM + 1,000 consolidation + 1,000
fixed candidate-compression updates, and 72,000 AC updates. Nominal collection
is 5,898,240 raw frames (1,474,560 decisions); replay capacity and byte storage
are unchanged from the original S1. Keep all 80 pruning-validation rollouts,
including failed candidates and dense fallback. Save the complete post-boundary
checkpoint and immutable boundary inference snapshot before ending.

Periodic evaluations at epochs 0–80 are unchanged (all six scheduled tasks).
After the saved boundary, evaluate task 0 once on its **existing S1 periodic
validation seed 1981893629**, 16 nominal rollouts, eligible registry `[0]`, no
oracle policy ID. This replaces the parent's task-0 periodic evaluation at
epoch 90 without acquiring task 1; the parent has `[0,1]` then. Keep this
distinction explicit. Existing cross-evaluation verified identical S1 task-0
returns under those two eligibility sets for its prior boundary, but this is
not a blanket equivalence assumption. Do not perform final-heldout evaluation
or report a completed six-task score. Save actual episode count and routing.

## Execution and interpretation

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 1 \
  --cpu-threads 8 --stop-after-first-task --classification pilot \
  --python /absolute/pinned/python --output-dir /absolute/new/run
```

First validate clean pushed/fetched Git state, target GPU smoke, resolved-config
equality and dependency/backend parity. Save original S1 provenance alongside
the new run, plus current host/GPU/RAM/disk/occupancy. Run in an independent
checkout/output directory. Resource contention is **observed**, not controlled;
one same-seed cross-host repeat cannot identify CPU model versus contention
versus nondeterministic kernels. Never claim all seeds favor one machine.

Compare identical periodic checkpoints, full learning curves, boundary
compression decisions and the fixed boundary validation. Preserve negative
results. No seed-specific tuning or favorable checkpoint selection is allowed.
The prefix's `run_status.json` reports its own completion separately from
`full_curriculum_complete=false`; it must not be auto-resumed to 540 epochs.
