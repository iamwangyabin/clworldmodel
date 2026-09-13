# DV3/FIFO CPU float32 replay execution profile v1

Classification: pilot. This explicitly named storage-only execution profile
extends `dv3_fifo_atari.md`; it does not redefine that frozen CUDA-replay baseline.

On 2026-09-11 the user requested immediate experiment allocation on four
two-GPU S4 instances. Supplement Atari seeds S1/S2/S3 (1337/31337/42), selected
before their results. Existing S0 and previous unsuccessful attempts remain
separate. Do not launch duplicate seeds merely to occupy remaining devices.

On 2026-09-12 the user requested additional experiments after S1/S2/S3 had
completed. Extend the allocation to the remaining original published seed
S4 (987654321), not a new or result-selected seed. The experiment registry and
cloud launch records contain no completed or active Atari DV3 S4 run. Existing
S0 stays separate; do not rerun it simply to fill an idle card. All execution
settings and budgets below are unchanged. This completes the planned seed
coverage only if S4 succeeds; the CPU/GPU replay distinction still applies.

The full FIFO observation allocation alone is 25,769,803,776 bytes (24 GiB),
larger than the 24,555 MiB logical GPU capacity before model/workspace memory.
Store FIFO on CPU without shrinking capacity or changing dtype: 1024 trajectories,
512 time steps, RGB 64x64 float32; observation/action/reward/done/first tensors
total 25,813,843,968 bytes. Python/index metadata, allocator overhead and sampled
batches are additional. No compression, mmap or sampling change is introduced.
This is analogous to the existing ARROW CPU float32 replay profile. Cross-device
bitwise trajectory identity is not claimed; report the storage difference.

Preserve all original config values except `replay_buffers[0].rb_device=cpu`
and explicit `replay_observation_dtype=float32`. In particular preserve the
six-task order, 90-epoch duration, seed, preprocessing, reward transforms,
advancing stochastic evaluation cohort and 541-epoch upstream schedule. The
epoch-539 six-task snapshot and epoch-540 wraparound training/evaluation are
distinct. Budgets remain 541,000 WM updates and 432,800 AC updates, with no
boundary consolidation or compression. Evaluate only through the unchanged
baseline evaluator; training never receives task identity or evaluation data.

Use the pinned PyTorch 2.3.0 CUDA 11.8 Atari environment, explicit Python,
8 CPU threads and one logical GPU per process. Preserve the launcher's compile,
fused Adam and TF32 settings. Save runtime/config/provenance and stage timings.
Require clean, pushed, fetched, upstream-synchronized source before smoke or
training. Target tests and a separately labelled tiny execution smoke must pass.
Analysis snapshots are inference-only, not resumable checkpoints; no automatic
resume or result-selected seed replacement is authorized.

```sh
python scripts/run_dv3_fifo_atari.py --seed 1 --cpu-threads 8 \
  --replay-device cpu --profile-stages --dry-run
```

For VirtAI SSH development instances load the platform's login environment,
then select the explicit pinned interpreter. Store logs, weights, snapshots
and manifests beneath the mounted project code directory, in a new run path.
Do not interpret the storage service's global filesystem totals as a user's
quota, overwrite old project snapshots, or use the ephemeral root disk as the
only result copy. Long-run acceptance requires adequate project quota. The
same source and output structure can be used with offline result mounts.
