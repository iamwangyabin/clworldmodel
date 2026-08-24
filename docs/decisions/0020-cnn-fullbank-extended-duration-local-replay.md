# Decision 0020: Extended task duration with node-local mmap replay

## Status

Accepted for a named task-1 acquisition ablation. It does not change the
90-epoch CNN-FullBank protocol.

## Context

The sample-matched `x4-linear-lr` profile reduced ordinary epoch time from
about 163 seconds to about 58 seconds. As replay filled, rank 0 was observed in
`fuse_wait_on_page_writeback` while the other DDP ranks waited. The run-local
uint8 mmap replay was backed by the `/gemini/code` network FUSE mount. The
Replay observations alone reserve about 6.44 GB, so network writeback can erase
the compute improvement and make long runs unstable.

The faster profile also creates room for an explicitly larger acquisition
budget. Extending task duration is not a transparent optimization: it adds
environment interactions and optimization samples.

## Decision

- Add a task-1-only `--task-duration-multiplier` for named CNN-FullBank batch
  ablations. The first extended run uses multiplier 2, or 180 MsPacman epochs.
- Record the source and resolved task duration, environment/sample multipliers,
  and update-count relation to the 90-epoch fixed-batch baseline.
- Add `--replay-mmap-root`. The launcher creates a unique node-local backing
  directory and an absolute `mmap_replay` symlink inside the persistent run.
- Refuse an existing scratch directory rather than reuse or overwrite it.
- Keep logs, manifests, evaluation results, and model weights in the persistent
  output directory. The replay backing store remains non-checkpointed and is
  never described as resumable state.

## Consequences

The 180-epoch result cannot pass or replace the 90-epoch protocol gate. It is a
larger-budget acquisition diagnostic and must retain its qualified protocol
name. Node-local replay improves runtime isolation but is ephemeral; a machine
loss still makes the run non-resumable, exactly as before.
