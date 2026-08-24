# Decision 0017: Add fixed-global-batch DDP to DINO-ConvBank

Scope note: Decision 0018 later validates the same fixed-global-batch execution
contract for the separately named CNN-FullBank method.

## Status

Accepted for 2- and 4-GPU DINO-ConvBank execution. This is an execution profile,
not a larger-batch method or a new training-budget ablation.

## Context

DINO-ConvBank repeatedly encodes replay observations during both world-model
and Actor-Critic training. Those stages dominate the intended multi-GPU work,
while collection uses only two synchronous environments and is too small to
justify changing the environment protocol. PyTorch does not automatically turn
the existing custom trainer into correct data parallelism: replay selection,
task routing, custom Actor-Critic losses, logging, evaluation, and filesystem
ownership all need explicit distributed semantics.

Increasing the global batch would change the optimization protocol and make a
speed comparison ambiguous. Independent replay sampling on each rank would also
change ARROW's one-sub-buffer-per-minibatch choice. The distributed path must
therefore preserve the original global samples and update counts.

## Decision

Support `data_parallel_world_size` values 1, 2, and 4, with multi-GPU execution
validated only for DINO-ConvBank. The launcher exposes `--devices 1|2|4` and
uses `torch.distributed.run`, one CUDA process per device, NCCL, and native
PyTorch `DistributedDataParallel` for both world-model and Actor-Critic
gradients.

Global batches remain fixed. The regular world-model batch stays `T=32, N=16`
and the Actor-Critic context stays `T=4, N=128`. Two GPUs receive local sequence
counts 8 and 64; four GPUs receive 4 and 32. Equal local loss means followed by
DDP gradient averaging reproduce the global mean objective. Return-normalization
statistics are gathered across ranks before their quantiles and EMA are
computed.

Rank 0 alone owns collection, FIFO/LTDM replay state, mmap files, logging, and
checkpoints. For every optimization step, rank 0 performs one unchanged global
ARROW buffer selection and global index draw, transfers the uint8-backed sample
to its device, and scatters contiguous slices of the sequence axis to the other
ranks. Each rank recomputes frozen DINO features for its local frames. Evaluation
tasks are divided by task index modulo world size and reduced back to rank 0.

## Consequences

- Environment interactions, global batch sizes, replay capacity and selection,
  optimizer updates, loss weights, and evaluation rollouts do not increase.
- Every GPU holds a complete model, Actor-Critic, optimizer state, and frozen
  DINO replica. DDP accelerates data-parallel compute; it does not pool device
  memory for a model that cannot fit on one GPU.
- Collection remains rank-0-only and replay scatter adds communication, so 2x
  or 4x wall-clock speedup is not promised. Stage timings on the target system
  must establish realized scaling.
- CPU contracts cover batch partitioning, configuration isolation, launcher
  topology, and unchanged budgets. A clean, pushed commit still requires 2-GPU
  and 4-GPU CUDA smoke runs before an official campaign.
