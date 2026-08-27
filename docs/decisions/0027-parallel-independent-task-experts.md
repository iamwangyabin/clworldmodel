# 0027: Parallel Independent Task Experts

## Status

Accepted for a single-seed pilot on 2026-08-25.

## Context

CNN-FullBank gives every task a complete world-model route and an independent
actor-critic. The sequential six-task pilot trained these routes one after
another with four-way data parallelism. That run increased sample use but did
not reduce wall-clock time relative to the original single-GPU workload.

Complete task isolation also permits a different experiment: train one expert
per task in independent single-GPU processes and append completed components to
a task-aware bank. This removes sequential warm starts and all shared-parameter
interference. It therefore answers an acquisition and systems question, not the
continual-learning retention question.

## Decision

Add the named `six-parallel-independent-single-gpu-experts-v1` campaign.

- Train MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and Enduro as six
  independent processes.
- Assign at most one process to each listed GPU. Four GPUs run the first four
  tasks concurrently; newly free GPUs receive the remaining tasks.
- Each child uses one GPU, BF16 autocast with the continuation FP32 stability
  path, uint8 mmap ARROW-50 replay, fixed validation and held-out final cohorts,
  and 180 epochs.
- Preserve the original single-device optimization batches: world-model `N=16`
  and actor context `N=128`, with 1000 and 800 updates per epoch respectively.
- Allocate the six-slot world-model topology in every child, train local slot 0,
  and record the original curriculum index as its immutable assembly slot.
- Advance the fixed validation and held-out seed streams to that original task
  index so child evaluation uses the same cohort slot as a six-task evaluator.
- Use fresh world-model and actor-critic initialization for every task. Do not
  claim forward transfer, backward transfer, retention, or forgetting.
- Keep every child run, Replay backing, log, and snapshot independent and
  non-overwriting. A child snapshot remains an inference artifact and is not a
  resumable checkpoint.

The campaign is an independently trained, task-aware expert-bank pilot. It is
not a sequential continual-learning run. Adding a completed expert to a bank is
operationally incremental, but it is not evidence of continual transfer or
resistance to forgetting.

## Budget and claims

Relative to original ARROW per task, every child has matched single-GPU batch
sizes and device count but twice the task duration, environment interaction,
optimizer updates, sampled contexts, and periodic evaluation opportunities.
BF16 compute and uint8 replay also differ from original FP32 training and
float32 replay. The pilot therefore cannot support a fair superiority claim.

The pre-run task-specific ARROW reference matrix remains frozen. Report raw
returns and taskwise normalized ratios separately; never average raw returns
across games.
