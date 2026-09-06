# 0056: One Dream Rehearsal implementation, two history capacities

## Status and correction

Implemented after the user's explicit clarification on 2026-09-05. CPU
synthetic fixtures and dry-run contracts pass; GPU smoke and training have
not run. This supersedes the assistant's overinterpretation in decision 0054.

The user wants the original Dream Rehearsal recipe transferred to our Atari
continual tasks, then the **same method with only random history compression**.
The bounded arm is not a different model, trainer, optimizer, grading rule,
rehearsal schedule, ordinary-update budget, or evaluation configuration.

## Single experimental variable

`Dream-Rehearsal-MemoryPair-v1-Atari` has one common typed configuration. Only
`history_capacity_transitions` differs:

- `null`: keep every collected transition (full-history reference arm);
- `524288`: keep an unbiased reservoir of 1,024 non-overlapping blocks of 512
  transitions (bounded arm), matching ARROW's declared sample capacity.

The source model, optimizers and `tunnel_update` remain unchanged at their
recorded pins. Both arms share one collector and the original sampler. The
launcher hashes the configuration after removing only the capacity key; these
hashes and all learning/evaluation budget fields must agree for a paired seed.

Do not “match” the bounded arm to ARROW by changing other Dream Rehearsal
settings. Sample capacity matches ARROW; full compute/parameter/evaluation/byte
matching to ARROW is **not** claimed. Comparisons between the two Dream
Rehearsal arms isolate the chosen history retention constraint.

## No hidden history

Both ordinary WM/actor/critic learning and old-phase rehearsal read the same
retained store. Per-phase libraries contain live index views, not old copied
episodes. An eviction removes the data from every eligible sampling view, and
slots are reused in fixed mmaps. No unbounded training NPZ archive is written
behind the bounded buffer. Evaluation data remains separate and is never
eligible for training.

The collector streams rows immediately. A reservoir admission decision is
made at the first transition of each 512-transition block; rejected blocks
are never added to replay. Therefore even within collection the bounded
transition count cannot exceed 524,288. Consecutive retained fragments are
reassembled into their original episodes, rather than creating new resets
every 512 transitions in the full-history arm. Gaps caused by eviction cannot
be joined into fictional transitions.

Context/reset observations and minibatch workspace are not extra stored
transitions; their bytes are separately accounted. The store uses one extra
starting-observation row per fragment, with zero reward/action, and no past
action/reward from an evicted block. The uncapped store reconstructs the exact
source episode arrays. Complete-run boundaries are block-aligned, so the
bounded final history is exactly 1,024 × 512 real transitions.

## Verification and limits

Tests cover the only-config-difference contract, unchanged source model config,
source collector/sampler trace equality without evictions, source rehearsal
generator behavior after evictions, exhaustive Algorithm-R inclusion
probabilities, capacity at every insertion, context/reset handling, episode
reconstruction and absence of an independently retained old-phase backup.
All tests use synthetic arrays/API responses and optimizer spies, not a GPU
experiment or an optimizer update.

Matching an algorithm does not guarantee matching the paper's MiniGrid scores
on Atari. The full-history arm is the first implementation/performance check;
the bounded arm then measures what the explicit storage restriction changes.
Old failed pilots retain their original identities and are not either new arm.

See `docs/protocols/dream_rehearsal_memory_pair_v1_atari.md` for the executable
protocol and shared budgets. New training still requires a clean, pushed,
upstream-synced commit and target-accelerator smoke.
