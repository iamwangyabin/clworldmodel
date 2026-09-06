# 0053: Use the original Dream Rehearsal learning source for the Atari reference

> See decision 0056 for the requested full/bounded memory-only pair. The
> original learning source remains shared; only the retention capacity changes.
> The interpretation in decision 0054 has been superseded.

## Status

Implemented on 2026-09-05 at the user's request for an original-code version.
CPU fixtures and dry runs only. No GPU experiment is authorized by the presence
of this decision; clean pushed provenance and target smoke remain required.

## Problem

The previous never-clear port retained old data but excluded it from ordinary
world-model and actor/critic training. The author's chain passes one shared
never-cleared episode dictionary to `D.make_dataset` for ordinary learning.
Those are substantively different algorithms. The prior port also used the
author's smoke batch rather than normal batch, ARROW model/optimizers, a
different bootstrap feature and delayed rehearsal. Calling its results a
faithful paper reproduction was incorrect.

Merely changing `current_task_only` is insufficient to establish source
fidelity. This audit does not prove which difference caused low Atari returns.

## Decision

Add a separately named `Dream-Rehearsal-OfficialCode-v1-Atari` reference path:

- vendor minimal, licensed, byte-identical source subsets at fixed pins;
- call author `tunnel_update` and NM512 Dreamer/model/sampler/optimizers directly;
- compose the author's normal substrate preset, without “fixing” grading,
  bootstrap convention, unused config keys, or actor distribution;
- keep the environment adapter and orchestration separate from learning;
- keep ordinary replay shared over every real episode, and old-phase views
  only for rehearsal;
- test grading, gradient ownership, normal preset, sampler/mmap equality,
  task-boundary state and update cadence on deterministic fixtures;
- reject protocol changes, dirty/unpushed launches, overwritten histories and
  ambiguous resumes rather than silently weakening the baseline; and
- retain previous results as evidence of the old adaptations, not relabel or
  delete their raw metrics.

## Consequences

Atari still differs from the paper's MiniGrid tasks. Duration is fixed instead
of using a MiniGrid score threshold; action handling, evaluation RNG isolation,
never-erasing capacity, storage device and final-boundary evaluation are named
adaptations. The author's exact NM512 commit is unavailable; our explicit pin
does not fill that historical evidence gap.

The reference's normal 16 × 64 batch and train ratio 512 substantially increase
ordinary update compute over the old ARROW port at this Atari budget. This is
an original-learning-code transfer, **not** a compute-matched ARROW comparison.
Future matched bounded baselines require another named protocol and controls.

Torch 2.4.1 reference dependencies live in an isolated environment, not the
ARROW torch 2.3.0 core package. The project does not refactor or modify the ARROW
trainer in this change. The exact original MiniGrid benchmark would be a
separate reproduction request, not an automatic interpretation of Atari scores.

See `docs/protocols/dream_rehearsal_official_code_v1_atari.md` and
`docs/experiments/dream_rehearsal_fidelity_audit_20260905.md`.
