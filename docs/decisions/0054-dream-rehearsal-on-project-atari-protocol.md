# 0054: Evaluate Dream Rehearsal under the project's Atari continual protocol

> **Superseded by decision 0056.** This note overinterpreted the user's scope
> as requiring another ordinary-training configuration. The user explicitly
> clarified that the bounded run must be the same Dream Rehearsal as the full
> history run, with only random retention at ARROW's sample capacity. Do not
> implement different models or budgets for these two arms.

## Status

Scope clarified by the user on 2026-09-05. This records the intended experiment;
implementation alignment and target-GPU validation remain pending. No run is
launched by this decision.

## Objective

Apply the paper's Dream Rehearsal **method** to our existing Atari continual
learning setting. The objective is neither a MiniGrid result reproduction nor
an unrestricted transfer of the entire reference training configuration.

Decision 0053's unmodified-source integration remains a useful algorithm
reference and parity oracle. Its full run is **not** the selected project-protocol
baseline: ordinary update and evaluation budgets differ from our controls.
Do not silently redefine that named reference protocol or label it matched.

## Experimental ownership

The project protocol owns the common DreamerV3 backbone and ordinary training
configuration, task order/duration, environment interactions, preprocessing,
action/reward/reset/termination handling, seeds and evaluation. Use the frozen
`arrow_ar50_atari.md` / `dv3_fifo_atari.md` contracts rather than importing
MiniGrid defaults for these settings. The existing 541-epoch schedule and its
distinct epoch-539 six-task endpoint and epoch-540 task-0 revisit must be
reported explicitly; changing that common schedule requires a new protocol
for the controls too.

Dream Rehearsal owns the method-specific changes:

- retain the complete real training history for the first, full-history arm;
- sample old and current real experience for ordinary WM and actor/critic
  training; never regress to `current_task_only` ordinary sampling;
- use one shared actor, with no per-task actor heads or inference task labels;
- use prior-phase replay starts, the reference imagined rollout and
  realized-first continuation-aware score, top-25% selection, and actor-only
  imitation of the selected sampled actions; and
- preserve and separately count the method's additional rehearsal work.

The method-specific batch is the author's normal 16 × 64, not its smoke
override. Bootstrap alignment and score/gradient ownership must be checked
against the imported author function. Porting these operations to the common
backbone does not authorize replacing that backbone's ordinary optimizer or
inflating its ordinary update budget.

Rehearsal cadence must be reconciled explicitly with our vector collector and
collection/update schedule. An epoch-boundary burst is not identical to the
author's interleaving; an exact-count test alone cannot establish timing parity.
This boundary remains an implementation task, not an assertion of alignment.

## Budgets and comparison claims

The current frozen 541-epoch controls have 541,000 ordinary WM updates and
432,800 ordinary actor/critic updates. The source-reference runner instead
projects 4,416,280 of each and evaluates ten episodes per seen task every
2,000 online decisions. Those settings must not become the comparison's
defaults merely because they came from the reference implementation.

For the intended baseline, common ordinary budgets and evaluation stay fixed;
extra actor-only rehearsal updates, imagined samples, wall time and storage
are reported separately. Full-history memory is intentionally **not** storage
matched to finite-memory ARROW. A bounded arm is a separately named ablation,
not a silent replacement of this requested full-history method.

Report Atari per-task raw returns, acquisition and retention/forgetting, not
MiniGrid success thresholds or an undefined cross-game “accuracy.” Old failed
ports remain evidence of those old implementations, not the new baseline's
performance. No new result exists until the aligned implementation and a
clean, pushed, target-smoke-validated launch have actually completed.
