# 0057: Retire historical method implementations

> Scope update 2026-09-07: Decision 0058 additionally retires StableTargets
> and F/D-AutoKAN. The original decision and evidence below are historical.

## Decision and status

The user explicitly requested code retirement on 2026-09-05/06. Keep
FastKAN StableTargets, D, D-AutoRoute and F/D-AutoKAN. Remove the early
ARROW-R2Rep/ReLU-KAN/non-stable FastKAN, KARROW, MoE/full-bank,
frozen-first-task LoRA/MB/REC, and other Evolving-Core execution paths.
ARROW/DV3 and independent R2-Dreamer/Dream Rehearsal references are not removed.

**Implementation is in progress, not accepted for training.** Safety review
blocked bulk runtime pruning and later bulk test migration. The work continued
with individually checked runtime repairs and a new fixed-input parity test.
Explicit confirmation for retiring old method-only tests and migrating shared
fixtures is pending. The full historical test discovery currently does not pass.
Do not relax numerical assertions or restore retired algorithms to hide that.

## Ownership

- Task-policy banks and fixed-budget update allocation move from `moe_arrow`
  to `continual/task_replay.py`. Existing checkpoint wire identifiers remain.
- The CNN feature projector moves from `rssm_lora` to `models/projector.py`.
- Native R2-Dreamer's retained Barlow objective moves to `r2dreamer/objective.py`;
  its removed ARROW representation-ablation wrappers do not remain.
- Replay placement is a launcher utility rather than an import from a retired
  experimental method.
- D, D-AutoRoute and F compose a retained D configuration, not a chain of old
  method presets. StableTargets likewise owns its complete preset.

## Research and compatibility

The ARROW pin, licenses, baseline config files, historical protocols, raw
metrics, registry records, failed-run evidence and reporting formulas are
preserved. Source retirement is not evidence that a method has no scientific
value. Historical runs remain attributable to their recorded revisions.

For retained methods, task schedules, replay budgets, Q/F/P compression gates,
optimizer hyperparameters and inference routing semantics must not change.
The current D-family dry runs retain 552,000 world-model optimizer steps and
432,000 actor-critic updates. D/D-AutoRoute have a 52,897,535-parameter dense
acquisition bound; F has 44,302,295. These are configuration contracts, not
new measured performance results.

Old resolved configs contain now-retired neutral fields. Cross-revision
checkpoint compatibility still requires explicit validation; no equivalence
claim is made from dry runs or from the tensor fixture alone.

## Evidence so far

`tests/fixtures/retained_method_parity.json` records hashes of the actual
pre-retirement working-tree source (including pre-existing uncommitted edits),
fixed CPU inputs and expected contracts. `test_retained_method_parity.py`
checks baseline/D initial state hashes, losses and gradient norms, plus
StableTargets state hashes and outputs. It exercises nonzero private
mechanisms and older-atom reuse without environment interaction or optimizer
steps. The fixture includes source hashes and the PyTorch version.

The retained ARROW, DV3, StableTargets, D, D-AutoRoute and F dry runs pass.
No experiment, GPU smoke, commit, push or performance claim accompanies this
intermediate cleanup. Full retained-suite and checkpoint validation remain
required before this decision's implementation can be marked complete.
