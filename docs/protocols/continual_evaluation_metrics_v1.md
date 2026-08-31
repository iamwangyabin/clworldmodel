# Continual evaluation metrics v1 (ARROW-compatible)

## Status and source

This is the mandatory metric schema for all new continual experiment reports in
this repository. It follows the definitions in ARROW v3, Section 4.4 and Tables
A.2, A.3, and A.15:

- paper: `https://arxiv.org/html/2603.11395v3`;
- metric implementation: `src/clworldmodel/evaluation/metrics.py`;
- reporting command: `scripts/summarize_continual_metrics.py`;
- Atari normalization constants:
  `references/arrow_v3_atari_normalization_v1.json`;
- published Atari comparison values:
  `references/arrow_v3_atari_reported_metrics_v1.json`.

Metric schema identifier: `arrow-paper-v1`. Changing a formula, task reference,
checkpoint selection rule, or cross-seed aggregation rule requires a new schema
identifier. Raw measurements are immutable inputs; derived reports may be
regenerated.

## Source measurements that every run must preserve

At every declared evaluation checkpoint, save the following for **every task in
the configured suite**, not only the active task:

- task name and fixed task index;
- completed epochs, agent decisions, and raw environment frames;
- raw episodic return mean and standard deviation;
- number of rollouts, policy mode, and evaluation-cohort identifier;
- whether the checkpoint is a task boundary;
- confirmation that evaluation transitions never entered Replay or training.

Never average raw returns across different Atari games. They have incompatible
scales. Never replace raw return fields with scaled optimization rewards or
normalized scores. The latter are additional derived fields.

The run manifest must also preserve task order/duration, all update counters,
model parameter counts, trainable parameter counts, incremental parameters per
task, parameter/buffer bytes, Replay sample capacity, Replay allocated bytes,
compute dtype, device count, and wall-clock/GPU measurements. ARROW matches
Replay capacity; a growing or task-private method also requires explicit model
capacity accounting.

## Normalized score

For task \(\tau\), ARROW defines

\[
q_\tau(n)=\frac{p_\tau(n)-p_{\mathrm{ST}_\tau}(0)}
{p_{\mathrm{ST}_\tau}(n)-p_{\mathrm{ST}_\tau}(0)}.
\]

`q=0` is random performance and `q=1` is matched-budget single-task ARROW
performance. Values below zero and above one are valid and must not be clipped.
Every report records the reference artifact ID and hash used to derive `q`.

Table A.15 provides the random and end-of-budget single-task medians needed for
Atari task-boundary metrics. Those six final constants are **not sufficient**
for exact forward-transfer curves before task completion. FT requires the
aligned single-task learning curves at the same within-task sample locations.

A candidate/local-ARROW raw-return ratio is not this metric and must never be
named `q`, `ACC`, or "paper-normalized accuracy".

## Single-pass metrics

After task \(k\), report:

### Average accuracy (ACC)

\[
\mathrm{ACC}_k=\frac{1}{k}\sum_{i=1}^{k}q_{\tau_i}(t_k).
\]

This is the average normalized performance of all tasks learned so far at the
boundary. For a partial curriculum, name it `ACC_1`, `ACC_2`, or `ACC_3`; do not
compare it directly with the paper's six-task `ACC_6`.

### Average minimum accuracy (min-ACC)

\[
\mathrm{min\text{-}ACC}_k=\frac{1}{k-1}\sum_{i=1}^{k-1}
\min_{t_i<n\le t_k}q_{\tau_i}(n).
\]

For each old task, take its minimum at all evaluations strictly after it was
learned through the current boundary, then average those taskwise minima.
`min-ACC_1` is undefined and is serialized as `null`, never as zero.

### Worst-case accuracy (WC-ACC)

\[
\mathrm{WC\text{-}ACC}_n=\frac{1}{k}q_{\tau_k}(n)+
\left(1-\frac{1}{k}\right)\mathrm{min\text{-}ACC}_k.
\]

At a task boundary this combines current-task plasticity with the historical
minimum of old tasks. Higher is better.

### Forgetting

\[
F=\frac{1}{T}\sum_{i=1}^{T}
\left(q_{\tau_i}(iN)-q_{\tau_i}(TN)\right).
\]

Lower is better. A negative value indicates backward improvement. Report the
per-task terms as well as their mean. Low forgetting alone is not evidence of a
good method if the method never acquired the tasks; read it together with ACC,
min-ACC, WC-ACC, and raw returns.

### Forward transfer (FT)

For each task, average its normalized acquisition curve in the continual and
matched single-task runs, then compute the relative difference specified by
ARROW Eq. 3. Higher is better. FT remains `null` with an explicit reason until
time-aligned single-task curves exist. It must not be approximated from a final
single-task number.

## Sample efficiency and two-cycle metrics

Paper-style sample efficiency is the earliest environment-frame count at which
a method's **five-seed median** curve reaches 85% of the global maximum median
performance across all compared methods. The threshold is shared across
methods. Record methods that never reach it as `null`/`never_reached`, not the
last frame.

For the named two-cycle protocol, additionally report:

- `Max-F`: first-exposure endpoint minus the last evaluation immediately before
  the second exposure, per task and averaged;
- `Recovery`: second-exposure endpoint divided by first-exposure endpoint, per
  task and averaged;
- cycle-1 and cycle-2 forgetting and FT.

These fields are `null` for a one-cycle run.

## Cross-seed aggregation

The per-run metric artifact is a seed-level record. An official table contains
five predeclared seeds and reports `median [q25, q75]` for every aggregate
metric. Preserve individual seed records. Do not discard or select seeds based
on performance. A single seed, smoke run, stopped run, or partial curriculum is
always labeled diagnostic/pilot and never substituted for a missing seed.

## Comparability tiers

1. **Published-protocol comparable**: same full task suite/order, 90 epochs per
   task, evaluation of every task every 10 epochs, stochastic policy, 16 Atari
   rollouts per task, matched counters/budgets, aligned normalization data, and
   five-seed median/IQR.
2. **Matched local per-seed comparison**: same task order, seed, evaluation
   policy/cohort, checkpoints, interaction/update budgets, and normalization.
   Useful for controlled ablations, but not a paper reproduction.
3. **Diagnostic only**: partial task suite, deterministic/task-aware evaluator,
   different cohorts, extra consolidation/updates, unmatched budgets, or a
   single final evaluation. Keep it in a separately labeled table.

Task-aware and task-agnostic methods must not share a headline table without an
explicit `task_identity_exposed_to_agent` column. Deterministic and stochastic
evaluation results must not be silently mixed.

## Required artifact and command

A completed continual run should produce `continual_metrics.json` containing:

- immutable source paths and SHA-256 hashes;
- raw checkpoint matrix;
- normalization reference and normalized matrix;
- boundary ACC/min-ACC/WC-ACC;
- final forgetting and per-task terms;
- FT/sample-efficiency/two-cycle values or explicit unavailability reasons;
- evaluation and budget comparability metadata;
- parameter, Replay-byte, and compute accounting when available.

Generate a matched local comparison with:

```bash
python scripts/summarize_continual_metrics.py \
  runs/RUN_A runs/RUN_B \
  --output comparison.json
```

The tool refuses to infer task-boundary metrics when a boundary evaluation is
missing. It also reports whether the supplied runs have identical local
comparison signatures.

## Frozen local seed-0 diagnostic currently available

`references/local_s0_continual_metric_comparison_v1.json` records the full raw
checkpoint matrices, hashes, normalized scores, and derived metrics for the
three complete local original-order runs presently available. Its compact table
is:

| method | F down | ACC up | min-ACC up | WC-ACC up | FT |
|---|---:|---:|---:|---:|---:|
| ARROW-50 | 0.849806 | 0.897343 | 0.622579 | 0.548055 | unavailable |
| DreamerV3/FIFO | 2.386320 | 0.230955 | -0.030802 | 0.012898 | unavailable |
| ARROW-FastKANAC-StableTargets-50 | 0.114152 | 0.827486 | 0.792280 | 0.675309 | unavailable |

These are matched local seed-0 diagnostics, not the paper's five-seed numbers.
Within this local comparison FastKAN StableTargets has much lower forgetting
and higher min-ACC/WC-ACC than local ARROW, while local ARROW has higher final
ACC. FT is deliberately absent because aligned single-task curves have not been
preserved locally.
