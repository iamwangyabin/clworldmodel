# Frozen CoinRun task-ID prefix classifiers — pilot v2

Protocol: `CoinRun-Frozen-TaskID-PrefixClassifier-Pilot-v2`.
Entrypoint: `scripts/eval_coinrun_temporal_task_id.py`.

All classifier definitions, train/validation/test seeds, data budgets, head
optimization budgets, model-selection rules and reporting populations are
exactly as specified in [pilot v1](coinrun_frozen_task_id_prefix_classifier_pilot_v1.md),
with the following explicit numerical execution correction.

## Match production batch size, not just nominal dtype

The v1 launch at `899573359b86c049c1a544d70d157c7e1521ba62` aborted after
collecting its training split and before any router optimization or any
validation/test collection. Its batch-16 offline first-frame scores failed the
comparison with batch-1 production routing. On 16 saved training examples,
batch-16 error differences reached 0.00204563 while replay at batch 1 matched
every production error bit-for-bit. No test labels were inspected to choose
this correction. This is numerical execution mismatch, not an accuracy result.

V2 fixes feature extraction at **batch 1**, the production collector's size,
for every candidate, frame, and split. Configuration rejects another score
batch size. The first-frame score audit is strengthened to exact array equality;
there is no first-frame substitution and no relaxation of tolerance. A larger
feature-extraction batch is not assumed equivalent in this BF16 categorical
RSSM, and no claim of batch-size-invariant deterministic inference is made.

The full predeclared data are recollected using the **same** v1 seeds; the failed
collection is not an extra pool of training examples. Preserve its failure,
trajectories, numerical audit and actual interaction count as campaign overhead.
Report both v2 interaction use and the extra failed-attempt interaction count.
The earlier attempt had zero router, WM and AC updates. V1's raw artifacts must
not be overwritten or retroactively assigned the v2 commit.

V2 retains single-instance reference behavior at extra feature-extraction cost.
It makes no FPS improvement claim. The fixed-input tests and real-checkpoint
no-update numerical checks must pass before the clean/pushed v2 launch.

## Read-only result verification

`scripts/report_coinrun_temporal_task_id.py RUN_DIRECTORY --plot` independently
recomputes every all-cohort accuracy from saved raw predictions and verifies
first-frame preservation. For the predeclared 1/2/5/9/17-frame columns, it also
reports paired GRU-minus-control accuracy intervals using 4,000 bootstrap draws
over equally sized environment-seed groups, with fixed seed 20260909. Each
group contains the matched task variants. Router seeds are averaged, not counted
as extra independent test examples. Intervals condition on this one WM checkpoint
and the fitted heads; there is no multiple-comparison adjustment. Plot shading
is the router-seed range, **not** a confidence interval. The report performs no
environment interaction or parameter update and cannot select another model.
