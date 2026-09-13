# 0062 — Maintain one AWM-AutoRoute

Date: 2026-09-08. Status: user-authorized retirement of old runtime variants.

The user requests exactly one method named **AWM-AutoRoute**, retaining the
two-frame probability-reconstruction behavior introduced by Decision 0061.
Do not present v1/v2/v3 as independently supported choices. The internal v3
protocol ID remains unchanged to distinguish old experiment provenance.

Remove the first-frame router class and its adapter branch. The typed config
and inference boundary reject `first_frame_reconstruction`, while collection
and evaluation default to the two-frame method. Keep the existing method key
and standalone launcher path. No changes to the retained router's equations,
two-observation limit, private Actor behavior, reset/state handling, training
budgets, model parameters or checkpoint schema.

Preserve historical protocols/results and read-only report interpretation.
Reproduction of retired behavior requires its recorded Git revision. Do not
restore old code to satisfy old tests or silently reinterpret old checkpoints.

Validation: 44 fixed-tensor/canned-response tests pass across
`test_two_frame_autoroute`, `test_d_autoroute`, `test_reconstruction_router`,
`test_method_retirement`, `test_retained_method_parity`, `test_continual_metrics`.
The three old first-frame unit tests are replaced by the retained router's
existing cumulative/reset/invalid-input tests, plus single-route/tie and
retired-interface rejection checks. Shared policy tests now assert two-frame
decisions rather than the retired first-frame lock. No environment interaction,
optimizer updates, CUDA smoke or performance measurement was launched.
The unrelated historical launcher-test migration limits recorded in Decision
0061 remain; the full repository suite is not claimed green.
