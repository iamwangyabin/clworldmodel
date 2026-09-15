# 0066 — Freeze four AWM-AutoRoute single-variable ablations

Date: 2026-09-16. Status: accepted for execution by the user.

The paper plan's twelve mechanism cells are fixed as four variants times the
three predeclared Atari seeds. They remain variants of the sole formal method,
AWM-AutoRoute, not new paper methods. The typed `awm_ablation` selector rejects
unknown values and combinations.

The exact controls, unchanged quantities, and the explicitly non-compute-
matched `no_rcc` interpretation are defined in
`docs/protocols/awm_autoroute_mechanism_ablations_v1_atari.md`. This freeze is
needed so results cannot influence which loss, budget, seed, or interpretation
is chosen after training. The user explicitly requested deployment of the
frozen matrix on all available cloud GPUs.

