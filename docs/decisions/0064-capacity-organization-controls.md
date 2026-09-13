# Decision 0064 — Five separately named capacity-organization controls

Date: 2026-09-11. Status: authorized for implementation and validation; launch
only after tests and target-accelerator smoke pass from clean, pushed code.

The user explicitly approved: “允许，按新协议实现五组对照后启动”.
Implement shared, wider shared, full per-task WMs, frozen shared core with
residuals, and independent residuals by composing the retained WM API and
common Atari trainer. This exception permits the new FullBank control, not
restoration of retired method implementations or historical runtime switches.

The invariant is equal interaction, online optimizer-step, replay and evaluation
budgets **among these five controls**, with oracle private AC throughout.
Architectures deliberately differ in parameter ownership and gradient reach.
These pilots omit AWM protection/projection/consolidation/RCC. AWM has 12,000
additional boundary WM updates and these controls are not compute-matched
single-switch AWM ablations. Do not use them alone for that claim.

See `docs/protocols/capacity_organization_v1_atari.md`. S0 is the prospective
seed, selected before observing results. No favorable-seed selection, equivalent
resume, automatic retry, or multi-seed reproduction is implied.
