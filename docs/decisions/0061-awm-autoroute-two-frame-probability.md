# 0061 — Update AWM-AutoRoute in place to a two-frame routing protocol

Date: 2026-09-07. Status: implementation, not a performance validation.

Compatibility update (2026-09-08): Decision 0062 retires the v1/v2 runtime
branches and keeps only the two-frame method named AWM-AutoRoute. References
below to loadable v2 configs describe the implementation at this decision's
original date, not current support.

The user keeps the **AWM-AutoRoute** name and explicitly chooses "only accumulate
two observations, then hold" rather than continuous comparison. Retain the
existing method key and entry point, but advance its protocol/default to v3.
Do not present this bounded window as naturally converging continuous inference.

Reuse the existing RSSM posterior probabilities as the routing decoder input;
use independent hard candidate histories and the executed first action to
accumulate two reconstruction scores. Apply the first choice immediately,
allow one revision after the second observation, then do no candidate probing.
Keep the private Actor, training, replay and oracle compression/consolidation
gates. Reinitialize the selected policy at an episode start, restore a new
candidate's own history on a switch, and version these reset/state differences.
The selected stochastic policy call is not replaced by a soft latent.

The [v3 protocol](../protocols/awm_autoroute_v3_atari.md) defines formulas,
eligibility, NextStep timing, compute accounting and diagnostic denominators.
Old v1/v2 evidence remains unchanged; explicit v2 configs still work and cannot
silently resume under v3. No new parameters, training budget or dependency.

Verification uses fixed real-RSSM tensors and canned vector responses, including
stochastic-vs-deterministic probe independence, route-specific switch state,
per-worker resets, no later probes, checkpoint/RNG round trips, launcher/config
isolation and unchanged baseline/AWM tensor parity. The existing v2 test failure
recorded in Decision 0060 was an off-by-one expected stored-action prefix: it
omitted ARROW's initial dummy action. The corrected assertion checks five
entries `[0,0,0,0,1]` (dummy plus all four env.step calls), retaining the original
env.step assertions. No old collector behavior was changed to satisfy it.

No environment experiment, GPU smoke or training was launched. Earlier
diagnostic CoinRun numbers are not v3 production performance measurements.

## Verification outcome

On the pinned CPU runtime (PyTorch 2.3.0, NumPy 1.26.4, Gymnasium 1.1.1),
45 tests passed across `test_two_frame_autoroute`, `test_d_autoroute`,
`test_reconstruction_router`, `test_method_retirement`,
`test_retained_method_parity` and `test_continual_metrics`. This includes a
mixed-dtype cache fixture, not a claim of CUDA/BF16 hardware validation.
The standalone dry-run resolves the v3 mode/protocol, two-observation bound,
zero router parameters and unchanged training budgets. Vendor fingerprints
and whitespace checks pass.

Broader historical launcher suites (`test_evolving_atomic_rssm_launcher`,
`test_training_launchers`) still fail: 64 tests report 18 failures and 43
errors, including retired methods and obsolete defaults. Reloading the saved
pre-change launcher in memory reproduces the identical failure/error signatures.
Those migration issues are not repaired or hidden by this routing change;
the full repository suite is not claimed green.
