# 0063 — Separate route-window resets from AWM policy state

Date: 2026-09-08. User-authorized behavioral correction and two fresh Atari pilots.

The user requires AWM-AutoRoute to preserve AWM's collection/evaluation behavior
except where inference actually changes the expert. The v3 adapter copied a
candidate's zero state and dummy action into the policy at every episode start,
even with one eligible expert. This violates that requirement: NextStep already
has the preserved AWM reset-mask timing. It is not a required consequence of
two-frame routing, nor proof that all observed return differences arise here.

Keep independent episode-local scoring, but substitute policy history only on
an actual ID change. Keep the collector's history/action on unchanged IDs and
use its exact observation conversion. Do not modify the oracle AWM collector,
reset masks, return extraction, updates, private AC or boundary selection gates.

Name the correction protocol v4, retain exactly one implementation, and reject
old AutoRoute configs/checkpoints rather than silently migrating their meaning.
Keep all v3 evidence. Stop only the two user-selected new Atari attempts and
restart S0/S1 from scratch on their assigned hosts; old ten experiments remain
read-only. No evidence-based claim that returns will recover to 4000+ is made.

The [v4 protocol](../protocols/awm_autoroute_v4_atari.md) specifies invariants,
switch semantics, version/checkpoint contracts and regression coverage.

Local verification: 75 focused retained-method, routing, recovery, replay,
precision and parity tests ran: 74 passed, one CUDA-only test skipped. The two
new state-isolation regressions fail on the original adapter and pass after
the correction. Vendor fingerprints and whitespace checks pass. The historical
`test_evolving_atomic_rssm` suite still has obsolete constructor/config fixtures
(`full_task_experts`, `dino_fullbank_current_task_fraction`); this change does
not restore retired interfaces or claim the complete historical suite is green.
Target-CUDA checks and actual run provenance belong to each launch manifest.
