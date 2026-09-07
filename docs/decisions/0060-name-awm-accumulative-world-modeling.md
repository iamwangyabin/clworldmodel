# 0060: Name D AWM — Accumulative World Modeling

## Status

Accepted by the user on 2026-09-07. Naming only; no research behavior changes.

## Decision

- D's formal method name is **AWM — Accumulative World Modeling**.
- Its automatic-routing variant is **AWM-AutoRoute**, formerly D-AutoRoute.
- Use these names in current documentation, launcher help and the human-readable
  `method` field of newly produced launch manifests.

The name expresses the aim of accumulating learned prediction and control
functionality within a continually trainable world model. Full-width acquisition,
protection of old functions and return-gated structural compaction support that
aim. The name is not a claim of zero forgetting or monotonic improvement.

## Compatibility and evidence

This decision does not rename launcher files, internal Python symbols, config
method keys, serialized checkpoint fields, default run directories or protocol
IDs. The retained keys remain:

- AWM: `evolving_atomic_rssm_adaptive_compression_shared_heads_arrow`.
- AWM-AutoRoute:
  `evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`.

Preserve historical decisions, experiment names, run IDs, manifests, raw results
and provenance as recorded. Historical D is AWM under its original protocol;
the earlier D-AutoRoute v1 and maintained AWM-AutoRoute v2 remain distinct
protocols under Decision 0059. No result is promoted or reinterpreted by naming.

Architecture, training, routing, evaluation, budgets and checkpoint contracts
are unchanged. No training or environment interaction is needed for this change.

## Verification

The new launcher naming regression test failed for both old display names before
the rename and passed afterward. Both pre/post-rename dry-run manifests were
compared: only the human-readable `method` field changed; resolved configs were
identical. Python syntax and touched-file whitespace checks passed.

The focused command passed all 10 tests in the pinned PyTorch 2.3.0 environment:

```bash
PYTHONPATH=src:tests:scripts python -m unittest \
  test_d_autoroute.DAutorouteLauncherTests test_method_retirement
```

The broader `test_d_autoroute test_reconstruction_router test_method_retirement`
run passed 29 of 30 tests. The existing
`test_collection_locks_each_worker_with_arrow_next_step_autoreset` test failed
on its expected action sequence (`[0, 0, 0, 0]` observed versus `[0, 0, 0, 1]`
expected). Loading the pre-rename launcher and test snapshots in memory reproduced
that same failure (28 of 29 tests passed, without the new naming test). No router
or collector code was changed for naming. Existing retention/cleanup validation
limits in Decisions 0057 and 0058 are unaffected; the full suite is not claimed
green. No training was launched.
