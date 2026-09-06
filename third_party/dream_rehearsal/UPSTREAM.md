# Dream Rehearsal reference subset

- Repository: <https://github.com/gurpnijjer/dream-rehearsal>
- Full pin: `7680778f798be3a27a17c320cc875b573c45f0e1`
- Imported: 2026-09-05, using `git show <pin>:<path>` without text rewriting.
- License: the original Apache-2.0 `LICENSE` is retained.
- Paper: <https://arxiv.org/abs/2607.19749>.

`MANIFEST.sha256` and `provenance.json` fingerprint every imported source,
license, and reference-document file. They use paths relative to this directory.
This `UPSTREAM.md` and the provenance files are project-written metadata, not
upstream source. No imported file has a local modification.

## Scope

The integration calls the **unchanged** `tunnel_update` in
`src/orchestrator_chain_nm512.py` with `cont_grading=True`, and the author's
`random_prefill` imported from `orchestrator_ab_nm512.py`. The normal
`substrate/SETUP.md` model preset is merged with pinned NM512 defaults. Margin
probe source is retained to make the published source boundary inspectable;
the Atari runner does not perform probe, composite-router, frozen-head,
real-BC, competent-anchor, or re-graduation experiments.

This is a subset, not a standalone copy of the author's runnable repository.
The upstream README references additional figures, results, and files that are
not vendored here. Open those through the pinned upstream repository.

## Known fidelity limits

The artifact points to NM512 but does not identify its exact historical commit
or provide a complete environment lock. Our NM512 pin is explicit, not claimed
to be the author's verified runtime. The substrate's `value: {layers: 5}` key
is unused by the pinned NM512 trainer, which consumes `critic`. The integration
preserves this fact instead of changing the network silently.

The original normal batch is 16 sequences of 64 steps; 4 by 16 belongs to the
author's **smoke** override. The official scorer bootstraps from `feats[-1]`,
not a newly computed post-horizon feature. Source behavior is preserved even
where a different implementation might be preferable.

Environment, scheduling, storage-device and evaluation adaptations are in
`docs/protocols/dream_rehearsal_official_code_v1_atari.md`. They must not be
confused with modifications to the imported learning source.

## Verification

`scripts/dream_rehearsal_reference_support.py` checks hashes before import and
launch. Tests include hand-computed grading, real NM512 posterior/imagination
and actor-gradient fixtures, shared-history sampler/mmap input equality,
strict protocol configuration, and chunk cadence. CPU fixtures do not
establish CUDA readiness or reproduce paper returns.
