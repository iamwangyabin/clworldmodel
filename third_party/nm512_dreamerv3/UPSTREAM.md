# Pinned NM512 DreamerV3 source subset

- Repository: <https://github.com/NM512/dreamerv3-torch>
- Commit: `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`
- Imported: 2026-09-06 using `git show <commit>:<path>`.
- License: MIT, copyright (c) 2023 NM512; original LICENSE retained.
- `MANIFEST.sha256` and `provenance.json` list every imported artifact.
- **No upstream source changes.** Provenance files are project-owned metadata.

This is a community PyTorch implementation, not the Continual-Dreamer authors'
V3 release (their code uses TF DreamerV2). The project already uses this same
pin in its independent reference integration. Neither source is relabeled as
ARROW's original implementation. This branch imports the source from Git, not
from an uncommitted working tree.

`scripts/continual_dreamer_v3_support.py` is the explicit private-API adapter.
It calls `models.WorldModel._train` and two `models.ImagBehavior._train`
instances directly. It does not run NM512's unrelated environment constructors,
episode sampler, or train-ratio scheduler. Native learning math is untouched.

Native `exploration.Plan2Explore` is included as reference but not instantiated:
its default action conditioning, log disagreement, activation/normalization,
standard-deviation correction and model-loss reduction differ from the audited
Continual-Dreamer recipe. The independent project ensemble preserves the latter
formula; V3's actors/critics are reused for both behaviors.

Install `requirements/continual_dreamer_v3.txt`, not the broad, obsolete suite
dependencies in the unmodified source `requirements.txt`. The resolved config,
installed versions, source hashes, PyTorch 2.4.1/CUDA runtime and adaptations are
recorded per run. See `docs/protocols/cd_dv3_p2e_episode_slots_v1.md`.

Known native differences retained include the V3 actor/critic learning rates,
KL loss and representation loss, symlog two-hot reward/value distributions,
unimix, stochastic latent evaluation, 15-state imagination with native return
indexing, and predicted continuation at initial imagination states. In
particular, this is not a claim of tensor parity with TF DreamerV2 or with ARROW.
