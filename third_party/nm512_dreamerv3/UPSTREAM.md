# NM512 DreamerV3 reference subset

- Repository: <https://github.com/NM512/dreamerv3-torch>
- Full pin: `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`
- Imported: 2026-09-05, by `git show <pin>:<path>` without source rewriting.
- License: MIT, copyright (c) 2023 NM512. Original `LICENSE` retained.

`MANIFEST.sha256` and `provenance.json` record hashes relative to this directory
for all imported files. Project provenance metadata and this note are not
upstream files. **No imported source has been modified.**

## Pin selection and boundary

Dream Rehearsal's substrate instructions identify this repository but not an
exact commit. This project pins the inspected upstream state (commit dated
2026-03-08). It is not evidence of the exact version used by the paper's authors.

The subset contains the complete imported model, policy, optimizer, sampler,
simulation utilities, configuration, and wrapper files required by our Atari
integration. It is not a general-purpose installation of all upstream suites.
`D.Dreamer`, `D.make_dataset`, `tools.simulate` and the reference Dream Rehearsal
function execute directly; the project does not translate their losses into
ARROW or monkey-patch learning behavior. Only the script-layer integration
uses these upstream private APIs.

`requirements.txt` is the unmodified source artifact. Install the separate
project compatibility constraints `requirements/dream_rehearsal_official.txt`
instead of combining this source with the ARROW environment. PyTorch 2.4.1,
NumPy 1.26.4 and old Gym 0.22.0 are preserved. Modern Atari uses explicitly
pinned Gymnasium/ALE in a project-owned API adapter; obsolete upstream Atari
imports are not used. Full installed versions are saved with each run.

## Known behavior retained

- 16 by 64 normal batches, train ratio 512, first-call pretrain 100.
- Per-dataset NumPy sampler seed 0, episode-length-weighted selection and
  reset-aware concatenation. Ordinary updates use the shared full-history dict.
- Both world model and actor/critic continue training on old real data.
- Actor unimix 0.01, learning rate 3e-5, and upstream optimizers/target behavior.
- `feats[-1]` bootstrap and stochastic RSSM latent sampling at evaluation.
- The substrate's unused `value` key is not renamed to `critic`.
- Analysis snapshots are not resumable training checkpoints.

The runtime uses `dataset_size=0` (no erasure), because the stock one-million
transition limit would evict episodes at our much larger Atari budget. Inactive
pixels are in a lossless CPU mmap with original sampler outputs tested equal;
the source sampler is untouched. See the named protocol for all adaptations.

The later `Dream-Rehearsal-MemoryPair-v1-Atari` uses a common project-owned
streaming collector and episode-preserving fixed-slot store in **both** arms.
Its collection order follows this source's MIT `tools.simulate`; fixed-response
fixtures compare the complete uncapped collection and original-sampler trace
against the source. It writes no unbounded training NPZ backup. The only
full/bounded configuration difference is an optional reservoir capacity.
Learning files and the original sampler are still byte-identical; see
`docs/protocols/dream_rehearsal_memory_pair_v1_atari.md` for the storage boundary.
