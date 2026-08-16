# ARROW vendored source

- Source: `https://github.com/Cerenaut/ARROW`
- Commit: `cb05e7d97ed83c3cf6e528960db0da6868e29232`
- Commit date: 2026-06-28
- Imported: 2026-08-12
- License: MIT, copyright Cerenaut
- Local policy: project-maintained vendor

`MANIFEST.sha256` fingerprints the current vendored tree. This `UPSTREAM.md`
file and the manifest itself are project metadata and are not part of upstream.
The pristine source remains recoverable from the commit above.

Reference launchers execute this directory directly. Local changes must be
documented here, covered by focused parity tests, and followed by regenerating
`MANIFEST.sha256`. Clean project-owned implementations still belong under
`src/clworldmodel/`.

## Local changes

1. Add the seven fields present in every published Atari configuration to the
   typed configuration model.
2. Make `--arrow-replay-ratio` an explicit override instead of silently
   replacing the config value.
3. Express categorical sampling, straight-through gradients, KL divergence,
   and diagnostic masked means as fixed-shape tensor operations.
4. Add optional stage timing, compiled world-model loss, fused Adam, TF32, and
   `zero_grad(set_to_none=True)`. The project launcher enables these runtime
   optimizations by default.
5. Remove the unused AutoROM dependency because the pinned `ale-py` wheel
   already supplies the required ROMs.
6. Add optional explicit run and analysis-snapshot directories to the Atari
   trainer. When requested, it atomically saves CPU-portable world-model and
   actor-critic weights with checksums at sequential task boundaries and at
   training end. These artifacts are labeled non-resumable and the default
   upstream execution remains unchanged when the options are omitted.
7. Add an opt-in Atari `r2` observation objective for the named
   `ARROW-R2Rep-50` ablation. The thin vendored integration removes the decoder,
   reuses one encoder pass, and calls the project-owned bias-free projector and
   stop-gradient Barlow Twins objective under `src/clworldmodel/`, based on
   R2-Dreamer commit `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`. The
   default remains pixel reconstruction. Fixed-formula, stop-gradient,
   precomputed-embedding parity, head-replacement, and training-gradient tests
   cover the change.
8. Persist world-model and observation-head parameter counts and parameter
   bytes in `model_parameter_accounting.json` for both objectives. The artifact
   explicitly excludes gradients, optimizer state, and activations.
9. Add an opt-in Atari `relu_kan` actor bridge for the named
   `ARROW-KANActor-50` ablation. The critic and all world-model modules remain
   unchanged. The bridge calls the independently implemented project-owned
   fixed-grid ReLU-KAN actor under `src/clworldmodel/`; the default remains the
   upstream MLP actor. Fixed-formula, tensor-shape, gradient, parameter-budget,
   config, critic-isolation, and default-MLP parity tests cover the change.
10. Persist actor, critic, and combined parameter, persistent-buffer, and byte
    accounting in `actor_critic_parameter_accounting.json` after actor creation.
    Optimizer state, gradients, and activations are explicitly out of scope.
11. Add opt-in explicit epoch and final-evaluation controls for named truncated
    pilots. Final evaluation uses the upstream stochastic-policy semantics,
    evaluates only tasks seen during a sequential prefix, and never adds its
    transitions to replay. Persist both optimization-scaled returns and raw
    game returns recovered from the fixed task reward scales; regular
    evaluation also logs both units. Omitting the pilot flags preserves
    upstream duration and training behavior.

## Known issues at import

1. Every Atari ARROW/DV3 JSON config contains seven keys missing from
   `Code/ARROW_and_DV3/Atari/config.py`, so config construction raises
   `TypeError` before training.
2. `--arrow-replay-ratio` defaults to `50-50` rather than `None`, which makes
   the CLI overwrite a value loaded from a config even when no override was
   supplied.
3. The world model and actor use unconditional `.cuda()` calls, and the
   published replay configuration stores approximately 24 GiB of float32
   observations on the accelerator.
4. The upstream commit has no automated test suite or CI definition.

Items 1 and 2 are corrected by the documented local changes. Items 3 and 4
remain constraints of the vendored implementation.
