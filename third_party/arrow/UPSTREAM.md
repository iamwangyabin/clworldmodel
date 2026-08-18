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
12. Seed Python `random` from the resolved run seed because ARROW's mixed replay
    uses it to select FIFO versus LTDM. Derive separate owned collection and
    evaluation seed streams, seed every Atari worker reset and action space,
    and restore parent Python, NumPy, and PyTorch CPU/CUDA RNG states after
    evaluation so its stochastic actions cannot alter later training draws.
    CUDA remains outside deterministic-only mode and is recorded as such.
13. Add opt-in `relu_kan_bounded`, a separate named KAN actor variant that
    preserves the historical direct `relu_kan` bridge and inserts a project-owned
    LayerNorm--sigmoid adapter between its two fixed-grid KAN layers. This keeps
    second-layer inputs inside the declared grid support after the direct pilot
    exposed an out-of-support, inactive-basis failure. The default MLP remains
    unchanged; fixed-grid support and gradient-reachability tests cover the new
    bridge.
14. Add opt-in `relu_kan_adaptive` for the separately named trainable-anchor
    ReLU-KAN actor protocol. The thin vendored bridge accepts the new validated
    config and CLI flag while the project-owned actor keeps the bounded
    LayerNorm--sigmoid interface and learns every per-input, per-basis support
    start and positive width. Widths use `softplus(raw_width)` rather than
    unconstrained endpoints so the basis normalization cannot become singular.
    Fixed-grid initialization, support ordering, anchor gradients, actor
    interface, parameter accounting, config validation, and launcher contracts
    cover the intentional deviation. The default MLP remains unchanged.
15. Add opt-in `fast_kan_ac` for the separately named
    `ARROW-FastKANAC-KDAligned-50` behavior pilot. The vendored bridge replaces
    both actor and critic with project-owned, fixed-grid Gaussian FastKAN heads
    and exposes the protocol's actor-critic optimizer, imagination, return
    normalization, and slow-critic settings without changing default MLP runs.
    LaProp and FastKAN live under `src/clworldmodel/`; fixed-center, branch,
    initialization, tensor-shape, actor/critic replacement, parameter,
    optimizer, config, and launcher contracts cover the bridge. ARROW replay
    and world-model training remain unchanged, and the omitted DreamerV3
    replay-value loss is explicitly recorded as a protocol deviation.
16. Add opt-in `fast_kan_ac_param_matched` for the separately named
    `ARROW-FastKANAC-ParamMatchedRepVal-50` trainability extension. It uses the
    same project-owned FastKAN Actor/Critic bridge at width 53, within 0.83% of
    the published ARROW MLP behavior-head parameter count, and applies a
    `0.3` replay critic loss to the four posterior context frames already
    sampled for imagination. The replay target follows ARROW's same-index
    reward convention and draws no additional minibatch. The vendored trainer
    also exposes per-update actor/critic diagnostics, explicit analysis
    milestones, and optional credential-free SwanLab TensorBoard mirroring.
    The default MLP and the completed width-34 FastKAN protocol are unchanged;
    parameter, target-return, config, logging, snapshot, and launcher contracts
    cover the new behavior.
17. Add opt-in `fast_kan_ac_stable` for the separately named
    `ARROW-FastKANAC-StableTargets-50` correction pilot. It preserves both
    width-53 project-owned FastKAN behavior heads and all per-epoch budgets,
    while using the existing EMA critic for imagination targets, replay-value
    bootstraps, and the detached actor advantage baseline. It also evaluates
    the horizon bootstrap on the final post-transition imagined state rather
    than duplicating the last pre-transition value. Historical MLP and FastKAN
    names preserve their prior target and bootstrap semantics. Focused tests
    cover the terminal state, slow baseline, config isolation, parameter
    accounting, and launcher contract.

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
