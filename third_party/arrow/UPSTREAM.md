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
18. Add metadata-only replay methods and return values for the project-owned
   native R2-Dreamer adapter. The original `add` writes, FIFO overwrite order,
   LTDM random-key decisions, and `minibatch` tensor outputs remain unchanged;
   the optional metadata exposes accepted slots plus sampled time and sequence
   indices so an external R2 posterior-state sidecar can stay aligned. A
   deterministic FIFO parity test checks the unchanged minibatch values and
   wraparound slot map.
19. Add the opt-in Atari `dinov3_next_feature` path and fixed-capacity residual
    hooks for the separately named KARROW-FrozenCore-v1 protocol. The path accepts frozen
    project-owned DINOv3 integration features, removes the pixel decoder, and
    predicts the stop-gradient observation feature from the one-step RSSM prior.
    The unchanged GRUCell and MLP behavior heads may receive independent,
    zero-initialized project-owned KAN or exactly core-parameter-matched MLP
    corrections. ARROW write and sampling metadata align a byte-accounted frozen
    feature sidecar without changing replay decisions. Defaults remain the
    published CNN, reconstruction objective, and no residual. Focused tests cover
    default parity, correction matching, feature-target gradients, sidecar
    alignment, configuration isolation, and launcher accounting.
20. Add the opt-in `freeze_after_first_task` shared-core mode for
    KARROW-FrozenCore-v1. Task 1 updates the original RSSM, reward/continue,
    feature-prediction, and actor-critic MLP bases together with their residual
    adapters. At the first sequential task boundary, the base modules and their
    optimizer state are removed from future updates; the single residual set
    remains trainable. The mode also applies residual corrections to posterior
    logits, latent-prior logits, reward, continuation, and feature prediction so
    new tasks retain a plastic path after the shared core is frozen. The default
    ARROW and DINO-only paths remain unchanged.
21. Add the separately named opt-in `dinov3_posterior_feature` observation path
    for `KARROW-SpatialFrozenCore-v2`. The project-owned frozen encoder excludes
    CLS and register tokens and pools the native patch grid to `4 x 4`. Before
    the first world-model update, 512 uniformly selected frames from the initial
    random Task-1 collection fit one 384-to-64 PCA channel projection, which is
    then frozen and stored in checkpoints. The resulting 1,024-dimensional
    float16 sidecar follows the unchanged ARROW write and sample decisions. The
    feature head reconstructs stopped spatial targets from posterior RSSM states
    with batch-standardized SmoothL1 and logs a constant-prediction baseline.
    The completed v1 prior/CLS/cosine behavior remains selectable and unchanged.
    Focused tests cover token selection, spatial pooling and learned projection,
    constant-shortcut rejection, target stop-gradient, configuration isolation,
    and byte accounting. Rollout access to optional reward and continuation
    residuals defaults to none, preserving the pre-residual WorldModel test
    interface without changing production behavior.
22. Add the separately named opt-in `replay_functional` KAN-consolidation hook
    for `KARROW-ReplayConsolidated-v3`. At each sequential task boundary, before
    new-task collection, the trainer visits every project-owned KAN residual on
    unchanged ARROW replay posteriors and short deterministic imagination. It
    estimates per-RBF-coefficient squared local output-Jacobian importance,
    restores all training RNG states, and performs no environment interaction or
    optimizer update. After Task 1, shared bases and KAN coordinate maps are
    frozen; only RBF coefficients remain trainable under cumulative gradient
    protection, post-Adam parameter-delta scaling, and an anchor loss. Defaults
    remain no consolidation. Focused
    tests cover configuration isolation, importance persistence, coordinate
    freezing, value-preserving gradient scaling, launcher budgets, and offline
    latent-region metrics.
23. Add an opt-in `snapshot_adaptation` shared-core mode and Task-2 acquisition
    path initialized from a non-resumable Task-1 analysis snapshot. The trainer
    can construct the loaded actor-critic before the first collection, reset
    replay/optimizer/RNG state, freeze the shared core, and leave all KAN
    residual parameters plastic. A separately named `kan_plus_heads` diagnostic
    opens only the final latent, reward/continuation, actor, and critic
    readouts. The default continual trainer behavior remains unchanged.
24. Add the separately named `KARROW-InputAligned-v4` residual topology. The
    recurrent, posterior, prior, actor, and critic corrections consume their
    corresponding module inputs rather than only a base output or frozen trunk
    feature; reward, continuation, and feature corrections retain their
    existing full-state inputs. Task 1 jointly optimizes unchanged bases and
    zero-output residuals. Private RNG construction preserves the same-seed
    base initialization and subsequent training stream. The existing
    first-boundary freeze then leaves the complete residual branches plastic.
    The default and v1-v3 `base_output` behavior remains unchanged. Focused
    tests cover configuration isolation, exact zero-init parity, residual input
    tensors, Task-1 trainability, and the v4 launcher contract.
25. Add the opt-in, separately named `MoE-ARROW-v1-Atari-TaskAware` path. The
    sequential scheduler's scalar task index hard-routes complete recurrent
    dynamics, latent-prior, reward, and continuation experts and selects an
    independent per-task Actor-Critic/optimizer. New task modules copy the
    preceding task once and then remain independent. ARROW replay stores one
    int64 task ID per trajectory slot and can condition its otherwise unchanged
    uniform sequence sampling on a homogeneous task; FIFO/LTDM selection keeps
    its configured weights whenever both contain that task and renormalizes only
    over eligible sub-buffers otherwise. Fixed world-model and Actor-Critic
    update totals are split 50 percent current task and 50 percent uniformly
    across replay-available old tasks. A frozen, seeded orthogonal DINOv3 patch
    projection replaces Task-1 PCA and no pixel decoder or residual correction
    is used. Defaults remain one unlabelled RSSM, one Actor-Critic, and the exact
    upstream replay RNG path. Focused tests cover config isolation, task-filtered
    replay, selected-expert gradients, fixed-budget allocation, deterministic
    projection, actor-bank warm starts, and launcher accounting.

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
