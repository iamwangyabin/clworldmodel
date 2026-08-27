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
26. Add the opt-in, separately named
    `DINO-FullBank-ARROW-v2-Atari-TaskAware` correction without changing V1.
    The hard task route now owns the posterior representation and DINO feature
    predictor in addition to recurrent dynamics, prior, reward, and
    continuation modules. The selected posterior reconstructs the current
    stopped spatial DINO feature with batch-standardized SmoothL1 and logs the
    constant-prediction baseline. On each task boundary the complete prior
    world-model expert is copied once with fresh parameter optimizer state,
    while the new Actor-Critic uses fresh deterministic weights. Old experts
    and policies are frozen, and 100 percent of the unchanged world-model and
    Actor-Critic update totals go to the current task. Every new task starts
    with random collection; the launcher accounts for interactions under the
    source pretrain setting. Focused tests cover protocol rejection,
    posterior/feature-head routing and gradients, frozen old parameters, fresh
    actors, current-only allocation, and launch-manifest semantics.
27. Add the opt-in, separately named
    `DINO-PatchBank-ARROW-v3-Atari-TaskAware` path without changing V2. The
    frozen DINOv3 encoder retains its complete `16 x 16 x 384` patch tensor and
    supplies the flattened 98,304 coordinates to the existing task-routed
    Dreamer posterior projection. The fixed pooling and channel projection are
    removed, and each task route owns a restored pixel reconstruction decoder
    instead of a DINO feature head. Frozen float16 features remain aligned to
    unchanged ARROW replay decisions; the launcher records their 96-GiB
    sidecar cost. Focused tests cover full-token preservation, fixed-protocol
    rejection, selected posterior/decoder gradients, and launch accounting.
28. Back only the `DINO-PatchBank-ARROW-v3-Atari-TaskAware` replay observations
    and frozen feature sidecars with run-local shared mmap files. This avoids
    exceeding the target container's 32-GiB memory cgroup while preserving the
    existing float32/float16 dtypes, tensor shapes, FIFO/LTDM retention, write
    maps, and sampled indices. Other methods retain their prior in-memory replay
    allocation. Focused tests compare mmap writes and samples with the existing
    CPU tensor semantics and verify launcher byte accounting.
29. Replace the V3 full-patch replay sidecar with on-the-fly frozen DINOv3
    encoding of the already sampled ARROW observation minibatch. The interrupted
    `d52bb17` Task-1 pilot showed that the 96-GiB sidecar on the target SeaweedFS
    FUSE mount exceeded the 32-GiB cgroup working set and caused sustained major
    faults rather than usable accelerator throughput. Float32 ARROW observations
    remain file-backed with unchanged shapes and replay decisions. Recomputed
    features round through the configured float16 cache dtype before returning
    to float32 RSSM input, matching the superseded sidecar's numeric interface.
    Focused tests compare cached and on-the-fly samples under identical replay
    RNG state and verify the launch manifest reports zero feature-storage bytes.
30. Add the opt-in, separately named
    `DINO-ConvBank-ARROW-v4-Atari-TaskAware` path without changing V3. A single
    project-owned trainable adapter reshapes each detached full
    `16 x 16 x 384` DINO patch grid, applies standard
    `Conv2d(384,64,3,2,1)`, channel-only LayerNorm, and SiLU, then supplies the
    flattened `8 x 8 x 64` result to the existing posterior. The 4,096-wide
    interface restores the original Dreamer CNN embedding width while retaining
    the V3 pixel decoder, KL, reward/continue, replay, and Actor-Critic losses.
    The adapter is one shared plastic module across task-banked RSSMs; DINO and
    old task experts remain frozen, so shared-adapter drift is an explicit
    retention risk rather than hidden task isolation. Focused tests cover exact
    shape/layer/parameter contracts, stopped DINO gradients, adapter and active
    expert gradients, frozen old experts, config isolation, and launch resource
    accounting.
31. Add the opt-in, separately named
    `DINO-ConvBank-ARROW-v4-BF16AMP-Atari-TaskAware` execution profile without
    changing the V4 FP32/TF32 default. CUDA DINO, RSSM, decoder, Actor, and
    Critic kernels run under BF16 autocast while model parameters, Adam state,
    categorical sampling and KL, symlog transforms, reconstruction and behavior
    losses, lambda returns, and value targets remain FP32. On-the-fly DINO
    features stay BF16 through the shared convolution instead of performing the
    V4 float16-to-float32 round trip, and the DINO execution chunk grows from
    128 to the unchanged 512-frame optimization batch. Focused tests cover
    config isolation, launcher budget and dtype accounting, BF16 feature flow,
    FP32-sensitive math, and unchanged FP32 defaults.
32. Supersede the DINO-ConvBank FP32 default with the separately reported
    `DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-Atari-TaskAware` execution and
    storage profile. The DINO-ConvBank launcher now selects and requires BF16
    autocast with a 512-frame encoder chunk while retaining FP32 parameters,
    Adam state, and sensitive probability/target math. Add an explicit replay
    observation dtype whose default remains float32 for all prior methods;
    DINO-ConvBank alone stores source Atari pixels as uint8 mmap tensors and
    restores the prior float32 `[0,1]` model interface after sampling. FIFO
    overwrite order, LTDM random-key decisions, task filtering, replay
    capacity, and update budgets remain unchanged. Focused tests cover exact
    sampled-value parity, FIFO wraparound, LTDM retention, mmap bytes, config
    isolation, launcher accounting, and FP32 rejection.
33. Add separately named fixed-global-batch `DP2` and `DP4` execution profiles
    for DINO-ConvBank only. The launcher uses `torch.distributed.run`, NCCL, and
    native PyTorch DistributedDataParallel. Rank 0 retains sole ownership of
    collection and the authoritative FIFO/LTDM replay; it makes one unchanged
    global buffer choice and index draw, then scatters equal sequence-axis
    slices to all ranks. The regular world-model `N=16` becomes local `N=8/4`,
    and Actor-Critic context `N=128` becomes local `N=64/32`, so global sample
    counts and optimizer-update budgets remain fixed. Frozen DINO encoding and
    model losses run per rank, Actor-Critic normalization uses gathered global
    returns, DDP averages gradients, evaluation tasks are sharded, and only
    rank 0 writes artifacts. Focused CPU tests cover partitioning, configuration
    isolation, torchrun commands, and manifest budgets; target-GPU 2/4-rank
    smoke runs remain required before official use.
34. Add the opt-in, separately named
    `CNN-FullBank-ARROW-v1-BF16AMP-Uint8Replay-Atari-TaskAware` path. It restores
    the vendored four-layer Dreamer CNN and pixel reconstruction objective while
    extending the hard task route to own the image encoder as well as posterior,
    recurrent dynamics, prior, decoder, reward/continue heads, and Actor-Critic.
    A new task copies the preceding complete world-model route once; its
    Actor-Critic is fresh, only the current route receives the unchanged update
    budgets, and all old parameters remain frozen. The method requires uint8
    file-backed observation replay and BF16 autocast and reuses the fixed-global-
    batch 1/2/4-device DDP execution without DINO or a feature sidecar. Focused
    tests cover strict config isolation, encoder copy/routing/gradients, frozen
    old encoders and decoders, 2/4-device acceptance, no-DINO launching, exact
    parameter accounting, and unchanged global budgets.
35. Add the opt-in CNN-FullBank `task_cosine_decay` Actor-Critic schedule and
    fixed-cohort evaluation snapshots for the separately named late-stability
    pilot. The default schedule and advancing evaluation-seed stream remain
    unchanged. The pilot keeps all interaction, replay, batch, sampled-frame,
    and update budgets, holds Actor-Critic LR/entropy constant through task
    epoch 40, and cosine decays them to fixed endpoints at task epoch 90. A
    fixed periodic-validation cohort is disjoint from the held-out final
    cohort. Exact evaluated world-model and complete Actor-Critic-bank weights
    are atomically saved with checksums and explicitly omit the state required
    for resumable training. Focused tests cover seed-stream separation, the
    task-local schedule and reset, non-resumable snapshot payloads, protocol
    rejection, and launcher accounting.
36. Keep the continuation Bernoulli head numerically stable under the opt-in
    BF16 profiles. The head now exposes logits, computes sigmoid probabilities
    and `binary_cross_entropy_with_logits` in FP32, and returns FP32 continue
    probabilities to imagined rollouts. Previously the terminal sigmoid ran
    inside autocast and could round a large positive logit to exactly `1.0`
    before the nominal FP32 BCE. A stopped CNN-FullBank diagnostic showed the
    resulting loss quantized in `100 / (T * N)` increments and 28.8 times the
    single-device FP32 ARROW reference at world-model step 10,000. Parameter
    shapes, state-dict tensor keys, update budgets, and FP32 behavior remain
    unchanged. A focused regression test forces a saturated BF16 logit and
    verifies a finite FP32 probability, logit-space loss, and gradient.
37. Require immutable per-task inference snapshots for CNN-FullBank training.
    Immediately after each task's final updates and before schedule advance,
    rank 0 atomically saves the complete world-model bank, complete
    Actor-Critic bank, completed task actor, counters, resolved config, and the
    exact project Git commit. A SHA256 sidecar and atomic index record every
    boundary and reject overwrites or a mid-run commit change. These artifacts
    explicitly omit optimizer, replay, RNG, and schedule state and are not
    resumable checkpoints. Focused tests cover payload completeness, atomic
    files, duplicate rejection, commit provenance, and launcher accounting.
38. Add an opt-in independent-expert task index and non-negative fixed-
    evaluation seed offset for separately named CNN-FullBank components. The
    explicit index permits exactly one scheduled environment while retaining
    the complete six-slot world-model allocation; only local route 0 trains and
    its original assembly slot is recorded. The offset advances only the
    isolated periodic-validation and held-out-final seed generators before a
    child task chooses its cohort, so the local task uses its original six-task
    cohort slot. Collection RNG, interaction, gradients, and default sequential
    behavior remain unchanged. The offset is rejected with advancing seeds or
    when it disagrees with the assembly slot. Focused tests cover config
    validation, global-slot equivalence, and launcher provenance.
39. Add the separately named, task-aware
    `CNN-Projector-RSSM-LoRA-ARROW-v1-Task1SnapshotSeeded` path. It imports only
    a completed CNN-FullBank Task-1 inference route and policy, explicitly
    resets Replay/optimizers/RNG/schedule state, and begins new environment
    interaction at Task 2. The Task-1 CNN and RSSM become immutable. Every later
    task owns a zero-effect residual spatial projector, affine RSSM LoRA at
    recurrent/representation/transition ranks `128/128/32`, private pixel and
    reward/continuation heads, and a fresh Actor-Critic. Normal Dreamer losses
    and current-task-only ARROW updates replace the privileged teacher used by
    the preceding posthoc capacity probe. Later LoRA routes share the exact
    frozen Task-1 base parameters rather than storing another full RSSM copy.
    Focused tests cover zero-effect initialization, base sharing,
    selected-route gradients, frozen Task-1 tensors, strict config isolation,
    and snapshot-seeded launcher semantics.
40. Add the separately named, task-aware
    `CNN-Projector-CompactRSSM-SharedActor-ARROW-v1-Task1SnapshotSeeded`
    feasibility path. Later tasks retain the zero-effect spatial projector,
    reduce representation LoRA to rank 32, keep transition LoRA at rank 32,
    and replace recurrent matrix LoRA with a project-owned bottleneck-32
    correction applied only to the frozen GRU output. One Actor is shared
    across tasks. Before each new task it is copied once as a transient frozen
    teacher; every fourth current-task Actor update adds an FP32
    `KL(teacher || student)` target on states generated from zero initialization
    by frozen old RSSM routes. No old real or evaluation transition enters this
    loss. The launcher separately accounts for the extra imagined states and
    makes no matched-compute or task-agnostic claim. Focused tests cover
    zero-effect recurrent adaptation, base sharing, strict config isolation,
    Actor KL targets, single-Actor launch contracts, and launch budgets.
41. Add the fixed `32/32/16` recurrent/representation/transition LoRA profile
    for the separately named
    `CNN-Projector-RSSM-CompactLoRA-ARROW-v2-Task1SnapshotSeeded` ablation.
    The strong Task-1 snapshot, zero-effect spatial projector, private
    world-model heads, independent Actor-Critics, current-task ARROW Replay,
    interaction and optimizer budgets, precision, and fixed evaluation cohorts
    remain identical to the matched `128/128/32` pilot. Configuration rejects
    unnamed intermediate ranks. Runtime accounting must report 643,648 FP32
    RSSM adapter parameters per later task, a 73.8 percent reduction. Focused
    tests cover exact profile selection, config isolation, and rejection of
    unnamed rank tuples.
42. Add the separately named, task-aware
    `CNN-MechanismBank-RSSM-ARROW-v1-Task1SnapshotSeeded` path. It retains one
    frozen Task-1 CNN and base recurrent/posterior/prior RSSM, then applies
    project-owned zero-output nonlinear residual mechanisms at the recurrent
    state and raw posterior/prior logits. Every later task owns one full new
    mechanism in each location. Independent zero-initialized tanh gates may add
    or subtract frozen older mechanisms without scaling the new mechanism; the
    capacity-matched NoReuse ablation stores the same route tensors but freezes
    them at zero. Spatial projectors, decoder/reward/continue heads, and
    Actor-Critics remain task-private, with fresh Actors and previous-task head
    initialization. The original Dreamer losses, current-task ARROW Replay,
    budgets, and default methods remain unchanged. Precomputed observation
    features now retain their task id when entering a task projector, fixing a
    latent routing error exposed by the new path. Focused parity and method tests
    cover exact zero effect, one-time normalization of composed logits, strict
    configuration isolation, gradient allowlists, frozen old tensors, nonzero
    zero-gate gradients, parameter accounting, and launcher semantics.

## Known issues at import

1. Every Atari ARROW/DV3 JSON config contains seven keys missing from
   `Code/ARROW_and_DV3/Atari/config.py`, so config construction raises
   `TypeError` before training.
2. `--arrow-replay-ratio` defaults to `50-50` rather than `None`, which makes
   the CLI overwrite a value loaded from a config even when no override was
   supplied.
3. The upstream world model and actor use unconditional `.cuda()` calls, and
   the published replay configuration stores approximately 24 GiB of float32
   observations on the accelerator.
4. The upstream commit has no automated test suite or CI definition.

Items 1 through 3 are corrected by the documented local compatibility and
runtime profiles; item 4 remains a constraint of the upstream implementation.
