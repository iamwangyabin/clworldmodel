# Research Scripts

This directory contains runnable launchers and offline analysis tools. Test
modules live in the repository-level `tests/` directory so this folder only
contains experiment-facing code.

## Launchers

- `run_arrow_ar50_atari.py`: canonical ARROW-50 launcher and named ARROW actor/objective ablations.
- `run_dv3_fifo_atari.py`: matched DreamerV3/FIFO control.
- `run_bounded_dream_rehearsal_atari.py`: DreamerV3 with one shared actor,
  realized-first actor-only dream self-imitation, and a configurable bounded
  reservoir whose default 524,288-transition capacity matches ARROW-50.
- `run_r2dreamer_arrow_atari.py`: native R2-Dreamer plus ARROW replay launcher.
- `train_r2dreamer_arrow_atari.py`: native R2-Dreamer training implementation used by the launcher.
- `run_karrow_ar50_atari.py`: all KARROW visual versions and `dino`/`mlp`/`kan` arms.
- `run_karrow_task2_from_snapshot.py`: isolated Task-2 acquisition diagnostic.
- `run_cnn_projector_lora_incremental.py`: Task-1-snapshot-seeded true
  Boxing/CrazyClimber acquisition with a frozen CNN/RSSM core, per-task
  projectors/RSSM LoRA, and independent Actor-Critics. Named capacity and
  compact `32/32/16` LoRA profiles keep the remaining protocol matched.
- `run_cnn_compact_shared_actor_incremental.py`: Task-1-snapshot-seeded compact
  RSSM adaptation with smaller representation/prior LoRA, GRU-output-only
  correction, and one shared Actor protected by frozen-route imagination
  distillation.
- `run_cnn_mechanism_bank_incremental.py`: Task-1-snapshot-seeded MB-RSSM with
  one frozen CNN/base RSSM, per-task spatial projectors and nonlinear recurrent,
  posterior, and prior mechanisms, optional zero-initialized reuse of frozen old
  mechanisms, private world-model heads, and independent Actor-Critics. The
  `no-reuse` mode is the capacity-matched routing ablation. The same launcher
  owns REC-RSSM through `--method-profile rec-rssm`; there is no separate
  wrapper launcher.
- `run_evolving_atomic_rssm.py`: from-scratch three-task Evolving-Core Atomic
  RSSM with a continually updated shared CNN/RSSM, symmetric per-task
  projectors/atoms/heads, independent Actor-Critics, replay-protected
  component gradients, and boundary consolidation. It supports the three
  predeclared three-task orders plus the separately named original-six-task
  seed-0 pilot. The explicit `compact_128_128_64` original-six profile is a
  separately named mechanism-capacity ablation; the default remains the
  `512/512/256` protocol. The launcher always disables world-model compilation.
  Three-task full curricula default to `fixed_v2`
  (`first_task_shared_core_lr=3e-4`); pass `--task0-profile fixed_v1` to
  reproduce the original `2e-4` protocol. The original-six pilot remains
  fixed to its preregistered `fixed_v1` acquisition setting. The separately
  named `--behavior-profile shared_fastkan_stable` three-task composition uses
  Shared-Frozen-Down Q/F/P mechanisms and one replay-rehearsed FastKAN Actor
  and Critic across all task routes without adding behavior updates. The
  separately named `--prediction-head-profile shared_distilled` profile instead
  preserves full-width Dense Q/F/P and independent MLP Actor-Critics while
  sharing one replay-, distillation-, projection-, and consolidation-protected
  decoder/reward/continue set. Adding `--adaptive-qfp-compression` to that
  original-six profile creates the separately named Dense-acquire protocol:
  every task is learned at full width, all fixed structured-pruning candidates
  receive equal LTDM recovery compute, and a dedicated raw-return cohort selects
  the smallest acceptable physical Q/F/P width with Dense fallback.
- `smoke_evolving_atomic_rssm.py`: target-CUDA, production-shaped synthetic
  update covering the fixed 12-current/4-memory split, component projection,
  frozen old private state, and shared/private/route Adam steps. Its explicit
  compact-mechanism profile allocates the complete six-task compact topology.
  It performs no environment interaction and is evidence of execution
  correctness only. With
  `--behavior-profile shared_fastkan_stable` it also exercises the frozen shared
  down bases and one old/current route update of the single FastKAN pair. With
  `--prediction-head-profile shared_distilled` it verifies prediction-head
  teacher losses, separate projection groups, and shared-head optimizer steps.
  With `--method-profile atomic_lora_shared_heads` it allocates method C's
  production six-task Rank-128 atomic topology and exercises one Task-1 update
  through the shared heads, private residuals, and active reuse routes. With
  `--method-profile adaptive_qfp_compression` it physically compacts Task 1,
  takes a Q/F/P-only recovery step, and strictly reloads the heterogeneous
  checkpoint topology.
- `run_evolving_task0_sweep.py`: fixed-order, seed-0 MsPacman acquisition
  launcher for the preregistered single-LR profiles or the 120/150/180/240
  duration profiles. It omits held-out-final evaluation and stops at the
  declared first-task boundary.
- `run_evolving_shared_down_task0.py`: seed-0 acquisition gate for the
  full-width shared-frozen-down plus private LayerNorm/FiLM/up mechanism. It
  allocates all six original-order routes, trains only MsPacman for 90 epochs,
  and omits held-out-final evaluation.
- `run_evolving_learned_base_adapters.py`: the recorded original-six seed-0
  Rank-32 learned-base/private-prediction-adapter pilot. Its stopped negative
  run remains evidence; do not relabel it as the newer method.
- `run_evolving_atomic_lora_shared_heads.py`: method C, a strictly validated
  post-Task-0 boundary bootstrap that keeps A-style plastic shared prediction
  heads and private MLP behavior while replacing Tasks 1-5 Dense Q/F/P with
  independent Rank-128 four-atom residuals. It restores only the immutable
  Task-0 checkpoint/replay and labels the run non-equivalent/non-from-scratch.
- `select_evolving_task0_profile.py`: require and rank the unchanged control
  plus a complete LR or duration family using only the fixed-cohort
  pre-consolidation raw return. Duration selection chooses the shortest run
  within five percent of the observed maximum.
- `run_moe_arrow_atari.py`: task-aware MoE, CNN-FullBank, DINO-FullBank, DINO-PatchBank, and DINO-ConvBank launchers.
- `verify_arrow_environment.py`: pinned dependency, CUDA, and Atari registration check.

The former version-specific KARROW and DINO wrapper files were removed. Use
explicit selectors instead:

```bash
python scripts/run_karrow_ar50_atari.py --visual-version v3 ...
python scripts/run_moe_arrow_atari.py --method cnn-fullbank ...
python scripts/run_moe_arrow_atari.py --method dino-fullbank ...
python scripts/run_moe_arrow_atari.py --method dino-patchbank ...
python scripts/run_moe_arrow_atari.py --method dino-convbank ...
```

## Audits And Reporting

- `component_forgetting_audit.py`: collect/evaluate frozen held-out audit data.
- `input_fixed_module_forgetting_audit.py`: fixed-input module drift audit.
- `encoder_feature_forgetting_audit.py`: direct image-encoder drift audit.
- `decoder_forgetting_audit.py`: fixed-input decoder drift audit.
- `component_swap_audit.py`: frozen parameter-group restoration audit.
- `latent_region_audit.py`: held-out task-region and RBF-support analysis.
- `component_audit_metrics.py`: shared numerical definitions for the audits.
- `summarize_component_audit.py`: paired audit conclusion data.
- `render_p1_full_audit_dossier.py`: render the completed P1 audit dossier.
- `extract_arrow_baseline_results.py`: convert ARROW logs into a result bundle.
- `summarize_continual_metrics.py`: preserve raw checkpoint matrices and compute
  versioned ARROW-style ACC, min-ACC, WC-ACC, and forgetting reports; it leaves
  FT/sample-efficiency unavailable unless their required reference curves exist.
- `experiment_registry.py`: validate the small, text-only records under
  `docs/experiments/records/` and deterministically rebuild `registry.json` plus
  the human-readable `RESULTS.md`; it rejects weights, Replay, TensorBoard, full
  logs, binaries, and oversized evidence.
- `evaluate_cnn_mechanism_bank_reuse.py`: fixed-cohort Task-3 gate ablations,
  functional mechanism contribution ratios, shared-trajectory latent/reward
  diagnostics, and the epoch-260/270 cross-cohort check. It does not train or
  write evaluation transitions to Replay.

## Shared Support

- `artifact_io.py`: dependency-free checksums and atomic artifact writers used
  by audits and post-hoc probes.
- `git_provenance.py`: clean-commit and upstream-sync checks used by training
  launchers and audits.
- `launcher_support.py`: dependency-free runtime-manifest, JSON, and
  subprocess-log helpers used by standalone launchers.

Run offline audit commands from the repository root. Training launchers should
be inspected with `--dry-run` before any environment interaction.
