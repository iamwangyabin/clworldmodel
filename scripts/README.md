# Research Scripts

This directory contains runnable launchers and offline analysis tools. Test
modules live in the repository-level `tests/` directory so this folder only
contains experiment-facing code.

## Launchers

- `run_arrow_ar50_atari.py`: canonical ARROW-50 launcher and named ARROW actor/objective ablations.
- `run_dv3_fifo_atari.py`: matched DreamerV3/FIFO control.
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
  `no-reuse` mode is the capacity-matched routing ablation.
- `run_evolving_atomic_rssm.py`: from-scratch three-task Evolving-Core Atomic
  RSSM with a continually updated shared CNN/RSSM, symmetric per-task
  projectors/atoms/heads, independent Actor-Critics, replay-protected
  component gradients, and boundary consolidation. It supports the three
  predeclared three-task orders plus the separately named original-six-task
  seed-0 pilot. The explicit `compact_128_128_64` original-six profile is a
  separately named mechanism-capacity ablation; the default remains the
  `512/512/256` protocol. The launcher always disables world-model compilation.
- `smoke_evolving_atomic_rssm.py`: target-CUDA, production-shaped synthetic
  update covering the fixed 12-current/4-memory split, component projection,
  frozen old private state, and shared/private/route Adam steps. Its explicit
  compact-mechanism profile allocates the complete six-task compact topology.
  It performs no environment interaction and is evidence of execution
  correctness only.
- `run_evolving_task0_sweep.py`: fixed-order, seed-0 MsPacman acquisition
  launcher for the preregistered single-LR profiles or the 120/150/180/240
  duration profiles. It omits held-out-final evaluation and stops at the
  declared first-task boundary.
- `run_evolving_shared_down_task0.py`: seed-0 acquisition gate for the
  full-width shared-frozen-down plus private LayerNorm/FiLM/up mechanism. It
  allocates all six original-order routes, trains only MsPacman for 90 epochs,
  and omits held-out-final evaluation.
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
- `evaluate_cnn_mechanism_bank_reuse.py`: fixed-cohort Task-3 gate ablations,
  functional mechanism contribution ratios, shared-trajectory latent/reward
  diagnostics, and the epoch-260/270 cross-cohort check. It does not train or
  write evaluation transitions to Replay.

## Shared Support

- `git_provenance.py`: clean-commit and upstream-sync checks used by training
  launchers and audits.

Run offline audit commands from the repository root. Training launchers should
be inspected with `--dry-run` before any environment interaction.
