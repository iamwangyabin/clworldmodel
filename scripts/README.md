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

## Shared Support

- `git_provenance.py`: clean-commit and upstream-sync checks used by training
  launchers and audits.

Run offline audit commands from the repository root. Training launchers should
be inspected with `--dry-run` before any environment interaction.
