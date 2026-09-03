# Adaptive Q/F/P No-Atom-Output-Regularization Ablation v1 (Atari)

## Scope

The protocol is
`Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFP-SharedDistilledHeads-PrivateMLPAC-NoAtomOutputReg-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.
Its config method key is
`evolving_atomic_rssm_adaptive_compression_shared_heads_no_atom_reg_arrow`.

It inherits the complete curriculum, topology, Replay, optimizer, teacher,
gradient projection, boundary consolidation, adaptive compression, evaluation,
checkpoint, parameter, and compute contracts from
`evolving_core_dense_acquire_adaptive_qfp_compression_v1_atari.md`.

## Sole behavioral change

For an online current-task batch, the control uses

```text
L_current = L_Dreamer + 1e-4 * sum_c E[||A_k^c(x)||_2^2].
```

This ablation uses

```text
L_current = L_Dreamer.
```

The change applies from Task 0 onward. It does not remove old-task real-target
Dreamer loss, posterior/hidden/frozen-Actor interface protection, shared-head
distillation, component-gradient projection, or compression-time Q/F/P output
distillation. It changes no sample, interaction, update, parameter, Replay, or
evaluation budget.

## Launch

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order arrow-original-six \
  --prediction-head-profile shared_distilled \
  --adaptive-qfp-compression \
  --disable-atom-output-regularization \
  --seed 0 \
  --classification pilot \
  --dry-run
```

A non-dry launch requires a clean pushed commit and a successful target-CUDA
smoke. The run is pilot evidence only.

The corresponding synthetic target-CUDA smoke is:

```bash
python scripts/smoke_evolving_atomic_rssm.py \
  --method-profile adaptive_qfp_compression_no_atom_reg \
  --device cuda:0
```
