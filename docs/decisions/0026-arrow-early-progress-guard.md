# Decision 0026: Predeclare an early ARROW progress guard

## Status

Accepted as a diagnostic stop rule for the replacement CNN-FullBank Task 1
pilot. It is not a performance-parity or acquisition claim.

## Reference

`tests/fixtures/arrow_ar50_original_s0_early_metrics.json` preserves early
world-model scalars from `arrow_ar50_original_s0_analysis`, project commit
`d812d54049ddc1c289a0dd951c015940094b2ab5`. The source TensorBoard event file
has SHA256
`8538732307542c10822bd4414ba7ffefd7ba3b361b971d3323dac128e527d5f7`.

The reference used one GPU, FP32 compute, float32 GPU replay, and the original
global batch. The current pilot uses BF16, uint8 mmap replay, DDP4, task banks,
and four times the sampled frames per update. Exact loss equality is neither
expected nor claimed.

## Guard

After at least three aligned points between world-model steps 1,000 and 5,000,
the diagnostic compares the median absolute current/reference ratio. Reference
values are floored at `0.001` when forming ratios.

| Metric | Accepted median ratio |
| --- | ---: |
| continuation loss | `0.05` to `10` |
| reconstruction loss | `0.2` to `5` |
| KL loss | `0.2` to `5` |
| world-model gradient norm | `0.1` to `5` |

Any non-finite required value fails immediately. Reward loss and imagined
performance remain informational because their early minibatch variance is too
large for this stop decision.

`scripts/compare_arrow_training_progress.py` reads the active TensorBoard
events, writes a structured comparison, and returns exit code 2 only when
`--enforce-guard` is supplied and the predeclared guard fails. It never kills a
training process itself. A monitor may stop a failed pilot only after recording
the comparison and preserving the run directory; it must not overwrite,
resume, or silently restart that run.

The first replacement run keeps the prior hyperparameters unchanged so the
BF16 continuation fix is isolated. Hyperparameter changes are considered only
after the numerical guard passes and acquisition evidence still warrants a
separately named controlled pilot.
