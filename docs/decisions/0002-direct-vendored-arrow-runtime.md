# Decision 0002: direct vendored ARROW runtime

- Status: accepted
- Date: 2026-08-12

## Decision

Maintain the project's runnable ARROW source directly under
`third_party/arrow` and execute it without generating patched reference trees.
The vendor is based on upstream commit
`cb05e7d97ed83c3cf6e528960db0da6868e29232`; local changes and current file
hashes are recorded inside that directory.

The canonical launcher enables the tensor categorical kernels, compiled
world-model loss, fused Adam, TF32, and set-to-none gradients by default. It
does not change task order, environment steps, update counts, minibatch sizes,
replay capacities, or buffer-selection probabilities.

## Evidence

On the target P4.gpu.large host, the world-model stage decreased from 127.696
seconds eager to 61.504 seconds after compilation warmup. A steady non-eval
epoch took 139.447 seconds. Fixed-seed tests compare sampling, KL values,
gradients, and masked metrics against the corresponding upstream operations.

These runs are labeled `vendored-optimized`; the pristine upstream source is
recoverable from the recorded commit when strict upstream comparison is needed.
