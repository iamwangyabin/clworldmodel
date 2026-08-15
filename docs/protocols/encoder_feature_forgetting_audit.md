# Direct Encoder-Feature Perturbation Audit

## Scope

This supplementary, offline diagnostic measures the direct output drift of the
DreamerV3 image encoder in the completed DreamerV3/FIFO P1 pilot. It replaces
neither the full module-forgetting profile nor a future causal intervention.
It gives a more direct answer than a geometry-only similarity score:

> For exactly the same old observations, how much do later image-encoder
> outputs depart from the old image-encoder outputs in the old coordinate
> system?

For old task `T_i`, held-out observations `x(t)`, old boundary checkpoint
`C_i`, and later boundary checkpoint `C_j`, the audit computes:

```text
e_i(t) = E_i(x(t))
e_j(t) = E_j(x(t))
```

The primary per-chunk metric is:

```text
relative RMS perturbation = RMS(e_j - e_i) / max(RMS(e_i), 1e-8)
```

The raw numerator is also retained. No feature centering, CKA, Procrustes
alignment, decoder, posterior, RSSM state, imagined rollout, parameter update,
or environment interaction occurs after the direct image-embedder evaluation.

## Experimental Control

The exact held-out natural chunks, old observations, task boundaries,
checkpoint selection, burn-in convention, and episode-cluster bootstrap are
shared with the input-fixed module audit. `Cfinal_e540` is excluded by default
because it follows one additional Task 1 update after the sixth-task boundary.

For each old task, compare `C_i` with every later task-boundary checkpoint.
Aggregate over the retained chunks with an episode-cluster bootstrap; preserve
the raw per-chunk values and confidence intervals.

## Interpretation

A value of zero means that the direct encoder outputs are identical. A relative
RMS perturbation of one means that the encoder-output change has the same RMS
magnitude as the old encoder output itself on those old images.

This is the primary descriptive encoder-forgetting measurement because it does
not deliberately treat a coordinate change as equivalent. CKA and Procrustes
remain useful secondary diagnostics of feature geometry, but they are not
substitutes for direct coordinate perturbation.

The audit establishes encoder-output drift, not its causal impact on control or
return. To test functional compatibility separately, freeze the old posterior,
old RSSM state, and old decoder while substituting only the later encoder output
into that old world-model path.

## Reporting Limits

The P1 source launch had a dirty worktree and uses a single seed. This audit is
pilot evidence only; it is not an official multi-seed result.
