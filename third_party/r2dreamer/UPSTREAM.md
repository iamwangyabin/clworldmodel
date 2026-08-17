# R2-Dreamer vendored source

- Source: `https://github.com/NM512/r2dreamer`
- Commit: `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`
- Imported: 2026-08-17
- License: MIT, copyright Naoki Morihira
- Local policy: project-maintained vendor used by the named
  `R2Dreamer-ARROW-50` method.

`MANIFEST.sha256` fingerprints this current vendored tree. This file and the
manifest are project metadata, not upstream source. The pristine upstream
source remains recoverable from the commit above.

## Imported upstream files

- `networks.py`
- `rssm.py`
- `distributions.py`
- `tools.py`
- `optim/agc.py`
- `optim/laprop.py`
- `optim/__init__.py`
- `LICENSE`

The project-owned agent and ARROW replay adapter live under
`src/clworldmodel/`; the upstream online trainer, environment wrappers,
TorchRL buffer, and Hydra launch files are intentionally not imported.

## Local modifications

1. Convert internal imports to package-relative imports so this vendor can
   coexist with ARROW's flat Python modules.
2. Add `compat.py` and route RMSNorm uses through it. The fallback preserves
   the upstream affine parameter layout on environments whose pinned PyTorch
   version predates `torch.nn.RMSNorm`.
3. Exclude the upstream TorchRL buffer and online trainer from this vendor.
   The project-owned ARROW adapter reproduces R2's one-step context and latent
   replay-state update contract while leaving ARROW retention policy separate.

These changes are covered by project-owned structural and fixed-formula tests.
No algorithmic weights, layer shapes, loss formulae, or optimizer equations
are changed in the imported source.
