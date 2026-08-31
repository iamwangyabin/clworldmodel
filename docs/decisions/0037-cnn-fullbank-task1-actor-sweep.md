# Decision 0037: Budget-matched Task-1 actor sweep

## Status

Accepted for a single-seed Task-1 acquisition pilot. It does not replace a
multi-seed protocol comparison.

## Context

The from-scratch MsPacman run used by later snapshot-seeded continual methods
fixes a single-GPU x4 double-sample profile. Earlier results showed a strong
periodic peak followed by lower heldout performance, so actor optimization is a
more targeted first sweep than changing the world-model capacity or training
budget.

## Decision

Run five concurrent, fixed-seed arms on identical RTX 4080 SUPER devices. The
existing arm remains the control at actor learning rate `2e-4` and entropy
scale `3e-4`. Four named arms form a small grid:

| Profile | Actor LR | Entropy scale |
| --- | ---: | ---: |
| `aclr1e4` | `1e-4` | `3e-4` |
| `aclr5e5` | `5e-5` | `3e-4` |
| `aclr1e4-ent1e4` | `1e-4` | `1e-4` |
| `aclr5e5-ent1e4` | `5e-5` | `1e-4` |

All arms keep the task, seed, environment interactions, ARROW replay, world
model, world-model learning rate, sampled-frame use, optimizer update counts,
precision, and fixed evaluation cohorts unchanged. Each arm has its own output
directory and file-backed Replay directory.

The predeclared selection order is:

1. highest heldout-final raw-return mean after 90 epochs;
2. highest best fixed-validation raw-return mean;
3. lower heldout-final raw-return standard deviation.

No intermediate score may be used to stop, extend, or restart an arm. The final
choice remains a single-seed pilot result and must not be described as a
general hyperparameter optimum.

## Consequences

The sweep can select a stronger Task-1 boundary snapshot for the later REC-RSSM
pilot without confounding extra data or updates. It cannot establish robustness
across seeds, downstream tasks, or accelerator types. Downstream REC-RSSM work
must record exactly which selected snapshot and profile it uses.
