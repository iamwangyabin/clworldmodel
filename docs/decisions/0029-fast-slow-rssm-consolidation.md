# 0029: Fast–Slow RSSM Consolidation World Model (Proposal)

## Status

Proposed on 2026-08-25. Under discussion; not yet accepted for any run. The
working method name is `FastSlow-WM-ARROW-v1` and is provisional.

## Context

The module-level audits of the canonical seed-0 runs localize residual
forgetting under ARROW-50 (`runs/arrow_vs_dv3_fifo_s0_input_fixed_audit_summary`):

- Encoder geometry is nearly perfectly retained by replay alone
  (CKA loss 0.005), while encoder coordinates still drift (relative RMS
  0.42–0.76). Encoder-targeted protection (e.g. frozen pretrained encoders)
  addresses an already-solved subproblem.
- Residual drift concentrates downstream: posterior symmetric KL 22.0, RSSM
  prior symmetric KL 18.9, critic distribution KL 15.3, and actor top-1 action
  disagreement 0.71 on old tasks at `C6_e539`. Replay preserves perception but
  not dynamics or control.
- The parallel independent-expert campaign (0027) is task-aware systems work
  with no continual-learning claim; it serves as an upper-bound reference, not
  a method.

The proposed method targets exactly the residual: RSSM-level dynamics drift.
Its story is a world-model instantiation of Complementary Learning Systems
(CLS) theory: a fast learner absorbs the current task; a slow backbone
consolidates knowledge through ARROW replay; LTDM replay becomes the
consolidation channel rather than merely a retention device.

## Literature positioning

Scan artifacts: `runs/litscan_fast_slow_cl.csv`, `runs/litscan_lora_cl.csv`,
`runs/litscan_fastslow_rl_wm.csv`, `runs/litscan_cls_cl.csv` (2026-08-25).

- Fast/slow parameter-efficient tuning is established in supervised continual
  learning: Safe (NeurIPS 2024), MoRAL (2024), EWC-LoRA (ICLR 2026), Merge
  before Forget (ICLR 2026), and related LoRA-CL work. These assume a frozen
  pretrained backbone; the slow side never has to learn.
- Fast–slow structure inside a world model's dynamics (RSSM) for model-based
  RL appears unoccupied. Dreamer-style "slow" usage is limited to the EMA
  target critic.
- The mechanism alone is therefore not the claim. The claim is: a from-scratch
  fast–slow dual system in which the slow backbone must itself be consolidated
  from the fast learner over time, with ARROW-50 replay driving consolidation,
  applied to RSSM dynamics in continual MBRL.

## Proposed mechanism

- Shared CNN encoder, trained normally. The audits show replay keeps its
  geometry stable; no freezing, no pretrained backbone.
- RSSM backbone = slow learner. It receives no direct gradients from the
  current task stream.
- Low-rank (LoRA-style) adapters on the RSSM = fast learner. All new-task
  learning is directed into the fast adapters.
- Periodic consolidation into the slow backbone, two candidate variants:
  - A: weight merge of the fast adapters into the backbone (cheap; risks
    accumulating cross-task interference in the backbone);
  - B: replay-driven consolidation — the backbone is trained as a student on
    ARROW replay sequences to match the frozen fast+slow composite teacher
    (prior, posterior, reward, continue), with a geometry-aware objective as
    an option given the observed latent re-coordinatization.
- Fast adapters reset after consolidation. Pilot v1 uses scheduler-known task
  boundaries for consolidation timing (orchestration knowledge only, per
  project rules); learned task-free boundary detection and routing are a later
  protocol and must not be claimed from v1.
- ARROW-50 replay is unchanged: equal FIFO/LTDM capacity and selection.

Actor-critic handling in v1: one shared actor-critic trained in imagination
from the composite model. Actor/critic retention is measured, not yet
protected; per-task or fast–slow control structure is out of scope for v1.

## Budgets, accounting, and claims discipline

- Environment interaction, sampled contexts, and world-model updates match the
  frozen ARROW-50 protocol per task. Consolidation updates are world-model
  updates: they are counted, reported, and budgeted explicitly.
- Parameter accounting: one backbone plus transient fast adapters plus a
  frozen teacher copy during consolidation. Total and peak parameter/byte
  overhead is reported alongside replay-byte accounting.
- v1 supports at most a retention claim on dynamics modules and old-task
  returns under scheduler-known boundaries. It does not support task-agnostic,
  forward-transfer, or backward-transfer claims.
- Matched controls: ARROW-50 (no consolidation) and, as the honesty ablation,
  a geometry-aware dynamics-consistency anchor without the fast–slow split.
  If the anchor baseline closes most of the drift gap, the structural claim
  weakens accordingly and must be reported as such.

## Evaluation plan

- Module metrics per existing audit harness: posterior symmetric KL, RSSM
  prior symmetric KL, encoder CKA, critic distribution KL, actor action
  disagreement, computed against each old task's boundary checkpoint.
- Behavioral metrics: raw per-task returns at every boundary; final average
  performance and forgetting per the frozen metric formulas. Module-drift
  reductions must be connected to old-task return retention.
- Ablations: merge (A) vs distill (B) consolidation; EMA-only slow track as a
  minimal variant; consolidation cadence.
- Report fast-adapter rank/capacity utilization to test whether cross-task
  dynamics differences are low-rank at all.

## Risks

- Low-rank sufficiency of Atari dynamics differences is unverified.
- Merge variant A may accumulate interference; distill variant B spends update
  budget and may inherit teacher drift.
- EMA-only slow tracking is a low-pass filter on drift, not a lock; over a
  six-task curriculum it may still accumulate.
- Consolidation timing in v1 uses known boundaries; task-free detection is the
  next open problem and the main later claim.

## Minimal pilot

Two tasks (MsPacman then Boxing), seed 0, matched budgets against the frozen
ARROW-50 reference data, variant B with variant A as ablation. Success:
old-task prior/posterior KL materially below ARROW-50 at the second boundary
with Task-1 return retention no worse than ARROW-50, at matched update and
interaction budgets. A single-seed pilot establishes mechanism plausibility
only; multi-seed evidence is required before any method claim.
