# Evolving-Core Atomic RSSM ARROW v1 (Atari)

## Scope

`Evolving-Core-Atomic-RSSM-ARROW-v1-Atari-TaskAware` is a separately named,
from-scratch continual-learning pilot. It does not replace ARROW-50,
CNN-FullBank, MB-RSSM, or REC-RSSM. Scheduler task identity selects private
routes and Actor-Critics, so this protocol is task-aware and makes no
task-agnostic claim.

The main order is MsPacman, Boxing, CrazyClimber. Two predeclared order checks
use Boxing, MsPacman, CrazyClimber and CrazyClimber, Boxing, MsPacman. Each task
lasts 90 epochs. Observation preprocessing, reward scaling, action semantics,
termination handling, environment repeat, collection size, and the ARROW-50
FIFO/LTDM capacity and selection probabilities remain inherited from the
published Atari config.

## Model Ownership

All tasks share one continually updated CNN encoder and one base posterior,
recurrent, and prior RSSM. `zh_transform` is also in the shared interface group;
in the current Dreamer implementation it is parameter-free concatenation, so
that group has zero trainable tensors. No task receives a copied base RSSM.

Every task, including Task 0, owns:

- a zero-effect residual spatial projector;
- one zero-output recurrent, posterior, and prior mechanism divided into four
  lossless atoms;
- private decoder, reward, and continuation heads; and
- a fresh independent MLP Actor-Critic and optimizer.

Mechanism widths are `512/512/256`, residual scale is `0.1`, and the projector
bottleneck is 64. Task `k` may reuse each older task's atoms through independent
zero-initialized `tanh` gates. The current task mechanism always has coefficient
one. Completed projectors, mechanisms, heads, gates, and Actor-Critics are
frozen; only the shared core and current task private state remain plastic.

## Online World-Model Update

Task 0 uses all 16 sequences for its current-task Dreamer loss. Later tasks use
12 task-homogeneous current sequences from the normal ARROW mixed replay and
four task-homogeneous memory sequences from one uniformly selected completed
task's LTDM entries. The two batches use separate forwards because the RSSM
accepts one scalar task route.

The current objective is the unchanged Dreamer loss plus
`1e-4 * sum_c E[||A_k^c(x)||_2^2]`. Memory uses the old task's Dreamer loss and
three fixed interface terms against the previous accepted boundary teacher:
posterior categorical KL `0.1`, layer-normalized hidden MSE `0.05`, and KL
between the frozen old Actor's policies on teacher/student latent states `0.05`.
Teacher and student memory forwards receive paired latent-sampling RNG states.
The old Actor is evaluated through its inputs but is never updated.

For each named shared component—encoder, posterior, recurrent, prior, and the
latent interface—the current gradient is projected only when its dot product
with the memory gradient is negative. The final shared gradient is the
projected current gradient plus the memory gradient. Current projectors,
mechanisms, heads, and reuse gates receive the complete unprojected current
gradient. Component dot products, conflict flags, and projected norms are
persisted.

One Adam optimizer owns the shared core from epoch 0 onward. Its state is never
rebuilt; its learning rate is `2e-4` on Task 0 and `1e-4` thereafter. Each task
gets one persistent private Adam at `2e-4`; reuse routes use a separate Adam at
`1e-3`. Actor-Critic optimizers remain inside the independent task bank.

## Boundary Consolidation And Safety

After every completed task, the trainer first writes a complete resumable
pre-consolidation checkpoint. It then freezes all private state and performs
1,000 round-robin, task-balanced LTDM Dreamer updates on only the shared core at
`2e-5`. The same fixed 16-rollout validation cohort evaluates every seen task
before and after. If any raw return falls by more than five percent relative to
`max(abs(pre_return), 1)`, both shared weights and shared Adam state roll back.
An accepted core becomes the next boundary teacher.

Evaluation, JSON, or consolidation failures restore the completed
pre-consolidation shared state and optimizer. They are recorded separately and
do not turn atom pruning or route validation into an online training failure.
Atom pruning and route ablation are offline analyses, not part of this method.

Each pre/post checkpoint stores the world model (including routes and masks),
boundary teacher, shared/private/route optimizers, complete Actor-Critic bank
and optimizers, FIFO/LTDM contents and retention indices, Python/NumPy/PyTorch
CPU/CUDA RNG states, environment-seed generators, schedule position, task,
epoch, and distinct frame/world-model/Actor-Critic counters. Mutable replay
mmaps are copied to checksum-protected checkpoint-owned assets; restore copies
them into new working storage so resumed training cannot mutate its checkpoint.

## Budgets And Comparison Rules

Online acquisition retains the original 16-sequence world-model batch and
optimizer-update count. Across 270 epochs this is 270,000 online world-model
updates and 216,000 Actor-Critic updates. The method additionally performs
3,000 consolidation updates and samples 48,000 consolidation sequences. This
extra compute must be reported and is not matched to plain ARROW, FullBank, or
frozen MB/REC unless a comparison explicitly adds the same budget.

Replay reports both trajectory capacity and actual bytes. Observations use
uint8 CPU file-backed storage; actions, rewards, continuation, reset, task-id,
priority, and indexing overhead are reported separately. Evaluation
transitions never enter Replay or affect optimization.

The first campaign compares plain shared ARROW, FullBank, frozen MB/REC, and
this method on all three declared orders. Reports retain raw per-task returns,
acquisition curves, final average performance, forgetting, shared-core drift,
per-component conflict rates, and atom-reuse ablations. A smoke or one seed is
execution/pilot evidence only, never a reproduction or superiority claim.

## Launch

Inspect a resolved launch without environment interaction:

```bash
python scripts/run_evolving_atomic_rssm.py \
  --task-order mspacman-boxing-crazyclimber \
  --seed 0 \
  --classification pilot \
  --dry-run
```

Non-dry runs enforce the repository clean, committed, pushed, and synchronized
Git provenance gate. Evolving-Core deliberately disables world-model
compilation because its component-wise `autograd.grad` ownership is explicit.

Run the production-shaped synthetic CUDA gate before starting a campaign:

```bash
python scripts/smoke_evolving_atomic_rssm.py --device cuda:0
```

This smoke performs one optimizer update without environment interaction. It
must be recorded as `smoke`, never as pilot or performance evidence.
