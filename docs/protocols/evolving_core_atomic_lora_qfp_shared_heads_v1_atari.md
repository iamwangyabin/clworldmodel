# Evolving-Core Atomic Rank-128 Q/F/P + Shared Heads v1 (Atari)

## Scope and protocol name

This document defines the task-aware, seed-0, single-seed pilot:

`Evolving-Core-Task0BoundaryBootstrap-AtomicRank128QFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot`.

It is a **post-Task-0 boundary bootstrap**, not an equivalent checkpoint resume
and not a from-scratch C result. The immutable source is Task 0 from the
learned-base Rank-32 pilot at project commit
`6fef9bdde01b77110606d50b3fa7f9449aae60ac`. Only the post-consolidation Task-0
checkpoint is admissible. A later checkpoint, live Replay containing Boxing,
or the A experiment's post-Task-1 Replay would leak future-task experience and
is forbidden.

## Fixed topology

The continually plastic replay-protected state matches the shared-head Dense
experiment A: CNN encoder, base recurrent/posterior/prior RSSM, latent
interface, decoder, reward head, and continuation head. Each task has one
projector and one independent standard DreamerV3 MLP Actor-Critic.

Task 0 owns the full Dense Q/F/P residual mechanisms with hidden widths
`512/512/256` and four atoms. A later task owns an independent nonlinear
Rank-128 residual at each interface:

`LayerNorm -> in-to-128 -> 128-to-hidden -> SiLU -> hidden-to-128 -> 128-to-out`.

The final `128-to-out` projection is zero initialized. Each rank dimension is
partitioned losslessly into four Rank-32 atoms. The mechanism itself contains
only its private delta. Older Task-0 or later-task atoms enter through the
existing learned `tanh` route gates, initialized at zero. Thus a new task begins
from the plastic shared RSSM base, not from a forced Task-0 Q/F/P correction.

All tasks use one plastic decoder/reward/continue set. Old-task LTDM sequences
retain real pixel/reward/continuation losses and match the frozen cumulative
boundary teacher's outputs at scale `0.1`. No private prediction adapter is
allocated. Prediction heads are separate component-projection groups and are
included in boundary consolidation and rollback.

## Boundary transition contract

The launcher and trainer must verify the source checkpoint SHA-256 sidecar,
artifact/schema, source method, resolved-config difference set, schedule, and
counters. The only allowed config changes are:

- method name;
- Q/F/P reuse, parameterization, and rank;
- private prediction-adapter enable/rank; and
- whether shared prediction heads freeze after Task 0.

The source must report exactly 90 completed epochs, Task ID 0, `5,898,240` raw
frames, `91,000` world-model updates including Task-0 consolidation, and
`72,000` Actor-Critic updates. Source Replay must contain only Task 0. Loading
copies checkpoint-owned mmap observations into new writable mmap files; the
source assets remain immutable.

Shared modules, Task-0 projector, Task-0 Dense Q/F/P, and prediction heads load
by exact name, shape, and dtype. Source-only future Rank-32 mechanisms and
prediction adapters are omitted. Target Rank-128 mechanisms and routes retain
their deterministic target initialization until Task 1 is activated. The
Task-0 Actor-Critic bank entry, return statistics, slow target if present, and
optimizer state load exactly.

World-model Adam state is reset because prediction-head ownership and
future-task parameter groups change. All Python, NumPy, PyTorch CPU/CUDA,
task-selection, collection, validation, and final-evaluation RNG streams are
restored only after target topology construction. A target-shaped frozen
boundary teacher receives the source Task-0 shared/Task-0 state.

The run writes `task0_transition_initialization.json`. Its source checkpoint,
checksum, allowed config delta, copied-state report, optimizer reset, Replay
semantics, counters, and non-equivalent-resume scope are required provenance.

## Training, curriculum, and Replay

Tasks are MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, and Enduro, 90
epochs each. Seed index 0 resolves to `123456789`. The combined curriculum is
540 epochs; the new process begins at completed epoch 90 and executes the
remaining 450 epochs.

ARROW-50 uses FIFO/LTDM capacities `512/512`, 50/50 whole-minibatch selection,
uint8 CPU mmap observations, and task labels only for this named task-aware
protocol. Later world-model updates preserve 12 current-task and four
task-homogeneous LTDM sequences. Environment/evaluation transitions, reward
scaling, action semantics, frame repeat, 1,000 updates per epoch, and 800
Actor-Critic updates per epoch are unchanged. Evaluation data never enters
Replay.

At every remaining task boundary, 1,000 balanced LTDM consolidation updates at
`2e-5` update the shared core and shared prediction heads. Fixed-cohort raw
return gates acceptance with the existing 5% drop rule; rollback restores both
weights and persistent Adam state. The source Task-0 consolidation record and
five target-run records jointly form six boundaries.

## Compute ledger

Combined source plus target protocol budgets remain:

| quantity | value |
|---|---:|
| raw emulator frames | `35,389,440` |
| online world-model updates | `540,000` |
| boundary-consolidation updates | `6,000` |
| total world-model optimizer steps | `546,000` |
| Actor-Critic updates | `432,000` |

The target process inherits `90` epochs/`5,898,240` frames/`91,000` world-model
updates/`72,000` behavior updates and executes the remaining `450` epochs. The
optimizer reset is a semantic deviation even though step counts remain matched.

## Exact online parameter ledger

| component | parameters |
|---|---:|
| base ARROW world model | `19,498,853` |
| six projectors | `205,440` |
| Task-0 Dense Q/F/P | `3,816,192` |
| five Rank-128 Q/F/P residual sets | `6,956,800` |
| active route scalars | `180` |
| **world model** | **`30,477,465`** |
| six private MLP Actor-Critic pairs | `10,295,910` |
| **online total** | **`40,773,375`** |
| FP32 parameter bytes | `163,093,500` |

Per-task world-model additions are `3,850,432`, `1,425,612`, `1,425,624`,
`1,425,636`, `1,425,648`, and `1,425,660`. Optimizer state, gradients,
activations, Replay, checkpoints, and the one training-only boundary teacher
are outside the online ledger.

## Metrics and evidence boundary

Fixed evaluation reports episodic return. Report raw mean and standard
deviation per seen task first, reward-scaled values separately, and ARROW-paper
normalization only with the fixed cited constants. The evaluation at completed
epoch 90 supplies the inherited Task-0 boundary row before any Boxing model
update; future-task diagnostic returns are not learned-task performance.

After three tasks the pilot may reject or provisionally support the mechanism.
Final ACC, minimum/worst-case ACC, forgetting, and valid transfer metrics
require all six boundary rows. One composite seed cannot establish superiority
over A, Dense Evolving-Core, or ARROW.

## Launch gate

The real launch requires a clean target commit already pushed and synchronized,
verified source checkpoint and boundary-snapshot checksums, a passing focused
test suite, a CUDA smoke on the 3090, sufficient independent Replay storage,
and a dry-run manifest. The launcher refuses to overwrite an existing output
directory.
