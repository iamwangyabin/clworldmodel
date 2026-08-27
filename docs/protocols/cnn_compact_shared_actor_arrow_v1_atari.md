# CNN-Compact-SharedActor-ARROW-v1 Atari Task-Aware Protocol

## Claim boundary

This is a Task-1-snapshot-seeded, task-aware, single-seed feasibility pilot.
The scheduler supplies task identity to select a world-model route during
training and evaluation. It is not a task-agnostic result, not an equivalent
resume of the source run, and not compute-matched to the earlier projector/LoRA
pilot because it adds old-route imagination.

## Persistent model

Task 0 imports the completed MsPacman CNN, base RSSM, world-model heads, and
Actor-Critic. Later tasks reuse the frozen CNN and base RSSM. Each later route
adds:

```text
pixels -> frozen Task-0 CNN -> private residual spatial projector
       -> frozen Task-0 RSSM
          + rank-32 representation LoRA
          + rank-32 transition LoRA
          + bottleneck-32 correction of the GRU output
       -> one shared Actor
```

There is no recurrent matrix LoRA. The GRU-output adapter is
`LayerNorm(512) -> Linear(512,32) -> SiLU -> Linear(32,512)` with a zero
initialized final layer, so a new route initially equals the Task-0 recurrent
function. The actor has one persistent copy and does not grow per task.

Pixel decoder, reward, continuation, and critic state support training. The
critic does not select evaluation actions. Parameter reports separate
persistent Actor state from the one transient teacher Actor.

## Actor retention

Before a new task, the current shared Actor is copied once, frozen, and used as
a temporary teacher. For every fourth one of the unchanged 400 current-task
Actor-Critic optimizer updates:

1. Select an old task route, alternating old routes during Task 3.
2. Start the frozen old RSSM route from zero state.
3. Roll the teacher policy for 16 burn-in transitions.
4. Retain the next 16 imagined states for a batch of 128 streams.
5. Add `KL(teacher || shared_actor)` with scale 1.0 to the current update.

The temporary teacher is replaced at the next boundary and is never a
per-task inference asset. No old real observation, action, reward, Replay
sample, or evaluation transition enters this retention loss. Zero-state
imagination may cover less of the old policy state distribution than
replay-conditioned imagination; this limitation must be reported.

## Budgets and data flow

The source boundary contributes weights only. Replay, optimizer, RNG, and
schedule state restart at Boxing. World-model and current-task Actor-Critic
budgets remain 500 and 400 optimizer updates per epoch. Across 180 post-Task-1
epochs, actor retention adds 18,000 distillation batches, 36,864,000 retained
imagined states, and 36,864,000 burn-in state uses. This extra model compute is
recorded separately and is not treated as a fair-compute improvement.

ARROW-50 FIFO/LTDM capacity, current-task environment interaction, BF16
autocast, FP32 parameters/Adam/sensitive math, uint8 mmap Replay, and fixed
16-rollout deterministic evaluation cohorts remain unchanged. Evaluation data
never enters Replay or updates.

## Required evidence

- clean pushed Git provenance and exact source-boundary SHA256;
- one GPU training context and explicit CPU thread limits;
- exact adapter, shared Actor, transient teacher, and imagination accounting;
- no `save_ac_bank.pt` and exactly one persistent `save_ac.pt`;
- frozen old world-model routes at every later-task update;
- fixed-cohort taskwise raw acquisition, retention, and forgetting;
- final and boundary weights, evaluation snapshots, manifests, TensorBoard,
  parameter accounting, and SHA256 sidecars; and
- `resumable=false` on snapshots that omit optimizer, Replay, RNG, and schedule
  position.

## Completed Seed-0 Result

The three-task pilot completed successfully at 270 epochs. On the held-out
final cohort of 16 deterministic rollouts per task, raw return mean +/- standard
deviation was MsPacman `1393.125 +/- 201.4857`, Boxing
`32.125 +/- 10.1049`, and CrazyClimber `20400.0004 +/- 4241.6300`.

Runtime accounting reported 37,691,055 world-model parameters and one
1,715,985-parameter shared Actor-Critic, for 39,407,040 parameters in total.
The result remains single-seed and task-aware, and the additional old-route
imagination is not compute matched to the independent-Actor pilots. Curated
metrics and provenance are in
[`references/cnn_incremental_seed0_results_20260827.json`](references/cnn_incremental_seed0_results_20260827.json).
No checkpoint, inference snapshot, or weight file is committed to Git.
