# Decision 0012: isolate the complete DINO world model by task

## Status

Accepted for the separately named
`DINO-FullBank-ARROW-v2-Atari-TaskAware` protocol. The implementation is ready
for a target-GPU smoke and acquisition screen; it has no performance result yet.
The failed `MoE-ARROW-v1` path remains unchanged and reproducible.

## Context

The seed-0 two-task MoE-ARROW-v1 pilot failed before retention could be tested.
Its first-task MsPacman policy remained weak while the prior cosine feature loss
became nearly zero and the KL reached its free-bits floor. V1 routed recurrent
dynamics and prediction heads, but its posterior representation and DINO
feature predictor were shared. It also gave a new task only half of the fixed
world-model and Actor-Critic update budgets and copied the preceding policy.

This combination was neither a complete per-task RSSM upper bound nor a strong
acquisition protocol. A separate task expert could still be bottlenecked by a
shared posterior that was not directly grounded in the current observation.

## Decision

Add `dino_fullbank_arrow` as a new fixed protocol with hard scheduler task-ID
routing. Every task independently owns:

- posterior representation;
- recurrent dynamics;
- latent prior;
- posterior DINO feature predictor;
- reward and continuation heads;
- Actor-Critic networks, optimizer, and return statistics.

The only learned-model input component shared across tasks is the frozen DINOv3
ViT-S/16 encoder. The parameter-free `[z,h]` concatenation is also reused.
Task 1 uses only expert 0; allocating six experts does not ensemble or jointly
train multiple experts on the first game.

The observation objective predicts the current stopped `4 x 4 x 64` spatial
DINO feature from the current posterior state. Prediction and target are
standardized independently over time and batch for each feature coordinate,
then compared with SmoothL1. The constant-prediction loss and model-to-constant
ratio remain logged. There is no prior cosine feature objective or pixel
decoder.

On first arrival, a new complete world-model expert copies the preceding expert
once but receives fresh per-parameter Adam state. Its Actor-Critic is initialized
from its own deterministic seed and does not copy the preceding policy. All old
world-model and policy parameters are frozen. Every fixed world-model and
Actor-Critic update in the epoch goes to the current task; old-task rehearsal
gets zero updates. The first collection for every new task is random.

ARROW-50 replay capacity, FIFO/LTDM selection, task duration, total update
counts, and evaluation cadence remain unchanged. Replay still retains task IDs
and old samples, but v2 retention comes from parameter isolation rather than
continued old-task optimization.

## Consequences

This is a task-aware, storage-expanding reference, not a fair task-agnostic
replacement for ARROW-50. It can remove parameter interference but cannot
guarantee good acquisition, transfer, or optimal control. Warm-starting the
world model may still cause negative transfer and requires a fresh-world-model
ablation if v2 fails after the observation path is validated.

Because all updates move to the current task, v2 changes V1's update allocation
but not the total budget. Because the published Atari config disables pretrain
collection multiplication, random collection at later task boundaries adds no
environment interactions. The launcher nevertheless accounts for extra
interactions if a different source config enables that multiplier.

The next evidence gate is a one-task MsPacman acquisition run. A low feature
loss alone is not success: the posterior predictor must beat its constant
baseline and the policy return must recover substantially toward ARROW-50 before
starting a continual campaign.
