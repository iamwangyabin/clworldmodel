# AWM-AutoRoute v4 CoinRun pilot

## Identity

- Method: **AWM-AutoRoute**; this is the same maintained two-frame router as
  [the Atari v4 protocol](awm_autoroute_v4_atari.md), not a new method.
- Protocol: `Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-TwoFrameProbabilityRouter-ARROWParity-v4-OriginalSix-ProcgenCoinRun-541EpochRevisit-TaskAwareTraining-TaskIDFreeInference-Pilot`.
- Entry point: `scripts/run_evolving_atomic_rssm_d_autoroute_coinrun.py`.
- Runs start from scratch. Historical first-frame CoinRun checkpoints and
  results cannot resume or be relabelled as this protocol.

## Benchmark contract

The fixed task sequence is `CoinRun`, `CoinRun+NB`, `CoinRun+NB+RT`,
`CoinRun+NB+RT+GA`, `CoinRun+NB+RT+GA+MA`, and
`CoinRun+NB+RT+GA+MA+CA`. Each task receives 90 epochs. Epoch 541 revisits
Task 0 without acquiring, consolidating, or compressing a seventh expert.

Observations are native `64 x 64 x 3` uint8 frames with no resizing. The
action space has 15 native actions, frame repeat is 1, and action 4 (the empty
Procgen action) is the dummy previous action. Rewards are unscaled. Native
autoreset frames are cached and consumed once under the shared collector's
NextStep contract. Procgen is pinned to commit
`5e1dbf341d291eff40d1f9e0c0a0d5003643aebf`.

## Routing and training

Training and Replay use scheduler task labels. Interaction and reported
evaluation do not. For every episode, acquired candidates score the first two
observations using their deterministic RSSM posterior probability vectors and
the shared reconstruction decoder. The first choice acts immediately, the
second observation may revise it, and no candidate is rescored afterward.
The v4 policy-state rules, ties, eligibility, and diagnostic denominators are
identical to Atari v4, except for CoinRun's 15-action interface and dummy
action 4. On the final Task-0 revisit all six acquired routes remain eligible.

The six-task AWM topology, update rates, Replay capacity, BF16 compute, uint8
Replay, oracle boundary consolidation, and oracle Q/F/P compression gate are
unchanged. The additional revisit epoch adds 1,000 world-model updates and 800
Task-0 actor-critic updates. It does not add a boundary operation.

## Reporting and claims

Keep raw per-task returns. Report first-pass retention at epoch 540 separately
from the post-revisit endpoint at epoch 541 using `raw-retention-v1`; do not
apply Atari normalization constants. Evaluation transitions never enter
Replay. This campaign is a pilot until multiple complete, predeclared seeds
exist. Routing accuracy and return are both required; neither substitutes for
the other.

Before a real launch, run the fixed-tensor contracts and the real Procgen
adapter/CUDA smoke from a clean pushed commit. Record the exact project commit,
runtime, hardware, seed, counters, storage preflight, and resolved config.
