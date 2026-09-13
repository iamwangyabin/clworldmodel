# AWM-AutoRoute: two-observation probability reconstruction

> Historical protocol: superseded by [v4](awm_autoroute_v4_atari.md) on
> 2026-09-08 (Decision 0063). The behavior below is preserved for provenance,
> not current runtime support. Reproduce it with its recorded revision.

## Identity

- Method remains **AWM-AutoRoute** (Accumulative World Modeling).
- Method key remains `evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow`.
- Entry point remains `scripts/run_evolving_atomic_rssm_d_autoroute.py`.
- Protocol: `Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-TwoFrameProbabilityRouter-ARROWParity-v3-OriginalSix-Atari-TaskAwareTraining-TaskIDFreeInference-Pilot`.
- Resolved `task_route_inference=two_frame_probability_reconstruction` fixes
  the horizon at two; it is not an adjustable test-set-selected threshold.

This is the only maintained AWM-AutoRoute implementation. The user selected
a two-observation budget, not continuous convergence. The internal protocol
identifier retains v3 for provenance; v1/v2 are not selectable alternatives.
Historical `first_frame_reconstruction` configs are rejected, never silently
promoted. Old protocols/results remain archived and require recorded revisions
for reproduction. This removes compatibility branches without changing the
two-observation algorithm or its checkpoint contract (Decision 0062).

## Routing rule

For each acquired route k, initialize its own hard latent z and hidden h.
The first observation uses a dummy action (Atari action 0). On the second,
advance each candidate with the action actually executed after observation 1:

\[
h_t^k=f_k(h_{t-1}^k,z_{t-1}^k,a_{t-1}),\qquad
\pi_t^k=q_k(z_t\mid h_t^k,o_t),\qquad
z_t^k=\operatorname{onehot}(\arg\max\pi_t^k).
\]

Argmax/one-hot is applied independently to each categorical latent group.
The RSSM returns normalized log probabilities; the adapter uses `q.exp()`:

\[
e_t(k)=\operatorname{MSE}\left(D_k([\operatorname{vec}(\pi_t^k),h_t^k]),o_t\right),
\qquad
c_t=\arg\min_{k\in\mathcal K_{\rm acquired}}
\frac{1}{m_t}\sum_{s=1}^{m_t}e_s(k),\qquad m_t=\min(t,2).
\]

Scores are computed only when t <= 2. The first decision is used immediately;
the second can revise it; subsequent decisions reuse c_2 with no further scoring.
Ties choose the lowest eligible ID. MSE is FP32 and sums are FP64. Probabilities
are decoder inputs only, not task probabilities or calibrated likelihoods;
decoding an expected latent is not the expected decoded image.

Routing histories are deterministic hard states, independent of stochastic
policy sampling. The selected policy still makes its ordinary RSSM call and
uses hard states (sampled in collection, modal in evaluation). If its ID stays
the same, it retains its policy history; if it changes at observation 2, its
previous state comes from that candidate's own observation-1 history before
processing observation 2. It never inherits the old expert's hidden state.
No new trainable module, prior computation, Monte Carlo vote or privileged
task label enters route selection.

## Reset, training and evaluation contracts

Each vector worker has independent counts, scores and candidate histories.
Pinned Gymnasium NextStep semantics and the collector's legacy stored actions,
reset masks, reward/continue shifts and return extraction are unchanged.
Terminal observations immediately before ignored autoreset actions are not
second routing observations. The returned reset observation starts a fresh
two-observation window, using fresh policy/candidate state and the dummy action.
This policy initialization is an explicit v3 routing change: identical actions
to oracle AWM/v2 across resets are **not** promised, even with one candidate.
An episode ending before observation 2 retains its first choice.

Task order, durations, preprocessing, environment/update budgets, replay
allocation/dtypes, world-model losses, private Actor-Critics, and oracle boundary
selection gates remain [v2's](evolving_core_d_autoroute_v2_atari.md). Training is
task-aware; inference is task-ID-free, not fully task-agnostic learning.
Evaluation is frozen/deterministic and never enters replay. Updates remain
552,000 world-model and 432,000 Actor-Critic; selection uses 480 nominal oracle
rollouts. No training or extra evaluation interaction is authorized by a dry-run.

Route diagnostics preserve each first/second decision with worker and episode
indices, instantaneous/cumulative MSE and margin. Primary `accuracy` and
confusion count the last available decision per episode. First/second accuracy
and denominators are also separate; unfinished one-observation windows are not
silently discarded. Labels are attached only after inference. Boundary resume
starts fresh environments/windows; no mid-episode router state is checkpointed.

## Compute and evidence limits

For L actionable observations and K candidates, U=min(2,L): candidate RSSM
work is K U and decoder work is K U, **in addition to** L selected-policy RSSM
and Actor steps (plus legacy ignored-reset policy steps). This implementation
deliberately preserves the ordinary selected-policy call rather than the
diagnostic prototype's deterministic-call reuse. At most two full comparisons
occur per episode; after that only the selected path runs. Cached candidate
states are bounded by K times worker count and cleared when all workers finish
their windows. No actual GPU latency/FPS result is claimed for this integration.

Earlier frozen CoinRun trajectory diagnostics motivated the probability-input
and two-frame choice, but used historical v1 checkpoints and evaluated several
windows. They are exploratory, not held-out v3 accuracy or return evidence.
Neither monotonic accuracy improvement nor natural convergence is claimed.

Dry-run and fixed/canned-input checks (no ROM interaction):

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute.py --seed 0 --dry-run
PYTHONPATH=src:tests:scripts python -m unittest \
  test_two_frame_autoroute test_d_autoroute test_reconstruction_router \
  test_method_retirement test_retained_method_parity
```

A real pilot still requires clean, pushed, upstream-synchronized provenance
and a target-CUDA smoke. There is no official v3 performance claim.
