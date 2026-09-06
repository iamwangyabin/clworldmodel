# Dream Rehearsal fidelity correction — 2026-09-05

## Conclusion and evidence scope

The two completed Atari pilots named `Bounded-Dream-Rehearsal-v1-Atari` and
`Never-Clear-Dream-Rehearsal-v1-Atari` are **not faithful paper reproductions**.
The bounded run was already a storage/framework adaptation. More importantly,
the never-clear run's ordinary data sampling contradicts the official chain.
Statements equating its Atari scores with the original method are retracted.

This is an implementation-fidelity audit, not a causal experiment. Low returns
do not establish that unlimited memory fails, that the official method is weak,
or that a particular scoring/world-model failure caused the old result. No
corrected full Atari result exists as of this audit.

## Preserved runs

The original local artifacts remain under the ignored run collection
`runs/dream_rehearsal_20260905/`; none are changed or deleted by this correction.
Their immutable launch provenance:

| Run directory name | Recorded launch commit |
|---|---|
| `bounded_dream_rehearsal_original_s0` | `75f8ff0ea0539f74a38f0232147d46e4c898666f` |
| `never_clear_dream_rehearsal_original_s0` | `bd1cabaf596babe324e7b092872de7e18731c372` |

Both manifests recorded clean, upstream-synced launches. This does not validate
the research semantics. Source artifact hashes, in the same table order:

| File | Bounded SHA256 | Never-clear SHA256 |
|---|---|---|
| `launch.json` | `cbe0314e3edd3dd07282136a04a35edd5beb464787858fd2d378d6a55f77b0f8` | `2e3ddde1228e30ebfbb376047759f953c345f77492251247db30ed208cccc03b` |
| `resolved_training_config.json` | `341fbaa414ea38fc25e08841b5686bc7db435f992565f971616b292e8b609156` | `5e2d1d18f573c18e770373a3bb3b259a579ed925bac9295dd489822aa836f714` |

## Verified discrepancies

1. The never-clear launch manifest explicitly says
   `ordinary_world_model_sampling=current_task_only`,
   `ordinary_actor_critic_sampling=current_task_only`, and
   `old_task_sampling=dream_rehearsal_only`. Official
   `orchestrator_chain_nm512.py` gives `D.make_dataset(shared_eps, config)` to
   ordinary training at every phase. Saved old data must also train the WM and
   ordinary actor/critic. **Storage retention alone is not sampling fidelity.**
2. Both port manifests/configs use four sequences of 16 context frames. In the
   author source those are `args.smoke` overrides. The normal substrate inherits
   NM512's 16 sequences of 64 frames.
3. The old port collects 16,384 decisions then batches all overdue rehearsal
   events. The author interleaves 50 rehearsal updates for each old phase after
   every 2,000 online decisions. Equal total counts do not imply equal training
   trajectories.
4. The old implementation uses the ARROW DreamerV3 stack. NM512's shared actor
   has unimix 0.01 and learning rate 3e-5, and its own optimizer/loss/target
   conventions. Reusing “DreamerV3” as a name does not imply numerical parity.
5. The old scoring port/documentation uses a post-horizon `V(s_H)`. The exact
   author function bootstraps `behavior.value(feats[-1])`. The new reference
   preserves the latter rather than silently substituting a preferred index.
6. The old 541-epoch schedule revisits task 0 after the six nominal 90-epoch
   phases. It is not a simple six-phase paper chain.

The prior bounded protocol/decision's phrase “preserve the artifact's 4 × 16
batch layout” was misleading: it preserved only a smoke layout. The historical
protocol values remain frozen; this note corrects the claim, not its identity.

## Replacement and verification

The new implementation is `Dream-Rehearsal-OfficialCode-v1-Atari`, documented in
`docs/protocols/dream_rehearsal_official_code_v1_atari.md`. It directly imports
the fixed official learning sources and rejects algorithm overrides. CPU
fixtures check normal configuration, all-history sampling, exact chunk cadence,
reach/survival grading, top-k, bootstrap feature, actor-only gradients and
lossless mmap input parity. They are not environment training or optimizer runs.

Atari adaptation, fixed interaction budget, full-history storage and additional
compute remain explicit. Do not compare this new protocol to the old bounded
run as a storage-only ablation. GPU smoke and full/multi-seed results are still
required before any performance/reproduction claim.
