# DV3-RS MiniGrid mechanism-reproduction smoke

## Status and claim boundary

`DV3-RS-MiniGrid-3Task-Smoke-v1` is an execution-correctness smoke. It ports
the defining replay mechanism of Continual-Dreamer onto the vendored
DreamerV3 backbone and runs it on the paper's three-task MiniGrid curriculum.
It is **not** a reproduction of a published Continual-Dreamer result: the
paper uses DreamerV2, and its reservoir experiments are principally reported
on MiniHack. A single smoke seed supports no performance or forgetting claim.

The reference inspected for the port is:

- paper: `https://arxiv.org/abs/2211.15944`;
- public implementation: `https://github.com/skezle/continual-dreamer`;
- inspected `main` commit: `77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6`;
- relevant upstream behavior: `dreamerv2/common/replay.py`, configured with
  `--reservoir_sampling` and no recent-biased `50:50` sampling.

No source from that repository is copied into this Apache-2.0 project. The
repository does not expose a top-level license in its root file listing, so
this port independently implements the paper's stated mechanism.

## Method definition

The method is named **DV3-RS**, not Continual-Dreamer:

- world model and actor-critic: the same vendored DreamerV3 implementation as
  the matched ARROW-50 smoke;
- replay: one full-capacity `LongTermReplay`;
- retention: attach an independent continuous random key to every incoming
  fixed-length trajectory and retain the highest keys, which is an unbiased
  uniform reservoir over all eligible trajectories observed so far;
- replay sampling: uniform over the retained trajectories;
- Plan2Explore: disabled;
- task identity: available only to the schedule and evaluator, never supplied
  to the model, policy, or replay sampler.

The public Continual-Dreamer implementation stores variable-length episodes
under a transition capacity and uses the number of loaded episodes as a
surrogate in its reservoir acceptance rule. DV3-RS intentionally does not
copy that approximation. ARROW's native replay boundary is a fixed-length
collected trajectory, so the port uses exact random-key reservoir sampling at
that boundary. This difference must remain visible in all result labels.

## Matched smoke comparison

DV3-RS and `ARROW-50-MiniGrid-3Task-Smoke-v1` are checked at launch to match
outside algorithm and replay retention. Both use:

1. `MiniGrid-DoorKey-9x9-v0`;
2. `MiniGrid-LavaCrossingS9N1-v0`;
3. `MiniGrid-SimpleCrossingS9N1-v0`.

Each task receives one epoch of 128 agent decisions, for 384 decisions and raw
frames in total. Both methods receive six world-model and six actor-critic
updates. Observations are agent-centred partial RGB, with mission text
removed, resized from 56 x 56 to 64 x 64 with OpenCV `INTER_AREA`, and stored
as float32 on CUDA. The discrete action space has seven actions and action
repeat is one.

The replay allowance is exactly matched:

| Method | FIFO slots | Reservoir slots | Total slots | Sequence length |
| --- | ---: | ---: | ---: | ---: |
| DV3-RS | 0 | 16 | 16 | 16 |
| ARROW-50 | 8 | 8 | 16 | 16 |

Each method therefore stores 256 trajectory frames. Tensor storage excluding
allocator overhead is 12,593,152 bytes in both cases. ARROW-50 chooses a whole
minibatch from FIFO or reservoir with probability 0.5 each; DV3-RS always
samples its reservoir.

Final evaluation uses 16 isolated rollouts for every task. Evaluation
transitions never enter replay or training. The vendored base policy evaluates
stochastically, which is recorded as a protocol property rather than silently
changed for this smoke.

## Deliberate differences from the paper

- DreamerV3 replaces DreamerV2.
- The replay retention unit is a fixed-length trajectory rather than a
  variable-length episode.
- Modern MiniGrid/Gymnasium replaces the repository's older Gym stack.
- Images are resized to the vendored DreamerV3 model's 64-pixel input.
- The interaction, update, replay, evaluation, and seed budgets are tiny smoke
  budgets rather than the paper's budgets.
- Plan2Explore is disabled to isolate replay retention.

Consequently, report only execution correctness. A later full protocol must
freeze a multi-seed design and match environmental interaction, update count,
capacity and actual bytes before comparing performance.

The proposed run counts and decision gates for that campaign are recorded in
`docs/protocols/dv3_rs_minigrid_campaign_plan.md`.

## Commands

Dry-run without environment interaction or gradient updates:

```bash
python scripts/run_dv3_rs_minigrid_smoke.py --seed 0 --dry-run
```

After the commit is clean, pushed, fetched, and verified synchronized with its
configured GitHub upstream, run on one selected GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_dv3_rs_minigrid_smoke.py \
  --seed 0 \
  --profile-stages \
  --output-dir /persistent/path/dv3_rs_minigrid_3task_smoke_s0
```
