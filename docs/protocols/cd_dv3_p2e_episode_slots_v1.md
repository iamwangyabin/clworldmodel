# CD-DV3-P2E-EpisodeSlots-v1

Status: implementation and execution-contract validation; **pilot only**. No
paper performance reproduction or method ranking is claimed. Historical
MiniGrid formal-v1 and autoreset diagnostics are immutable and not replaced.

## Hypothesis and source

The old ARROW MiniGrid training observed no positive reward on DoorKey despite
750k nominal rows. Its absence of Plan2Explore and collection/training protocol
differences prevent interpreting those results as a V2-to-V3 substitution.
This pilot asks whether a source-recipe exploration actor restores reward
discovery and acquisition. It does **not** assume that P2E guarantees success.

References inspected locally at full pins:

- [Continual-Dreamer](https://github.com/skezle/continual-dreamer/tree/77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6):
  `train_minigrid.py`, `dreamerv2/configs.yaml`, `dreamerv2/api.py`,
  `dreamerv2/expl.py`, `dreamerv2/agent.py`, `dreamerv2/common/replay.py`.
- [Paper](https://proceedings.mlr.press/v232/kessler23a/kessler23a.pdf).
- [NM512 V3](https://github.com/NM512/dreamerv3-torch/tree/6ef8646d807cd10ce0c88e10a7e943211e7fc44c),
  MIT, unmodified vendored files. It is a community V3 implementation, not a
  V3 checkpoint or release supplied by the Continual-Dreamer authors.

The Continual-Dreamer repository has no top-level license at the inspected pin.
No source is copied from it: orchestration, replay and disagreement formulas
are independently implemented. Its vendored gym-minigrid has a separate
license; this integration instead uses pinned modern MiniGrid.

## Shared recipe and arms

Both `arrow50` and `rs` use **identical V3 + P2E**, architecture, collection,
training/evaluation schedules and seeds. Only retention allocation and
whole-minibatch buffer selection differ. Here ARROW means the **ARROW-50 replay
strategy port**, not the original ARROW model/trainer or paper reproduction.

| Setting | Both arms |
|---|---|
| Default pilot | DoorKey only, seed 0, 750,000 actual actions including prefill |
| Continual pilot | DoorKey, LavaCrossingS9N1, SimpleCrossingS9N1; 750k actions each |
| Environment | 1 env, 7 native actions, repeat 1, limit 100 actions |
| DoorKey geometry | Released source actual 8x8 despite its `9x9` registration name |
| Observation | Agent-centred 7x7 RGB view, tile 8, PIL nearest 56x56 → 64x64 uint8 |
| Reward | Raw shaped native reward in logs/replay; tanh at learning boundary only |
| Episode semantics | Incoming action/reward; zero reset row; retain final image; timeout bootstraps |
| Warmup | 10k random actions, then two initial updates (source initialization + pretrain) |
| Regular updates | First action after prefill, then every 10 actions; B=16, T=50 |
| Per update | 1 WM + 1 task actor/critic + 1 ensemble + 1 exploration actor/critic |
| V3 | NM512 defaults; discrete one-hot actor; discount .99; entropy .003 |
| Imagined training | Native V3 15 states per posterior start; all 16×50 starts |
| Collection | Exploration actor throughout, no task ID |
| Evaluation | Task actor argmax, stochastic posterior, separate environments and RNG |
| Evaluation budget | Every 10k actual actions starting at prefill, 1k actions per seen task |
| Snapshots | Atomic inference-only files, not resumable; no resume switch |

V3 retains native KL/free-bits, symlog/two-hot heads, normalization/unimix,
model/actor/critic optimizers and return indexing. The manifest saves the entire
resolved native YAML, including unused native-driver keys. No hidden V3 size
override is accepted by the pilot launcher. Run counters are distinct: actual
actions/frames/transitions, reset observations, each learner's updates, and
evaluation actions. The 750k first-task pilot has **74,002 updates per learner**;
the 2.25M three-task pilot has **224,002**. Each actor update imagines 12,000
states; P2E doubles actor/critic updates relative to a one-controller setup.

### Plan2Explore formula

Ten independently initialized MLPs, four 400-wide ELU hidden layers, no
normalization, Glorot-uniform kernels/zero biases. Inputs are detached posterior
features plus same-row incoming one-hot action; targets are next-row flattened
stochastic latents. Both are stop-gradient. The loss is the sum over ten models
of `0.5 * squared_error.sum(target_coordinates).mean(batch,time)`; the fixed
Gaussian NLL constant is omitted (identical gradients, offset loss logs).
Intrinsic reward is population standard deviation across models averaged over
target coordinates, **without log**. Mixed exploration reward is
`(.9 * disagreement + .9 * predicted_extrinsic) / (1 + 1e-8)`, reproducing the
source momentum-1 (disabled normalization) setting. Task policy gets only
predicted extrinsic reward. Ensemble Adam uses 3e-4, eps 1e-5, global clip 100,
source-style multiplicative decay 1e-6 per update, independent of LR.

Source imagination labels actions as incoming; NM512 returns outgoing actions.
The adapter shifts only the P2E reward's action argument by one row. Its first
incoming action is zero; native V3 excludes that first reward from targets.
This prevents silently changing the source action/feature pairing.

### Replay capacity: a named, unavoidable port deviation

Source replay stores complete episodes, discards episodes shorter than 50
actions, chooses episodes uniformly, and clamps an end-prioritized start to
sample 50 rows. These rules are retained, including discarding short successful
episodes; that questionable choice must not be silently optimized away.

Source capacity is a 2M-action sum over variable-length episodes. Its released
RS implementation admits the new episode before the reservoir draw and can
then apply additional random eviction; it is not a clean unbiased Algorithm R.
This project must not label biased retention as LTDM. The named port therefore
uses **20,000 episode slots**, each holding at most 100 actions plus one reset
row, with no padding. `rs` is standard Algorithm R over eligible episodes;
`arrow50` has 10k FIFO + 10k independent reservoir slots and p=.5 to choose an
entire minibatch from FIFO. FIFO overwrites oldest admitted episode first.

This is an **upper bound of 2M actions**, not an assertion of 2M occupied
transitions: shorter episodes reduce occupancy. ARROW stores separate copies
when an episode occupies both buffers; duplicates consume both capacity and
bytes. Both arms share representation/budgets, but actual occupancy and byte
usage can differ and are logged. CPU payload upper bound is 24,892,460,000
bytes (~23.18 GiB), plus array/dict/list overhead and model/runtime memory.
No compressed/GPU replay, mmap, unbounded NPZ archive or dataset download.
Memory is allocated lazily; live payload/metadata and process peak RSS are
logged. Replay is deliberately not checkpointed: resumption is prohibited.

Do not call this a literal 'only V2→V3' reproduction. Before 10k eligible
episodes, neither ARROW subbuffer evicts anything; first-task acquisition is
therefore especially useful without interpreting continual retention rankings.
The standard five-seed comparison must wait for acquisition evidence and an
explicit review of this episode-capacity harmonization; no official campaign
launcher is enabled by this change.

## Further differences from released execution

- Modern MiniGrid 3.0.0 vs legacy gym-minigrid: geometry/observation conventions
  are matched, not byte-for-byte environment generator/render parity.
- FP32 without JIT vs source FP16/JIT; compute and wall time are reported.
- Exact task budgets instead of chunk overshoot; incomplete tails at prefill
  or task boundary are excluded and counted. Complete episodes never cross
  tasks. The model/optimizers persist, without source per-task reconstruction,
  disposable initialization updates or reloading.
- Evaluation begins after prefill and resets its own streams each checkpoint.
  Fixed 1k-action budgets produce variable numbers of complete episodes; raw
  returns and lengths are saved individually. The last partial return/length
  is logged separately and excluded from means, never treated as a failure.
  Only seen tasks are evaluated, with fixed seed streams across checkpoints;
  evaluation cannot advance training RNG, replay or learners.
- Native V3 initial predicted continuation/imagination return math is retained,
  rather than copying V2 terminal-start discount overriding. Changing it is a
  separate named adaptation, not something to slip into a 'V3 baseline'.

Raw evaluation arrays permit final average performance and forgetting across
seen-task checkpoints using the project's existing metric implementation.
This protocol does not support a forward-transfer claim (future tasks are not
evaluated). No normalized paper score is manufactured from these returns.

## Commands and validation

All relative paths are resolved from the repository containing the script.
Use a dedicated Python 3.10–3.12 environment. Install the pinned requirements
and the project package. Training and optimizer checks must run from a clean,
already-pushed branch; both commands fetch then refuse dirty/ahead/behind code.
The `continual-dreamer-v3` extra declares this runtime; legacy `atari` and
`minigrid` extras remain pinned to PyTorch 2.3.0. The core compatibility range
allows 2.3.0–2.4.1 so the isolated 2.4.1 installation also passes `pip check`.
Do not combine the old and new environment extras in one virtual environment.

```sh
python -m pip install -r requirements/continual_dreamer_v3.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -p test_continual_dreamer_v3.py -v
python scripts/run_continual_dreamer_v3.py --dry-run --output-dir runs/cd-v3-plan
python scripts/check_continual_dreamer_v3.py --device cpu --output-dir runs/cd-v3-cpu-contract
CUDA_VISIBLE_DEVICES=0 python scripts/check_continual_dreamer_v3.py --device cuda:0 --full-size --output-dir runs/cd-v3-gpu-contract
CUDA_VISIBLE_DEVICES=0 python scripts/run_continual_dreamer_v3.py --method arrow50 --seed 0 --output-dir runs/cd-v3-arrow50-seed0
```

The check uses synthetic replay, not game interaction or a performance smoke
result. It exercises real WM/task/exploration/ensemble optimizer updates,
frozen evaluation/RNG isolation, and inference snapshot round trips. A full
size GPU check is distinct from the tiny CPU fixture; all effective fixture
settings and commit provenance are saved. Never count either as training.

`--task-count 3` selects the three-task pilot, `--method rs` the matched arm.
Seeds and short pilot budgets require explicit CLI/config values and are
manifested. Unknown/frozen keys fail; CLI omissions cannot override JSON.
The unit suite covers source hashes, geometry interpolation, terminal and
action alignment, ensemble loss/std, stopped WM gradients, uniform reservoir
retention, FIFO wraparound, CPU bytes, sampler ownership, cadence, matched
configs, dry-run side effects and evaluation RNG isolation.
