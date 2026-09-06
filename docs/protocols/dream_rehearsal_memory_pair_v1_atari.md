# Dream-Rehearsal-MemoryPair-v1-Atari

## The requested comparison

**One Dream Rehearsal method. Only the history cap changes.**

| Setting | Full-history arm | Bounded arm |
|---|---|---|
| Real history retention | All collected transitions | Uniform Algorithm-R reservoir |
| Maximum real transitions | No retention cap within the declared interaction budget | **524,288 = 1,024 × 512** |
| Model / WM / actor / critic / optimizers | Same pinned original source | Same |
| Normal and rehearsal minibatch | Same 16 × 64 | Same |
| Dream grading / top fraction / horizon | Original 0.3 / +10, top 25%, H=15, `feats[-1]` bootstrap | Same |
| Rehearsal schedule | 50 actor-only updates per old phase per complete 2,000-decision chunk | Same |
| Ordinary real-data training | All currently retained real history | All currently retained real history |
| Actor ownership | One shared actor, no inference task label | Same |
| Tasks / steps / evaluation / seeds | Same Atari protocol below | Same |

Status: implemented and CPU-fixture tested, **not GPU-tested or trained**.
Decision 0056 corrects the overly broad configuration changes proposed in
decision 0054. This protocol does not launch the old bounded ARROW port.

## Common learning and Atari settings

The fixed Dream Rehearsal and NM512 commits, normal substrate preset and
learning invariants are the ones documented in
`dream_rehearsal_official_code_v1_atari.md`. The original learning files are
still byte-identical. Ordinary training remains shared-history training,
not `current_task_only`.

Both arms use MsPacman → Boxing → CrazyClimber → Frostbite → Seaquest → Enduro,
each for 1,474,560 decisions including 2,500 random prefill. RGB64, all 18 actions,
repeat 4, reset no-ops up to 30, no sticky actions, raw rewards and separate
MDP-terminal/time-limit flags are unchanged between arms. Collector state
persists across chunks and resets at task switches. No MiniGrid score threshold
controls the Atari task boundaries. There is no extra task-0 training epoch.

Each arm projects exactly the following common budgets:

| Counter | Per arm |
|---|---:|
| Collected real decisions including prefill | 8,847,360 |
| Ordinary WM updates | 4,416,280 |
| Ordinary actor updates | 4,416,280 |
| Ordinary critic updates | 4,416,280 |
| Extra actor-only rehearsal updates | 552,000 |
| Evaluation episodes over all seen tasks | 154,980 |

These are inherited original-learning-recipe budgets, not a claim of compute
matching to ARROW. Only the bounded real-sample capacity is matched to ARROW.
The final 60-decision fragment of each task does not add an extra rehearsal
event. Both arms evaluate ten episodes per seen task before online training
and after each chunk. Actor argmax with original stochastic latent inference,
evaluation RNG isolation, raw per-episode returns, distinct step/update/frame
counters and the last-checkpoint/last-three-round summaries are shared.

## Retention semantics

The same `EpisodeHistory` implementation streams both arms into non-overlapping
512-transition storage blocks. The full arm retains every block. The bounded
arm uses unbiased Algorithm R with 1,024 slots and an independent Python
`Random(seed)` instance. A new block is admitted or rejected when its first
transition arrives; rejected blocks never appear in the sampler, even briefly.
Every seen block has the same inclusion probability. Blocks may span episode
boundaries, which remain explicitly represented by reset/terminal flags.

At every insertion, retained transitions are at most the requested capacity.
At the full experiment's block-aligned endpoint, bounded history contains
exactly 524,288 actual collected transitions, while full history contains all
8,847,360. Partial blocks during collection can temporarily use less capacity.
No per-task quota, reward priority, competent-episode filter, FIFO preference
or replacement resampling is added.

The original NM512 episode sampler is used unchanged. A read-only array-like
view joins consecutive fragments of the same original episode, dropping
duplicate block-prefix context rows. Thus block storage does **not** introduce
artificial 512-step RSSM resets in the full arm. Under bounded retention,
missing intervals create separate valid retained segments; no sequence bridges
an eviction gap. The leading observation of a segment is context, with zero
action/reward; it is not another collected transition.

Per-phase rehearsal dictionaries are live index views into this same store.
Eviction removes entries from ordinary replay and all rehearsal views. Old
sampler objects cannot retain complete evicted episodes: their episode series
own only indices and reject stale slot reads. Already materialized minibatches
are ordinary finite workspace, not a separate history available for future
sampling. If an old phase loses all retained data, fail explicitly rather than
inventing a backup or silently dropping its rehearsal updates.

Both arms use one storage-aware collector with the same reset → policy → step
→ replay-insert order as NM512. Source parity fixtures cover the random-prefill
policy, partial episodes, chunk continuation, observations, actions, reset
flags and the original sampler's batches before each synthetic decision.
No learning function is copied or changed. The storage-aware collector writes
no complete training NPZ archive in **either** arm, avoiding a hidden full
history on disk behind the bounded memory. The older original-source runner
is preserved separately for inspection, not used as the other member of the pair.

## Resource and output accounting

One fixed-slot mmap per data field backs the entire replay. Image dtype is
uint8 in both arms. Auxiliary dtypes/shapes are inferred and checked at the
input boundary; action count and image size are not hardcoded in storage.
Up to two observation rows per transition are reserved to accommodate resets
and fragment context even for one-step episodes. Actual retained transitions,
used context rows, used/allocated tensor bytes, allocated filesystem bytes,
retained-block IDs and per-phase live-view counts are recorded. Byte use need
not equal ARROW's different storage representation and must be reported.

The current observation/context and upstream minibatch/optimizer workspace
are separate from historical replay capacity. A current reset-only episode has
zero transitions and is ineligible for the original sampler. Python/index/OS
metadata overhead is explicitly not included in tensor-byte totals.

Evaluation output is kept separately and is never a training source. It can
consume substantial additional disk and time. Analysis snapshots are atomic
and checksummed but **not resumable**, because optimizer, environment and RNG
states are not included. No old checkpoint is reused to initialize either arm.
Existing run directories cannot be overwritten or resumed.

## Invocation and verification

Use the isolated environment from `requirements/dream_rehearsal_official.txt`,
not the ARROW torch 2.3.0 environment. From the repository checkout, inspect:

```bash
python scripts/run_dream_rehearsal_memory_pair_atari.py --history full --dry-run
python scripts/run_dream_rehearsal_memory_pair_atari.py --history bounded --dry-run
```

The two manifest configs differ only in `history_capacity_transitions`; their
common algorithm-and-schedule hash and projected budgets are equal. Output
directories include the arm name so they cannot collide. The `--history`
argument changes no ordinary optimizer or evaluation setting.
`--seed` is the actual RNG seed, not an index into ARROW's seed list; the
default is 123456789. Use the same actual seed for each member of a pair.

```bash
python -m unittest discover -s tests -p 'test_dream_rehearsal*py'
```

The parity wrapper runs its synthetic reference tests in a fresh interpreter
to avoid ARROW/NM512 module-name collisions. Tests establish code/storage
contracts, not Atari learning performance. After a reviewed, clean, pushed,
upstream-synced commit, run target-GPU `--smoke` first. Smoke preserves the same
normal learning constants and arm capacities, but shortens interactions and
evaluation identically; forced reservoir replacement is covered by the CPU
fixtures. Only then consider each full pilot by removing `--dry-run`.

The original MiniGrid return values are not Atari targets or guaranteed scores.
Preserve raw per-task Atari acquisition/retention, negative results and all
seeds. Do not relabel either old faulty port as one of these corrected arms.
