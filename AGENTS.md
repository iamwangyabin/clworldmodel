# clworldmodel Agent Instructions

## Scope and precedence

This file applies to the entire repository. A nested `AGENTS.md` may add stricter
rules for its subtree, but it must not weaken the research, licensing, or
reproducibility requirements defined here.

Treat this file as the project's engineering contract. Update it only when a
project-level decision changes. Do not edit it merely to accommodate a shortcut
in an implementation.

## Project mission

`clworldmodel` is a research platform for continual model-based reinforcement
learning. Its purpose is to develop and evaluate world models that learn a
sequence of tasks while retaining prior capabilities and enabling transfer.

The initial primary method and architectural base is ARROW, "Augmented Replay
for RObust World models", built on DreamerV3. Unless a protocol states
otherwise, `ARROW` means the paper's main ARROW-50 setting: equal FIFO/LTDM
capacity and equal buffer-selection probability. DV3/FIFO is a matched control,
not the project main line. AR25 and AR75 are named ablations, not defaults.

ARROW remains the behavioral base rather than the final clean architecture of
this repository. The vendored source contains the documented compatibility and
runtime optimizations used by the project. The pristine upstream state remains
recoverable from the recorded commit rather than as a second local source tree.

The project must support three kinds of work without conflating them:

1. Faithful reproduction of published baselines.
2. Clean, tested implementations of shared continual world-model components.
3. New research methods evaluated under controlled, comparable protocols.

This is not intended to become a general-purpose RL framework. Add only the
abstractions needed for continual world-model research and its reproducibility.

## Non-negotiable research rules

- Preserve the distinction between environment interaction, world-model
  training, imagined rollouts, controller training, and evaluation.
- Never improve a method by silently giving it more environment steps, replay
  capacity, update steps, evaluation opportunities, privileged task labels, or
  compute.
- Compare replay methods using both trajectory/sample capacity and actual byte
  usage. Report any difference in observation dtype, compression, storage
  device, metadata, or indexing overhead.
- Keep training and evaluation data flows separate. Evaluation transitions must
  never enter replay or affect model updates.
- Do not expose task identity to the agent unless the named protocol explicitly
  studies task-aware learning. The scheduler may know task boundaries for
  orchestration and evaluation; the model and policy may not receive them in a
  task-agnostic experiment.
- Task order, task duration, action handling, observation preprocessing, reward
  transformation, termination handling, and reset handling are part of the
  protocol. Changes to any of them create a new protocol and must be named and
  documented.
- Keep raw per-task returns. Any normalization must be a separate derived metric
  with fixed, cited constants.
- Report retention as well as current-task performance. At minimum, continual
  experiments must retain enough evaluation data to compute final average
  performance and forgetting. Forward transfer and backward transfer should be
  reported when the protocol supports valid reference points.
- A paper claim is not considered reproduced from a single seed or a smoke run.
  Smoke runs establish execution correctness only.

## ARROW vendoring policy

The canonical upstream is:

- Repository: `https://github.com/Cerenaut/ARROW`
- Initial pinned commit: `cb05e7d97ed83c3cf6e528960db0da6868e29232`
- Upstream license: MIT, copyright Cerenaut
- Paper: `https://arxiv.org/abs/2603.11395`

When importing or changing ARROW:

- Keep the project-maintained vendored source under `third_party/arrow/`.
- Record the source URL, full base commit, import date, file hash manifest,
  local modifications, and known upstream issues in
  `third_party/arrow/UPSTREAM.md`.
- Preserve `third_party/arrow/LICENSE` and all upstream copyright notices.
- Direct edits to the vendored source are allowed, but every behavioral change
  must update `UPSTREAM.md` and regenerate `MANIFEST.sha256`. Once research
  development begins, add the smallest relevant parity test in the same change.
- Reference launchers execute `third_party/arrow` directly. The later clean
  implementation still belongs under `src/clworldmodel/` and must not depend on
  undocumented vendored internals.
- Keep upstream parity tests small and deterministic. Compare local tensor
  behavior against upstream semantics or checked fixtures where practical.
- Changing the pinned upstream commit is an explicit research decision. Review
  its diff, update the manifest, rerun parity tests, and document the change.

The repository itself is Apache-2.0 licensed. New project code uses that
license. MIT-derived material must retain the MIT notice; do not relabel copied
upstream code as solely Apache-2.0.

## Target repository layout

The baseline-only repository starts with this minimal layout:

```text
AGENTS.md
README.md
LICENSE
NOTICE
scripts/            # Reproducible orchestration and environment checks
docs/               # Protocols, architecture decisions, and reproduction notes
third_party/arrow/  # Project-maintained ARROW vendor based on the pinned commit
```

Create `pyproject.toml`, `src/clworldmodel/`, project configs, and tests when
the clean implementation or new method begins. Do not add toy implementations
or placeholder tests merely to populate a planned directory tree. At that
point, use the subsystem boundaries described below.

Do not create separate copies of the model or trainer for Atari, CoinRun, or a
new environment. Environment-specific behavior belongs in adapters and config.

## Architecture boundaries

### Configuration

- Use one validated, typed configuration model. Unknown keys are errors.
- Configuration files must be declarative. Do not encode Python class names and
  recover them with unrestricted `getattr` or dynamic imports.
- Every behavior-affecting default must live in the schema and appear in the
  resolved config saved with a run.
- CLI values override config values only when explicitly supplied. An argparse
  default must not accidentally overwrite a config-file value.
- Paths must be resolved relative to a documented root, never implicitly from
  whichever directory launched the process.
- Baseline configs are frozen after validated reproduction. A changed baseline
  receives a new protocol name or version.
- Prefer composed presets over hundreds of hand-copied JSON files. Generate a
  run matrix from task order, method, seed, and ablation dimensions.

### Environments and schedules

- Normalize environment APIs behind project-owned adapters.
- Adapters define observation shape and dtype, action semantics, reward
  transformation, frame skip/repeat, reset semantics, and episode termination.
- Schedules own task order and transition timing. Models, replay buffers, and
  trainers must not contain environment-name checks.
- Environment constructors must be seedable and safe for multiprocessing.
- Atari ROMs, proprietary assets, and downloaded datasets must never be
  committed. Document acquisition and integrity checks instead.

### Models and algorithms

- Keep the world model independent from the continual-learning strategy.
- Keep actor-critic training independent from replay retention policy.
- ARROW should be composition of a Dreamer-style agent and a mixed replay
  policy, not a forked copy of the entire trainer.
- Tensor shapes, dtypes, and time/batch axis order must be documented at public
  boundaries and checked where data enters a subsystem.
- Do not hardcode image size, channel count, or action count inside replay or
  model storage.
- Do not call `.cuda()` in library code. Accept a `torch.device` or infer it
  from owned tensors. CPU must remain usable for unit and integration tests.
- Avoid global mutable state. Random number generators used by sampling logic
  must be injectable or owned, seedable objects.

### Replay

- Replay is a first-class, independently tested subsystem.
- FIFO replay preserves the most recent valid sequences according to a precise,
  documented overwrite order.
- LTDM must implement unbiased uniform retention over all eligible sequences,
  equivalent to reservoir sampling or random-key top-k sampling. Any weighting
  or prioritization is a different named method.
- Mixed replay must define capacity allocation and sampling allocation
  separately. Do not assume they are identical unless the config says so.
- Sampling a whole minibatch from one sub-buffer and mixing examples within a
  minibatch are different semantics. Make the choice explicit and test it.
- Validate capacity, sequence length, burn-in/context, valid starts, episode
  boundaries, and partial-buffer behavior.
- Storage dtype and device are explicit config. Faithful ARROW presets preserve
  upstream behavior; optimized presets must be labeled and byte-accounted.
- Replay checkpoint behavior must be explicit. If replay is not checkpointed,
  a resumed run is not equivalent and must not be presented as such.

### Training and evaluation

- Maintain distinct counters for raw environment frames, agent decisions,
  collected transitions, world-model updates, and actor-critic updates.
- Schedule and log metrics against the correct counter. Never use an ambiguous
  `step` field in persisted results.
- Evaluation uses frozen parameters and deterministic policy behavior unless a
  protocol explicitly defines stochastic evaluation.
- Evaluate every task seen so far at protocol checkpoints. When measuring
  forward transfer, evaluate future tasks without training on them and keep that
  data isolated.
- Centralize metric formulas in `src/clworldmodel/evaluation/metrics.py` and
  test them with small hand-computed matrices.
- Reward scaling used for optimization must not silently replace raw returns in
  reports.

## Reproducibility contract

Every non-smoke run must save a self-contained run manifest containing:

- project git commit and dirty-worktree status;
- resolved config and protocol version;
- upstream ARROW pin when relevant;
- Python and dependency versions or lockfile digest;
- OS, CPU, accelerator model, accelerator count, CUDA, and driver versions;
- all seeds and deterministic/nondeterministic backend settings;
- task order and task-boundary schedule;
- interaction, update, replay-capacity, and replay-byte budgets;
- observation and replay dtypes/devices;
- metric schema version and evaluation schedule.

Seed Python, NumPy, PyTorch CPU/CUDA, environments, action spaces, workers, and
replay samplers. Record any operation known to remain nondeterministic.

A resumable checkpoint must include model and target parameters, optimizers,
scalers, replay or replay provenance, all RNG states, scheduler/task position,
and every step counter. Write checkpoints atomically and test round trips.

Keep generated runs, checkpoints, videos, TensorBoard events, ROMs, and large
datasets out of git. Small deterministic test fixtures are allowed under
`tests/fixtures/` with documented provenance.

## Testing requirements

These requirements apply when project-owned method development begins. Use the
lightest test that can catch the relevant failure, but do not create unrelated
toy code merely to make a test suite exist.

Required unit coverage includes:

- typed config parsing, unknown-key rejection, and CLI override semantics;
- FIFO wraparound and partially filled buffers;
- deterministic LTDM updates under an injected RNG;
- statistical or exhaustive checks of uniform LTDM retention;
- mixed replay capacity and sampling ratios;
- sample shapes, dtypes, valid sequence starts, and episode boundaries;
- schedule transitions and task-identity isolation;
- continual metric formulas;
- checkpoint and RNG-state round trips.

Required integration coverage includes:

- a CPU-only tiny environment collection, replay, world-model update, imagined
  rollout, actor update, evaluation, checkpoint, and resume path;
- one tiny continual schedule crossing at least one task boundary;
- config-to-run-manifest serialization.

Required parity coverage includes fixed-input comparisons against upstream
ARROW semantics for replay retention, replay sampling, model losses,
and one short training trace where practical. Document intentional deviations
instead of weakening assertions until they pass.

GPU smoke tests may be separate from the default suite, but an official
baseline run must pass one on the target accelerator before a campaign starts.

## Dependency and tooling rules

- Once the project-owned package exists, declare its runtime/test dependencies
  in `pyproject.toml`.
- Use a reproducible lock or constraints mechanism for official experiments.
- Separate core, Atari, Procgen, development, and reporting dependencies where
  practical. Importing the core package must not require every benchmark.
- Do not depend on an unpinned Git branch for a published result.
- Do not rely on a user's global Python environment or shell initialization.
- Keep commands non-interactive and cluster-safe. SLURM scripts must call the
  same CLI and resolved configs as local execution.
- Formatters, linters, and type checking support correctness; they do not
  replace behavioral tests.

## Coding rules

- Prefer explicit, typed interfaces and small composable modules.
- Use package-relative imports inside `clworldmodel`; never rely on adding a
  source directory to `sys.path`.
- Use structured logging for persisted run data. Console output should be
  concise and must not be the only record of an experiment.
- Raise specific errors with actionable context. Do not swallow training,
  checkpoint, evaluation, or logging failures with broad exception handlers.
- Assertions may document internal invariants, but user/config validation must
  raise explicit exceptions that remain active under optimized Python.
- Do not add environment-name conditionals to shared trainers or models.
- Do not duplicate a module to make a small benchmark-specific change.
- Keep comments focused on research semantics, invariants, and non-obvious
  tensor transformations.
- Avoid unrelated refactors in baseline-fidelity changes. A parity fix and an
  architectural cleanup should be reviewable separately.

## Documentation and experiment claims

- `README.md` explains environment setup, baseline execution, and the current
  supported protocols. It must not claim a reproduction before validated
  results exist.
- Put protocol definitions and deviations in `docs/protocols/`.
- Record consequential design choices as short files in `docs/decisions/`.
- Every result table must be reproducible from preserved raw metrics and a
  versioned reporting command.
- Label results as `smoke`, `debug`, `pilot`, `ablation`, or `official`.
- Never select or discard seeds based on favorable outcomes without reporting
  the selection rule.
- Negative and failed runs that influence a conclusion belong in the experiment
  record, with failure reasons.

## Development workflow

Before changing research behavior:

1. Read the relevant protocol, config schema, tests, and upstream reference.
2. State the invariant or hypothesis being changed.
3. Add or update the smallest test that exposes the intended difference.
4. Implement within the existing subsystem boundary.
5. Run focused tests, then the broader affected suite.
6. Record any protocol, parity, resource, or reproducibility impact.

Do not mix baseline import, upstream fixes, clean-port refactors, and new-method
changes in one indistinguishable patch.

Explicit project-level review is required before changing:

- the pinned ARROW commit or vendored behavioral changes;
- benchmark task sets, curricula, or task durations;
- preprocessing, reward scaling, termination semantics, or metric formulas;
- baseline budgets or seed sets;
- licensing or third-party provenance;
- the meaning of an already published config or protocol name.

## Bootstrap phases

Work through these phases in order. Do not claim later-phase readiness early.

1. **Governance and baseline:** establish this contract, a minimal repository,
   the target GPU environment, and a reproducible baseline command.
2. **Vendored upstream:** import and fingerprint ARROW; document the base
   commit, upstream failures, and every local compatibility or runtime change.
3. **Reference execution:** run tiny Atari and CoinRun reference jobs, validate
   configs, logging, budgets, checkpoints, and evaluation isolation.
4. **Clean implementation:** port shared components into `clworldmodel`, remove
   environment duplication, and establish parity on deterministic fixtures.
5. **Baseline campaign:** reproduce ARROW-50 first, then selected DreamerV3
   control results with matched budgets and multiple seeds.
6. **New research:** introduce new methods and ablations only after baseline
   behavior and reporting are stable.

## Definition of done

A change is done only when:

- its behavior and ownership boundary are clear;
- relevant unit and integration tests pass;
- baseline parity is preserved or the deviation is explicit and tested;
- resolved configs and manifests capture all new behavior;
- documentation and licensing notices are current;
- no generated artifacts, credentials, local paths, or hidden machine
  assumptions were added;
- claims in code, docs, and reports do not exceed the evidence produced.
