#!/usr/bin/env python3
"""Run the named V3+P2E acquisition/continual pilot; not a paper reproduction.

Paths are repository-relative. Training requires a clean, fetched, pushed
commit. --dry-run does not instantiate an environment or take optimizer steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

from continual_dreamer_v3_support import (
    CD_PIN, ROOT, SourceRecipeAgent, isolated_evaluation_rng,
    resolve_native_config, verify_source,
)
from git_provenance import git_state, require_synced_training_git_state
from launcher_support import write_json
from clworldmodel.continual_dreamer import ContinualDreamerConfig, Counters, should_update

CONFIG = ROOT / "configs/minigrid/cd_dv3_p2e_episode_slots_v1.json"
TASKS = ("MiniGrid-DoorKey-9x9-v0", "MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-SimpleCrossingS9N1-v0")


def resolve_path(path: Path) -> Path:
    return (ROOT / path.expanduser()).resolve()


def build_manifest(recipe, native, output: Path) -> dict:
    lock = ROOT / "requirements/continual_dreamer_v3.txt"
    return dict(
        manifest_version=1, metric_schema_version=1,
        protocol=recipe.protocol, evidence_level=recipe.evidence_level,
        claim_scope="V3+P2E EpisodeSlots pilot; not literal Continual-Dreamer or ARROW paper reproduction",
        git=git_state(ROOT), source=verify_source(),
        continual_dreamer_reference=dict(repository="https://github.com/skezle/continual-dreamer", commit=CD_PIN, copied_source=False),
        resolved_config=recipe.as_dict(), resolved_native_config=vars(native),
        dependency_constraints_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        task_schedule=dict(tasks=TASKS[:recipe.task_count], decisions_per_task=recipe.decisions_per_task),
        budgets=dict(environment_decisions=recipe.total_decisions,
                     raw_environment_frames=recipe.total_decisions,
                     collected_transitions=recipe.total_decisions,
                     world_model_updates=recipe.expected_updates,
                     task_actor_updates=recipe.expected_updates,
                     task_critic_updates=recipe.expected_updates,
                     exploration_actor_updates=recipe.expected_updates,
                     exploration_critic_updates=recipe.expected_updates,
                     ensemble_updates=recipe.expected_updates,
                     initial_updates=recipe.initial_updates,
                     observed_rows_per_world_model_update=recipe.batch_size * recipe.batch_length,
                     imagined_states_per_actor_update=recipe.batch_size * recipe.batch_length * native.imag_horizon),
        replay=dict(capacity_episode_slots=recipe.replay_episode_slots,
                    fifo_slots=recipe.replay_episode_slots // 2 if recipe.method == "arrow50" else 0,
                    reservoir_slots=recipe.replay_episode_slots // 2 if recipe.method == "arrow50" else recipe.replay_episode_slots,
                    fifo_minibatch_probability=0.5 if recipe.method == "arrow50" else 0.0,
                    capacity_action_upper_bound=recipe.replay_episode_slots * recipe.episode_limit,
                    capacity_includes_duplicate_occupancy=True, storage_device="cpu",
                    observation_dtype="uint8", action_dtype="float32", reward_dtype="float32", flag_dtype="bool",
                    compression="none", payload_byte_upper_bound=recipe.replay_episode_slots * (recipe.episode_limit + 1) * (64 * 64 * 3 + 7 * 4 + 4 + 3),
                    overhead="live array/dict/list bytes plus process peak RSS logged separately",
                    eligibility="complete episodes with at least 50 actions; partial/short episodes excluded",
                    retention="FIFO/Algorithm R over episodes; not source variable-transition-cap RS",
                    sampling="whole-batch buffer selection; uniform episodes; source end-prioritized 50-row windows"),
        environment=dict(doorkey_geometry="released_source_8x8", library="minigrid==3.0.0",
                         image="7x7 agent view, tile_size=8, PIL NEAREST resize 56->64",
                         reward="native shaped raw reward retained; tanh only for optimization",
                         action_semantics="seven native actions, repeat=1, one-hot incoming action",
                         termination="terminal observation retained; timeout is_last but not is_terminal",
                         agent_inputs=["image", "is_first", "is_terminal"], task_identity_to_agent=False),
        evaluation=dict(policy="task actor mode, stochastic posterior (source eval_state_mean=False)",
                        every_environment_decisions=recipe.eval_every_decisions,
                        decisions_per_seen_task=recipe.eval_decisions_per_task,
                        seen_tasks_only=True, future_tasks_evaluated=False,
                        separate_environments=True, isolated_rng=True,
                        boundary="fresh evaluation episodes per checkpoint; incomplete final episode logged but excluded from mean",
                        transitions_enter_replay=False, model_updates=False),
        seeds=dict(base=recipe.seed, python=recipe.seed, numpy=recipe.seed, torch=recipe.seed,
                   replay="SeedSequence(base).spawn(3): retention, buffer choice, windows",
                   train_env="base + 1000 * task_index", eval="base + 50000 + 1000 * task_index; fixed across checkpoints"),
        checkpoint=dict(kind="inference_only", resumable=False, replay_checkpointed=False,
                        resume_command_supported=False),
        known_deviations=[
            "NM512 PyTorch V3 backbone, native KL/two-hot/return targets/optimizers, rather than TF V2",
            "FP32, no compile; source uses mixed precision/JIT",
            "Modern MiniGrid renderer/generator, not bitwise legacy gym-minigrid",
            "Fixed episode slots bound <=2M retained actions, not variable episode count under a 2M-action cap",
            "Seeded unbiased Algorithm R replaces known released source RS admission/eviction inconsistency",
            "Exact action budgets; incomplete tails excluded; no source chunk overshoot or per-task dummy update",
            "Preserve one model and optimizer state across tasks; no per-task reconstruction/loading",
            "Evaluation only seen tasks, separate resettable streams, frozen training RNG; no eval recon-loss updates",
        ], output_dir=str(output), status="dry_run",
    )


def runtime_info() -> dict:
    from importlib import metadata
    import torch

    packages = {}
    for line in (ROOT / "requirements/continual_dreamer_v3.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==")
        actual = metadata.version(package)
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(f"Pinned runtime mismatch: {package} expected {expected}, found {actual}")
        packages[package] = actual
    accelerators = [dict(index=i, name=torch.cuda.get_device_name(i), memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
    driver = None
    if accelerators:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    return dict(python=sys.version, os=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
                packages=packages,
                all_installed_distributions={dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
                cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                visible_accelerators=accelerators, accelerator_count=len(accelerators), nvidia_smi=driver,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                deterministic_algorithms=False, cudnn_benchmark=False, tf32=False,
                nondeterminism="native CUDA floating-point kernels; seeded but bitwise determinism not guaranteed")


def run(recipe, native, output: Path, manifest: dict) -> None:
    import gymnasium as gym
    import numpy as np
    import resource
    import torch
    from clworldmodel.environments.episode_stream import EpisodeStream
    from clworldmodel.environments.minigrid import make_minigrid_environment
    from clworldmodel.replay.episode_slots import EpisodeSlotsReplay

    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(recipe.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no accelerator is available")
    replay = EpisodeSlotsReplay(
        recipe.replay_episode_slots, method=recipe.method, min_decisions=recipe.batch_length,
        max_decisions=recipe.episode_limit, sequence_length=recipe.batch_length, seed=recipe.seed,
    )
    counts = Counters()
    started = time.monotonic()
    agent = None
    latest_metrics = {}
    trained_positive_episodes = 0

    def emit(kind, **values):
        record = dict(metric_schema_version=1, kind=kind, wall_seconds=time.monotonic() - started,
                      counters=asdict(counts), **values)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if kind in {"progress", "evaluation", "complete"}:
            print(json.dumps(record, allow_nan=False), flush=True)

    def make_env(index):
        return make_minigrid_environment(TASKS[index], max_episode_steps=recipe.episode_limit,
            doorkey_geometry="released_source_8x8", resize_interpolation="nearest")

    def update():
        nonlocal latest_metrics
        latest_metrics = agent.update(replay.sample(recipe.batch_size))
        for key in ("world_model_updates", "task_actor_updates", "task_critic_updates", "exploration_actor_updates", "exploration_critic_updates", "ensemble_updates"):
            setattr(counts, key, getattr(counts, key) + 1)

    def evaluate(seen):
        # There is deliberately no replay reference or update call in this path.
        for index in range(seen):
            seed = recipe.seed + 50_000 + 1_000 * index
            env = make_env(index)
            completed, partial = [], 0.0
            lengths = []
            try:
                with isolated_evaluation_rng(seed):
                    env.action_space.seed(seed)
                    image, _ = env.reset(seed=seed)
                    obs = dict(image=image, is_first=np.bool_(True), is_terminal=np.bool_(False))
                    state, length = None, 0
                    for _ in range(recipe.eval_decisions_per_task):
                        action, state = agent.act(obs, state, evaluation=True)
                        image, reward, terminal, timeout, _ = env.step(action)
                        counts.evaluation_decisions += 1
                        partial += float(reward)
                        length += 1
                        if terminal or timeout:
                            completed.append(partial)
                            lengths.append(length)
                            partial, length, state = 0.0, 0, None
                            image, _ = env.reset()
                            obs = dict(image=image, is_first=np.bool_(True), is_terminal=np.bool_(False))
                        else:
                            obs = dict(image=image, is_first=np.bool_(False), is_terminal=np.bool_(False))
                if not completed:
                    raise RuntimeError("Evaluation produced no complete episodes within its action budget")
                emit("evaluation", task_index=index, task_name=TASKS[index], raw_returns=completed,
                     episode_lengths=lengths, mean_raw_return=float(np.mean(completed)),
                     success_rate=float(np.mean(np.asarray(completed) > 0)),
                     partial_episode_raw_return=partial, partial_episode_decisions=length)
            finally:
                env.close()

    for task_index in range(recipe.task_count):
        env = make_env(task_index)
        stream = EpisodeStream(env, recipe.seed + 1_000 * task_index)
        state = None
        try:
            for _ in range(recipe.decisions_per_task):
                if stream.complete:
                    stream.reset()
                    counts.reset_observations += 1
                    state = None
                if agent is None:
                    action = int(env.action_space.sample())
                else:
                    action, state = agent.act(stream.observation, state)
                _, episode, reward = stream.step(action)
                counts.environment_decisions += 1
                counts.raw_environment_frames += 1
                counts.collected_transitions += 1
                if episode is not None:
                    accepted = replay.add(episode)
                    raw_return = float(episode["reward"].astype(np.float64).sum())
                    trained_positive_episodes += int(raw_return > 0)
                    emit("train_episode", task_index=task_index, raw_return=raw_return,
                         episode_decisions=len(episode["reward"]) - 1, replay_eligible=accepted)
                if counts.environment_decisions == recipe.prefill_decisions:
                    emit("prefill_complete", partial_decisions_excluded=stream.discard_partial(), replay=replay.accounting())
                    obs_space = gym.spaces.Dict(dict(image=env.observation_space,
                        is_first=gym.spaces.Box(0, 1, (), np.bool_), is_terminal=gym.spaces.Box(0, 1, (), np.bool_)))
                    agent = SourceRecipeAgent(recipe, native, obs_space, env.action_space)
                    for _ in range(recipe.initial_updates):
                        update()
                elif should_update(counts.environment_decisions, recipe):
                    update()
                if counts.environment_decisions % recipe.log_every_decisions == 0:
                    emit("progress", task_index=task_index, replay=replay.accounting(), latest_update_metrics=latest_metrics,
                         positive_training_episodes=trained_positive_episodes,
                         process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                         gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
                if counts.environment_decisions % recipe.eval_every_decisions == 0:
                    evaluate(task_index + 1)
                    # Keep only the latest snapshot plus final task snapshots.
                    agent.save_inference_snapshot(output / "latest_inference.pt", asdict(counts), manifest["resolved_config"])
            emit("task_boundary", task_index=task_index, partial_decisions_excluded=stream.discard_partial())
            agent.save_inference_snapshot(output / f"task_{task_index}_inference.pt", asdict(counts), manifest["resolved_config"])
        finally:
            env.close()
    if counts.environment_decisions != recipe.total_decisions or counts.world_model_updates != recipe.expected_updates:
        raise RuntimeError("Observed training counters differ from the declared budgets")
    manifest.update(status="complete", final_counters=asdict(counts), replay_final=replay.accounting(), wall_seconds=time.monotonic() - started)
    write_json(output / "manifest.json", manifest)
    emit("complete", replay=replay.accounting())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--method", choices=("arrow50", "rs"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-count", type=int, choices=(1, 3))
    parser.add_argument("--decisions-per-task", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    recipe = ContinualDreamerConfig.from_json(resolve_path(args.config), **{
        key: getattr(args, key) for key in ("method", "seed", "task_count", "decisions_per_task", "device")})
    native = resolve_native_config(recipe)
    output = resolve_path(args.output_dir)
    manifest = build_manifest(recipe, native, output)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
    manifest["git"] = require_synced_training_git_state(ROOT)
    manifest["runtime"] = runtime_info()
    output.mkdir(parents=True, exist_ok=False)
    manifest["status"] = "running"
    write_json(output / "manifest.json", manifest)
    try:
        run(recipe, native, output, manifest)
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
