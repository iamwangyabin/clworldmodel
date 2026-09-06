#!/usr/bin/env python3
"""Native V3+P2E gradient/checkpoint contract on synthetic replay, not training evidence.

This check performs optimizer updates, so it also enforces pushed Git provenance.
Use --full-size on the target GPU before launching a pilot. No game is stepped.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time

from clworldmodel.continual_dreamer import ContinualDreamerConfig, Counters
from continual_dreamer_v3_support import ROOT, SourceRecipeAgent, resolve_native_config, isolated_evaluation_rng
from git_provenance import require_synced_training_git_state
from launcher_support import write_json
from run_continual_dreamer_v3 import resolve_path, runtime_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
    git = require_synced_training_git_state(ROOT)
    runtime = runtime_info()

    import gymnasium as gym
    import numpy as np
    import torch
    from clworldmodel.replay.episode_slots import EpisodeSlotsReplay

    recipe = ContinualDreamerConfig(device=args.device)
    native = resolve_native_config(recipe)
    if not args.full_size:
        # Named synthetic contract fixture, never accepted by the run launcher.
        native.dyn_hidden = native.dyn_deter = native.units = 32
        native.dyn_stoch = native.dyn_discrete = 4
        native.encoder["cnn_depth"] = native.decoder["cnn_depth"] = 2
        native.actor["layers"] = native.critic["layers"] = 1
        native.reward_head["layers"] = native.cont_head["layers"] = 1
        native.batch_size, native.batch_length, native.imag_horizon = 2, 6, 5
    torch.manual_seed(recipe.seed)
    torch.set_num_threads(recipe.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(git=git, runtime=runtime, evidence_level="synthetic_execution_fixture",
                    real_environment_decisions=0, recipe=recipe.as_dict(), native=vars(native),
                    full_size=args.full_size, status="running")
    write_json(output / "manifest.json", manifest)
    image_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    obs_space = gym.spaces.Dict(dict(image=image_space, is_first=gym.spaces.Box(0, 1, (), np.bool_),
        is_terminal=gym.spaces.Box(0, 1, (), np.bool_)))
    agent = SourceRecipeAgent(recipe, native, obs_space, gym.spaces.Discrete(7))
    replay = EpisodeSlotsReplay(4, method="arrow50", min_decisions=50,
        max_decisions=100, sequence_length=native.batch_length, seed=0)
    rng = np.random.default_rng(0)
    for terminal in (False, True):
        ep = dict(image=rng.integers(0, 256, (101, 64, 64, 3), dtype=np.uint8),
                  action=np.eye(7, dtype=np.float32)[rng.integers(0, 7, 101)],
                  reward=np.zeros(101, np.float32), is_first=np.zeros(101, bool),
                  is_last=np.zeros(101, bool), is_terminal=np.zeros(101, bool))
        ep["action"][0] = 0
        ep["is_first"][0] = True
        ep["is_last"][-1] = True
        ep["is_terminal"][-1] = terminal
        ep["reward"][-1] = float(terminal) * 0.5
        replay.add(ep)
    observation = dict(image=ep["image"][0], is_first=np.bool_(True), is_terminal=np.bool_(False))
    before = {name: next(module.parameters()).detach().clone() for name, module in (
        ("wm", agent.wm), ("task_actor", agent.task.actor), ("explore_actor", agent.explore.actor), ("ensemble", agent.ensemble))}
    started = time.monotonic()
    metrics = [agent.update(replay.sample(native.batch_size)) for _ in range(2)]
    changed = {name: not torch.equal(before[name], next(module.parameters()).detach()) for name, module in (
        ("wm", agent.wm), ("task_actor", agent.task.actor), ("explore_actor", agent.explore.actor), ("ensemble", agent.ensemble))}
    if not all(changed.values()):
        raise AssertionError(f"A required learner failed to update: {changed}")
    # Evaluation must leave all model parameters and training RNG unchanged.
    frozen = {k: v.detach().clone() for k, v in agent.wm.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    with isolated_evaluation_rng(99):
        action, _ = agent.act(observation, evaluation=True)
    if not 0 <= action < 7 or not torch.equal(rng_before, torch.get_rng_state()):
        raise AssertionError("Evaluation policy/action/RNG contract failed")
    if any(not torch.equal(v, agent.wm.state_dict()[k]) for k, v in frozen.items()):
        raise AssertionError("Evaluation changed world-model parameters")
    counts = Counters(world_model_updates=2, task_actor_updates=2, task_critic_updates=2,
        exploration_actor_updates=2, exploration_critic_updates=2, ensemble_updates=2)
    agent.save_inference_snapshot(output / "inference.pt", asdict(counts), recipe.as_dict())
    saved = torch.load(output / "inference.pt", map_location=args.device, weights_only=True)
    if saved["resumable"] or saved["replay_checkpointed"]:
        raise AssertionError("Weights-only snapshot incorrectly advertises resumability")
    agent.wm.load_state_dict(saved["world_model"], strict=True)
    agent.task.actor.load_state_dict(saved["task_actor"], strict=True)
    with isolated_evaluation_rng(99):
        reloaded_action, _ = agent.act(observation, evaluation=True)
    if action != reloaded_action:
        raise AssertionError("Snapshot round trip changed the seeded evaluation action")
    manifest.update(status="passed", elapsed_seconds=time.monotonic() - started,
                    changed=changed, metrics=metrics, replay=replay.accounting(),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    write_json(output / "manifest.json", manifest)
    print(f"V3+P2E execution contract passed: {output}")


if __name__ == "__main__":
    main()
