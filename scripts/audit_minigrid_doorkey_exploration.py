#!/usr/bin/env python3
"""Validate DoorKey action semantics, solvability, and blind exploration.

This diagnostic uses privileged grid state only to build an oracle action
sequence. It is an environment-contract check, never agent performance or a
training result.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from git_provenance import git_state, require_synced_training_git_state


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
GEOMETRIES = ("released_source_8x8", "paper_label_9x9")


@dataclass(frozen=True)
class OracleState:
    x: int
    y: int
    direction: int
    key_present: bool
    carrying_key: bool
    door_open: bool
    door_locked: bool


def _objects(
    raw: Any,
) -> tuple[
    set[tuple[int, int]],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    walls: set[tuple[int, int]] = set()
    locations: dict[str, tuple[int, int]] = {}
    for x in range(raw.width):
        for y in range(raw.height):
            obj = raw.grid.get(x, y)
            if obj is None:
                continue
            if obj.type == "wall":
                walls.add((x, y))
            elif obj.type in {"key", "door", "goal"}:
                locations[obj.type] = (x, y)
    missing = {"key", "door", "goal"} - set(locations)
    if missing:
        raise RuntimeError(f"DoorKey grid is missing objects: {sorted(missing)}")
    return walls, locations["key"], locations["door"], locations["goal"]


def _oracle_plan(raw: Any) -> list[int]:
    """Return a shortest full-state plan under native MiniGrid actions."""

    walls, key_pos, door_pos, goal_pos = _objects(raw)
    door = raw.grid.get(*door_pos)
    initial = OracleState(
        int(raw.agent_pos[0]),
        int(raw.agent_pos[1]),
        int(raw.agent_dir),
        True,
        False,
        bool(door.is_open),
        bool(door.is_locked),
    )
    direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    queue: deque[tuple[OracleState, list[int]]] = deque([(initial, [])])
    visited = {initial}

    while queue:
        state, plan = queue.popleft()
        dx, dy = direction_vectors[state.direction]
        front = (state.x + dx, state.y + dy)
        for action in (0, 1, 2, 3, 5):
            success = False
            if action == 0:
                nxt = replace(state, direction=(state.direction - 1) % 4)
            elif action == 1:
                nxt = replace(state, direction=(state.direction + 1) % 4)
            elif action == 2:
                blocked = front in walls
                blocked |= front == key_pos and state.key_present
                blocked |= front == door_pos and not state.door_open
                nxt = state if blocked else replace(state, x=front[0], y=front[1])
                success = not blocked and front == goal_pos
            elif action == 3:
                if front == key_pos and state.key_present and not state.carrying_key:
                    nxt = replace(state, key_present=False, carrying_key=True)
                else:
                    nxt = state
            else:
                if front != door_pos:
                    nxt = state
                elif state.door_locked and state.carrying_key:
                    nxt = replace(state, door_open=True, door_locked=False)
                elif not state.door_locked:
                    nxt = replace(state, door_open=not state.door_open)
                else:
                    nxt = state

            candidate = [*plan, action]
            if success:
                return candidate
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, candidate))
    raise RuntimeError("No DoorKey oracle plan exists for the generated grid")


def _oracle_audit(geometry: str, seeds: list[int]) -> dict[str, Any]:
    from clworldmodel.environments.minigrid import make_minigrid_environment

    env = make_minigrid_environment(
        "MiniGrid-DoorKey-9x9-v0", doorkey_geometry=geometry
    )
    records = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            raw = env.unwrapped
            plan = _oracle_plan(raw)
            total_reward = 0.0
            terminated = truncated = False
            for action in plan:
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            if not terminated or truncated or total_reward <= 0:
                raise RuntimeError(
                    "Oracle failed: "
                    f"geometry={geometry} seed={seed} plan={plan} "
                    f"reward={total_reward} terminated={terminated} "
                    f"truncated={truncated}"
                )
            records.append(
                {
                    "seed": seed,
                    "plan_length": len(plan),
                    "return": total_reward,
                    "actions": dict(
                        Counter(ACTION_NAMES[action] for action in plan)
                    ),
                }
            )

        action_mapping = {
            name: int(value) for name, value in raw.actions.__members__.items()
        }
        expected_mapping = {
            name: index for index, name in enumerate(ACTION_NAMES)
        }
        if action_mapping != expected_mapping:
            raise RuntimeError(f"Unexpected native action mapping: {action_mapping}")
        lengths = np.asarray([record["plan_length"] for record in records])
        returns = np.asarray([record["return"] for record in records])
        return {
            "width": int(raw.width),
            "height": int(raw.height),
            "underlying_reward_horizon": int(raw.max_steps),
            "outer_episode_limit": 100,
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "action_mapping": action_mapping,
            "oracle_uses_privileged_state": True,
            "oracle_successes": len(records),
            "oracle_episodes": len(records),
            "oracle_plan_length": {
                "minimum": int(lengths.min()),
                "median": float(np.median(lengths)),
                "maximum": int(lengths.max()),
            },
            "oracle_return": {
                "minimum": float(returns.min()),
                "mean": float(returns.mean()),
                "maximum": float(returns.max()),
            },
        }
    finally:
        env.close()


def _random_audit(size: int, episodes: int, base_seed: int) -> dict[str, Any]:
    import gymnasium as gym
    from minigrid.envs import DoorKeyEnv

    env = gym.wrappers.TimeLimit(DoorKeyEnv(size=size), max_episode_steps=100)
    seed_sequence = np.random.SeedSequence(base_seed).spawn(episodes + 1)
    action_rng = np.random.default_rng(seed_sequence[0])
    successes = 0
    return_sum = 0.0
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    executed = 0
    try:
        for episode_seed in seed_sequence[1:]:
            seed = int(episode_seed.generate_state(1, dtype=np.uint32)[0])
            env.reset(seed=seed)
            for _ in range(100):
                action = int(action_rng.integers(0, len(ACTION_NAMES)))
                actions[action] += 1
                executed += 1
                _, reward, terminated, truncated, _ = env.step(action)
                return_sum += float(reward)
                if terminated or truncated:
                    successes += int(float(reward) > 0)
                    break
        return {
            "policy": "uniform over all seven native actions",
            "episodes": episodes,
            "executed_actions": executed,
            "successes": successes,
            "success_rate": successes / episodes,
            "return_sum": return_sum,
            "action_counts": {
                name: int(actions[index])
                for index, name in enumerate(ACTION_NAMES)
            },
            "rule_of_three_95pct_upper_success_rate_if_zero": (
                3.0 / episodes if successes == 0 else None
            ),
        }
    finally:
        env.close()


def _runtime() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in ("gymnasium", "minigrid", "numpy", "opencv-python"):
        result[package] = importlib.metadata.version(package)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-seeds", type=int, default=100)
    parser.add_argument("--random-episodes", type=int, default=6000)
    parser.add_argument("--base-seed", type=int, default=123456789)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "runs"
        / "diagnostics"
        / "doorkey_exploration_contract.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.oracle_seeds < 1 or args.random_episodes < 1 or args.base_seed < 0:
        parser.error(
            "seed and episode counts must be positive; base seed must be non-negative"
        )

    if not args.dry_run:
        subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
        project_git = require_synced_training_git_state(ROOT)
    else:
        project_git = git_state(ROOT)
    declaration: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "minigrid_doorkey_environment_contract_diagnostic",
        "evidence_level": "diagnostic",
        "claim_scope": (
            "environment action/solvability and blind-exploration audit; "
            "not agent performance"
        ),
        "project_git": project_git,
        "base_seed": args.base_seed,
        "oracle_seed_count_per_geometry": args.oracle_seeds,
        "uniform_random_episodes_per_geometry": args.random_episodes,
        "geometries": list(GEOMETRIES),
        "source_reference": {
            "repository": "https://github.com/skezle/continual-dreamer",
            "commit": "77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6",
            "vendored_gym_minigrid_version": "1.0.2",
            "registered_name": "MiniGrid-DoorKey-9x9-v0",
            "executable_size": 8,
            "reason": (
                "registration supplies no kwargs and DoorKeyEnv defaults to size=8"
            ),
        },
        "uses_training_or_evaluation_data": False,
        "uses_optimizer_updates": False,
    }
    if args.dry_run:
        print(json.dumps(declaration, indent=2))
        return 0

    oracle_seed_sequence = np.random.SeedSequence(args.base_seed).spawn(2)
    oracle_seeds = [
        [
            int(item.generate_state(1, dtype=np.uint32)[0])
            for item in root.spawn(args.oracle_seeds)
        ]
        for root in oracle_seed_sequence
    ]
    sizes = {"released_source_8x8": 8, "paper_label_9x9": 9}
    declaration["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    declaration["runtime"] = _runtime()
    declaration["results"] = {}
    for index, geometry in enumerate(GEOMETRIES):
        declaration["results"][geometry] = {
            "oracle": _oracle_audit(geometry, oracle_seeds[index]),
            "uniform_random": _random_audit(
                sizes[geometry],
                args.random_episodes,
                args.base_seed + 1000 + index,
            ),
        }
    declaration["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(declaration, indent=2))
    print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
