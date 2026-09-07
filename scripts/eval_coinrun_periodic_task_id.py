#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen closed-loop task-ID pilot: hold the route between sparse rechecks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from artifact_io import sha256_file, write_json_atomic, write_sha256_sidecar
from eval_coinrun_prefix_refinement import recheck_prefix
from git_provenance import require_synced_training_git_state
from probe_coinrun_sequence_routing import ROOT, load_model, rooted, seed_for, verify_launch, weight_digest
from report_coinrun_temporal_task_id import cluster_delta

PROTOCOL = "CoinRun-Frozen-Periodic-TaskID-ClosedLoop-Pilot-v1"


@dataclass(frozen=True)
class EvaluationConfig:
    episodes_per_task: int = 128
    intervals: tuple[int, ...] = (16, 32)
    window_frames: int = 9
    seed: int = 20260910
    environment_seed_domain: int = 7401
    action_seed_domain: int = 7501
    cpu_threads: int = 2
    device: str = "cuda:0"

    def __post_init__(self):
        if min(self.episodes_per_task, self.cpu_threads) < 1 or self.window_frames < 2:
            raise ValueError("Require positive episode/thread budgets and at least two window frames")
        if not self.intervals or any(type(n) is not int or n < 1 for n in self.intervals):
            raise ValueError("Recheck intervals must be positive integers")
        if tuple(sorted(set(self.intervals))) != self.intervals or not 0 <= self.seed < 2**31:
            raise ValueError("Intervals must be sorted/unique and seed in [0, 2**31)")


def frame_tensor(frames, device):
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(device).float() / 255


def state_digest(z, h):
    return hashlib.sha256(b"".join(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() for x in (z, h))).hexdigest()


@torch.inference_mode()
def restore_route_state(wm, frames, actions, route, dummy):
    """Restore only the selected route from the full actual episode, no decoder.

    frames [T,C,H,W] include the current observation; actions [T-1] are the
    executed actions, NOT hypothetical actions of the newly selected actor.
    """
    from generate_trajectory import _autocast_context
    if frames.ndim != 4 or actions.shape != (len(frames) - 1,) or len(frames) < 1:
        raise ValueError("Require full episode frames [T,C,H,W] and actions [T-1]")
    previous = torch.nn.functional.one_hot(torch.cat((actions.new_tensor([dummy]), actions)), wm.a_dim).float()
    z, h = wm.rssm.initial_state(1)
    with _autocast_context(frames.device, wm.compute_dtype):
        for t, frame in enumerate(frames):
            _, z, h = wm.rssm(z, previous[t:t + 1], h, frame[None],
                             frames.new_full((1, 1), float(t == 0)), task_id=route, stochastic=False)
    return z, h


@torch.inference_mode()
def evaluate_episode(wm, actors, env, *, env_seed: int, interval: int, cfg: EvaluationConfig,
                     max_decisions: int, dummy: int):
    """No true task label, reward gate, new head, prior, or inactive state cache."""
    from ac import zh_to_ac_state
    from clworldmodel.routing import EpisodeReconstructionRouter
    from generate_trajectory import _autocast_context, _routed_policy_step
    if interval not in (0, *cfg.intervals) or max_decisions < 1:
        raise ValueError("Unknown interval or invalid episode safety cap")
    device = next(wm.parameters()).device
    router = EpisodeReconstructionRouter(actors.route_ids)
    z, h = wm.rssm.initial_state(1)
    previous = torch.nn.functional.one_hot(torch.tensor([dummy], device=device), wm.a_dim).float()
    actions, rewards, trace, seconds, rechecks = [], [], [], [], []
    calls = dict(initial_probe=len(actors.route_ids), ordinary_filter=0, window_probe=0, selected_history_restore=0)
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=env_seed)
        # Full uint8 history is retained only for exact selected-route recovery.
        history = [obs.copy()]
        for decision in range(max_decisions):
            if device.type == "cuda": torch.cuda.synchronize(device)
            tick = time.perf_counter()
            switched, event = False, None
            if interval and decision and decision % interval == 0:
                start = max(0, decision + 1 - cfg.window_frames)
                frames = frame_tensor(history[start:], device)
                route, _, _, scores = recheck_prefix(wm, frames, torch.tensor(actions[start:], device=device),
                                                    actors.route_ids, dummy)
                old = int(router.routes[0])
                switched = route != old
                calls["window_probe"] += len(actors.route_ids) * len(frames)
                event = {"after_agent_decisions": decision, "window_start": start,
                         "window_frames": len(frames), "old_route": old, "selected_route": route,
                         "reconstruction_mse": scores, "switched": switched}
                if switched:
                    z, h = restore_route_state(wm, frame_tensor(history, device),
                                               torch.tensor(actions, device=device), route, dummy)
                    calls["selected_history_restore"] += len(history)
                    router.routes.fill_(route)
                    with _autocast_context(device, wm.compute_dtype):
                        action = actors(zh_to_ac_state(z, h), router.routes).float().argmax(-1)
                rechecks.append(event)
            if not switched:
                # A matching recheck must NOT overwrite the incumbent's full-history state.
                z, h, action = _routed_policy_step(wm, actors, router, frame_tensor([obs], device),
                    z, h, previous, torch.full((1, 1), float(decision == 0), device=device),
                    stochastic=False, dummy_previous_action=dummy)
                calls["ordinary_filter"] += 1
            value = int(action.item())
            previous = torch.nn.functional.one_hot(action, wm.a_dim).float()
            if device.type == "cuda": torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - tick)
            # Audit hashing is outside inference latency, but included in episode wall time.
            if event is not None: event["executed_state_sha256"] = state_digest(z, h)
            trace.append(int(router.routes[0]))
            obs, reward, terminated, truncated, _ = env.step(value)
            if not np.isfinite(float(reward)): raise FloatingPointError("Nonfinite raw reward")
            actions.append(value); rewards.append(float(reward))
            if terminated or truncated: break  # Terminal output is the next episode's reset image.
            history.append(obs.copy())
        else:
            raise RuntimeError(f"Episode exceeded safety cap {max_decisions}; no partial episode accepted")
        calls["total"] = sum(calls.values())
        return {"interval": interval, "environment_seed": env_seed, "raw_return": sum(rewards),
                "raw_rewards": rewards, "actions": actions, "route_trace": trace,
                "agent_decisions": len(actions), "initial_route": trace[0], "final_route": trace[-1],
                "initial_routing": router.events[0], "rechecks": rechecks, "rssm_calls": calls,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "model_step_seconds": seconds, "elapsed_seconds": time.perf_counter() - started}, np.stack(history)
    finally:
        env.close()


def audit_pair(baseline, baseline_frames, candidate, candidate_frames):
    """Before the first possible recheck the executions must agree exactly."""
    n = candidate["interval"]
    if (baseline["initial_route"] != candidate["initial_route"] or
            baseline["actions"][:n] != candidate["actions"][:n] or
            baseline["raw_rewards"][:n] != candidate["raw_rewards"][:n] or
            not np.array_equal(baseline_frames[:n + 1], candidate_frames[:n + 1])):
        raise RuntimeError("Paired execution diverged before the authorized recheck")
    if not any(e["switched"] for e in candidate["rechecks"]) and (
        baseline["actions"] != candidate["actions"] or baseline["raw_rewards"] != candidate["raw_rewards"] or
        not np.array_equal(baseline_frames, candidate_frames)
    ):
        raise RuntimeError("An unchanged route altered baseline execution")


def summarize(episodes, task_names, cfg):
    from clworldmodel.evaluation.metrics import task_id_trace_accuracy
    tables, comparisons = [], []
    for task in (*range(len(task_names)), None):
        rows = [r for r in episodes if task is None or r["task_index_for_audit_only"] == task]
        arm_rows = {}
        for interval in (0, *cfg.intervals):
            records = sorted((r for r in rows if r["interval"] == interval),
                             key=lambda r: (r["task_index_for_audit_only"], r["episode_index"]))
            labels = [r["task_index_for_audit_only"] for r in records]
            metrics = task_id_trace_accuracy([r["route_trace"] for r in records], labels)
            events = [(r, e) for r in records for e in r["rechecks"]]
            check_seconds = [r["model_step_seconds"][e["after_agent_decisions"]] for r, e in events]
            normal_seconds = [s for r in records for t, s in enumerate(r["model_step_seconds"])
                              if t > 0 and (not interval or t % interval)]
            confusion = np.zeros((len(task_names), len(task_names)), dtype=int)
            for r, label in zip(records, labels): confusion[label, r["final_route"]] += 1
            row = {"task": task_names[task] if task is not None else "all_seen_tasks", "interval": interval,
                   **metrics, "final_confusion": confusion.tolist(),
                   "episodes_rechecked": sum(bool(r["rechecks"]) for r in records), "rechecks": len(events),
                   "switches": sum(e["switched"] for _, e in events),
                   "wrong_to_correct_events": sum(e["old_route"] != r["task_index_for_audit_only"] == e["selected_route"] for r, e in events),
                   "correct_to_wrong_events": sum(e["old_route"] == r["task_index_for_audit_only"] != e["selected_route"] for r, e in events),
                   "initial_wrong_final_correct": sum(r["initial_route"] != y == r["final_route"] for r, y in zip(records, labels)),
                   "initial_correct_final_wrong": sum(r["initial_route"] == y != r["final_route"] for r, y in zip(records, labels)),
                   "agent_decisions": sum(r["agent_decisions"] for r in records),
                   "model_seconds": sum(sum(r["model_step_seconds"]) for r in records),
                   "rssm_calls": {k: sum(r["rssm_calls"][k] for r in records) for k in records[0]["rssm_calls"]}}
            for name, seconds in (("recheck", check_seconds), ("ordinary", normal_seconds)):
                row[name + "_median_ms"] = float(np.median(seconds) * 1000) if seconds else None
                row[name + "_p95_ms"] = float(np.percentile(seconds, 95) * 1000) if seconds else None
            row["model_ms_per_decision"] = row["model_seconds"] * 1000 / row["agent_decisions"]
            tables.append(row); arm_rows[interval] = (records, metrics)
        for interval in cfg.intervals:
            records, metrics = arm_rows[interval]
            comparisons.append({"task": tables[-1]["task"], "interval": interval,
                **cluster_delta(metrics["episode_accuracy"], arm_rows[0][1]["episode_accuracy"],
                                [r["environment_seed"] for r in records], seed=cfg.seed)})
    return {"tables": tables, "paired_execution_accuracy": comparisons}


def render_report(result):
    lines = ["# CoinRun 稀疏复核 task-ID 精度（pilot）", "",
             "仅固定 checkpoint、三个已完成视觉任务；未训练新路由器，不是多训练 seed 结论。",
             "执行中准确率对每回合先求正确路由占比、再等权平均；micro 列按决策数加权。", "",
             "| 任务 | 复核间隔 | 回合数 | 执行中准确率 | micro | 最终准确率 | 复核回合 | 切换 | 纠正/改错事件 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["tables"]:
        lines.append(f"| {r['task']} | {r['interval'] or '不复核'} | {r['episodes']} | {r['episode_mean_accuracy']:.2%} | "
                     f"{r['decision_weighted_accuracy']:.2%} | {r['final_accuracy']:.2%} | {r['episodes_rechecked']} | "
                     f"{r['switches']} | {r['wrong_to_correct_events']}/{r['correct_to_wrong_events']} |")
    lines.extend(["", "## 配对执行中准确率差值", "",
                  "环境种子分组 bootstrap 95% 区间；只条件于当前 checkpoint，未作多重比较修正。"])
    for r in result["paired_execution_accuracy"]:
        if r["task"] == "all_seen_tasks":
            lo, hi = r["ci95"]
            lines.append(f"- 每 {r['interval']} 步：{r['paired_delta'] * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]。")
    lines.extend(["", "## 计算成本", "", "共享 GPU、batch=1、逐次 CUDA 同步；不是专用设备速度基准。"])
    for r in result["tables"]:
        if r["task"] == "all_seen_tasks":
            lines.append(f"- 间隔 {r['interval']}：平均 {r['model_ms_per_decision']:.2f} ms/决策；"
                         f"复核决策 median/p95={r['recheck_median_ms']}/{r['recheck_p95_ms']} ms；"
                         f"RSSM 调用 {r['rssm_calls']['total']} / 决策 {r['agent_decisions']}。")
    lines.extend(["", "最近 9 帧从零状态评分是有限上下文近似，不等于 episode 前缀评分。"
                  "仅切换时恢复新路径的完整真实历史；不切换则保留原状态。",
                  "原始回报已保留但不作为选路或本实验成败标准；不同路由会改变后续轨迹。",
                  f"环境决策总数 {result['actual_agent_decisions']}；WM/Actor/Router 更新均为 0。", ""])
    return "\n".join(lines)


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    result = json.loads((output / "results.json").read_text())
    if not result["complete"] or manifest["frozen_state_sha256_before"] != result["frozen_state_sha256_after"]:
        raise ValueError("Require complete evaluation and unchanged model parameters/buffers")
    cfg = EvaluationConfig(**{**manifest["resolved_evaluation_config"], "intervals": tuple(manifest["resolved_evaluation_config"]["intervals"])})
    episodes = json.loads((output / "episodes.json").read_text())["episodes"]
    expected = {(task, episode, interval) for task in range(len(manifest["task_names"]))
                for episode in range(cfg.episodes_per_task) for interval in (0, *cfg.intervals)}
    keys = [(r["task_index_for_audit_only"], r["episode_index"], r["interval"]) for r in episodes]
    if len(keys) != len(set(keys)) or set(keys) != expected: raise ValueError("Incomplete or duplicated cohort")
    for row in episodes:
        path = output / row["trajectory_file"]
        if sha256_file(path) != row["trajectory_sha256"]: raise ValueError("Raw trajectory checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            for key, field in (("actions", "actions"), ("rewards", "raw_rewards"), ("routes", "route_trace")):
                np.testing.assert_array_equal(data[key], row[field])
            if len(data["observations"]) != row["agent_decisions"]: raise ValueError("Observation/action alignment failed")
        n, interval = row["agent_decisions"], row["interval"]
        if [e["after_agent_decisions"] for e in row["rechecks"]] != (list(range(interval, n, interval)) if interval else []):
            raise ValueError("Task ID was not checked at exactly the declared sparse schedule")
        current, event_by_t = row["initial_route"], {e["after_agent_decisions"]: e for e in row["rechecks"]}
        for t, route in enumerate(row["route_trace"]):
            if t in event_by_t:
                event = event_by_t[t]
                winner = manifest["eligible_routes"][int(np.argmin(event["reconstruction_mse"]))]
                if event["old_route"] != current or event["selected_route"] != winner: raise ValueError("Recheck score/route mismatch")
                current = winner
            if route != current: raise ValueError("Route changed between authorized rechecks")
    recomputed = summarize(episodes, manifest["task_names"], cfg)
    if any(result[k] != recomputed[k] for k in recomputed): raise ValueError("Raw task-ID metric recomputation failed")
    (output / "SUMMARY.md").write_text(render_report(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", allow_abbrev=False)
    for name in ("checkpoint", "output-dir", "upstream-verification"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    sub.add_parser("report", allow_abbrev=False).add_argument("run_directory", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(rooted(args.run_directory)); return
    checkpoint, output, verification = map(rooted, (args.checkpoint, args.output_dir, args.upstream_verification))
    cfg = EvaluationConfig()
    git = require_synced_training_git_state(ROOT)
    receipt = json.loads(verification.read_text()); verify_launch(receipt, git)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device, started, context = torch.device(cfg.device), time.perf_counter(), {"phase": "loading"}
    try:
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        wm, actors, config, routes, payload, digest = load_model(checkpoint, device)
        from clworldmodel.environments.coinrun import COINRUN_TASKS, CoinRunFactory, PROCGEN_COMMIT
        task_names = COINRUN_TASKS[:len(routes)]
        before, max_decisions = weight_digest(wm, actors), config.evaluation_max_agent_decisions_per_episode
        manifest = {"protocol": PROTOCOL, "classification": "pilot", "project_git": git, "upstream_verification": receipt,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "resolved_evaluation_config": asdict(cfg),
            "resolved_training_config": config.to_dict(), "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
            "checkpoint_source_commit": payload["project_git_commit"], "checkpoint_completed_epochs": payload["completed_epochs"],
            "checkpoint_training_seed": payload["seed"], "eligible_routes": routes, "task_names": task_names,
            "environment_options": {name: CoinRunFactory(name).options for name in task_names}, "procgen_commit": PROCGEN_COMMIT,
            "arrow_base_commit": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
            "vendored_manifest_sha256": sha256_file(ROOT / "third_party/arrow/MANIFEST.sha256"),
            "budgets": {"episodes_per_arm": len(routes) * cfg.episodes_per_task, "intervals": [0, *cfg.intervals],
                        "max_agent_decisions_per_episode": max_decisions,
                        "max_total_agent_decisions": (1 + len(cfg.intervals)) * len(routes) * cfg.episodes_per_task * max_decisions,
                        "world_model_updates": 0, "actor_critic_updates": 0, "router_updates": 0,
                        "replay_writes": 0, "replay_capacity": 0, "replay_bytes": 0},
            "order": "task then paired episode; rotate arm order by episode index",
            "task_identity_input": False, "observations": "uint8 NHWC saved, float BCHW /255 inference, repeat=1",
            "state_on_switch": "Selected route only, full actual episode, no decoder; no hidden inheritance or double current-frame processing",
            "state_without_switch": "Keep incumbent full-history state; ignore candidate window states",
            "scoring": "batch1 deterministic posterior reconstruction MSE mean of last9frames, zero context at window start, no new parameters",
            "determinism": {"latent": "mode", "actor": "argmax", "tf32": False, "cudnn_benchmark": False,
                            "cudnn_deterministic": True, "strict_deterministic_algorithms": False,
                            "caveat": "Seeded same-runtime only, not cross-platform bitwise guarantee"},
            "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_inventory": subprocess.check_output(["lscpu"], text=True),
                        "packages": {k: metadata.version(k) for k in ("torch", "numpy", "gymnasium", "gym3", "procgen")},
                        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)},
            "metric_schema_version": "task-id-execution-trace-v1", "frozen_state_sha256_before": before,
            "selection": "interval16/32 and last9frames fixed before this independent cohort; no tuning on its results"}
        write_json_atomic(output / "manifest.json", manifest)
        (output / "trajectories").mkdir()
        episodes = []
        intervals = (0, *cfg.intervals)
        for task, name in enumerate(task_names):
            factory = CoinRunFactory(name)
            for episode in range(cfg.episodes_per_task):
                env_seed = seed_for(cfg.seed, cfg.environment_seed_domain, episode)
                action_seed = seed_for(cfg.seed, cfg.action_seed_domain, episode)
                rotate = episode % len(intervals)
                pair = {}
                for interval in intervals[rotate:] + intervals[:rotate]:
                    context = {"phase": "evaluation", "task_for_audit": task, "episode": episode, "interval": interval}
                    row, frames = evaluate_episode(wm, actors, factory.prepare(1, action_seed), env_seed=env_seed,
                        interval=interval, cfg=cfg, max_decisions=max_decisions, dummy=factory.dummy_previous_action)
                    path = output / "trajectories" / f"task{task}_episode{episode}_interval{interval}.npz"
                    np.savez_compressed(path, observations=frames, actions=np.array(row["actions"], np.int64),
                                        rewards=np.array(row["raw_rewards"], np.float64), routes=np.array(row["route_trace"], np.int64))
                    row.update(task_index_for_audit_only=task, task_name=name, episode_index=episode, action_seed=action_seed,
                               trajectory_file=str(path.relative_to(output)), trajectory_sha256=sha256_file(path))
                    episodes.append(row); pair[interval] = (row, frames)
                for interval in cfg.intervals: audit_pair(*pair[0], *pair[interval])
                write_json_atomic(output / "episodes.json", {"episodes": episodes})
                if (episode + 1) % 32 == 0:
                    print(json.dumps({"phase": "progress", "task": name, "paired_episodes": episode + 1,
                                      "actual_agent_decisions": sum(r["agent_decisions"] for r in episodes)}), flush=True)
        after = weight_digest(wm, actors)
        if before != after: raise RuntimeError("Frozen model/actor state changed")
        result = {"protocol": PROTOCOL, "classification": "pilot", "complete": True,
                  **summarize(episodes, task_names, cfg), "frozen_state_sha256_after": after,
                  "actual_agent_decisions": sum(r["agent_decisions"] for r in episodes),
                  "actual_raw_environment_frames": sum(r["agent_decisions"] for r in episodes),
                  "world_model_updates": 0, "actor_critic_updates": 0, "router_updates": 0,
                  "elapsed_seconds": time.perf_counter() - started,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "paired_precheck_audit_passed": True}
        write_json_atomic(output / "results.json", result)
        for name in ("manifest.json", "episodes.json", "results.json"): write_sha256_sidecar(output / name)
        report(output)
        print(json.dumps({"complete": True, "output_dir": str(output)}), flush=True)
    except BaseException:
        write_json_atomic(output / "failure.json", {"context": context, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
