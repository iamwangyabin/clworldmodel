#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen paired evaluation: first-frame lock versus one short-prefix recheck."""
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
from git_provenance import require_synced_training_git_state
from probe_coinrun_sequence_routing import ROOT, load_model, rooted, seed_for, verify_launch, weight_digest

PROTOCOL = "CoinRun-Frozen-Prefix4-Once-ClosedLoop-Debug-v1"
ARMS = ("first_frame", "prefix_once")


@dataclass(frozen=True)
class EvaluationConfig:
    episodes_per_task: int = 16
    refine_after_decisions: int = 4
    seed: int = 20260908
    cpu_threads: int = 2
    device: str = "cuda:0"

    def __post_init__(self):
        if min(self.episodes_per_task, self.refine_after_decisions, self.cpu_threads) < 1:
            raise ValueError("Episode, refinement and CPU budgets must be positive")
        if not 0 <= self.seed < 2**31:
            raise ValueError("Seed must be in [0, 2**31)")


@torch.inference_mode()
def recheck_prefix(wm, frames: torch.Tensor, actions: torch.Tensor, routes: tuple[int, ...], dummy: int):
    """[T,C,H,W] frames and [T-1] action IDs -> winner and its own final z/h.

    Every candidate filters the identical observed prefix from zero. The final
    state already includes the current frame: do NOT process that frame again.
    No actor, old hidden state, task label, reward, or prior is needed to score.
    """
    from ac import zh_to_ac_state
    from generate_trajectory import _autocast_context
    if frames.ndim != 4 or actions.shape != (len(frames) - 1,) or len(frames) < 2:
        raise ValueError("Require frames [T,C,H,W] and actions [T-1], T >= 2")
    if not routes or tuple(sorted(set(routes))) != routes:
        raise ValueError("Require sorted unique eligible routes")
    device = frames.device
    scores, states = [], []
    previous = torch.nn.functional.one_hot(torch.cat((actions.new_tensor([dummy]), actions)), wm.a_dim).float()
    with _autocast_context(device, wm.compute_dtype):
        for route in routes:
            z, h = wm.rssm.initial_state(1)
            errors = []
            for t, frame in enumerate(frames):
                _, z, h = wm.rssm(z, previous[t:t + 1], h, frame[None],
                                 frames.new_full((1, 1), float(t == 0)), task_id=route, stochastic=False)
                decoded = wm.decoder_for(route)(zh_to_ac_state(z, h))
                errors.append((decoded.float() - frame[None].float()).square().mean())
            scores.append(torch.stack(errors).mean())
            states.append((z, h))
    scores = torch.stack(scores)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite prefix reconstruction score")
    winner = int(scores.argmin())
    return routes[winner], *states[winner], scores.cpu().tolist()


@torch.inference_mode()
def evaluate_episode(wm, actors, env, *, env_seed: int, arm: str, cfg: EvaluationConfig,
                     max_decisions: int, dummy: int):
    """Single complete episode; no true task ID is accepted at this boundary."""
    from ac import zh_to_ac_state
    from clworldmodel.routing import EpisodeReconstructionRouter
    from generate_trajectory import _autocast_context, _routed_policy_step
    if arm not in ARMS or max_decisions < 1:
        raise ValueError("Unknown arm or invalid safety cap")
    device = next(wm.parameters()).device
    router = EpisodeReconstructionRouter(actors.route_ids)
    z, h = wm.rssm.initial_state(1)
    previous = torch.nn.functional.one_hot(torch.tensor([dummy], device=device), wm.a_dim).float()
    actions, rewards, step_seconds, event = [], [], [], None
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=env_seed)
        prefix = [obs.copy()]
        for decision in range(max_decisions):
            if device.type == "cuda": torch.cuda.synchronize(device)
            tick = time.perf_counter()
            if arm == "prefix_once" and decision == cfg.refine_after_decisions:
                frames = torch.from_numpy(np.stack(prefix)).permute(0, 3, 1, 2).to(device).float() / 255
                route, z, h, scores = recheck_prefix(
                    wm, frames, torch.tensor(actions, device=device), actors.route_ids, dummy)
                old = int(router.routes[0])
                router.routes.fill_(route)
                with _autocast_context(device, wm.compute_dtype):
                    action = actors(zh_to_ac_state(z, h), router.routes).float().argmax(-1)
                event = {"after_agent_decisions": decision, "old_route": old, "selected_route": route,
                         "reconstruction_mse": scores, "switched": route != old}
            else:
                x = torch.from_numpy(obs).permute(2, 0, 1)[None].to(device).float() / 255
                z, h, action = _routed_policy_step(
                    wm, actors, router, x, z, h, previous,
                    torch.full((1, 1), float(decision == 0), device=device),
                    stochastic=False, dummy_previous_action=dummy)
            value = int(action.item())
            previous = torch.nn.functional.one_hot(action, wm.a_dim).float()
            if device.type == "cuda": torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - tick)
            obs, reward, terminated, truncated, _ = env.step(value)
            if not np.isfinite(float(reward)):
                raise FloatingPointError("Nonfinite raw environment reward")
            actions.append(value); rewards.append(float(reward))
            if terminated or truncated:
                break  # Native autoreset image is not a prefix frame.
            if len(prefix) <= cfg.refine_after_decisions:
                prefix.append(obs.copy())
        else:
            raise RuntimeError(f"Episode exceeded safety cap {max_decisions}; no partial return accepted")
        prefix = np.stack(prefix)
        prefix_actions = np.array(actions[:len(prefix) - 1], dtype=np.int64)
        return {"arm": arm, "environment_seed": env_seed, "raw_return": sum(rewards),
                "raw_rewards": rewards, "actions": actions, "agent_decisions": len(actions),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "initial_route": router.events[0]["selected_route_id"], "initial_routing": router.events[0],
                "final_route": int(router.routes[0]), "refinement": event,
                "model_step_seconds": step_seconds, "elapsed_seconds": time.perf_counter() - started,
                "prefix_sha256": hashlib.sha256(prefix.tobytes() + prefix_actions.tobytes()).hexdigest()}, prefix
    finally:
        env.close()


def summarize(episodes, task_names, cfg):
    from clworldmodel.evaluation.metrics import paired_raw_return_summary
    tables = []
    for task in (*range(len(task_names)), None):
        rows = [r for r in episodes if task is None or r["task_index_for_audit_only"] == task]
        arms = [[r for r in rows if r["arm"] == arm] for arm in ARMS]
        baseline, candidate = arms
        paired = paired_raw_return_summary([r["raw_return"] for r in baseline], [r["raw_return"] for r in candidate])
        table = {"task": task_names[task] if task is not None else "all_seen_tasks", **paired}
        for arm, records in zip(ARMS, arms):
            at_event = [r["model_step_seconds"][cfg.refine_after_decisions] for r in records
                        if r["agent_decisions"] > cfg.refine_after_decisions]
            table[arm] = {"initial_route_accuracy": float(np.mean([r["initial_route"] == r["task_index_for_audit_only"] for r in records])),
                          "final_route_accuracy": float(np.mean([r["final_route"] == r["task_index_for_audit_only"] for r in records])),
                          "model_seconds": sum(sum(r["model_step_seconds"]) for r in records),
                          "agent_decisions": sum(r["agent_decisions"] for r in records),
                          "episode_seconds": sum(r["elapsed_seconds"] for r in records),
                          "step_at_recheck_count": len(at_event),
                          "step_at_recheck_median_ms": float(np.median(at_event) * 1000) if at_event else None,
                          "step_at_recheck_p95_ms": float(np.percentile(at_event, 95) * 1000) if at_event else None}
        table.update(rechecks=sum(r["refinement"] is not None for r in candidate),
                     switches=sum(r["initial_route"] != r["final_route"] for r in candidate),
                     wrong_initial_corrected=sum(r["initial_route"] != r["task_index_for_audit_only"] == r["final_route"] for r in candidate),
                     correct_initial_broken=sum(r["initial_route"] == r["task_index_for_audit_only"] != r["final_route"] for r in candidate))
        tables.append(table)
    return tables


def render_report(result):
    lines = ["# CoinRun 首帧启动＋一次短窗复核：闭环小实验", "",
             "单 checkpoint、独立于先前短轨迹诊断的配对种子；debug，不是多 seed 正式结论。", "",
             "| 任务 | 配对数 | 原版平均原始回报 | 复核版平均原始回报 | 配对增量 | 胜/平/负 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in result["tables"]:
        lines.append(f"| {row['task']} | {row['pairs']} | {row['baseline_mean']:.3f} | {row['candidate_mean']:.3f} | "
                     f"{row['mean_paired_delta']:+.3f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    all_tasks = result["tables"][-1]
    lines.extend(["", f"首帧选路准确率 {all_tasks['first_frame']['initial_route_accuracy']:.1%}；"
                  f"一次复核后 {all_tasks['prefix_once']['final_route_accuracy']:.1%}。"
                  f"复核 {all_tasks['rechecks']} 次，切换 {all_tasks['switches']} 次；"
                  f"纠正 {all_tasks['wrong_initial_corrected']} 次，改坏 {all_tasks['correct_initial_broken']} 次。", "",
                  "## 复核所在决策的模型耗时（含历史恢复）", ""])
    for arm in ARMS:
        row = all_tasks[arm]
        lines.append(f"- {arm}: median={row['step_at_recheck_median_ms']} ms, p95={row['step_at_recheck_p95_ms']} ms; "
                     f"总模型时间 {row['model_seconds']:.3f} s / {row['agent_decisions']} 次决策。")
    lines.extend(["", f"主体总耗时 {result['elapsed_seconds']:.2f} s，真实环境决策 {result['actual_agent_decisions']}，"
                  "WM/AC 更新均为 0，前后参数和缓冲区哈希一致。", "",
                  "共享 GPU、batch=1、逐步同步计时；两臂交错且奇偶配对反转顺序。"
                  "回合长度可因行为不同而改变，不把总耗时比直接当作路由开销。", "",
                  "窗口来自上一组探索数据，当前组只检验固定的 4 步复核，不再挑窗口。"
                  "路由正确不保证回报提高；本实验没有部署到训练或修改已有实验。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "output-dir", "upstream-verification"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    checkpoint, output, verification = map(rooted, (args.checkpoint, args.output_dir, args.upstream_verification))
    cfg = EvaluationConfig()
    git = require_synced_training_git_state(ROOT)
    receipt = json.loads(verification.read_text()); verify_launch(receipt, git)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg.device)
    started = time.perf_counter()
    context = {"phase": "loading"}
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        wm, actors, config, routes, payload, digest = load_model(checkpoint, device)
        from clworldmodel.environments.coinrun import COINRUN_TASKS, CoinRunFactory, PROCGEN_COMMIT
        task_names = COINRUN_TASKS[:len(routes)]
        before = weight_digest(wm, actors)
        max_decisions = config.evaluation_max_agent_decisions_per_episode
        manifest = {"protocol": PROTOCOL, "classification": "debug", "project_git": git, "upstream_verification": receipt,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(), "resolved_evaluation_config": asdict(cfg),
                    "resolved_training_config": config.to_dict(), "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
                    "checkpoint_source_commit": payload["project_git_commit"], "checkpoint_completed_epochs": payload["completed_epochs"],
                    "checkpoint_training_seed": payload["seed"], "eligible_routes": routes, "task_names": task_names,
                    "environment_options": {name: CoinRunFactory(name).options for name in task_names}, "procgen_commit": PROCGEN_COMMIT,
                    "arrow_base_commit": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
                    "vendored_manifest_sha256": sha256_file(ROOT / "third_party/arrow/MANIFEST.sha256"),
                    "budgets": {"episodes_per_arm": len(routes) * cfg.episodes_per_task, "arms": ARMS,
                                "max_agent_decisions_per_episode": max_decisions,
                                "max_total_agent_decisions": 2 * len(routes) * cfg.episodes_per_task * max_decisions,
                                "world_model_updates": 0, "actor_critic_updates": 0, "replay_writes": 0},
                    "seed_domains": {"root": cfg.seed, "environment": 4001, "action_space": 5001},
                    "order": "Interleaved paired episodes; first arm reversed on odd episode indices",
                    "task_identity_input": False, "state_on_switch": "Winner's own replayed posterior state, already including current frame",
                    "determinism": {"latent": "mode", "actor": "argmax", "tf32": False, "cudnn_benchmark": False,
                                    "cudnn_deterministic": True, "strict_deterministic_algorithms": False,
                                    "caveat": "Same-runtime seeded evaluation, not cross-platform bitwise guarantee"},
                    "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_inventory": subprocess.check_output(["lscpu"], text=True),
                                "packages": {k: metadata.version(k) for k in ("torch", "numpy", "gymnasium", "gym3", "procgen")},
                                "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)},
                    "metric_schema_version": "paired-raw-returns-v1", "frozen_state_sha256_before": before,
                    "window_selection": "4 transitions selected as shortest of tied 4/8-transition candidates on prior exploratory cohort; independent seeds here"}
        write_json_atomic(output / "manifest.json", manifest)
        episodes, prefixes = [], {}
        for task, name in enumerate(task_names):
            factory = CoinRunFactory(name)
            for episode in range(cfg.episodes_per_task):
                env_seed, action_seed = seed_for(cfg.seed, 4001, episode), seed_for(cfg.seed, 5001, episode)
                pair = []
                for arm in (ARMS if episode % 2 == 0 else ARMS[::-1]):
                    context = {"phase": "evaluation", "task_for_audit": task, "episode": episode, "arm": arm}
                    row, prefix = evaluate_episode(wm, actors, factory.prepare(1, action_seed), env_seed=env_seed,
                                                   arm=arm, cfg=cfg, max_decisions=max_decisions, dummy=factory.dummy_previous_action)
                    row.update(task_index_for_audit_only=task, task_name=name, episode_index=episode, action_seed=action_seed)
                    episodes.append(row); pair.append(row); prefixes[f"task{task}_episode{episode}_{arm}"] = prefix
                    write_json_atomic(output / "episodes.json", {"episodes": episodes})
                if (pair[0]["prefix_sha256"] != pair[1]["prefix_sha256"]
                        or pair[0]["initial_route"] != pair[1]["initial_route"]
                        or pair[0]["actions"][:cfg.refine_after_decisions] != pair[1]["actions"][:cfg.refine_after_decisions]):
                    raise RuntimeError("Paired initial histories/routes differ before the authorized refinement")
            print(json.dumps({"phase": "task_complete", "task": name, "paired_episodes": cfg.episodes_per_task}), flush=True)
        np.savez_compressed(output / "prefixes.npz", **prefixes); write_sha256_sidecar(output / "prefixes.npz")
        after = weight_digest(wm, actors)
        if before != after: raise RuntimeError("Frozen model/actor state changed")
        result = {"protocol": PROTOCOL, "classification": "debug", "complete": True,
                  "tables": summarize(episodes, task_names, cfg), "frozen_state_sha256_after": after,
                  "actual_agent_decisions": sum(r["agent_decisions"] for r in episodes),
                  "actual_raw_environment_frames": sum(r["agent_decisions"] for r in episodes),
                  "world_model_updates": 0, "actor_critic_updates": 0,
                  "elapsed_seconds": time.perf_counter() - started, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        write_json_atomic(output / "results.json", result)
        (output / "SUMMARY.md").write_text(render_report(result))
        print(json.dumps({"complete": True, "output_dir": str(output), "summary": result["tables"][-1]}), flush=True)
    except BaseException:
        write_json_atomic(output / "failure.json", {"context": context, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
