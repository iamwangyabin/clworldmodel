#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the v4 S0/S1 seed audit from preserved raw logs and diagnostic cells."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean

from artifact_io import sha256_file, write_json_atomic, write_text_atomic


def periodic_task0(log):
    """Epoch here is completed online epochs before the next scheduled update."""
    epoch, values = None, {}
    for line in log.splitlines():
        match = re.fullmatch(r"Starting Epoch\s+(\d+)", line.strip())
        if match:
            epoch = int(match[1])
        if line.startswith("Eval raw means:") and epoch is not None and epoch <= 90:
            values[epoch] = float(ast.literal_eval(line.split(":", 1)[1].strip())[0])
    return values


def chart(curves):
    # Fixed scientific axes; no smoothing or seed selection.
    x = lambda epoch: 65 + epoch / 90 * 625
    y = lambda value: 315 - value / 5500 * 270
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="405" viewBox="0 0 760 405">',
           '<rect width="760" height="405" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#25324a">',
           '<text x="65" y="23" font-size="18">AWM-AutoRoute v4 · MsPacman learning curves</text>']
    for score in range(0, 5501, 1000):
        svg += [f'<path d="M65 {y(score)}H690" stroke="#e4e8ef"/>',
                f'<text x="52" y="{y(score)+4}" text-anchor="end">{score}</text>']
    for epoch in range(0, 91, 10):
        svg.append(f'<text x="{x(epoch)}" y="337" text-anchor="middle">{epoch}</text>')
    for index, (curve, color) in enumerate(zip(curves, ("#1868d5", "#e57817"))):
        points = " ".join(f"{x(epoch)},{y(value)}" for epoch, value in sorted(curve.items()))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for epoch, value in curve.items():
            svg.append(f'<circle cx="{x(epoch)}" cy="{y(value)}" r="3.5" fill="{color}"/>')
        svg.append(f'<text x="{470+index*110}" y="48" fill="{color}">S{index}</text>')
    svg += ['<text x="65" y="364">x: completed epochs · y: raw return mean · no smoothing</text>',
            '<text x="65" y="386">Each training seed uses its own fixed validation cohort; not a cross-seed average.</text>',
            '</g></svg>']
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = [args.metrics_dir / f"4090-{i}.json" for i in (1, 2)]
    effective_files = [args.audit_dir / f"4090-{i}.effective.json" for i in (1, 2)]
    raw = [json.loads(path.read_text()) for path in files]
    effective = [json.loads(path.read_text()) for path in effective_files]
    curves = [periodic_task0(value["train.log"]["text"]) for value in raw]
    configs = [value["effective_config"] for value in effective]
    differences = {key: [config.get(key) for config in configs]
                   for key in configs[0].keys() | configs[1].keys()
                   if configs[0].get(key) != configs[1].get(key)}
    collections = []
    for value in raw:
        series = value["scalars"][".::Perf/rews_eps_mean"]
        # Exclude the initial random-policy epoch 0; these are scaled diagnostic
        # proxies, including reset markers/partial episodes, NOT official returns.
        collections.append({f"{lo}-{lo+9}": mean([row["value"] for row in series
                            if lo * 1000 <= row["step"] <= (lo + 9) * 1000])
                            for lo in range(1, 62, 10)})
    manifests = []
    for index in (0, 1):
        path = args.audit_dir / "cross_eval" / f"diagnostic_s{index}" / "manifest.json"
        if path.exists():
            files.append(path)
            value = json.loads(path.read_text())
            if value["status"] != "complete":
                raise ValueError(f"Incomplete cross evaluation: {path}")
            manifests.append(value)
    summary = {
        "classification": "debug", "sample_times_utc": [v["sample_time_utc"] for v in raw],
        "effective_config_differences": differences,
        "actual_import_hashes_equal": effective[0]["hashes"] == effective[1]["hashes"],
        "source_commits": [v["source_commit"] for v in raw],
        "source_status": [v["source_status"] for v in raw],
        "periodic_raw_return_means": curves, "scaled_collection_proxies": collections,
        "first_800_update_mean_actor_entropy": [v["scalars"][".::ActorCritic/actor_entropy"][0]["value"] for v in raw],
        "rng_isolation_probes": [{key: v[key] for key in ("rng_cpu_only_fork", "rng_cpu_cuda_fork")} for v in effective],
        "cross_evaluation": [{"training_seed": m["training_seed"],
                              "layout": m["adaptive_compression_layout"],
                              "cells": [{key: c[key] for key in ("evaluation_seed", "raw_return_mean", "raw_return_std")} for c in m["cells"]]} for m in manifests],
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in files + effective_files},
    }
    output = args.output_dir.resolve()
    write_json_atomic(output / "summary.json", summary)
    write_text_atomic(output / "learning_curves.svg", chart(curves))
    lines = ["# AWM-AutoRoute v4：S0/S1 差异审计", "", "仅为 debug/pilot 诊断，不是论文跨 seed 均值或最终成绩。", "",
             "## 同进度的第一任务原始回报", "", "每格是该模型在它自己的固定验证种子上的回报均值，不是跨 seed 均值。",
             "", "| 已完成轮数 | S0 | S1 | S0/S1 |", "|---:|---:|---:|---:|"]
    for epoch in sorted(curves[0].keys() & curves[1].keys()):
        a, b = curves[0][epoch], curves[1][epoch]
        lines.append(f"| {epoch} | {a:.3f} | {b:.3f} | {a/b:.2f} |")
    lines += ["", f"![原始学习曲线]({output / 'learning_curves.svg'})", "", "## 固定第 90 轮交叉评估", "",
              "同一台 4090-1、同一 GPU、相同评估实现；不更新参数、不进入训练。", "",
              "| 固定模型 | S0 验证种子 | S1 验证种子 |", "|---|---:|---:|"]
    for m in manifests:
        means = [c["raw_return_mean"] for c in m["cells"]]
        lines.append(f"| seed={m['training_seed']} | {means[0]:.3f} | {means[1]:.3f} |")
    lines += ["", "边界后的任务 0 Q/F/P 隐层宽度（不是整个模型参数量）："]
    for m in manifests:
        layout = m["adaptive_compression_layout"]
        widths = [layout[key][0] for key in ("posterior", "recurrent", "prior")]
        lines.append(f"- seed={m['training_seed']}: Q/F/P = {' / '.join(map(str, widths))}。")
    lines.append("两行因此比较的是各 seed 完整方法的边界结果，包含自适应压缩选择，不能称为相同拓扑只换权重；但第 40–80 轮差距先于首次压缩，压缩不能解释最初的分叉。")
    if len(manifests) < 2:
        lines.append("\nS1 第 90 轮检查点尚待交叉评估，不能据此宣布原因已经完全确定。")
    else:
        lines.append("\n在相同验证种子下仍可比较两行模型的差距；同一行跨列则显示对验证种子的敏感度。这不是训练 seed 总体方差的估计。")
    lines += ["", "## 已核对与仍未确定", "",
              f"- 有效配置差异：`{json.dumps(differences)}`；实际导入模块哈希一致：{summary['actual_import_hashes_equal']}。",
              "- 第一任务只有候选路由 0，因此这段差距不是选错 task ID；差距出现时也尚未发生任务间遗忘或边界压缩。",
              "- 训练采集代理指标也出现差距，因此不能将所有差距归于验证种子；该代理包含部分回合和 reset 标记，不能当论文成绩。",
              "- seed 同时控制权重、环境、随机动作、Replay/任务采样和潜状态/想象采样，此外两次训练的 CPU 和资源竞争不同。",
              "- 已通过两台主机的独立 CUDA 状态探针验证：Actor 初始化的 CPU-only fork 恢复了 CPU RNG，却没有恢复 CUDA RNG。原 D 同样存在，不是 AutoRoute 新引入；不能据此断言它导致 S1 低分。正式训练未热改。",
              "- 首批 800 次更新的平均 actor entropy 不同，但这不是初始化 entropy，也不证明永久坍塌；S1 后续 entropy 曾回升。",
              "- 当前结果不分离世界模型与 Actor 的贡献，也没有同 seed 跨主机重训，不能把所有因果都归到某一个随机源。两条训练曲线不足以估计总体 seed 方差。",
              "", "精确训练代理值、RNG 探针、输入文件 SHA-256 和交叉评估原始均值/标准差见 `summary.json`。",
              "独立评估的每格还保留完整 legacy 输出张量、实际回合数、权重不变校验和单独的 Git/runtime 清单。"]
    write_text_atomic(output / "report.md", "\n".join(lines) + "\n")
    print(output / "report.md")


if __name__ == "__main__":
    main()
