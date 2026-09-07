#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the complete, predeclared routing table without selecting a window."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(directory: Path) -> str:
    result = json.loads((directory / "results.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if not result["complete"] or manifest["frozen_state_sha256_before"] != result["frozen_state_sha256_after"]:
        raise ValueError("Require a completed diagnostic with unchanged frozen state")
    task_count = len(manifest["eligible_route_ids"])
    episodes = manifest["resolved_probe_config"]["episodes_per_task"]
    lines = ["# CoinRun 冻结 checkpoint 路由诊断", "",
             "**debug，单 checkpoint；不是闭环游戏得分实验或多 seed 结论。**", "",
             f"运行提交：`{manifest['project_git']['commit']}`。训练 seed：{manifest['checkpoint']['seed']}，"
             f"checkpoint：{manifest['checkpoint']['completed_epochs']} epochs，{task_count} 条已学习路径。",
             f"实际 {result['actual_agent_decisions']:,} 次环境决策，{result['elapsed_seconds']:.2f} 秒；"
             f"峰值 PyTorch allocated 显存 {result['gpu_peak_allocated_bytes'] / 2**20:.1f} MiB；"
             "WM/AC 更新均为 0。耗时来自共享 GPU，不代表各路由器的在线 FPS。", "",
             f"每个 cohort 原定 {task_count * episodes} 条短轨迹（每任务 {episodes} 条）；种子跨任务和 cohort 配对，"
             "不能把两个 cohort 的重复首帧算成独立样本。W 表示真实转移数，B 使用 W+1 帧。", ""]
    names = ("A_first_frame", "B_multiframe_reconstruction", "C_action_conditioned_prediction", "C_shuffled_actions")
    for cohort in manifest["resolved_probe_config"]["cohorts"]:
        lines.extend([f"## {cohort}", "",
                      "| W | 有效/排除 | A 首帧 | B 多帧重建 | C 动作预测 | C 打乱动作 | B 纠错/改坏 |",
                      "|---:|---:|---:|---:|---:|---:|---:|"])
        for window in manifest["resolved_probe_config"]["windows"]:
            rows = {r["router"]: r for r in result["tables"] if r["cohort"] == cohort and r["window_transitions"] == window}
            if not rows:
                continue
            a, b = rows[names[0]], rows[names[1]]
            values = " | ".join(f"{100 * rows[n]['accuracy']:.1f}%" if rows[n]['accuracy'] is not None else "NA" for n in names)
            lines.append(f"| {window} | {a['sample_count']}/{a['short_prefixes_excluded']} | {values} | "
                         f"{b['wrong_first_frame_corrected']}/{b['correct_first_frame_broken']} |")
        lines.append("")
    lines.extend(["## 解读边界", "",
                  "- 同一行四种方法使用完全相同的有效短轨迹；不同 W 的样本可能不同。"
                  "较长窗口可能排除提前终止轨迹，应查看逐行分母，不能跨窗口直接归因比较。",
                  "- B 是各候选自己的时序 posterior 重建；C 是 prior-mode 解码 MSE，不是精确似然。"
                  "打乱动作对照不等于学习到了跨任务不同的动力学。NB/RT 在该协议里是视觉变化。",
                  "- 未选择最优窗口、调阈值、实现 CUSUM 或部署在线切换；全部预定窗口均列出。"
                  "KL 原始分数已保留，但尚未验证其作为低成本异常门控的区分能力。",
                  "- `results.json` 保留逐任务准确率、混淆矩阵及 C 的纠错/改坏数；"
                  "`scores.npz`、`trajectories.npz`、`episodes.json` 保留原始证据。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run_directory", type=Path)
    path = parser.parse_args().run_directory.expanduser()
    print(report(path if path.is_absolute() else ROOT / path))
