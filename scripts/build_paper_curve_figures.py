#!/usr/bin/env python3
"""Render the current Atari and CoinRun paper learning-curve figures."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_results_draft as results  # noqa: E402
import plot_arrow_atari_ours_only as plot  # noqa: E402
import plot_arrow_atari_paper_aligned_ours as paper  # noqa: E402
from clworldmodel.evaluation.metrics import median_iqr  # noqa: E402
from summarize_continual_metrics import build_run_report  # noqa: E402


OUTPUT = ROOT / "docs" / "experiments" / "figures"
METHODS = ("DreamerV3/FIFO", "ARROW-50", "AWM-AutoRoute")
METHOD_TITLES = {
    "ARROW-50": "ARROW (AR50)",
    "DreamerV3/FIFO": "DreamerV3",
    "AWM-AutoRoute": "AWM",
}
ATARI_TASKS = (
    "Ms. Pac-Man",
    "Boxing",
    "Crazy Climber",
    "Frostbite",
    "Seaquest",
    "Enduro",
)
COINRUN_TASKS = (
    "CoinRun",
    "CoinRun+NB",
    "CoinRun+NB+RT",
    "CoinRun+NB+RT+GA",
    "CoinRun+NB+RT+GA+MA",
    "CoinRun+NB+RT+GA+MA+CA",
)
COINRUN_RANDOM = (2.78, 2.45, 2.70, 2.62, 2.50, 2.69)
COINRUN_SINGLE = (6.09, 7.14, 6.89, 6.85, 7.89, 5.78)
EPOCHS = tuple(range(0, 541, 10))
METHOD_COLORS = {
    "DreamerV3/FIFO": "#9A9A9A",
    "ARROW-50": "#4C78A8",
    "AWM-AutoRoute": "#F58518",
}


def _atari_runs() -> dict[str, list[dict[str, Any]]]:
    groups = {method: [] for method in METHODS}
    for report in results._json(results.S0_COMPARISON)["reports"]:
        if report["method"] in groups:
            groups[report["method"]].append(report)
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        for relative in results.BASELINE_ARCHIVES:
            report = build_run_report(
                results._extract_run(ROOT / relative, destination)
            )
            groups[report["method"]].append(report)
    for seed_dir in results.AWM_ATARI_SEEDS:
        groups["AWM-AutoRoute"].append(
            results._json(
                ROOT
                / "runs"
                / "main_results"
                / "atari"
                / "awm_autoroute"
                / seed_dir
                / "continual_metrics.json"
            )
        )
    if any(len(groups[method]) != 5 for method in METHODS):
        raise ValueError("Atari curve figure requires five runs per method")
    return groups


def _coinrun_runs() -> dict[str, list[dict[int, list[float]]]]:
    groups: dict[str, list[dict[int, list[float]]]] = {
        method: [] for method in METHODS
    }
    for path in sorted(
        (ROOT / "docs" / "experiments" / "records").glob(
            "coinrun-*/record.json"
        )
    ):
        record = results._json(path)
        if record["record_id"] in results.COINRUN_EXCLUDED_RECORD_IDS:
            continue
        groups[str(record["method"])].append(
            {
                int(row["completed_epochs"]): [
                    float(task["raw_return_mean"]) for task in row["tasks"]
                ]
                for row in record["evaluation"]["checkpoints"]
            }
        )
    for seed_dir in results.AWM_COINRUN_SEEDS:
        awm = results._json(
            ROOT
            / "runs"
            / "main_results"
            / "coinrun"
            / "awm_autoroute"
            / seed_dir
            / "continual_metrics.json"
        )
        groups["AWM-AutoRoute"].append(
            {
                int(row["completed_epochs"]): [
                    float(value) for value in row["raw_return_mean"]
                ]
                for row in awm["evaluation_checkpoints"]
                if int(row["completed_epochs"]) in EPOCHS
            }
        )
    if [len(groups[method]) for method in METHODS] != [4, 5, 3]:
        raise ValueError("CoinRun curve figure requires 4/5/3 runs after exclusions")
    return {
        method: [
            {
                epoch: [
                    (value - random) / (single - random)
                    for value, random, single in zip(
                        values, COINRUN_RANDOM, COINRUN_SINGLE
                    )
                ]
                for epoch, values in run.items()
            }
            for run in runs
        ]
        for method, runs in groups.items()
    }


def _atari_matrices() -> dict[str, list[dict[int, list[float]]]]:
    return {
        method: [
            {
                int(row["completed_epochs"]): [
                    float(value) for value in row["normalized_score"]
                ]
                for row in report["evaluation_checkpoints"]
            }
            for report in reports
        ]
        for method, reports in _atari_runs().items()
    }


def _summary(
    runs: Sequence[dict[int, list[float]]], task_index: int
) -> tuple[list[float], list[float], list[float]]:
    median, q25, q75 = [], [], []
    for epoch in EPOCHS:
        values = [run[epoch][task_index] for run in runs]
        row = median_iqr(values)
        median.append(float(row["median"]))
        q25.append(float(row["q25"]))
        q75.append(float(row["q75"]))
    return median, q25, q75


def _draw_method_panel(
    image: Image.Image,
    box: tuple[float, float, float, float],
    method: str,
    runs: Sequence[dict[int, list[float]]],
    task_count: int,
    y_limits: tuple[float, float] | None,
    y_ticks: Sequence[float] | None,
    show_y_ticks: bool,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right = x0 + 105, x1 - 22
    top, bottom = y0 + 54, y1 - 84
    summaries = [_summary(runs, task_index) for task_index in range(task_count)]
    all_values = [value for summary in summaries for curve in summary for value in curve]
    if y_limits is None:
        y_low, y_high, panel_y_ticks = plot.nice_limits(
            all_values, include_zero=True, include_one=True
        )
    else:
        y_low, y_high = y_limits
        panel_y_ticks = list(y_ticks or ())

    def px(frame: float) -> float:
        return left + frame / (540 * 16_384) * (right - left)

    def py(value: float) -> float:
        projected = bottom - (value - y_low) / (y_high - y_low) * (bottom - top)
        return max(top, min(bottom, projected))

    draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG)
    draw = ImageDraw.Draw(image)
    for tick in panel_y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=paper.GRID, width=2)
        if show_y_ticks:
            plot.right_text(
                draw,
                (left - 12, y),
                plot.fmt_tick(tick),
                font=plot.FONTS.get(16),
                fill=paper.MUTED,
            )
    for epoch in range(90, 540, 90):
        x = px(epoch * 16_384)
        paper.dashed_line(
            draw, (x, top), (x, bottom), fill="#B8B8B8", width=2, dash=8, gap=8
        )
    if y_low <= 0 <= y_high:
        paper.dashed_line(
            draw,
            (left, py(0)),
            (right, py(0)),
            fill="#8C8C8C",
            width=2,
            dash=9,
            gap=8,
        )
    frames = [epoch * 16_384 for epoch in EPOCHS]
    for task_index, (median, q25, q75) in enumerate(summaries):
        color = str(paper.PAPER_TASK_STYLES[task_index]["color"])
        upper = [(px(frame), py(value)) for frame, value in zip(frames, q75)]
        lower = [
            (px(frame), py(value))
            for frame, value in reversed(list(zip(frames, q25)))
        ]
        plot.alpha_polygon(image, upper + lower, color, 42)
    draw = ImageDraw.Draw(image)
    for task_index, (median, _, _) in enumerate(summaries):
        style = paper.PAPER_TASK_STYLES[task_index]
        color = str(style["color"])
        points = [(px(frame), py(value)) for frame, value in zip(frames, median)]
        start, end = task_index * 90, (task_index + 1) * 90
        before = [point for epoch, point in zip(EPOCHS, points) if epoch <= start]
        active = [
            point for epoch, point in zip(EPOCHS, points) if start <= epoch <= end
        ]
        after = [point for epoch, point in zip(EPOCHS, points) if epoch >= end]
        thin_dash = paper.scaled_dash(style["dash"], scale=5.0)
        bold_dash = paper.scaled_dash(style["dash"], scale=11.0)
        paper.styled_polyline(
            draw, before, fill=color, width=4, dash_pattern=thin_dash
        )
        paper.styled_polyline(
            draw, active, fill=color, width=10, dash_pattern=bold_dash
        )
        paper.styled_polyline(
            draw, after, fill=color, width=4, dash_pattern=thin_dash
        )
    draw.line((left, top, left, bottom), fill=paper.INK, width=3)
    draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
    for frame in (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000):
        x = px(frame)
        draw.line((x, bottom, x, bottom + 8), fill=paper.INK, width=3)
        plot.center_text(
            draw,
            (x, bottom + 27),
            f"{frame / 1_000_000:g}",
            font=plot.FONTS.get(17),
            fill=paper.MUTED,
        )
    draw.text((right - 22, bottom + 40), "1e6", font=plot.FONTS.get(15), fill=paper.MUTED)
    plot.center_text(
        draw,
        ((x0 + x1) / 2, y0 + 19),
        METHOD_TITLES[method],
        font=plot.FONTS.get(22, bold=True),
    )
    plot.center_text(
        draw,
        ((left + right) / 2, y1 - 21),
        "Agent decisions",
        font=plot.FONTS.get(18),
        fill=paper.INK,
    )


def _render(
    output: Path,
    tasks: Sequence[str],
    groups: dict[str, list[dict[int, list[float]]]],
    y_label: str,
    y_limits: tuple[float, float] | None,
    y_ticks: Sequence[float] | None,
) -> None:
    size = (2450, 820)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "A", font=plot.FONTS.get(31), fill=paper.INK)
    left, right, gap = 55, 30, 22
    panel_width = (size[0] - left - right - gap * 2) / 3
    boxes = [
        (
            left + index * (panel_width + gap),
            18,
            left + index * (panel_width + gap) + panel_width,
            650,
        )
        for index in range(3)
    ]
    for method_index, (method, box) in enumerate(zip(METHODS, boxes)):
        _draw_method_panel(
            image,
            box,
            method,
            groups[method],
            len(tasks),
            y_limits,
            y_ticks,
            show_y_ticks=method_index == 0,
        )
    paper.draw_vertical_text(
        image, (22, 335), y_label, size=22, fill=paper.INK
    )
    draw = ImageDraw.Draw(image)
    legend_font = plot.FONTS.get(16)
    for task_index, task in enumerate(tasks):
        column, row = task_index // 2, task_index % 2
        x = 300 + column * 760
        y = 700 + row * 48
        style = paper.PAPER_TASK_STYLES[task_index]
        paper.styled_polyline(
            draw,
            [(x, y), (x + 76, y)],
            fill=str(style["color"]),
            width=5,
            dash_pattern=paper.scaled_dash(style["dash"], scale=6.0),
        )
        draw.text((x + 92, y - 9), task, font=legend_font, fill=paper.INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def _metric_tuple(value: dict[str, float]) -> tuple[float, float, float]:
    return float(value["median"]), float(value["q25"]), float(value["q75"])


def _draw_metric_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    rows: Sequence[tuple[str, tuple[float, float, float] | None]],
    limits: tuple[float, float],
    ticks: Sequence[float],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 72, x1 - 18, y0 + 60, y1 - 92
    low, high = limits

    def py(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG)
    for tick in ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=paper.GRID, width=2)
        plot.right_text(
            draw, (left - 10, y), f"{tick:g}", font=plot.FONTS.get(15), fill=paper.MUTED
        )
    if low <= 0 <= high:
        draw.line((left, py(0), right, py(0)), fill="#777777", width=2)
    slot = (right - left) / len(rows)
    zero = py(0) if low <= 0 <= high else bottom
    for index, (label, value) in enumerate(rows):
        center = left + slot * (index + 0.5)
        if value is None:
            y = bottom - 28
            paper.dashed_line(
                draw,
                (center - 28, y),
                (center + 28, y),
                fill="#888888",
                width=3,
                dash=6,
                gap=5,
            )
            plot.center_text(
                draw,
                (center, y - 23),
                "PENDING",
                font=plot.FONTS.get(12, bold=True),
                fill=paper.MUTED,
            )
        else:
            median, q25, q75 = value
            y_median = py(median)
            draw.rectangle(
                (center - 31, min(zero, y_median), center + 31, max(zero, y_median)),
                fill=METHOD_COLORS[label],
                outline="#555555",
                width=2,
            )
            draw.line((center, py(q25), center, py(q75)), fill="#333333", width=3)
            draw.line((center - 10, py(q25), center + 10, py(q25)), fill="#333333", width=3)
            draw.line((center - 10, py(q75), center + 10, py(q75)), fill="#333333", width=3)
        display = {"DreamerV3/FIFO": "DV3", "ARROW-50": "AR50", "AWM-AutoRoute": "AWM"}[label]
        plot.center_text(
            draw, (center, bottom + 29), display, font=plot.FONTS.get(15, bold=True)
        )
    draw.line((left, top, left, bottom), fill=paper.INK, width=3)
    draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
    plot.center_text(
        draw, ((x0 + x1) / 2, y0 + 23), title, font=plot.FONTS.get(19, bold=True)
    )


def _render_metric_summary(output: Path) -> None:
    atari, _, _ = results._atari_results()
    coinrun, coinrun_awm, _ = results._coinrun_results()
    metric_specs = (
        ("Forgetting ↓", "forgetting", (-0.5, 2.5), (-0.5, 0, 0.5, 1, 1.5, 2, 2.5)),
        ("ACC ↑", "acc", (-0.5, 3.5), (-0.5, 0, 1, 2, 3)),
        ("min-ACC ↑", "min_acc", (-0.75, 3.5), (-0.5, 0, 1, 2, 3)),
        ("WC-ACC ↑", "wc_acc", (-0.5, 3.0), (-0.5, 0, 1, 2, 3)),
    )
    size = (2400, 1260)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    margin, gap = 42, 22
    width = (size[0] - margin * 2 - gap * 3) // 4
    row_boxes = []
    for row in range(2):
        top = 74 + row * 585
        row_boxes.append(
            [
                (margin + col * (width + gap), top, margin + col * (width + gap) + width, top + 500)
                for col in range(4)
            ]
        )
    draw.text((12, 18), "A  Atari", font=plot.FONTS.get(27, bold=True), fill=paper.INK)
    draw.text((12, 603), "B  CoinRun", font=plot.FONTS.get(27, bold=True), fill=paper.INK)
    methods = ("DreamerV3/FIFO", "ARROW-50", "AWM-AutoRoute")
    for box, (title, key, limits, ticks) in zip(row_boxes[0], metric_specs):
        _draw_metric_panel(
            image,
            box,
            title,
            [(method, _metric_tuple(atari[method]["metrics"][key])) for method in methods],
            limits,
            ticks,
        )
    coinrun_limits = {
        "forgetting": (-0.2, 0.5, (-0.2, 0, 0.2, 0.4)),
        "acc": (0.5, 1.2, (0.6, 0.8, 1.0, 1.2)),
        "min_acc": (-0.7, 1.1, (-0.5, 0, 0.5, 1.0)),
        "wc_acc": (-0.4, 1.1, (-0.25, 0, 0.5, 1.0)),
    }
    for box, (title, key, _, _) in zip(row_boxes[1], metric_specs):
        low, high, ticks = coinrun_limits[key]
        _draw_metric_panel(
            image,
            box,
            title,
            [
                ("DreamerV3/FIFO", _metric_tuple(coinrun["DreamerV3/FIFO"]["metrics"][key])),
                ("ARROW-50", _metric_tuple(coinrun["ARROW-50"]["metrics"][key])),
                ("AWM-AutoRoute", _metric_tuple(coinrun_awm["metrics"][key])),
            ],
            (low, high),
            ticks,
        )
    note = "AWM CoinRun uses three remaining seeds from the post-hoc top-five subset; descriptive only, not a predeclared cohort estimate."
    plot.center_text(
        draw,
        (size[0] / 2, 1210),
        note,
        font=plot.FONTS.get(17),
        fill=paper.MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def _awm_parameter_summary() -> dict[str, float]:
    values = []
    for seed_dir in results.AWM_ATARI_SEEDS:
        run = ROOT / "runs" / "main_results" / "atari" / "awm_autoroute" / seed_dir
        wm = results._json(run / "model_parameter_accounting.json")
        values.append(float(wm["world_model"]["parameters"]))
    return median_iqr(values)


def _render_capacity_pareto(output: Path) -> None:
    capacity = results._capacity_results()
    awm, _, _ = results._atari_results()
    awm_params = _awm_parameter_summary()
    methods = (
        "Shared WM + private AC",
        "Wider shared + private AC",
        "FullBank + private AC",
        "Frozen core + residuals",
        "Independent residuals",
    )
    colors = ("#4C78A8", "#72B7B2", "#E45756", "#B279A2", "#54A24B")
    size = (1600, 920)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 1530, 85, 775
    x_low, x_high, y_low, y_high = 15.0, 135.0, 0.0, 3.4

    def px(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

    draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG)
    for tick in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        y = py(tick)
        draw.line((left, y, right, y), fill=paper.GRID, width=2)
        plot.right_text(draw, (left - 14, y), f"{tick:g}", font=plot.FONTS.get(17), fill=paper.MUTED)
    for tick in (20, 40, 60, 80, 100, 120):
        x = px(tick)
        draw.line((x, bottom, x, bottom + 8), fill=paper.INK, width=2)
        plot.center_text(draw, (x, bottom + 27), str(tick), font=plot.FONTS.get(17), fill=paper.MUTED)
    offsets = {
        "Shared WM + private AC": (12, -38),
        "Wider shared + private AC": (12, 10),
        "FullBank + private AC": (-235, 8),
        "Frozen core + residuals": (12, 8),
        "Independent residuals": (12, -28),
    }
    for method, color in zip(methods, colors):
        x_value = results.CAPACITY_WM_PARAMS[method] / 1e6
        acc = capacity[method]["metrics"]["acc"]
        x, y = px(x_value), py(float(acc["median"]))
        draw.line((x, py(float(acc["q25"])), x, py(float(acc["q75"]))), fill=color, width=5)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="#333333", width=2)
        dx, dy = offsets[method]
        draw.text((x + dx, y + dy), method.replace(" + private AC", ""), font=plot.FONTS.get(16), fill=paper.INK)
    awm_acc = awm["AWM-AutoRoute"]["metrics"]["acc"]
    x = px(float(awm_params["median"]) / 1e6)
    y = py(float(awm_acc["median"]))
    draw.line((px(float(awm_params["q25"]) / 1e6), y, px(float(awm_params["q75"]) / 1e6), y), fill=METHOD_COLORS["AWM-AutoRoute"], width=5)
    draw.line((x, py(float(awm_acc["q25"])), x, py(float(awm_acc["q75"]))), fill=METHOD_COLORS["AWM-AutoRoute"], width=5)
    draw.rectangle((x - 10, y - 10, x + 10, y + 10), fill=METHOD_COLORS["AWM-AutoRoute"], outline="#333333", width=2)
    draw.text((x + 16, y - 38), "AWM pilot", font=plot.FONTS.get(17, bold=True), fill=paper.INK)
    draw.line((left, top, left, bottom), fill=paper.INK, width=3)
    draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
    plot.center_text(draw, ((left + right) / 2, 842), "World-model parameters (millions)", font=plot.FONTS.get(20))
    paper.draw_vertical_text(image, (35, (top + bottom) // 2), "ACC ↑", size=20, fill=paper.INK)
    draw.text((18, 18), "Atari capacity–performance diagnostic", font=plot.FONTS.get(25, bold=True), fill=paper.INK)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def _awm_parameter_growth(
    run_root: Path, seeds: Sequence[str]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    components = {name: [[] for _ in range(6)] for name in ("Shared", "Q", "F", "P", "Projectors")}
    allocated = [[] for _ in range(6)]
    for seed_dir in seeds:
        run = run_root / seed_dir
        wm = results._json(run / "model_parameter_accounting.json")
        banks = wm["rssm_task_mechanism_banks"]
        shared = (
            float(wm["world_model"]["parameters"])
            - sum(float(row["parameters"]) for row in wm["observation_projectors_per_task"].values())
            - sum(float(row["parameters"]) for row in banks.values())
        )
        for task in range(6):
            components["Shared"][task].append(shared)
            components["Projectors"][task].append(
                sum(
                    float(row["parameters"])
                    for index, row in wm["observation_projectors_per_task"].items()
                    if int(index) <= task
                )
            )
            for label, bank_name in (("Q", "representation"), ("F", "recurrent"), ("P", "transition")):
                bank = banks[bank_name]
                components[label][task].append(
                    sum(float(value) for value in bank["mechanism_parameters_per_task"][: task + 1])
                    + sum(float(value) for value in bank["route_parameters_per_later_task"][: task + 1])
                )
            boundary = results._json(run / "adaptive_qfp_compression" / f"task_{task:02d}_boundary.json")
            allocated[task].append(float(boundary["world_model_parameters_after"]))
    component_summary = [
        {name: float(median_iqr(values[task])["median"]) for name, values in components.items()}
        for task in range(6)
    ]
    retained_summary = []
    for task in range(6):
        totals = [sum(components[name][task][seed] for name in components) for seed in range(len(seeds))]
        retained_summary.append({key: float(value) for key, value in median_iqr(totals).items()})
        retained_summary[-1]["allocated"] = float(median_iqr(allocated[task])["median"])
    return component_summary, retained_summary


def _render_parameter_growth(
    output: Path,
    run_root: Path,
    seeds: Sequence[str],
    task_labels: Sequence[str],
    title: str,
) -> None:
    components, totals = _awm_parameter_growth(run_root, seeds)
    size = (1600, 950)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 1530, 145, 770
    y_max = 45.0
    colors = {
        "Shared": "#4C78A8",
        "Q": "#F58518",
        "F": "#54A24B",
        "P": "#E45756",
        "Projectors": "#B279A2",
    }

    def py(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG)
    for tick in (0, 10, 20, 30, 40):
        y = py(tick)
        draw.line((left, y, right, y), fill=paper.GRID, width=2)
        plot.right_text(draw, (left - 14, y), str(tick), font=plot.FONTS.get(17), fill=paper.MUTED)
    centers = [left + (index + 0.5) * (right - left) / 6 for index in range(6)]
    bar_width = 118
    for task, x in enumerate(centers):
        cumulative = 0.0
        for name in ("Shared", "Q", "F", "P", "Projectors"):
            value = components[task][name] / 1e6
            y0, y1 = py(cumulative), py(cumulative + value)
            draw.rectangle((x - bar_width / 2, y1, x + bar_width / 2, y0), fill=colors[name], outline="white", width=1)
            cumulative += value
        total = totals[task]
        low, high = total["q25"] / 1e6, total["q75"] / 1e6
        draw.line((x, py(low), x, py(high)), fill=paper.INK, width=4)
        draw.line((x - 10, py(low), x + 10, py(low)), fill=paper.INK, width=4)
        draw.line((x - 10, py(high), x + 10, py(high)), fill=paper.INK, width=4)
        draw.text((x - 34, py(total["median"] / 1e6) - 31), f'{total["median"] / 1e6:.2f}', font=plot.FONTS.get(15, bold=True), fill=paper.INK)
        short = task_labels[task].replace(" ", "\n", 1)
        plot.center_text(draw, (x, bottom + 34), f"T{task + 1}", font=plot.FONTS.get(17, bold=True))
        plot.center_text(draw, (x, bottom + 62), short, font=plot.FONTS.get(14), fill=paper.MUTED)

    allocated_points = [(x, py(totals[task]["allocated"] / 1e6)) for task, x in enumerate(centers)]
    for first, second in zip(allocated_points, allocated_points[1:]):
        draw.line((*first, *second), fill="#666666", width=4)
    for x, y in allocated_points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="white", outline="#555555", width=3)

    legend_x, legend_y = 160, 94
    for name, label in (("Shared", "Shared base"), ("Q", "Representation Q"), ("F", "Recurrent F"), ("P", "Transition P"), ("Projectors", "Projectors (0.03→0.21M)")):
        draw.rectangle((legend_x, legend_y - 9, legend_x + 22, legend_y + 9), fill=colors[name])
        draw.text((legend_x + 31, legend_y - 13), label, font=plot.FONTS.get(15), fill=paper.INK)
        legend_x += 215 if name != "Projectors" else 265
    draw.line((legend_x, legend_y, legend_x + 42, legend_y), fill="#666666", width=4)
    draw.ellipse((legend_x + 15, legend_y - 6, legend_x + 27, legend_y + 6), fill="white", outline="#555555", width=3)
    draw.text((legend_x + 51, legend_y - 13), "Physical checkpoint allocation", font=plot.FONTS.get(15), fill=paper.INK)

    draw.line((left, top, left, bottom), fill=paper.INK, width=3)
    draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
    paper.draw_vertical_text(image, (35, (top + bottom) // 2), "World-model parameters (millions)", size=20, fill=paper.INK)
    draw.text((18, 18), title, font=plot.FONTS.get(25, bold=True), fill=paper.INK)
    plot.center_text(
        draw,
        ((left + right) / 2, 900),
        "Bars count shared + retained components for completed tasks; future task slots are excluded.",
        font=plot.FONTS.get(16),
        fill=paper.MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def _artifact_status(key: str) -> str:
    staged = len(list((ROOT / "runs" / "cloud_result_backups").glob(f"*/completed/**/{key}_s*.partial/continual_metrics.json")))
    failed = len(list((ROOT / "runs" / "cloud_result_backups").glob(f"*/metadata_snapshots/**/failed/{key}_s*/run_status.json")))
    if staged:
        return f"{staged} staged\nnot admitted" + (f"; {failed} failed" if failed else "")
    if failed:
        return f"{failed} failed\nretry pending"
    return "not started"


def _render_ablation_placeholder(output: Path) -> None:
    atari, _, _ = results._atari_results()
    full = atari["AWM-AutoRoute"]["metrics"]
    variants = (
        ("AWM full", None),
        ("NoReuse", _artifact_status("no_reuse")),
        ("NoFuncProt", _artifact_status("no_functional_protection")),
        ("NoConflict", _artifact_status("no_conflict_projection")),
        ("NoRCC", _artifact_status("no_rcc")),
    )
    specs = (
        ("Forgetting ↓", "forgetting", (-0.5, 2.5), (-0.5, 0, 0.5, 1, 1.5, 2)),
        ("ACC ↑", "acc", (-0.5, 3.5), (-0.5, 0, 1, 2, 3)),
        ("min-ACC ↑", "min_acc", (-0.5, 3.5), (-0.5, 0, 1, 2, 3)),
        ("WC-ACC ↑", "wc_acc", (-0.5, 3.0), (-0.5, 0, 1, 2, 3)),
    )
    size = (2400, 760)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    gap, margin = 22, 30
    width = (size[0] - 2 * margin - 3 * gap) // 4
    for panel_index, (title, key, limits, ticks) in enumerate(specs):
        x0 = margin + panel_index * (width + gap)
        x1 = x0 + width
        left, right, top, bottom = x0 + 72, x1 - 15, 72, 565
        low, high = limits

        def py(value: float) -> float:
            return bottom - (value - low) / (high - low) * (bottom - top)

        draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG)
        for tick in ticks:
            y_tick = py(tick)
            draw.line((left, y_tick, right, y_tick), fill=paper.GRID, width=2)
            plot.right_text(draw, (left - 9, y_tick), f"{tick:g}", font=plot.FONTS.get(14), fill=paper.MUTED)
        zero = py(0) if low <= 0 <= high else bottom
        slot = (right - left) / len(variants)
        median, q25, q75 = _metric_tuple(full[key])
        for index, (label, status) in enumerate(variants):
            center = left + slot * (index + 0.5)
            if index == 0:
                y_median = py(median)
                draw.rectangle((center - 24, min(zero, y_median), center + 24, max(zero, y_median)), fill=METHOD_COLORS["AWM-AutoRoute"], outline="#555555", width=2)
                draw.line((center, py(q25), center, py(q75)), fill="#333333", width=3)
                draw.line((center - 8, py(q25), center + 8, py(q25)), fill="#333333", width=3)
                draw.line((center - 8, py(q75), center + 8, py(q75)), fill="#333333", width=3)
            else:
                paper.dashed_line(draw, (center - 22, bottom - 24), (center + 22, bottom - 24), fill="#888888", width=3, dash=5, gap=4)
                plot.center_text(draw, (center, bottom - 45), "PENDING", font=plot.FONTS.get(11, bold=True), fill=paper.MUTED)
            plot.center_text(draw, (center, bottom + 28), label, font=plot.FONTS.get(12, bold=True), fill=paper.INK)
            if status:
                for line_index, line in enumerate(status.split("\n")):
                    plot.center_text(draw, (center, bottom + 49 + line_index * 16), line, font=plot.FONTS.get(10), fill=paper.MUTED)
        draw.line((left, top, left, bottom), fill=paper.INK, width=3)
        draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
        plot.center_text(draw, ((x0 + x1) / 2, 28), title, font=plot.FONTS.get(19, bold=True))
    note = "Only the completed AWM pilot is plotted. Staged or failed ablation artifacts are placeholders, never numerical evidence."
    plot.center_text(draw, (size[0] / 2, 715), note, font=plot.FONTS.get(16), fill=paper.MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def _render_diagnostic_placeholder(output: Path) -> None:
    size = (2400, 1320)
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    titles = (
        ("A", "Auto-route vs oracle return", "Task", "Raw return", "paired checkpoint evaluation"),
        ("B", "Routing confusion", "Predicted route", "True task", "task × route matrix"),
        ("C", "Reset switching latency", "Frames after reset", "Correct-route rate", "median [IQR]"),
        ("D", "Historical reuse intervention", "Task", "Return difference", "reuse on − reuse off"),
        ("E", "Retained Q/F/P width", "Task", "Retained width", "mechanism capacity"),
        ("F", "Predictive retention", "Open-loop horizon H", "Prediction error", "H = 1, 2, 4, 8, 16"),
    )
    margin_x, margin_y, gap_x, gap_y = 55, 48, 34, 42
    width = (size[0] - margin_x * 2 - gap_x * 2) // 3
    height = (size[1] - margin_y * 2 - gap_y) // 2
    for index, (letter, title, x_label, y_label, detail) in enumerate(titles):
        row, col = divmod(index, 3)
        x0 = margin_x + col * (width + gap_x)
        y0 = margin_y + row * (height + gap_y)
        x1, y1 = x0 + width, y0 + height
        left, right, top, bottom = x0 + 90, x1 - 24, y0 + 76, y1 - 92
        draw.rectangle((left, top, right, bottom), fill=paper.PANEL_BG, outline=paper.GRID, width=2)
        draw.line((left, top, left, bottom), fill=paper.INK, width=3)
        draw.line((left, bottom, right, bottom), fill=paper.INK, width=3)
        draw.text((x0 + 4, y0 + 8), letter, font=plot.FONTS.get(24, bold=True), fill=paper.INK)
        plot.center_text(draw, ((x0 + x1) / 2, y0 + 27), title, font=plot.FONTS.get(19, bold=True))
        plot.center_text(draw, ((left + right) / 2, bottom + 30), x_label, font=plot.FONTS.get(15), fill=paper.MUTED)
        paper.draw_vertical_text(image, (x0 + 22, (top + bottom) // 2), y_label, size=15, fill=paper.MUTED)
        plot.center_text(draw, ((left + right) / 2, (top + bottom) / 2 - 12), "PENDING", font=plot.FONTS.get(28, bold=True), fill="#888888")
        plot.center_text(draw, ((left + right) / 2, (top + bottom) / 2 + 28), detail, font=plot.FONTS.get(14), fill=paper.MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=95)


def build(output_dir: Path) -> None:
    _render(
        output_dir / "atari_main_learning_curves.png",
        ATARI_TASKS,
        _atari_matrices(),
        "Norm. perf.",
        (-0.5, 1.85),
        (-0.5, 0.0, 0.5, 1.0, 1.5),
    )
    _render(
        output_dir / "coinrun_main_learning_curves.png",
        COINRUN_TASKS,
        _coinrun_runs(),
        "Norm. perf.",
        (-0.25, 1.75),
        (0.0, 0.5, 1.0, 1.5),
    )
    _render(
        output_dir / "atari_main_learning_curves_full_range.png",
        ATARI_TASKS,
        _atari_matrices(),
        "Norm. perf.",
        None,
        None,
    )
    _render(
        output_dir / "coinrun_main_learning_curves_full_range.png",
        COINRUN_TASKS,
        _coinrun_runs(),
        "Norm. perf.",
        None,
        None,
    )
    _render_metric_summary(output_dir / "main_metric_summary_with_pending.png")
    _render_capacity_pareto(output_dir / "capacity_performance_pareto.png")
    _render_parameter_growth(
        output_dir / "awm_parameter_growth.png",
        ROOT / "runs" / "main_results" / "atari" / "awm_autoroute",
        results.AWM_ATARI_SEEDS,
        ATARI_TASKS,
        "AWM Atari parameter growth across tasks",
    )
    _render_parameter_growth(
        output_dir / "coinrun_awm_parameter_growth.png",
        ROOT / "runs" / "main_results" / "coinrun" / "awm_autoroute",
        results.AWM_COINRUN_SEEDS,
        ("CoinRun", "+NB", "+RT", "+GA", "+MA", "+CA"),
        "AWM CoinRun parameter growth across tasks",
    )
    _render_ablation_placeholder(output_dir / "ablation_results_placeholder.png")
    _render_diagnostic_placeholder(output_dir / "diagnostic_panels_placeholder.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary)
            build(candidate)
            for name in (
                "atari_main_learning_curves.png",
                "coinrun_main_learning_curves.png",
                "atari_main_learning_curves_full_range.png",
                "coinrun_main_learning_curves_full_range.png",
                "main_metric_summary_with_pending.png",
                "capacity_performance_pareto.png",
                "awm_parameter_growth.png",
                "coinrun_awm_parameter_growth.png",
                "ablation_results_placeholder.png",
                "diagnostic_panels_placeholder.png",
            ):
                if not (OUTPUT / name).is_file() or (OUTPUT / name).read_bytes() != (candidate / name).read_bytes():
                    print(f"out of date: {OUTPUT / name}", file=sys.stderr)
                    return 1
        print(f"up to date: {OUTPUT}")
    else:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary)
            build(candidate)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            for path in candidate.iterdir():
                shutil.copy2(path, OUTPUT / path.name)
                print(f"wrote {OUTPUT / path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
