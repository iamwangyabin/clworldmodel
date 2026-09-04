#!/usr/bin/env python3
"""Render paper-aligned Atari ARROW figures using only local experiment data.

The layout follows the ARROW paper's Atari Figure 3A, Figure 4A, and selected
appendix tables.  No numerical result from the paper is copied into any output.
The report uses five local continual ARROW-50 seeds and three local single-task
ARROW-50 seeds per game.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import plot_arrow_atari_ours_only as base  # noqa: E402
from clworldmodel.evaluation.metrics import forward_transfer  # noqa: E402


DEFAULT_OUTPUT = base.DEFAULT_OUTPUT / "paper_aligned"
PAPER_URL = "https://arxiv.org/html/2603.11395v3"
FRAMES_PER_EPOCH = 32 * 512
FRAME_REPEAT = 4
# Palette and dash grammar sampled from the paper's vector Figure 3.  Only the
# appearance is reproduced here; every plotted value still comes from our runs.
PAPER_TASK_STYLES: tuple[dict[str, Any], ...] = (
    {"color": "#2E86C1", "dash": None},
    {"color": "#E74C3C", "dash": (3.7, 1.6)},
    {"color": "#8E44AD", "dash": (6.4, 1.6, 1.0, 1.6)},
    {"color": "#27AE60", "dash": (1.0, 1.65)},
    {"color": "#F39C12", "dash": (3.0, 1.0, 1.0, 1.0)},
    {"color": "#808080", "dash": (5.0, 1.0)},
)
TASK_COLORS = tuple(style["color"] for style in PAPER_TASK_STYLES)
PAPER_NEUTRAL_BAR = "#D4C8A8"
PAPER_ACC_COLORS = ("#9DB1C6", "#DFA17F", "#A3C69D")
PAPER_METHOD_LINE = "#6F91B2"
INK = "#262626"
MUTED = "#666666"
GRID = "#D5D5D5"
PANEL_BG = "#F8F8F8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the inventory without writing outputs.",
    )
    return parser.parse_args()


def load_sources() -> tuple[
    list[base.Run], tuple[str, ...], dict[str, list[base.Run]], dict[str, Any]
]:
    continual = [
        base.load_run(path, expected_epochs=base.EXPECTED_CONTINUAL_EPOCHS)
        for path in base.CONTINUAL_RUNS
    ]
    continual.sort(key=lambda run: run.seed_id)
    if [run.seed_id for run in continual] != list(range(5)):
        raise ValueError("Expected continual seed IDs 0 through 4")
    task_order = continual[0].task_names
    if any(run.task_names != task_order for run in continual):
        raise ValueError("Continual task order differs between seeds")
    if tuple(base.TASK_DISPLAY) != task_order:
        raise ValueError(f"Unexpected task order: {task_order}")

    grouped: dict[str, list[base.Run]] = {task: [] for task in task_order}
    for run in base.discover_single_task_runs():
        if len(run.task_names) != 1 or run.task_names[0] not in grouped:
            raise ValueError(f"Unexpected single-task run: {run.run_dir}")
        grouped[run.task_names[0]].append(run)
    for task, runs in grouped.items():
        runs.sort(key=lambda run: run.seed_id)
        if [run.seed_id for run in runs] != [0, 1, 2]:
            raise ValueError(f"Expected local single-task seeds 0,1,2 for {task}")

    config = base.json_object(base.config_path(grouped[task_order[0]][0].run_dir))
    expected_frames = (int(config["epochs"]) - 1) * int(config["data_n"]) * int(
        config["data_t"]
    )
    if expected_frames != 90 * FRAMES_PER_EPOCH:
        raise ValueError("Unexpected single-task interaction budget")
    return continual, task_order, grouped, config


def build_references(
    task_order: Sequence[str], grouped: dict[str, list[base.Run]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[int, float]]]:
    endpoint: dict[str, dict[str, float]] = {}
    curves: dict[str, dict[int, float]] = {}
    for task in task_order:
        runs = grouped[task]
        initial = base.stats([run.by_epoch[0].raw_mean[0] for run in runs])
        final = base.stats([run.by_epoch[90].raw_mean[0] for run in runs])
        endpoint[task] = {
            "initial_median": initial["median"],
            "initial_q25": initial["q25"],
            "initial_q75": initial["q75"],
            "final_median": final["median"],
            "final_q25": final["q25"],
            "final_q75": final["q75"],
        }
        curves[task] = {
            epoch: base.stats(
                [run.by_epoch[epoch].raw_mean[0] for run in runs]
            )["median"]
            for epoch in base.EXPECTED_SINGLE_TASK_EPOCHS
        }
        for epoch in base.EXPECTED_SINGLE_TASK_EPOCHS[1:]:
            if curves[task][epoch] == curves[task][0]:
                raise ValueError(
                    f"Zero time-aligned normalization denominator for {task} E{epoch}"
                )
    return endpoint, curves


def endpoint_score(raw: float, task: str, references: dict[str, dict[str, float]]) -> float:
    reference = references[task]
    return (raw - reference["initial_median"]) / (
        reference["final_median"] - reference["initial_median"]
    )


def aligned_score(
    raw: float,
    task: str,
    local_epoch: int,
    endpoint: dict[str, dict[str, float]],
    st_curves: dict[str, dict[int, float]],
) -> float:
    initial = endpoint[task]["initial_median"]
    return (raw - initial) / (st_curves[task][local_epoch] - initial)


def median_iqr(values: Sequence[float]) -> tuple[float, float, float]:
    summary = base.stats(values)
    return summary["median"], summary["q25"], summary["q75"]


def format_interval(values: Sequence[float], digits: int = 3) -> str:
    median, q25, q75 = median_iqr(values)
    return f"{median:.{digits}f} [{q25:.{digits}f} – {q75:.{digits}f}]"


def format_raw_interval(values: Sequence[float]) -> str:
    median, q25, q75 = median_iqr(values)
    return (
        f"{base.raw_format(median)} "
        f"[{base.raw_format(q25)} – {base.raw_format(q75)}]"
    )


def format_frame_interval(values: Sequence[float]) -> str:
    median, q25, q75 = median_iqr(values)
    return f"{median:,.0f} [{q25:,.0f} – {q75:,.0f}]"


def endpoint_matrix(
    run: base.Run,
    task_order: Sequence[str],
    references: dict[str, dict[str, float]],
) -> list[list[float]]:
    return [
        [
            endpoint_score(point.raw_mean[index], task, references)
            for index, task in enumerate(task_order)
        ]
        for point in run.evaluations
    ]


def compute_metrics(
    continual: Sequence[base.Run],
    task_order: Sequence[str],
    endpoint: dict[str, dict[str, float]],
    st_curves: dict[str, dict[int, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    ft_task_rows: list[dict[str, Any]] = []
    row_by_epoch = {
        epoch: index for index, epoch in enumerate(base.EXPECTED_CONTINUAL_EPOCHS)
    }
    ends = [row_by_epoch[epoch] for epoch in base.BOUNDARY_EPOCHS]

    for run in continual:
        matrix = endpoint_matrix(run, task_order, endpoint)
        metrics = base.single_pass_metrics(matrix, ends)
        continual_curves: list[list[float]] = []
        single_curves: list[list[float]] = []
        for task_index, task in enumerate(task_order):
            curve = []
            for local_epoch in base.EXPECTED_SINGLE_TASK_EPOCHS[1:]:
                global_epoch = task_index * 90 + local_epoch
                raw = run.by_epoch[global_epoch].raw_mean[task_index]
                curve.append(
                    aligned_score(raw, task, local_epoch, endpoint, st_curves)
                )
            continual_curves.append(curve)
            single_curves.append([1.0] * len(curve))
        transfer = forward_transfer(continual_curves, single_curves)
        metric_rows.append(
            {
                "seed_id": run.seed_id,
                "seed": run.seed,
                "forgetting": metrics["forgetting"],
                "forward_transfer": transfer["forward_transfer"],
                "acc": metrics["acc"],
                "min_acc": metrics["min_acc"],
                "wc_acc": metrics["wc_acc"],
            }
        )
        for task_index, task in enumerate(task_order):
            ft_task_rows.append(
                {
                    "seed_id": run.seed_id,
                    "seed": run.seed,
                    "task_index": task_index,
                    "task": base.TASK_DISPLAY[task],
                    "continual_area_discrete": transfer["continual_areas"][task_index],
                    "single_task_area_discrete": transfer["single_task_areas"][task_index],
                    "forward_transfer": transfer["per_task_forward_transfer"][task_index],
                    "evaluation_grid": "local epochs 10..90 every 10 epochs",
                }
            )
    return metric_rows, ft_task_rows


def first_crossing(
    epochs: Sequence[int], values: Sequence[float], threshold: float
) -> int | None:
    for epoch, value in zip(epochs, values):
        if value >= threshold:
            return epoch * FRAMES_PER_EPOCH
    return None


def compute_continual_sample_efficiency(
    continual: Sequence[base.Run],
    task_order: Sequence[str],
    endpoint: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_curves: dict[int, list[float]] = {}
    for run in continual:
        matrix = endpoint_matrix(run, task_order, endpoint)
        seed_curves[run.seed_id] = [sum(row) / len(row) for row in matrix]

    curve_rows: list[dict[str, Any]] = []
    median_curve: list[float] = []
    for point_index, epoch in enumerate(base.EXPECTED_CONTINUAL_EPOCHS):
        values = [seed_curves[seed_id][point_index] for seed_id in sorted(seed_curves)]
        median, q25, q75 = median_iqr(values)
        median_curve.append(median)
        curve_rows.append(
            {
                "epoch": epoch,
                "environment_frames": epoch * FRAMES_PER_EPOCH,
                "median": median,
                "q25": q25,
                "q75": q75,
            }
        )
    peak = max(median_curve)
    peak_index = median_curve.index(peak)
    threshold = 0.85 * peak
    median_crossing = first_crossing(
        base.EXPECTED_CONTINUAL_EPOCHS, median_curve, threshold
    )
    seed_rows: list[dict[str, Any]] = []
    reached: list[float] = []
    for run in continual:
        crossing = first_crossing(
            base.EXPECTED_CONTINUAL_EPOCHS,
            seed_curves[run.seed_id],
            threshold,
        )
        if crossing is not None:
            reached.append(float(crossing))
        seed_rows.append(
            {
                "seed_id": run.seed_id,
                "seed": run.seed,
                "maximum": max(seed_curves[run.seed_id]),
                "first_crossing_frames": crossing,
                "reached": crossing is not None,
            }
        )
    summary = {
        "threshold_rule": "0.85 * maximum of our ARROW-only five-seed median curve",
        "paper_equivalent": False,
        "reason_not_paper_equivalent": (
            "The paper uses a shared maximum across ARROW, DreamerV3, and TES-SAC; "
            "this report intentionally contains only our ARROW results."
        ),
        "median_curve_peak": peak,
        "peak_epoch": base.EXPECTED_CONTINUAL_EPOCHS[peak_index],
        "peak_environment_frames": base.EXPECTED_CONTINUAL_EPOCHS[peak_index]
        * FRAMES_PER_EPOCH,
        "threshold": threshold,
        "median_curve_first_crossing_frames": median_crossing,
        "runs_reached": len(reached),
        "runs_total": len(continual),
        "reached_seed_crossing_median": (
            base.stats(reached)["median"] if reached else None
        ),
        "reached_seed_crossing_q25": base.stats(reached)["q25"] if reached else None,
        "reached_seed_crossing_q75": base.stats(reached)["q75"] if reached else None,
    }
    return curve_rows, seed_rows, summary


def compute_single_task_efficiency(
    task_order: Sequence[str], grouped: dict[str, list[base.Run]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    crossings: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_order):
        median_curve = [
            base.stats(
                [run.by_epoch[epoch].raw_mean[0] for run in grouped[task]]
            )["median"]
            for epoch in base.EXPECTED_SINGLE_TASK_EPOCHS
        ]
        maximum = max(median_curve)
        peak_epoch = base.EXPECTED_SINGLE_TASK_EPOCHS[median_curve.index(maximum)]
        threshold = 0.85 * maximum
        reached: list[float] = []
        for run in grouped[task]:
            values = [
                run.by_epoch[epoch].raw_mean[0]
                for epoch in base.EXPECTED_SINGLE_TASK_EPOCHS
            ]
            crossing = first_crossing(
                base.EXPECTED_SINGLE_TASK_EPOCHS, values, threshold
            )
            if crossing is not None:
                reached.append(float(crossing))
            crossings.append(
                {
                    "task_index": task_index,
                    "task": base.TASK_DISPLAY[task],
                    "seed_id": run.seed_id,
                    "seed": run.seed,
                    "threshold": threshold,
                    "first_crossing_frames": crossing,
                    "reached": crossing is not None,
                }
            )
        reached_stats = base.stats(reached) if reached else None
        summaries.append(
            {
                "task_index": task_index,
                "task": base.TASK_DISPLAY[task],
                "threshold_85pct_ours_only": threshold,
                "arrow_max_median_raw_return": maximum,
                "peak_epoch": peak_epoch,
                "peak_environment_frames": peak_epoch * FRAMES_PER_EPOCH,
                "crossing_frames_median": (
                    reached_stats["median"] if reached_stats else None
                ),
                "crossing_frames_q25": reached_stats["q25"] if reached_stats else None,
                "crossing_frames_q75": reached_stats["q75"] if reached_stats else None,
                "runs_reached": len(reached),
                "runs_total": len(grouped[task]),
                "threshold_scope": "our ARROW single-task runs only",
                "paper_equivalent": False,
            }
        )
    return summaries, crossings


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    """Render multiline image-table cells safely in Markdown."""

    clean_headers = [value.replace("\n", "<br>") for value in headers]
    clean_rows = [
        [value.replace("\n", "<br>") for value in row]
        for row in rows
    ]
    return base.markdown_table(clean_headers, clean_rows)


def draw_vertical_text(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    *,
    size: int = 30,
    fill: str = INK,
) -> None:
    font = base.FONTS.get(size)
    scratch = Image.new("RGBA", (1000, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scratch)
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0] + 16, box[3] - box[1] + 16
    scratch = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    ImageDraw.Draw(scratch).text((8, 8), text, font=font, fill=fill)
    rotated = scratch.rotate(90, expand=True)
    image.alpha_composite(
        rotated,
        (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)),
    )


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int = 2,
    dash: int = 10,
    gap: int = 7,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    cursor = 0.0
    while cursor < length:
        stop = min(cursor + dash, length)
        draw.line(
            (x0 + ux * cursor, y0 + uy * cursor, x0 + ux * stop, y0 + uy * stop),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def styled_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    *,
    fill: str,
    width: int,
    dash_pattern: Sequence[float] | None = None,
) -> None:
    """Draw one continuous solid/dashed polyline without matplotlib.

    Pillow does not expose dash patterns for polylines.  Maintaining the dash
    cursor across every segment avoids the visual restarts that occur when each
    checkpoint-to-checkpoint edge is drawn independently.
    """

    if len(points) < 2:
        return
    if not dash_pattern:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    pattern = [max(0.5, float(length)) for length in dash_pattern]
    if len(pattern) % 2:
        pattern *= 2
    pattern_index = 0
    pattern_remaining = pattern[0]
    drawing = True
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        x1, y1 = end
        segment_length = math.hypot(x1 - x0, y1 - y0)
        if segment_length <= 1e-9:
            continue
        ux, uy = (x1 - x0) / segment_length, (y1 - y0) / segment_length
        cursor = 0.0
        while cursor < segment_length - 1e-9:
            advance = min(pattern_remaining, segment_length - cursor)
            if drawing:
                draw.line(
                    (
                        x0 + ux * cursor,
                        y0 + uy * cursor,
                        x0 + ux * (cursor + advance),
                        y0 + uy * (cursor + advance),
                    ),
                    fill=fill,
                    width=width,
                )
            cursor += advance
            pattern_remaining -= advance
            if pattern_remaining <= 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_remaining = pattern[pattern_index]
                drawing = not drawing


def scaled_dash(
    pattern: Sequence[float] | None,
    *,
    scale: float,
) -> tuple[float, ...] | None:
    if pattern is None:
        return None
    return tuple(float(value) * scale for value in pattern)


def fig3_summaries_from_rows(
    curve_rows: Sequence[dict[str, Any]],
    task_count: int,
) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(task_count)]
    for row in curve_rows:
        grouped[int(row["task_index"])].append(row)
    for rows in grouped:
        rows.sort(key=lambda row: int(row["epoch"]))
        if len(rows) != len(base.EXPECTED_CONTINUAL_EPOCHS):
            raise ValueError("Figure 3 curve row count does not match evaluation grid")
    return grouped


def draw_fig3a(
    path: Path,
    continual: Sequence[base.Run],
    task_order: Sequence[str],
    endpoint: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    width, height = 2400, 1300
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    base.center_text(
        draw,
        (width / 2, 50),
        "Atari normalized performance — our ARROW-50",
        font=base.FONTS.get(42, bold=True),
        fill=INK,
    )
    base.center_text(
        draw,
        (width / 2, 100),
        "Figure 3A-aligned subset · default order · median and IQR across 5 local seeds",
        font=base.FONTS.get(24),
        fill=MUTED,
    )

    legend = [(base.TASK_DISPLAY[task], TASK_COLORS[i]) for i, task in enumerate(task_order)]
    font = base.FONTS.get(23)
    widths = [base.text_size(draw, label, font)[0] + 78 for label, _ in legend]
    cursor = (width - sum(widths)) / 2
    for (label, color), cell_width in zip(legend, widths):
        draw.line((cursor, 160, cursor + 44, 160), fill=color, width=7)
        draw.text((cursor + 55, 147), label, font=font, fill=INK)
        cursor += cell_width

    left, right, top, bottom = 190, 2325, 220, 1085
    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)
    x_min, x_max = 0.0, 540 * FRAMES_PER_EPOCH
    y_min, y_max = -0.5, 2.25

    def px(frame: float) -> float:
        return left + (frame - x_min) / (x_max - x_min) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    y_ticks = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    for tick in y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 18, y),
            f"{tick:.1f}",
            font=base.FONTS.get(23),
            fill=MUTED,
        )
    dashed_line(draw, (left, py(1.0)), (right, py(1.0)), fill="#8C8C8C", width=3)

    for boundary in base.BOUNDARY_EPOCHS[:-1]:
        x = px(boundary * FRAMES_PER_EPOCH)
        dashed_line(draw, (x, top), (x, bottom), fill="#BDBDBD", width=2)

    x_ticks = (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000)
    for tick in x_ticks:
        x = px(tick)
        draw.line((x, bottom, x, bottom + 10), fill=INK, width=3)
        base.center_text(
            draw,
            (x, bottom + 38),
            f"{tick / 1_000_000:g}",
            font=base.FONTS.get(24),
            fill=MUTED,
        )

    curve_rows: list[dict[str, Any]] = []
    x_frames = [epoch * FRAMES_PER_EPOCH for epoch in base.EXPECTED_CONTINUAL_EPOCHS]
    summaries_by_task: list[list[tuple[float, float, float]]] = []
    for task_index, task in enumerate(task_order):
        task_summaries: list[tuple[float, float, float]] = []
        for epoch in base.EXPECTED_CONTINUAL_EPOCHS:
            values = [
                endpoint_score(run.by_epoch[epoch].raw_mean[task_index], task, endpoint)
                for run in continual
            ]
            median, q25, q75 = median_iqr(values)
            task_summaries.append((median, q25, q75))
            curve_rows.append(
                {
                    "epoch": epoch,
                    "environment_frames": epoch * FRAMES_PER_EPOCH,
                    "task_index": task_index,
                    "task": base.TASK_DISPLAY[task],
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "task_is_active": task_index * 90 <= epoch <= (task_index + 1) * 90,
                }
            )
        summaries_by_task.append(task_summaries)

    for task_index, summaries in enumerate(summaries_by_task):
        color = TASK_COLORS[task_index]
        upper = [(px(frame), py(row[2])) for frame, row in zip(x_frames, summaries)]
        lower = [
            (px(frame), py(row[1]))
            for frame, row in reversed(list(zip(x_frames, summaries)))
        ]
        base.alpha_polygon(image, upper + lower, color, 34)

    draw = ImageDraw.Draw(image)
    for task_index, summaries in enumerate(summaries_by_task):
        color = TASK_COLORS[task_index]
        points = [
            (px(frame), py(row[0])) for frame, row in zip(x_frames, summaries)
        ]
        draw.line(points, fill=color, width=4, joint="curve")
        start_epoch, end_epoch = task_index * 90, (task_index + 1) * 90
        active_points = [
            point
            for epoch, point in zip(base.EXPECTED_CONTINUAL_EPOCHS, points)
            if start_epoch <= epoch <= end_epoch
        ]
        draw.line(active_points, fill=color, width=11, joint="curve")

    draw.line((left, top, left, bottom), fill=INK, width=4)
    draw.line((left, bottom, right, bottom), fill=INK, width=4)
    base.center_text(
        draw,
        ((left + right) / 2, bottom + 92),
        "Environment frames",
        font=base.FONTS.get(30),
        fill=INK,
    )
    draw.text(
        (right - 48, bottom + 18),
        "× 10^6",
        font=base.FONTS.get(21),
        fill=MUTED,
    )
    draw_vertical_text(
        image,
        (54, int((top + bottom) / 2)),
        "Local normalized performance",
        size=29,
    )
    draw = ImageDraw.Draw(image)
    base.center_text(
        draw,
        (width / 2, 1245),
        "Fixed local E0→E90 single-task anchors; bold segment = task being trained; no paper result values used.",
        font=base.FONTS.get(22),
        fill=MUTED,
    )
    base.save_image(image, path)
    return curve_rows


def draw_fig3a_strict(
    path: Path,
    task_order: Sequence[str],
    curve_rows: Sequence[dict[str, Any]],
) -> None:
    """Render the ARROW/default-order cell with the paper's visual grammar.

    This intentionally uses the paper panel's fixed y-range.  A separate
    full-range rendering is generated alongside it so values outside the
    publication viewport are never hidden from the analysis bundle.
    """

    width, height = 1800, 1080
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 1760, 105, 885
    plot_width, plot_height = right - left, bottom - top
    x_min, x_max = 0.0, 540 * FRAMES_PER_EPOCH
    y_min, y_max = -0.5, 1.85

    def px(frame: float) -> float:
        return left + (frame - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    def local_px(frame: float) -> float:
        return (frame - x_min) / (x_max - x_min) * plot_width

    def local_py(value: float) -> float:
        return plot_height - (value - y_min) / (y_max - y_min) * plot_height

    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)
    for tick in (-0.5, 0.0, 0.5, 1.0, 1.5):
        y = py(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 16, y),
            f"{tick:.1f}",
            font=base.FONTS.get(24),
            fill=INK,
        )
    dashed_line(
        draw,
        (left, py(0.0)),
        (right, py(0.0)),
        fill="#808080",
        width=2,
        dash=9,
        gap=8,
    )
    for boundary in base.BOUNDARY_EPOCHS[:-1]:
        x = px(boundary * FRAMES_PER_EPOCH)
        dashed_line(
            draw,
            (x, top),
            (x, bottom),
            fill="#B8B8B8",
            width=2,
            dash=8,
            gap=8,
        )

    grouped = fig3_summaries_from_rows(curve_rows, len(task_order))
    plot_layer = Image.new("RGBA", (plot_width + 1, plot_height + 1), (0, 0, 0, 0))
    # IQR opacity is 0.2, matching the vector figure.
    for task_index, rows in enumerate(grouped):
        color = str(PAPER_TASK_STYLES[task_index]["color"])
        upper = [
            (local_px(float(row["environment_frames"])), local_py(float(row["q75"])))
            for row in rows
        ]
        lower = [
            (local_px(float(row["environment_frames"])), local_py(float(row["q25"])))
            for row in reversed(rows)
        ]
        base.alpha_polygon(plot_layer, upper + lower, color, 51)

    layer_draw = ImageDraw.Draw(plot_layer)
    for task_index, rows in enumerate(grouped):
        style = PAPER_TASK_STYLES[task_index]
        color = str(style["color"])
        points = [
            (local_px(float(row["environment_frames"])), local_py(float(row["median"])))
            for row in rows
        ]
        epochs = [int(row["epoch"]) for row in rows]
        start_epoch, end_epoch = task_index * 90, (task_index + 1) * 90
        before = [point for epoch, point in zip(epochs, points) if epoch <= start_epoch]
        active = [
            point
            for epoch, point in zip(epochs, points)
            if start_epoch <= epoch <= end_epoch
        ]
        after = [point for epoch, point in zip(epochs, points) if epoch >= end_epoch]
        thin_dash = scaled_dash(style["dash"], scale=9.0)
        bold_dash = scaled_dash(style["dash"], scale=22.5)
        if len(before) >= 2:
            styled_polyline(
                layer_draw,
                before,
                fill=color,
                width=7,
                dash_pattern=thin_dash,
            )
        styled_polyline(
            layer_draw,
            active,
            fill=color,
            width=18,
            dash_pattern=bold_dash,
        )
        if len(after) >= 2:
            styled_polyline(
                layer_draw,
                after,
                fill=color,
                width=7,
                dash_pattern=thin_dash,
            )
    image.alpha_composite(plot_layer, (left, top))
    draw = ImageDraw.Draw(image)

    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for tick in (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000):
        x = px(tick)
        draw.line((x, bottom, x, bottom + 9), fill=INK, width=3)
        base.center_text(
            draw,
            (x, bottom + 32),
            f"{tick / 1_000_000:g}",
            font=base.FONTS.get(23),
            fill=INK,
        )
    draw.text(
        (right - 35, bottom + 43),
        "1e6",
        font=base.FONTS.get(20),
        fill=INK,
    )
    draw.text((8, 13), "A", font=base.FONTS.get(34), fill=INK)
    base.center_text(
        draw,
        ((left + right) / 2, 45),
        "ARROW (ours)",
        font=base.FONTS.get(30, bold=True),
        fill=INK,
    )
    draw_vertical_text(
        image,
        (35, int((top + bottom) / 2)),
        "Norm. perf.",
        size=26,
        fill=INK,
    )

    # The original legend is a compact two-row, three-column block.
    draw = ImageDraw.Draw(image)
    legend_font = base.FONTS.get(21)
    for task_index, task in enumerate(task_order):
        column = task_index // 2
        row = task_index % 2
        sample_x = 235 + column * 515
        sample_y = 975 + row * 43
        style = PAPER_TASK_STYLES[task_index]
        styled_polyline(
            draw,
            [(sample_x, sample_y), (sample_x + 105, sample_y)],
            fill=str(style["color"]),
            width=7,
            dash_pattern=scaled_dash(style["dash"], scale=9.0),
        )
        draw.text(
            (sample_x + 122, sample_y - 13),
            base.TASK_DISPLAY[task],
            font=legend_font,
            fill=INK,
        )
    base.save_image(image, path)


def draw_bar_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[tuple[float, float, float]],
    colors: Sequence[str],
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 90, x1 - 25, y0 + 75, y1 - 100
    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for tick in y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 12, y),
            f"{tick:.1f}",
            font=base.FONTS.get(18),
            fill=MUTED,
        )
    zero = py(0.0) if y_min <= 0 <= y_max else bottom
    draw.line((left, zero, right, zero), fill="#777777", width=3)
    slot = (right - left) / len(labels)
    bar_width = min(95, slot * 0.48)
    for index, (label, (median, q25, q75), color) in enumerate(
        zip(labels, values, colors)
    ):
        x = left + slot * (index + 0.5)
        y_median = py(median)
        draw.rectangle(
            (x - bar_width / 2, min(zero, y_median), x + bar_width / 2, max(zero, y_median)),
            fill=color,
            outline=INK,
            width=2,
        )
        draw.line((x, py(q25), x, py(q75)), fill=INK, width=4)
        draw.line((x - 13, py(q25), x + 13, py(q25)), fill=INK, width=4)
        draw.line((x - 13, py(q75), x + 13, py(q75)), fill=INK, width=4)
        base.center_text(
            draw,
            (x, bottom + 37),
            label,
            font=base.FONTS.get(18),
            fill=INK,
        )
        label_y = py(q75) - 25 if q75 >= 0 else py(q25) + 25
        base.center_text(
            draw,
            (x, label_y),
            f"{median:.3f}",
            font=base.FONTS.get(17, bold=True),
            fill=INK,
        )
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    base.center_text(
        draw,
        ((x0 + x1) / 2, y0 + 28),
        title,
        font=base.FONTS.get(25, bold=True),
        fill=INK,
    )


def draw_sample_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    curve_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 90, x1 - 25, y0 + 75, y1 - 100
    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)
    x_max = 540 * FRAMES_PER_EPOCH
    y_min = min(-0.2, min(float(row["q25"]) for row in curve_rows))
    y_max = max(1.05, max(float(row["q75"]) for row in curve_rows) * 1.05)
    y_min = math.floor(y_min * 5) / 5
    y_max = math.ceil(y_max * 5) / 5

    def px(value: float) -> float:
        return left + value / x_max * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    y_tick = math.ceil(y_min * 2) / 2
    while y_tick <= y_max + 1e-9:
        y = py(y_tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 12, y),
            f"{y_tick:.1f}",
            font=base.FONTS.get(18),
            fill=MUTED,
        )
        y_tick += 0.5
    for tick in (0, 4_000_000, 8_000_000):
        x = px(tick)
        draw.line((x, bottom, x, bottom + 8), fill=INK, width=2)
        base.center_text(
            draw,
            (x, bottom + 34),
            f"{tick / 1_000_000:g}",
            font=base.FONTS.get(18),
            fill=MUTED,
        )
    upper = [(px(row["environment_frames"]), py(row["q75"])) for row in curve_rows]
    lower = [
        (px(row["environment_frames"]), py(row["q25"]))
        for row in reversed(curve_rows)
    ]
    base.alpha_polygon(image, upper + lower, "#4C78A8", 48)
    draw = ImageDraw.Draw(image)
    points = [(px(row["environment_frames"]), py(row["median"])) for row in curve_rows]
    draw.line(points, fill="#4C78A8", width=6, joint="curve")
    threshold = float(summary["threshold"])
    dashed_line(draw, (left, py(threshold)), (right, py(threshold)), fill="#D62728", width=3)
    crossing = summary["median_curve_first_crossing_frames"]
    if crossing is not None:
        dashed_line(draw, (px(crossing), top), (px(crossing), bottom), fill="#444444", width=3)
    draw.text(
        (left + 12, py(threshold) - 31),
        f"85% self-peak = {threshold:.3f}",
        font=base.FONTS.get(17, bold=True),
        fill="#B22222",
    )
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    base.center_text(
        draw,
        ((x0 + x1) / 2, y0 + 28),
        "Sample efficiency*",
        font=base.FONTS.get(25, bold=True),
        fill=INK,
    )
    base.center_text(
        draw,
        ((left + right) / 2, bottom + 74),
        "Environment frames (× 10^6)",
        font=base.FONTS.get(18),
        fill=INK,
    )


def draw_fig4a(
    path: Path,
    metric_rows: Sequence[dict[str, Any]],
    sample_curve: Sequence[dict[str, Any]],
    sample_summary: dict[str, Any],
) -> None:
    width, height = 2800, 1120
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    base.center_text(
        draw,
        (width / 2, 48),
        "A  Atari metrics — default task order (our ARROW-50)",
        font=base.FONTS.get(40, bold=True),
        fill=INK,
    )
    base.center_text(
        draw,
        (width / 2, 95),
        "Figure 4A-aligned subset · bars are median with 0.25–0.75 quantiles across 5 local seeds",
        font=base.FONTS.get(23),
        fill=MUTED,
    )

    left_margin, gap, top, bottom = 30, 28, 145, 900
    panel_widths = (570, 570, 790, 760)
    boxes: list[tuple[int, int, int, int]] = []
    x = left_margin
    for panel_width in panel_widths:
        boxes.append((x, top, x + panel_width, bottom))
        x += panel_width + gap

    forgetting = median_iqr([float(row["forgetting"]) for row in metric_rows])
    transfer = median_iqr([float(row["forward_transfer"]) for row in metric_rows])
    acc = median_iqr([float(row["acc"]) for row in metric_rows])
    min_acc = median_iqr([float(row["min_acc"]) for row in metric_rows])
    wc_acc = median_iqr([float(row["wc_acc"]) for row in metric_rows])
    draw_bar_panel(
        image,
        boxes[0],
        title="Forgetting ↓",
        labels=("AR50\nours",),
        values=(forgetting,),
        colors=("#4C78A8",),
        y_min=-0.1,
        y_max=0.35,
        y_ticks=(-0.1, 0.0, 0.1, 0.2, 0.3),
    )
    draw_bar_panel(
        image,
        boxes[1],
        title="Forward transfer ↑",
        labels=("AR50\nours",),
        values=(transfer,),
        colors=("#4C78A8",),
        y_min=-0.2,
        y_max=0.2,
        y_ticks=(-0.2, -0.1, 0.0, 0.1, 0.2),
    )
    draw_bar_panel(
        image,
        boxes[2],
        title="Stability–plasticity ↑",
        labels=("ACC", "min-ACC", "WC-ACC"),
        values=(acc, min_acc, wc_acc),
        colors=("#4C78A8", "#F58518", "#54A24B"),
        y_min=0.0,
        y_max=1.0,
        y_ticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    draw_sample_panel(image, boxes[3], sample_curve, sample_summary)
    draw = ImageDraw.Draw(image)
    note = (
        "Local references only: endpoint metrics use our 3-seed E0/E90 anchors; FT uses time-aligned E10–E90 "
        "single-task medians. *Self-peak threshold is diagnostic, not the paper's cross-method threshold."
    )
    for line_index, line in enumerate(textwrap.wrap(note, width=180)):
        base.center_text(
            draw,
            (width / 2, 988 + line_index * 31),
            line,
            font=base.FONTS.get(21),
            fill=MUTED,
        )
    base.save_image(image, path)


def draw_strict_bar_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    ylabel: str,
    values: Sequence[tuple[float, float, float]],
    colors: Sequence[str],
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
) -> None:
    """Draw one compact Figure-4-style bar panel for our ARROW group."""

    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 82, x1 - 14, y0 + 74, y1 - 80
    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for tick in y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 10, y),
            f"{tick:g}",
            font=base.FONTS.get(17),
            fill=INK,
        )
    if y_min <= 0 <= y_max:
        draw.line((left, py(0.0), right, py(0.0)), fill="#777777", width=2)

    count = len(values)
    bar_width = 58 if count > 1 else 70
    gap = 16
    group_width = count * bar_width + (count - 1) * gap
    group_left = (left + right - group_width) / 2
    zero = py(0.0) if y_min <= 0 <= y_max else bottom
    for index, ((median, q25, q75), color) in enumerate(zip(values, colors)):
        center = group_left + index * (bar_width + gap) + bar_width / 2
        y_median = py(median)
        draw.rectangle(
            (
                center - bar_width / 2,
                min(zero, y_median),
                center + bar_width / 2,
                max(zero, y_median),
            ),
            fill=color,
            outline="#555555",
            width=2,
        )
        draw.line((center, py(q25), center, py(q75)), fill="#444444", width=3)
        draw.line((center - 10, py(q25), center + 10, py(q25)), fill="#444444", width=3)
        draw.line((center - 10, py(q75), center + 10, py(q75)), fill="#444444", width=3)

    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    base.center_text(
        draw,
        ((x0 + x1) / 2, y0 + 27),
        title,
        font=base.FONTS.get(22),
        fill=INK,
    )
    base.center_text(
        draw,
        ((left + right) / 2, bottom + 31),
        "ARROW",
        font=base.FONTS.get(18, bold=True),
        fill=INK,
    )
    draw_vertical_text(
        image,
        (x0 + 16, int((top + bottom) / 2)),
        ylabel,
        size=18,
        fill=INK,
    )


def draw_strict_sample_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    curve_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 82, x1 - 14, y0 + 74, y1 - 80
    plot_width, plot_height = right - left, bottom - top
    x_max, y_min, y_max = 540 * FRAMES_PER_EPOCH, 0.0, 3.0

    def px(value: float) -> float:
        return left + value / x_max * plot_width

    def py(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    def local_px(value: float) -> float:
        return value / x_max * plot_width

    def local_py(value: float) -> float:
        return plot_height - (value - y_min) / (y_max - y_min) * plot_height

    draw.rectangle((left, top, right, bottom), fill=PANEL_BG)
    for tick in (0.0, 1.0, 2.0, 3.0):
        y = py(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        base.right_text(
            draw,
            (left - 10, y),
            f"{tick:.1f}",
            font=base.FONTS.get(17),
            fill=INK,
        )
    for tick in (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000):
        x = px(tick)
        draw.line((x, bottom, x, bottom + 7), fill=INK, width=2)
        base.center_text(
            draw,
            (x, bottom + 27),
            f"{tick / 1_000_000:g}",
            font=base.FONTS.get(16),
            fill=INK,
        )

    plot_layer = Image.new("RGBA", (plot_width + 1, plot_height + 1), (0, 0, 0, 0))
    upper = [
        (local_px(float(row["environment_frames"])), local_py(float(row["q75"])))
        for row in curve_rows
    ]
    lower = [
        (local_px(float(row["environment_frames"])), local_py(float(row["q25"])))
        for row in reversed(curve_rows)
    ]
    base.alpha_polygon(plot_layer, upper + lower, "#9BB7D4", 64)
    layer_draw = ImageDraw.Draw(plot_layer)
    styled_polyline(
        layer_draw,
        [
            (local_px(float(row["environment_frames"])), local_py(float(row["median"])))
            for row in curve_rows
        ],
        fill=PAPER_METHOD_LINE,
        width=5,
    )
    image.alpha_composite(plot_layer, (left, top))
    draw = ImageDraw.Draw(image)
    threshold = float(summary["threshold"])
    dashed_line(
        draw,
        (left, py(threshold)),
        (right, py(threshold)),
        fill="#C76B67",
        width=2,
        dash=9,
        gap=6,
    )
    draw.text(
        (left + 8, py(threshold) - 27),
        f"Threshold = {threshold:.2f} (ours)",
        font=base.FONTS.get(15),
        fill="#A75A56",
    )
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    base.center_text(
        draw,
        ((x0 + x1) / 2, y0 + 27),
        "Sample Efficiency",
        font=base.FONTS.get(22),
        fill=INK,
    )
    draw_vertical_text(
        image,
        (x0 + 16, int((top + bottom) / 2)),
        "Norm. Perf.",
        size=18,
        fill=INK,
    )
    draw.text(
        (right - 20, bottom + 39),
        "1e6",
        font=base.FONTS.get(15),
        fill=INK,
    )


def draw_fig4a_strict(
    path: Path,
    metric_rows: Sequence[dict[str, Any]],
    sample_curve: Sequence[dict[str, Any]],
    sample_summary: dict[str, Any],
) -> None:
    """Render the default-order Atari row using the paper Figure 4 layout."""

    width, height = 2600, 760
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((6, 10), "A", font=base.FONTS.get(30), fill=INK)
    widths = (520, 520, 680, 730)
    boxes: list[tuple[int, int, int, int]] = []
    cursor = 30
    for panel_width in widths:
        boxes.append((cursor, 20, cursor + panel_width, 670))
        cursor += panel_width + 28

    forgetting = median_iqr([float(row["forgetting"]) for row in metric_rows])
    transfer = median_iqr([float(row["forward_transfer"]) for row in metric_rows])
    acc = median_iqr([float(row["acc"]) for row in metric_rows])
    min_acc = median_iqr([float(row["min_acc"]) for row in metric_rows])
    wc_acc = median_iqr([float(row["wc_acc"]) for row in metric_rows])
    draw_strict_bar_panel(
        image,
        boxes[0],
        title="Forgetting ↓",
        ylabel="Forgetting",
        values=(forgetting,),
        colors=(PAPER_NEUTRAL_BAR,),
        y_min=0.0,
        y_max=1.5,
        y_ticks=(0.0, 0.5, 1.0, 1.5),
    )
    draw_strict_bar_panel(
        image,
        boxes[1],
        title="Forward Transfer ↑",
        ylabel="Forward Transfer",
        values=(transfer,),
        colors=(PAPER_NEUTRAL_BAR,),
        y_min=-1.0,
        y_max=1.0,
        y_ticks=(-1.0, 0.0, 1.0),
    )
    draw_strict_bar_panel(
        image,
        boxes[2],
        title="Stability-plasticity Tradeoff ↑",
        ylabel="Norm. Performance",
        values=(acc, min_acc, wc_acc),
        colors=PAPER_ACC_COLORS,
        y_min=-0.5,
        y_max=1.0,
        y_ticks=(-0.5, 0.0, 0.5, 1.0),
    )
    draw_strict_sample_panel(image, boxes[3], sample_curve, sample_summary)

    draw = ImageDraw.Draw(image)
    legend_y = 717
    legend_items = (
        ("ACC", PAPER_ACC_COLORS[0]),
        ("minACC", PAPER_ACC_COLORS[1]),
        ("WCACC", PAPER_ACC_COLORS[2]),
    )
    legend_x = 880
    for label, color in legend_items:
        draw.rectangle((legend_x, legend_y - 9, legend_x + 26, legend_y + 9), fill=color)
        draw.text(
            (legend_x + 35, legend_y - 11),
            label,
            font=base.FONTS.get(16),
            fill=INK,
        )
        legend_x += 145
    draw.line((legend_x + 10, legend_y, legend_x + 55, legend_y), fill=PAPER_METHOD_LINE, width=5)
    draw.text(
        (legend_x + 67, legend_y - 11),
        "ARROW",
        font=base.FONTS.get(16),
        fill=INK,
    )
    base.save_image(image, path)


def draw_acquisition_figure(
    path: Path,
    continual: Sequence[base.Run],
    task_order: Sequence[str],
    endpoint: dict[str, dict[str, float]],
    st_curves: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    """Draw the time-aligned acquisition curves that actually enter local FT."""

    width, height = 2400, 1620
    image = base.new_canvas(
        "Time-aligned local normalization during task acquisition",
        "Exact checkpoint inputs to local discrete FT; median/IQR across five continual seeds",
        size=(width, height),
    )
    boxes = base.grid_boxes(size=(width, height), rows=2, columns=3)
    rows: list[dict[str, Any]] = []
    for task_index, (task, box) in enumerate(zip(task_order, boxes)):
        curves: list[list[float]] = []
        for run in continual:
            curve = []
            for local_epoch in base.EXPECTED_SINGLE_TASK_EPOCHS[1:]:
                global_epoch = task_index * 90 + local_epoch
                value = aligned_score(
                    run.by_epoch[global_epoch].raw_mean[task_index],
                    task,
                    local_epoch,
                    endpoint,
                    st_curves,
                )
                curve.append(value)
            curves.append(curve)
        summaries = [
            base.stats([curve[index] for curve in curves])
            for index in range(len(base.EXPECTED_SINGLE_TASK_EPOCHS) - 1)
        ]
        for index, local_epoch in enumerate(base.EXPECTED_SINGLE_TASK_EPOCHS[1:]):
            rows.append(
                {
                    "task_index": task_index,
                    "task": base.TASK_DISPLAY[task],
                    "local_epoch": local_epoch,
                    "environment_frames": local_epoch * FRAMES_PER_EPOCH,
                    "median": summaries[index]["median"],
                    "q25": summaries[index]["q25"],
                    "q75": summaries[index]["q75"],
                }
            )
        base.draw_line_panel(
            image,
            box,
            x_values=[epoch * FRAMES_PER_EPOCH for epoch in base.EXPECTED_SINGLE_TASK_EPOCHS[1:]],
            seed_curves=curves,
            median=[row["median"] for row in summaries],
            q25=[row["q25"] for row in summaries],
            q75=[row["q75"] for row in summaries],
            title=base.TASK_DISPLAY[task],
            color=TASK_COLORS[task_index],
            x_ticks=(163_840, 655_360, 1_146_880, 1_474_560),
            reference_lines=(1.0,),
            include_zero=True,
            include_one=True,
            xlabel="Frames in current task",
        )
    base.save_image(image, path)
    return rows


def multiline_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    *,
    font_size: int,
    bold: bool = False,
    fill: str = INK,
) -> None:
    font = base.FONTS.get(font_size, bold=bold)
    lines = text.split("\n")
    line_height = font_size * 1.2
    center_y = (box[1] + box[3]) / 2
    start_y = center_y - line_height * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        base.center_text(
            draw,
            ((box[0] + box[2]) / 2, start_y + index * line_height),
            line,
            font=font,
            fill=fill,
        )


def draw_table(
    path: Path,
    *,
    title: str,
    subtitle: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[int],
    notes: Sequence[str],
) -> None:
    if len(headers) != len(widths) or any(len(row) != len(headers) for row in rows):
        raise ValueError("Table dimensions do not match")
    margin = 45
    title_height = 125
    header_height = 105
    row_height = 88
    note_height = 42 * max(1, len(notes)) + 35
    width = sum(widths) + margin * 2
    height = title_height + header_height + row_height * len(rows) + note_height + 55
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    base.center_text(
        draw,
        (width / 2, 42),
        title,
        font=base.FONTS.get(34, bold=True),
        fill=INK,
    )
    base.center_text(
        draw,
        (width / 2, 83),
        subtitle,
        font=base.FONTS.get(20),
        fill=MUTED,
    )
    y = title_height
    x = margin
    for header, cell_width in zip(headers, widths):
        draw.rectangle((x, y, x + cell_width, y + header_height), fill="#EAEAEA")
        multiline_center(
            draw,
            (x, y, x + cell_width, y + header_height),
            header,
            font_size=20,
            bold=True,
        )
        x += cell_width
    draw.line((margin, y, width - margin, y), fill=INK, width=4)
    draw.line((margin, y + header_height, width - margin, y + header_height), fill=INK, width=3)
    y += header_height
    for row_index, row in enumerate(rows):
        x = margin
        fill = "#FFFFFF" if row_index % 2 == 0 else "#F7F7F7"
        for value, cell_width in zip(row, widths):
            draw.rectangle((x, y, x + cell_width, y + row_height), fill=fill)
            multiline_center(
                draw,
                (x, y, x + cell_width, y + row_height),
                value,
                font_size=20,
                bold=(row_index == 0 and len(rows) == 1),
            )
            x += cell_width
        draw.line((margin, y + row_height, width - margin, y + row_height), fill="#CCCCCC", width=2)
        y += row_height
    draw.line((margin, y, width - margin, y), fill=INK, width=4)
    note_y = y + 28
    for note in notes:
        draw.text(
            (margin, note_y),
            note,
            font=base.FONTS.get(18),
            fill=MUTED,
        )
        note_y += 38
    base.save_image(image, path)


def metric_summary_rows(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key, label in (
        ("forgetting", "Forgetting"),
        ("forward_transfer", "FT"),
        ("acc", "ACC"),
        ("min_acc", "min-ACC"),
        ("wc_acc", "WC-ACC"),
    ):
        values = [float(row[key]) for row in metric_rows]
        summary = base.stats(values)
        output.append({"metric": label, **summary, "seed_count": len(values)})
    return output


def source_files(runs: Iterable[base.Run]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    seen: set[Path] = set()
    for run in runs:
        for path in (
            run.run_dir / "launch.json",
            run.run_dir / "run_status.json",
            base.config_path(run.run_dir),
            run.run_dir / "train.log",
        ):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append({"path": base.relative(path), "sha256": base.sha256(path)})
    return files


def build_report(output: Path, *, validate_only: bool = False) -> dict[str, Any]:
    continual, task_order, grouped, config = load_sources()
    endpoint, st_curves = build_references(task_order, grouped)
    inventory = {
        "continual_runs": len(continual),
        "continual_seed_ids": [run.seed_id for run in continual],
        "single_task_runs": sum(len(runs) for runs in grouped.values()),
        "single_task_seed_ids_per_task": {
            base.TASK_DISPLAY[task]: [run.seed_id for run in grouped[task]]
            for task in task_order
        },
        "tasks": [base.TASK_DISPLAY[task] for task in task_order],
        "environment_frames_per_epoch": FRAMES_PER_EPOCH,
        "continual_environment_frames": 540 * FRAMES_PER_EPOCH,
        "single_task_environment_frames": 90 * FRAMES_PER_EPOCH,
    }
    if validate_only:
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return inventory

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    metric_rows, ft_task_rows = compute_metrics(
        continual, task_order, endpoint, st_curves
    )
    metric_summaries = metric_summary_rows(metric_rows)
    sample_curve, sample_seed_rows, sample_summary = compute_continual_sample_efficiency(
        continual, task_order, endpoint
    )
    st_efficiency, st_crossings = compute_single_task_efficiency(task_order, grouped)

    fig3_rows = draw_fig3a(
        figures / "fig3A_atari_arrow50_default_ours_full_range.png",
        continual,
        task_order,
        endpoint,
    )
    draw_fig3a_strict(
        figures / "fig3A_atari_arrow50_default_ours.png",
        task_order,
        fig3_rows,
    )
    draw_fig4a(
        figures / "fig4A_atari_metrics_arrow50_ours_expanded.png",
        metric_rows,
        sample_curve,
        sample_summary,
    )
    draw_fig4a_strict(
        figures / "fig4A_atari_metrics_arrow50_ours.png",
        metric_rows,
        sample_curve,
        sample_summary,
    )
    acquisition_rows = draw_acquisition_figure(
        figures / "figS1_time_aligned_acquisition_for_FT_ours.png",
        continual,
        task_order,
        endpoint,
        st_curves,
    )

    write_csv(tables / "figure3A_curve_data.csv", fig3_rows)
    write_csv(tables / "figure4A_metrics_by_seed.csv", metric_rows)
    write_csv(tables / "figure4A_metric_summary.csv", metric_summaries)
    write_csv(tables / "figure4A_sample_efficiency_curve.csv", sample_curve)
    write_csv(tables / "figure4A_sample_efficiency_by_seed.csv", sample_seed_rows)
    write_csv(tables / "time_aligned_acquisition_curves.csv", acquisition_rows)
    write_csv(tables / "forward_transfer_by_task_and_seed.csv", ft_task_rows)
    write_csv(tables / "table_A11_style_single_task_efficiency_ours.csv", st_efficiency)
    write_csv(tables / "single_task_threshold_crossings_by_seed.csv", st_crossings)

    endpoint_rows = [
        {
            "task_index": index,
            "task": base.TASK_DISPLAY[task],
            **endpoint[task],
        }
        for index, task in enumerate(task_order)
    ]
    write_csv(tables / "local_endpoint_references.csv", endpoint_rows)

    # Table A.3-style: default order and our ARROW row only.
    by_metric = {
        row["metric"]: [float(item[row_key]) for item in metric_rows]
        for row, row_key in zip(
            metric_summaries,
            ("forgetting", "forward_transfer", "acc", "min_acc", "wc_acc"),
        )
    }
    a3_headers = (
        "Method",
        "Forgetting ↓",
        "FT ↑",
        "ACC ↑",
        "min-ACC ↑",
        "WC-ACC ↑",
    )
    a3_row = (
        "ARROW-50\n(ours)",
        format_interval(by_metric["Forgetting"]),
        format_interval(by_metric["FT"]),
        format_interval(by_metric["ACC"]),
        format_interval(by_metric["min-ACC"]),
        format_interval(by_metric["WC-ACC"]),
    )
    a3_csv = [
        {
            "setting": "Default task order",
            "method": "ARROW-50 (ours)",
            "forgetting_median_iqr": a3_row[1],
            "ft_median_iqr": a3_row[2],
            "acc_median_iqr": a3_row[3],
            "min_acc_median_iqr": a3_row[4],
            "wc_acc_median_iqr": a3_row[5],
            "continual_seeds": 5,
            "single_task_reference_seeds": 3,
        }
    ]
    write_csv(tables / "table_A3_style_default_arrow50_ours.csv", a3_csv)
    a3_md = (
        "# Table A.3-style — Atari metrics, default task order (ours only)\n\n"
        "> No numerical result from the ARROW paper is included. Values are median [q25–q75] across our five continual seeds.\n\n"
        + markdown_table(a3_headers, [a3_row])
        + "\nEndpoint metrics use local 3-seed single-task E0/E90 anchors. FT uses the time-aligned local E10–E90 single-task median curve and a discrete 10-epoch grid.\n"
    )
    (tables / "table_A3_style_default_arrow50_ours.md").write_text(
        a3_md, encoding="utf-8"
    )
    draw_table(
        tables / "table_A3_style_default_arrow50_ours.png",
        title="Table A.3-style · Atari metrics · default order",
        subtitle="Our ARROW-50 results only · median [q25–q75]",
        headers=a3_headers,
        rows=(a3_row,),
        widths=(260, 310, 310, 310, 310, 310),
        notes=(
            "Local endpoint references: 3 single-task seeds per game; continual summary: 5 seeds.",
            "FT is time-aligned on local epochs 10–90. No paper result values are used.",
        ),
    )

    # Table A.11-style: ARROW-only task-specific self thresholds.
    a11_headers = (
        "Task",
        "Task-specific\n85% threshold*",
        "Method",
        "Max perf.\n(raw return)",
        "Env. frames\nmedian [q25–q75]",
        "Runs ≥85%",
    )
    a11_rows: list[tuple[str, ...]] = []
    for row in st_efficiency:
        if row["crossing_frames_median"] is None:
            frame_text = "Never reached"
        else:
            frame_text = (
                f"{row['crossing_frames_median']:,.0f} "
                f"[{row['crossing_frames_q25']:,.0f} – {row['crossing_frames_q75']:,.0f}]"
            )
        a11_rows.append(
            (
                str(row["task"]),
                base.raw_format(float(row["threshold_85pct_ours_only"])),
                "ARROW\n(ours)",
                base.raw_format(float(row["arrow_max_median_raw_return"])),
                frame_text,
                f"{row['runs_reached']}/{row['runs_total']}",
            )
        )
    a11_md = (
        "# Table A.11-style — Atari single-task sample efficiency (ours only)\n\n"
        "> *Diagnostic threshold: 85% of each task's maximum **our-ARROW median**. The paper's threshold is shared across methods, so this table is layout-aligned but not numerically paper-equivalent.\n\n"
        + markdown_table(a11_headers, a11_rows)
        + "\nCrossings are observed only at our 10-epoch evaluation grid (163,840 environment-frame increments).\n"
    )
    (tables / "table_A11_style_single_task_efficiency_ours.md").write_text(
        a11_md, encoding="utf-8"
    )
    draw_table(
        tables / "table_A11_style_single_task_efficiency_ours.png",
        title="Table A.11-style · Atari single-task sample efficiency",
        subtitle="Our ARROW results only · raw returns · 3 seeds per task",
        headers=a11_headers,
        rows=a11_rows,
        widths=(300, 330, 230, 300, 540, 230),
        notes=(
            "*85% of the maximum median within our ARROW cohort; not the paper's shared cross-method threshold.",
            "Crossings are quantized to the 163,840-frame evaluation interval. No paper result values are used.",
        ),
    )

    # Table A.14-style: local resolved ARROW training parameters.
    single_frames = (int(config["epochs"]) - 1) * int(config["data_n"]) * int(
        config["data_t"]
    )
    replay_buffers = config["replay_buffers"]
    replay_capacity = len(replay_buffers) * int(config["data_n_max"]) * int(
        config["data_t"]
    )
    a14_headers = ("Method", "Env. frames", "Env. steps", "Replay buffer capacity")
    a14_rows = (
        (
            "ARROW\n(ours)",
            f"{single_frames / 1_000_000:.2f}M",
            f"{single_frames * int(config['env_repeat']) / 1_000_000:.2f}M",
            f"2 × {config['data_n_max']} sequences × T={config['data_t']}\n({replay_capacity:,} = 2^19 observations)",
        ),
    )
    a14_csv = [
        {
            "method": "ARROW (ours)",
            "environment_frames": single_frames,
            "emulator_frames_env_steps": single_frames * int(config["env_repeat"]),
            "frame_repeat": int(config["env_repeat"]),
            "replay_buffer_count": len(replay_buffers),
            "sequences_per_buffer": int(config["data_n_max"]),
            "sequence_length": int(config["data_t"]),
            "total_observation_capacity": replay_capacity,
            "replay_observation_dtype": config["replay_observation_dtype"],
            "replay_devices": ",".join(str(item["rb_device"]) for item in replay_buffers),
        }
    ]
    write_csv(tables / "table_A14_style_training_parameters_ours.csv", a14_csv)
    a14_md = (
        "# Table A.14-style — Atari single-task training parameters (ours only)\n\n"
        + markdown_table(a14_headers, a14_rows)
        + f"\nResolved local profile: `{config['replay_observation_dtype']}` observations on "
        + ", ".join(str(item["rb_device"]) for item in replay_buffers)
        + "; Atari frame repeat = "
        + str(config["env_repeat"])
        + ".\n"
    )
    (tables / "table_A14_style_training_parameters_ours.md").write_text(
        a14_md, encoding="utf-8"
    )
    draw_table(
        tables / "table_A14_style_training_parameters_ours.png",
        title="Table A.14-style · Atari single-task training parameters",
        subtitle="Resolved values from our completed ARROW runs",
        headers=a14_headers,
        rows=a14_rows,
        widths=(280, 330, 330, 760),
        notes=(
            f"Replay observations: {config['replay_observation_dtype']}; devices: "
            + ", ".join(str(item["rb_device"]) for item in replay_buffers)
            + ".",
        ),
    )

    # Table A.15-style: local untrained and final single-task returns.
    a15_headers = (
        "Task",
        "Reward scale",
        "Untrained E0\nmedian [q25–q75]",
        "ARROW E90\nmedian [q25–q75]",
        "Seeds",
    )
    a15_rows: list[tuple[str, ...]] = []
    a15_csv: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_order):
        initial_values = [run.by_epoch[0].raw_mean[0] for run in grouped[task]]
        final_values = [run.by_epoch[90].raw_mean[0] for run in grouped[task]]
        reward_scale = continual[0].reward_scales[task_index]
        a15_rows.append(
            (
                base.TASK_DISPLAY[task],
                f"{reward_scale:g}",
                format_raw_interval(initial_values),
                format_raw_interval(final_values),
                "3",
            )
        )
        initial_stats, final_stats = base.stats(initial_values), base.stats(final_values)
        a15_csv.append(
            {
                "task_index": task_index,
                "task": base.TASK_DISPLAY[task],
                "reward_scale": reward_scale,
                "untrained_e0_median": initial_stats["median"],
                "untrained_e0_q25": initial_stats["q25"],
                "untrained_e0_q75": initial_stats["q75"],
                "arrow_e90_median": final_stats["median"],
                "arrow_e90_q25": final_stats["q25"],
                "arrow_e90_q75": final_stats["q75"],
                "seed_count": 3,
            }
        )
    write_csv(tables / "table_A15_style_single_task_results_ours.csv", a15_csv)
    a15_md = (
        "# Table A.15-style — Atari single-task raw returns (ours only)\n\n"
        "> Our epoch-0 evaluation is reported as `Untrained E0`; it is not relabeled as an independently sampled random-agent baseline.\n\n"
        + markdown_table(a15_headers, a15_rows)
        + "\nAll entries are raw episodic return means summarized across our three completed single-task seeds.\n"
    )
    (tables / "table_A15_style_single_task_results_ours.md").write_text(
        a15_md, encoding="utf-8"
    )
    draw_table(
        tables / "table_A15_style_single_task_results_ours.png",
        title="Table A.15-style · Atari single-task experimental results",
        subtitle="Our ARROW results only · raw returns · median [q25–q75]",
        headers=a15_headers,
        rows=a15_rows,
        widths=(330, 260, 540, 540, 180),
        notes=(
            "E0 is our untrained-policy evaluation, not an independent random-agent cohort.",
            "E90 is the end of the 1,474,560-frame single-task budget. No paper result values are used.",
        ),
    )

    alignment_notes = f"""# ARROW Atari paper alignment — our data only

This directory mirrors the **presentation and metric organization** of the
[ARROW v3 paper]({PAPER_URL}), but every plotted or tabulated numerical result
comes from local artifacts in this repository. No paper result value is used.

## Direct correspondence

| Our artifact | Paper location mirrored | Scope retained |
|---|---|---|
| `figures/fig3A_atari_arrow50_default_ours.png` | Figure 3A | Strict visual subset: Atari, default one-cycle order, ARROW-50 only |
| `figures/fig4A_atari_metrics_arrow50_ours.png` | Figure 4A | Strict visual subset: Atari, default one-cycle order, ARROW-50 only |
| `figures/fig3A_atari_arrow50_default_ours_full_range.png` | Figure 3A supplement | Expanded annotations and an unclipped local data range |
| `figures/fig4A_atari_metrics_arrow50_ours_expanded.png` | Figure 4A supplement | Expanded axes, values, intervals, and caveat text |
| `tables/table_A3_style_default_arrow50_ours.*` | Table A.3 | Default-order ARROW row only |
| `tables/table_A11_style_single_task_efficiency_ours.*` | Table A.11 | ARROW rows only; self-threshold diagnostic |
| `tables/table_A14_style_training_parameters_ours.*` | Table A.14 | ARROW row only |
| `tables/table_A15_style_single_task_results_ours.*` | Table A.15 | Six Atari tasks, our E0 and E90 only |

## Data cohorts

- Continual ARROW-50: 5 completed, predeclared seeds (0–4), 55 evaluations per seed.
- Single-task ARROW-50: 3 completed seeds per game (0–2), 10 evaluations per seed.
- Default order: Ms. Pac-Man → Boxing → Crazy Climber → Frostbite → Seaquest → Enduro.
- Each task receives 90 epochs × 16,384 environment frames = 1,474,560 frames.
- Full curriculum: 8,847,360 environment frames.

## Normalization and metric rules

For full-duration curves and endpoint metrics, the local fixed endpoint score is

`(raw - median(local ST E0)) / (median(local ST E90) - median(local ST E0))`.

This is exact for the endpoint form of the paper normalization at task boundaries;
intermediate full-duration curve values are a fixed-anchor analogue and are named
**local normalized performance**, not a claim that external paper anchors were used.

The strict Figure 3 view follows the paper's fixed `[-0.5, 1.85]` viewport,
palette, dash patterns, 0.2-IQR opacity, and active-task line-width change. Its
full-range companion preserves all local IQR values (our maximum q75 is above
the strict paper viewport). The strict Figure 4 row likewise uses the paper's
compact panel geometry and fixed axes; the expanded companion is the better
artifact for reading exact local values.

## Why some normalized values are negative

A negative value is expected whenever a continual raw return is below the local
single-task E0 reference: `raw < median(local ST E0)`. It is not a negative raw
Atari score inserted by the renderer. The paper's own Figure 3 also exposes a
negative normalized range down to `-0.5`. Our negatives are visually prominent
because each E0 reference is estimated from only three local runs and is an
untrained-policy evaluation after initial collection, not a separately sampled
random-agent cohort.

The active interval is rendered once at the larger line width in the strict
Figure 3 output. It is not a second or duplicated result series.

FT uses the available time-aligned local single-task curves at E10, E20, …, E90:

`(CL raw at local epoch n - local ST E0 median) / (local ST median at n - local ST E0 median)`.

The nine checkpoint values are averaged per task, compared with the corresponding
single-task normalized area (=1 on the median reference), and then averaged across
tasks. It is therefore a discrete 10-epoch approximation of the paper's integral.

## Deliberate non-equivalences

1. The local normalization cohort has 3 single-task seeds, not 5.
2. Local E0 is an untrained-policy evaluation, not a separately run random-agent cohort.
3. The paper sample-efficiency threshold is shared across ARROW, DreamerV3, and
   TES-SAC. Because this report intentionally uses only our ARROW data, its 85%
   threshold is relative to our ARROW-only median peak and is labeled diagnostic.
4. Threshold crossings are visible only every 10 epochs (163,840 frames).
5. Continual seed 0 used GPU replay storage; seeds 1–4 used CPU float32 replay.
   The interaction/update protocol is matched, but this storage-profile deviation
   must remain disclosed in publication text.

## What is still needed for the complete paper grid

The paper's full Figure 3/4 grids also contain reversed-order, two-cycle, DreamerV3,
and TES-SAC experiments. Those cells are intentionally absent rather than filled
with published numbers. They can be added only after corresponding **local** runs
exist.
"""
    (output / "paper_alignment_notes.md").write_text(alignment_notes, encoding="utf-8")

    metric_lookup = {row["metric"]: row for row in metric_summaries}
    readme = f"""# Paper-aligned ARROW Atari report — ours only

This is a paper-shaped view of our local results, not a copy of the paper's values.

## Headline local numbers

- Forgetting: {format_interval([float(row['forgetting']) for row in metric_rows])}
- FT: {format_interval([float(row['forward_transfer']) for row in metric_rows])}
- ACC: {format_interval([float(row['acc']) for row in metric_rows])}
- min-ACC: {format_interval([float(row['min_acc']) for row in metric_rows])}
- WC-ACC: {format_interval([float(row['wc_acc']) for row in metric_rows])}

All values are median [q25–q75]. Read `paper_alignment_notes.md` before using a
number in prose, especially the normalization and sample-efficiency caveats.

## Main artifacts

- `figures/fig3A_atari_arrow50_default_ours.png`
- `figures/fig4A_atari_metrics_arrow50_ours.png`
- `figures/fig3A_atari_arrow50_default_ours_full_range.png`
- `figures/fig4A_atari_metrics_arrow50_ours_expanded.png`
- `figures/figS1_time_aligned_acquisition_for_FT_ours.png`
- `tables/table_A3_style_default_arrow50_ours.png`
- `tables/table_A11_style_single_task_efficiency_ours.png`
- `tables/table_A14_style_training_parameters_ours.png`
- `tables/table_A15_style_single_task_results_ours.png`

Machine-readable CSV files sit beside the rendered tables.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    cards = [
        ("figures/fig3A_atari_arrow50_default_ours.png", "Figure 3A strict visual subset: our ARROW-50"),
        ("figures/fig4A_atari_metrics_arrow50_ours.png", "Figure 4A strict visual subset: our ARROW-50 metrics"),
        ("figures/fig3A_atari_arrow50_default_ours_full_range.png", "Figure 3A full-range companion: no local IQR clipping"),
        ("figures/fig4A_atari_metrics_arrow50_ours_expanded.png", "Figure 4A expanded companion: exact labels and intervals"),
        ("figures/figS1_time_aligned_acquisition_for_FT_ours.png", "Supplement: exact time-aligned inputs to local FT"),
        ("tables/table_A3_style_default_arrow50_ours.png", "Table A.3-style: default-order metrics"),
        ("tables/table_A11_style_single_task_efficiency_ours.png", "Table A.11-style: single-task efficiency"),
        ("tables/table_A14_style_training_parameters_ours.png", "Table A.14-style: training budget"),
        ("tables/table_A15_style_single_task_results_ours.png", "Table A.15-style: single-task raw returns"),
    ]
    html_cards = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><a href="{html.escape(path)}"><img src="{html.escape(path)}" alt="{html.escape(title)}"></a></section>'
        for path, title in cards
    )
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ARROW Atari — ours only</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f2f2f2;color:#222}}
main{{max-width:1500px;margin:0 auto;padding:28px}}
.notice{{background:#fff7d6;border-left:6px solid #d6a800;padding:15px 18px;margin-bottom:24px}}
section{{background:white;padding:18px;margin:22px 0;box-shadow:0 2px 10px #0001}}
h1,h2{{margin-top:0}} img{{width:100%;height:auto;border:1px solid #ddd}}
a{{color:#185abc}}
</style></head><body><main>
<h1>Paper-aligned ARROW Atari report — our data only</h1>
<div class="notice"><b>Important:</b> layout and metric organization follow the ARROW paper; all numerical results are ours. Negative normalized values mean the raw return fell below our local E0 anchor. The sample-efficiency threshold is ARROW-only and therefore diagnostic.</div>
<p><a href="paper_alignment_notes.md">Alignment and caveat notes</a> · <a href="README.md">README</a></p>
{html_cards}
</main></body></html>"""
    (output / "index.html").write_text(index, encoding="utf-8")

    all_runs = list(continual) + [run for task in task_order for run in grouped[task]]
    generated = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "arrow-atari-paper-aligned-ours-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": base.relative(Path(__file__)),
        "paper_reference": {
            "url": PAPER_URL,
            "used_for": "layout, labels, and metric definitions only",
            "numerical_result_values_used": False,
        },
        "external_result_values_used": False,
        "inventory": inventory,
        "normalization": {
            "endpoint_formula": "(raw - local_ST_E0_median)/(local_ST_E90_median - local_ST_E0_median)",
            "endpoint_reference_seed_count_per_task": 3,
            "ft_formula": "time-aligned local Eq.1 analogue on E10..E90; discrete mean; then Eq.3",
            "numeric_values_unclipped": True,
            "strict_figure3_viewport": [-0.5, 1.85],
            "strict_figure4_viewports": {
                "forgetting": [0.0, 1.5],
                "forward_transfer": [-1.0, 1.0],
                "stability_plasticity": [-0.5, 1.0],
                "sample_efficiency": [0.0, 3.0],
            },
            "full_range_companion": "figures/fig3A_atari_arrow50_default_ours_full_range.png",
        },
        "continual_sample_efficiency": sample_summary,
        "metric_summary": metric_lookup,
        "source_files": source_files(all_runs),
        "outputs": [
            {"path": str(path.relative_to(output)), "sha256": base.sha256(path)}
            for path in generated
        ],
        "known_deviations": [
            "Only default-order ARROW-50 is present; no local reversed/two-cycle or baseline cells.",
            "Single-task reference cohort contains 3 seeds per game.",
            "E0 is an untrained-policy evaluation, not an independent random-agent cohort.",
            "ARROW-only self-peak sample-efficiency threshold is not the paper's shared cross-method threshold.",
            "Continual seed0 replay storage profile differs from seeds1-4.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for png in output.rglob("*.png"):
        with Image.open(png) as image:
            image.verify()
        if png.stat().st_size < 10_000:
            raise ValueError(f"Suspiciously small figure: {png}")
    print(json.dumps({**inventory, "output": str(output)}, ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    args = parse_args()
    build_report(args.output.resolve(), validate_only=args.validate_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
