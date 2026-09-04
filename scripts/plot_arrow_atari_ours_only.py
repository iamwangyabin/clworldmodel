#!/usr/bin/env python3
"""Render an Atari ARROW-only figure and table bundle from local run artifacts.

The report deliberately uses no published result values. Raw returns come from
the preserved local logs. The optional normalized view uses only the local
three-seed single-task cohort: for each game, its epoch-0 median is score 0 and
its epoch-90 median is score 1.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.evaluation.metrics import single_pass_metrics  # noqa: E402


CONTINUAL_RUNS = (
    ROOT / "runs" / "arrow_ar50_original_s0_analysis",
    ROOT
    / "runs"
    / "arrow_ar50_cpu_fp32_original_seeds1_4_2x4090"
    / "seed1_retry1",
    ROOT / "runs" / "arrow_ar50_cpu_fp32_original_seeds1_4_2x4090" / "seed2",
    ROOT / "runs" / "arrow_ar50_cpu_fp32_original_seeds1_4_2x4090" / "seed3",
    ROOT / "runs" / "arrow_ar50_cpu_fp32_original_seeds1_4_2x4090" / "seed4",
)
SINGLE_TASK_ROOTS = (
    ROOT / "runs" / "arrow_single_task_s0_cpu_fp32_368f440_virtai4x24g",
    ROOT / "runs" / "arrow_single_task_seeds1_2_cpu_fp32_368f440_virtai4x24g",
)
DEFAULT_OUTPUT = ROOT / "runs" / "analysis" / "arrow_atari_ours_only_20260902"

EXPECTED_CONTINUAL_EPOCHS = tuple(range(0, 541, 10))
EXPECTED_SINGLE_TASK_EPOCHS = tuple(range(0, 91, 10))
BOUNDARY_EPOCHS = (90, 180, 270, 360, 450, 540)

TASK_DISPLAY = {
    "ALE/MsPacman-v5": "Ms. Pac-Man",
    "ALE/Boxing-v5": "Boxing",
    "ALE/CrazyClimber-v5": "Crazy Climber",
    "ALE/Frostbite-v5": "Frostbite",
    "ALE/Seaquest-v5": "Seaquest",
    "ALE/Enduro-v5": "Enduro",
}
TASK_SHORT = {
    "ALE/MsPacman-v5": "MsPac",
    "ALE/Boxing-v5": "Boxing",
    "ALE/CrazyClimber-v5": "Crazy",
    "ALE/Frostbite-v5": "Frostbite",
    "ALE/Seaquest-v5": "Seaquest",
    "ALE/Enduro-v5": "Enduro",
}

TASK_COLORS = (
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#B279A2",
)
SEED_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
GROUP_COLORS = {
    "single": "#4C78A8",
    "acquisition": "#F58518",
    "final": "#54A24B",
}

FONT_REGULAR_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
)
FONT_BOLD_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
)


@dataclass(frozen=True)
class EvalPoint:
    epoch: int
    raw_mean: tuple[float, ...]
    raw_std: tuple[float, ...]
    raw_source: str


@dataclass(frozen=True)
class Run:
    run_dir: Path
    seed_id: int
    seed: int
    task_names: tuple[str, ...]
    reward_scales: tuple[float, ...]
    evaluations: tuple[EvalPoint, ...]
    runtime: str
    git_commit: str
    started_at_utc: str | None
    finished_at_utc: str | None
    gpu_name: str | None

    @property
    def by_epoch(self) -> dict[int, EvalPoint]:
        return {point.epoch: point for point in self.evaluations}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all source runs and print the inventory without writing files.",
    )
    return parser.parse_args()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def config_path(run_dir: Path) -> Path:
    for candidate in (
        run_dir / "resolved_training_config.json",
        run_dir / "config.json",
        run_dir / "tensorboard" / "config.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No training config found under {run_dir}")


def float_list(line: str, prefix: str) -> tuple[float, ...]:
    if not line.startswith(prefix):
        raise ValueError(f"Expected {prefix!r}: {line!r}")
    parsed = ast.literal_eval(line[len(prefix) :].strip())
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Expected a non-empty list after {prefix!r}")
    values = tuple(float(value) for value in parsed)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite value after {prefix!r}")
    return values


def parse_evaluations(
    log_path: Path, reward_scales: Sequence[float]
) -> tuple[EvalPoint, ...]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    points: list[EvalPoint] = []
    for index, line in enumerate(lines):
        if not line.startswith("Eval for epoch:"):
            continue
        match = re.search(r"Eval for epoch:\s*(\d+)", line)
        if match is None:
            raise ValueError(f"Malformed evaluation epoch in {log_path}: {line}")
        epoch = int(match.group(1))
        fields: dict[str, str] = {}
        for candidate in lines[index + 1 : index + 9]:
            for prefix in (
                "Eval means:",
                "Eval stds:",
                "Eval raw means:",
                "Eval raw stds:",
            ):
                if candidate.startswith(prefix):
                    fields[prefix] = candidate
        scaled_mean = float_list(fields["Eval means:"], "Eval means:")
        scaled_std = float_list(fields["Eval stds:"], "Eval stds:")
        if len(scaled_mean) != len(reward_scales):
            raise ValueError(f"Task count mismatch at epoch {epoch} in {log_path}")
        if "Eval raw means:" in fields:
            raw_mean = float_list(fields["Eval raw means:"], "Eval raw means:")
            raw_std = float_list(fields["Eval raw stds:"], "Eval raw stds:")
            raw_source = "explicit_log"
        else:
            if any(scale == 0 for scale in reward_scales):
                raise ValueError(f"Cannot recover raw return with zero scale: {log_path}")
            raw_mean = tuple(
                value / scale for value, scale in zip(scaled_mean, reward_scales)
            )
            raw_std = tuple(
                value / abs(scale) for value, scale in zip(scaled_std, reward_scales)
            )
            raw_source = "derived_from_local_scaled_log_and_config"
        points.append(EvalPoint(epoch, raw_mean, raw_std, raw_source))
    if not points:
        raise ValueError(f"No evaluation blocks found in {log_path}")
    epochs = [point.epoch for point in points]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Evaluation epochs are not unique and increasing: {log_path}")
    return tuple(points)


def load_run(run_dir: Path, *, expected_epochs: Sequence[int]) -> Run:
    status_path = run_dir / "run_status.json"
    launch_path = run_dir / "launch.json"
    log_path = run_dir / "train.log"
    if not (status_path.is_file() and launch_path.is_file() and log_path.is_file()):
        raise FileNotFoundError(f"Incomplete local artifact set under {run_dir}")
    status = json_object(status_path)
    if status.get("complete") is not True or int(status.get("return_code", -1)) != 0:
        raise ValueError(f"Run is not successfully complete: {run_dir}")
    launch = json_object(launch_path)
    config = json_object(config_path(run_dir))
    env_configs = config["esc"]["env_configs"]
    task_names = tuple(str(item["name"]) for item in env_configs)
    reward_scales = tuple(float(item["rew_scale"]) for item in env_configs)
    evaluations = parse_evaluations(log_path, reward_scales)
    epochs = tuple(point.epoch for point in evaluations)
    if epochs != tuple(expected_epochs):
        raise ValueError(
            f"Unexpected evaluation epochs in {run_dir}: {epochs}; "
            f"expected {tuple(expected_epochs)}"
        )
    git = launch.get("project_git", {})
    cuda = launch.get("cuda", {})
    return Run(
        run_dir=run_dir,
        seed_id=int(launch["seed_id"]),
        seed=int(config["seed"]),
        task_names=task_names,
        reward_scales=reward_scales,
        evaluations=evaluations,
        runtime=str(launch.get("runtime", "unknown")),
        git_commit=str(git.get("commit", "unknown")),
        started_at_utc=launch.get("started_at_utc"),
        finished_at_utc=status.get("finished_at_utc"),
        gpu_name=cuda.get("device_name"),
    )


def discover_single_task_runs() -> list[Run]:
    runs: list[Run] = []
    for root in SINGLE_TASK_ROOTS:
        for status_path in sorted(root.glob("*/run_status.json")):
            run_dir = status_path.parent
            if "failed" in run_dir.name.lower():
                continue
            launch_path = run_dir / "launch.json"
            if not launch_path.is_file():
                continue
            launch = json_object(launch_path)
            if launch.get("method") != "ARROW-50-SingleTask":
                continue
            runs.append(load_run(run_dir, expected_epochs=EXPECTED_SINGLE_TASK_EPOCHS))
    return runs


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": quantile(values, 0.5),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def iso_hours(start: str | None, finish: str | None) -> float | None:
    if not start or not finish:
        return None
    return (
        datetime.fromisoformat(finish).astimezone(timezone.utc)
        - datetime.fromisoformat(start).astimezone(timezone.utc)
    ).total_seconds() / 3600


def stage_hours(log_path: Path) -> float:
    pattern = re.compile(r"\[stage-time\].*?\btotal=([0-9.]+)s")
    total = 0.0
    for match in pattern.finditer(
        log_path.read_text(encoding="utf-8", errors="replace")
    ):
        total += float(match.group(1))
    return total / 3600


def normalize(value: float, anchor: dict[str, float]) -> float:
    denominator = anchor["final_median"] - anchor["initial_median"]
    if denominator == 0:
        raise ValueError("Local single-task anchors have a zero denominator")
    return (value - anchor["initial_median"]) / denominator


def rgb(color: str) -> tuple[int, int, int]:
    value = color.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend(left: str, right: str, amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    a = rgb(left)
    b = rgb(right)
    return tuple(round(x * (1 - amount) + y * amount) for x, y in zip(a, b))


def score_color(value: float) -> tuple[int, int, int]:
    value = max(-0.5, min(1.5, value))
    if value <= 0:
        return blend("#B24A4A", "#F3F4F6", (value + 0.5) / 0.5)
    if value <= 1:
        return blend("#F3F4F6", "#4C78A8", value)
    return blend("#4C78A8", "#235347", (value - 1) / 0.5)


class Fonts:
    def __init__(self) -> None:
        regular = next((path for path in FONT_REGULAR_CANDIDATES if path.is_file()), None)
        bold = next((path for path in FONT_BOLD_CANDIDATES if path.is_file()), None)
        self.regular_path = regular
        self.bold_path = bold or regular

    def get(self, size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        path = self.bold_path if bold else self.regular_path
        if path is None:
            return ImageFont.load_default()
        return ImageFont.truetype(str(path), size=size)


FONTS = Fonts()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "#111827",
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def right_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "#374151",
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width, xy[1] - height / 2), text, font=font, fill=fill)


def alpha_polygon(
    image: Image.Image,
    points: Sequence[tuple[float, float]],
    color: str,
    alpha: int,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(points, fill=(*rgb(color), alpha))
    image.alpha_composite(overlay)


def alpha_rectangle(
    image: Image.Image,
    box: tuple[float, float, float, float],
    color: str,
    alpha: int,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(*rgb(color), alpha))
    image.alpha_composite(overlay)


def nice_step(span: float, target_ticks: int = 5) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(1, target_ticks - 1)
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10**exponent)
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exponent)


def nice_limits(
    values: Sequence[float], *, include_zero: bool = False, include_one: bool = False
) -> tuple[float, float, list[float]]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if include_one:
        low = min(low, 1.0)
        high = max(high, 1.0)
    if low == high:
        margin = max(1.0, abs(low) * 0.1)
        low -= margin
        high += margin
    margin = (high - low) * 0.08
    low -= margin
    high += margin
    step = nice_step(high - low)
    tick_low = math.floor(low / step) * step
    tick_high = math.ceil(high / step) * step
    ticks: list[float] = []
    value = tick_low
    while value <= tick_high + step * 0.1:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return tick_low, tick_high, ticks


def fmt_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 10_000:
        return f"{value / 1_000:.0f}k"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def new_canvas(
    title: str,
    subtitle: str,
    *,
    size: tuple[int, int] = (2400, 1700),
) -> Image.Image:
    image = Image.new("RGBA", size, "white")
    draw = ImageDraw.Draw(image)
    center_text(
        draw,
        (size[0] / 2, 54),
        title,
        font=FONTS.get(42, bold=True),
        fill="#111827",
    )
    center_text(
        draw,
        (size[0] / 2, 103),
        subtitle,
        font=FONTS.get(23),
        fill="#4B5563",
    )
    return image


def draw_line_panel(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    x_values: Sequence[float],
    seed_curves: Sequence[Sequence[float]],
    median: Sequence[float],
    q25: Sequence[float],
    q75: Sequence[float],
    title: str,
    color: str,
    x_ticks: Sequence[float],
    train_span: tuple[float, float] | None = None,
    reference_lines: Sequence[float] = (),
    include_zero: bool = True,
    include_one: bool = False,
    xlabel: str | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right = x0 + 100, x1 - 25
    top, bottom = y0 + 58, y1 - 72
    all_values = [value for curve in seed_curves for value in curve]
    all_values.extend(q25)
    all_values.extend(q75)
    y_low, y_high, y_ticks = nice_limits(
        all_values, include_zero=include_zero, include_one=include_one
    )
    x_low, x_high = min(x_values), max(x_values)

    def px(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

    if train_span is not None:
        alpha_rectangle(
            image,
            (px(train_span[0]), top, px(train_span[1]), bottom),
            "#FDE68A",
            82,
        )
    draw = ImageDraw.Draw(image)
    for tick in y_ticks:
        y = py(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=2)
        right_text(
            draw,
            (left - 13, y),
            fmt_tick(tick),
            font=FONTS.get(18),
            fill="#6B7280",
        )
    for tick in x_ticks:
        x = px(tick)
        draw.line((x, bottom, x, bottom + 8), fill="#6B7280", width=2)
        center_text(
            draw,
            (x, bottom + 29),
            fmt_tick(tick),
            font=FONTS.get(18),
            fill="#6B7280",
        )
    for line in reference_lines:
        if y_low <= line <= y_high:
            y = py(line)
            draw.line((left, y, right, y), fill="#9CA3AF", width=3)
    band = [(px(x), py(y)) for x, y in zip(x_values, q75)]
    band.extend((px(x), py(y)) for x, y in reversed(list(zip(x_values, q25))))
    alpha_polygon(image, band, color, 55)
    draw = ImageDraw.Draw(image)
    for curve in seed_curves:
        points = [(px(x), py(y)) for x, y in zip(x_values, curve)]
        draw.line(points, fill=(*rgb(color), 75), width=2, joint="curve")
    points = [(px(x), py(y)) for x, y in zip(x_values, median)]
    draw.line(points, fill=color, width=6, joint="curve")
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.line((left, top, left, bottom), fill="#374151", width=3)
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    center_text(
        draw,
        ((x0 + x1) / 2, y0 + 20),
        title,
        font=FONTS.get(25, bold=True),
    )
    if xlabel:
        center_text(
            draw,
            ((left + right) / 2, y1 - 20),
            xlabel,
            font=FONTS.get(19),
            fill="#4B5563",
        )


def grid_boxes(
    *,
    size: tuple[int, int] = (2400, 1700),
    rows: int = 2,
    columns: int = 3,
) -> list[tuple[float, float, float, float]]:
    left, right, top, bottom = 70, 45, 155, 55
    horizontal_gap, vertical_gap = 35, 45
    width = (size[0] - left - right - horizontal_gap * (columns - 1)) / columns
    height = (size[1] - top - bottom - vertical_gap * (rows - 1)) / rows
    boxes = []
    for row in range(rows):
        for column in range(columns):
            x0 = left + column * (width + horizontal_gap)
            y0 = top + row * (height + vertical_gap)
            boxes.append((x0, y0, x0 + width, y0 + height))
    return boxes


def add_line_legend(
    image: Image.Image,
    *,
    labels: Sequence[tuple[str, str]],
    y: int = 133,
) -> None:
    draw = ImageDraw.Draw(image)
    font = FONTS.get(18)
    widths = [text_size(draw, label, font)[0] + 62 for label, _ in labels]
    x = (image.width - sum(widths)) / 2
    for (label, color), width in zip(labels, widths):
        draw.line((x, y, x + 38, y), fill=color, width=6)
        draw.text((x + 47, y - 10), label, font=font, fill="#374151")
        x += width


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", quality=95, dpi=(180, 180))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def raw_format(value: float) -> str:
    if abs(value) >= 10000:
        return f"{value:,.1f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_bundle(output: Path, *, validate_only: bool = False) -> dict[str, Any]:
    continual = [
        load_run(path, expected_epochs=EXPECTED_CONTINUAL_EPOCHS)
        for path in CONTINUAL_RUNS
    ]
    continual.sort(key=lambda run: run.seed_id)
    if {run.seed_id for run in continual} != set(range(5)):
        raise ValueError("Expected continual seed IDs 0 through 4")
    task_order = continual[0].task_names
    if any(run.task_names != task_order for run in continual):
        raise ValueError("Continual task orders differ across seeds")
    if tuple(TASK_DISPLAY) != task_order:
        raise ValueError(f"Unexpected Atari task order: {task_order}")

    single_runs = discover_single_task_runs()
    by_single_task: dict[str, list[Run]] = {task: [] for task in task_order}
    for run in single_runs:
        if len(run.task_names) != 1 or run.task_names[0] not in by_single_task:
            raise ValueError(f"Unexpected single-task run: {run.run_dir}")
        by_single_task[run.task_names[0]].append(run)
    for task, runs in by_single_task.items():
        runs.sort(key=lambda run: run.seed_id)
        if {run.seed_id for run in runs} != {0, 1, 2}:
            raise ValueError(f"Expected single-task seed IDs 0,1,2 for {task}")

    inventory = {
        "continual_runs": len(continual),
        "single_task_runs": len(single_runs),
        "continual_seed_ids": [run.seed_id for run in continual],
        "single_task_seed_ids_per_task": {
            TASK_DISPLAY[task]: [run.seed_id for run in runs]
            for task, runs in by_single_task.items()
        },
        "tasks": [TASK_DISPLAY[task] for task in task_order],
        "continual_evaluation_checkpoints": len(EXPECTED_CONTINUAL_EPOCHS),
        "single_task_evaluation_checkpoints": len(EXPECTED_SINGLE_TASK_EPOCHS),
    }
    if validate_only:
        print(json.dumps(inventory, indent=2))
        return inventory

    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    table_dir = output / "tables"
    figure_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    anchors: dict[str, dict[str, float]] = {}
    for task in task_order:
        runs = by_single_task[task]
        initial_values = [run.by_epoch[0].raw_mean[0] for run in runs]
        final_values = [run.by_epoch[90].raw_mean[0] for run in runs]
        initial_stats = stats(initial_values)
        final_stats = stats(final_values)
        anchors[task] = {
            "initial_median": initial_stats["median"],
            "initial_q25": initial_stats["q25"],
            "initial_q75": initial_stats["q75"],
            "final_median": final_stats["median"],
            "final_q25": final_stats["q25"],
            "final_q75": final_stats["q75"],
        }

    continual_long: list[dict[str, Any]] = []
    single_long: list[dict[str, Any]] = []
    for run in continual:
        for point in run.evaluations:
            for task_index, task in enumerate(task_order):
                continual_long.append(
                    {
                        "seed_id": run.seed_id,
                        "seed": run.seed,
                        "epoch": point.epoch,
                        "task_index": task_index,
                        "task": TASK_DISPLAY[task],
                        "raw_return_mean": point.raw_mean[task_index],
                        "raw_return_std": point.raw_std[task_index],
                        "local_st_aligned_score": normalize(
                            point.raw_mean[task_index], anchors[task]
                        ),
                        "raw_source": point.raw_source,
                    }
                )
    for task_index, task in enumerate(task_order):
        for run in by_single_task[task]:
            for point in run.evaluations:
                single_long.append(
                    {
                        "seed_id": run.seed_id,
                        "seed": run.seed,
                        "epoch": point.epoch,
                        "task_index": task_index,
                        "task": TASK_DISPLAY[task],
                        "raw_return_mean": point.raw_mean[0],
                        "raw_return_std": point.raw_std[0],
                        "local_st_aligned_score": normalize(point.raw_mean[0], anchors[task]),
                        "raw_source": point.raw_source,
                    }
                )
    write_csv(
        table_dir / "continual_evaluations_long.csv",
        continual_long,
        list(continual_long[0]),
    )
    write_csv(
        table_dir / "single_task_evaluations_long.csv",
        single_long,
        list(single_long[0]),
    )

    metric_rows: list[dict[str, Any]] = []
    metric_by_seed: dict[int, dict[str, Any]] = {}
    for run in continual:
        normalized_matrix = [
            [
                normalize(point.raw_mean[index], anchors[task])
                for index, task in enumerate(task_order)
            ]
            for point in run.evaluations
        ]
        row_by_epoch = {point.epoch: index for index, point in enumerate(run.evaluations)}
        ends = [row_by_epoch[epoch] for epoch in BOUNDARY_EPOCHS]
        metrics = single_pass_metrics(normalized_matrix, ends)
        boundary_forgetting = []
        for boundary_index, boundary_row in enumerate(ends):
            values = [
                normalized_matrix[ends[task_index]][task_index]
                - normalized_matrix[boundary_row][task_index]
                for task_index in range(boundary_index + 1)
            ]
            boundary_forgetting.append(sum(values) / len(values))
        metrics["boundary_forgetting"] = boundary_forgetting
        metric_by_seed[run.seed_id] = metrics
        metric_rows.append(
            {
                "seed_id": run.seed_id,
                "seed": run.seed,
                "forgetting": metrics["forgetting"],
                "acc": metrics["acc"],
                "min_acc": metrics["min_acc"],
                "wc_acc": metrics["wc_acc"],
                "normalization": "local_single_task_epoch0_to_epoch90_medians",
            }
        )
    write_csv(table_dir / "seed_metrics_local.csv", metric_rows, list(metric_rows[0]))

    metric_summary_rows = []
    for metric in ("forgetting", "acc", "min_acc", "wc_acc"):
        values = [float(row[metric]) for row in metric_rows]
        summary = stats(values)
        metric_summary_rows.append({"metric": metric, **summary, "seed_count": 5})
    write_csv(
        table_dir / "metric_summary_local.csv",
        metric_summary_rows,
        list(metric_summary_rows[0]),
    )

    anchor_rows = []
    for task_index, task in enumerate(task_order):
        row = {"task_index": task_index, "task": TASK_DISPLAY[task], **anchors[task]}
        anchor_rows.append(row)
    write_csv(table_dir / "local_normalization_anchors.csv", anchor_rows, list(anchor_rows[0]))

    final_raw_rows = []
    final_score_rows = []
    for run in continual:
        point = run.by_epoch[540]
        raw_row: dict[str, Any] = {"seed_id": run.seed_id, "seed": run.seed}
        score_row: dict[str, Any] = {"seed_id": run.seed_id, "seed": run.seed}
        for task_index, task in enumerate(task_order):
            raw_row[TASK_DISPLAY[task]] = point.raw_mean[task_index]
            score_row[TASK_DISPLAY[task]] = normalize(point.raw_mean[task_index], anchors[task])
        final_raw_rows.append(raw_row)
        final_score_rows.append(score_row)
    write_csv(
        table_dir / "continual_final_raw_by_seed.csv",
        final_raw_rows,
        list(final_raw_rows[0]),
    )
    write_csv(
        table_dir / "continual_final_local_score_by_seed.csv",
        final_score_rows,
        list(final_score_rows[0]),
    )

    task_summary_rows = []
    for task_index, task in enumerate(task_order):
        acquisition_epoch = BOUNDARY_EPOCHS[task_index]
        acquisition_raw = [
            run.by_epoch[acquisition_epoch].raw_mean[task_index] for run in continual
        ]
        final_raw = [run.by_epoch[540].raw_mean[task_index] for run in continual]
        acquisition_score = [normalize(value, anchors[task]) for value in acquisition_raw]
        final_score = [normalize(value, anchors[task]) for value in final_raw]
        forgetting = [left - right for left, right in zip(acquisition_score, final_score)]
        acq_raw_stats, final_raw_stats = stats(acquisition_raw), stats(final_raw)
        acq_score_stats, final_score_stats = stats(acquisition_score), stats(final_score)
        forgetting_stats = stats(forgetting)
        task_summary_rows.append(
            {
                "task_index": task_index,
                "task": TASK_DISPLAY[task],
                "acquisition_epoch": acquisition_epoch,
                "acquisition_raw_median": acq_raw_stats["median"],
                "acquisition_raw_q25": acq_raw_stats["q25"],
                "acquisition_raw_q75": acq_raw_stats["q75"],
                "final_raw_median": final_raw_stats["median"],
                "final_raw_q25": final_raw_stats["q25"],
                "final_raw_q75": final_raw_stats["q75"],
                "acquisition_local_score_median": acq_score_stats["median"],
                "acquisition_local_score_q25": acq_score_stats["q25"],
                "acquisition_local_score_q75": acq_score_stats["q75"],
                "final_local_score_median": final_score_stats["median"],
                "final_local_score_q25": final_score_stats["q25"],
                "final_local_score_q75": final_score_stats["q75"],
                "forgetting_median": forgetting_stats["median"],
                "forgetting_q25": forgetting_stats["q25"],
                "forgetting_q75": forgetting_stats["q75"],
            }
        )
    write_csv(
        table_dir / "task_acquisition_final_summary.csv",
        task_summary_rows,
        list(task_summary_rows[0]),
    )

    single_final_rows = []
    for task_index, task in enumerate(task_order):
        for run in by_single_task[task]:
            single_final_rows.append(
                {
                    "task_index": task_index,
                    "task": TASK_DISPLAY[task],
                    "seed_id": run.seed_id,
                    "seed": run.seed,
                    "raw_return_mean": run.by_epoch[90].raw_mean[0],
                    "raw_return_std": run.by_epoch[90].raw_std[0],
                }
            )
    write_csv(
        table_dir / "single_task_final_raw.csv",
        single_final_rows,
        list(single_final_rows[0]),
    )

    inventory_rows = []
    for run in continual:
        inventory_rows.append(
            {
                "run_type": "continual",
                "task": "six-task curriculum",
                "seed_id": run.seed_id,
                "seed": run.seed,
                "run_dir": relative(run.run_dir),
                "runtime": run.runtime,
                "git_commit": run.git_commit,
                "evaluation_checkpoints": len(run.evaluations),
                "wall_hours": iso_hours(run.started_at_utc, run.finished_at_utc),
                "logged_stage_hours": stage_hours(run.run_dir / "train.log"),
                "gpu_name": run.gpu_name,
            }
        )
    for task in task_order:
        for run in by_single_task[task]:
            inventory_rows.append(
                {
                    "run_type": "single_task",
                    "task": TASK_DISPLAY[task],
                    "seed_id": run.seed_id,
                    "seed": run.seed,
                    "run_dir": relative(run.run_dir),
                    "runtime": run.runtime,
                    "git_commit": run.git_commit,
                    "evaluation_checkpoints": len(run.evaluations),
                    "wall_hours": iso_hours(run.started_at_utc, run.finished_at_utc),
                    "logged_stage_hours": stage_hours(run.run_dir / "train.log"),
                    "gpu_name": run.gpu_name,
                }
            )
    write_csv(table_dir / "run_inventory.csv", inventory_rows, list(inventory_rows[0]))

    # Figure 1: raw continual curves.
    image = new_canvas(
        "ARROW-50 Atari continual learning — raw returns",
        "Our five completed seeds; thick line = median, band = IQR, thin lines = individual seeds",
    )
    add_line_legend(image, labels=(("5-seed median", "#111827"), ("task-specific color / IQR", "#6B7280")))
    for task_index, (task, box) in enumerate(zip(task_order, grid_boxes())):
        curves = [
            [run.by_epoch[epoch].raw_mean[task_index] for epoch in EXPECTED_CONTINUAL_EPOCHS]
            for run in continual
        ]
        summaries = [stats([curve[index] for curve in curves]) for index in range(len(EXPECTED_CONTINUAL_EPOCHS))]
        draw_line_panel(
            image,
            box,
            x_values=EXPECTED_CONTINUAL_EPOCHS,
            seed_curves=curves,
            median=[row["median"] for row in summaries],
            q25=[row["q25"] for row in summaries],
            q75=[row["q75"] for row in summaries],
            title=TASK_DISPLAY[task],
            color=TASK_COLORS[task_index],
            x_ticks=(0, 90, 180, 270, 360, 450, 540),
            train_span=(task_index * 90, (task_index + 1) * 90),
            xlabel="Continual epoch",
        )
    save_image(image, figure_dir / "fig01_continual_raw_per_task.png")

    # Figure 2: locally normalized continual curves.
    image = new_canvas(
        "ARROW-50 Atari continual learning — local single-task aligned score",
        "Only our runs: score 0 = local 3-seed single-task epoch-0 median; score 1 = epoch-90 median",
    )
    add_line_legend(image, labels=(("5-seed median", "#111827"), ("IQR + individual seeds", "#6B7280")))
    for task_index, (task, box) in enumerate(zip(task_order, grid_boxes())):
        curves = [
            [normalize(run.by_epoch[epoch].raw_mean[task_index], anchors[task]) for epoch in EXPECTED_CONTINUAL_EPOCHS]
            for run in continual
        ]
        summaries = [stats([curve[index] for curve in curves]) for index in range(len(EXPECTED_CONTINUAL_EPOCHS))]
        draw_line_panel(
            image,
            box,
            x_values=EXPECTED_CONTINUAL_EPOCHS,
            seed_curves=curves,
            median=[row["median"] for row in summaries],
            q25=[row["q25"] for row in summaries],
            q75=[row["q75"] for row in summaries],
            title=TASK_DISPLAY[task],
            color=TASK_COLORS[task_index],
            x_ticks=(0, 90, 180, 270, 360, 450, 540),
            train_span=(task_index * 90, (task_index + 1) * 90),
            reference_lines=(0, 1),
            include_zero=True,
            include_one=True,
            xlabel="Continual epoch",
        )
    save_image(image, figure_dir / "fig02_continual_local_score_per_task.png")

    # Figure 3: task-aligned continual acquisition versus single-task learning.
    image = new_canvas(
        "Task-aligned acquisition: continual ARROW-50 vs single-task ARROW-50",
        "All curves are ours; raw episodic return, with each continual task window re-indexed to epochs 0–90",
    )
    add_line_legend(
        image,
        labels=(("Continual: 5-seed median/IQR", GROUP_COLORS["acquisition"]), ("Single-task: 3-seed median/IQR", GROUP_COLORS["single"])),
    )
    for task_index, (task, box) in enumerate(zip(task_order, grid_boxes())):
        offsets = EXPECTED_SINGLE_TASK_EPOCHS
        continual_curves = [
            [run.by_epoch[task_index * 90 + offset].raw_mean[task_index] for offset in offsets]
            for run in continual
        ]
        single_curves = [
            [run.by_epoch[offset].raw_mean[0] for offset in offsets]
            for run in by_single_task[task]
        ]
        all_curves = continual_curves + single_curves
        all_values = [value for curve in all_curves for value in curve]
        y_low, y_high, y_ticks = nice_limits(all_values, include_zero=True)
        x0, y0, x1, y1 = box
        left, right, top, bottom = x0 + 100, x1 - 25, y0 + 58, y1 - 72

        def px(value: float) -> float:
            return left + value / 90 * (right - left)

        def py(value: float) -> float:
            return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

        draw = ImageDraw.Draw(image)
        for tick in y_ticks:
            y = py(tick)
            draw.line((left, y, right, y), fill="#E5E7EB", width=2)
            right_text(draw, (left - 13, y), fmt_tick(tick), font=FONTS.get(18), fill="#6B7280")
        for tick in (0, 30, 60, 90):
            x = px(tick)
            draw.line((x, bottom, x, bottom + 8), fill="#6B7280", width=2)
            center_text(draw, (x, bottom + 29), str(tick), font=FONTS.get(18), fill="#6B7280")
        for curves, color in ((continual_curves, GROUP_COLORS["acquisition"]), (single_curves, GROUP_COLORS["single"])):
            summaries = [stats([curve[index] for curve in curves]) for index in range(len(offsets))]
            band = [(px(x), py(row["q75"])) for x, row in zip(offsets, summaries)]
            band.extend((px(x), py(row["q25"])) for x, row in reversed(list(zip(offsets, summaries))))
            alpha_polygon(image, band, color, 48)
            draw = ImageDraw.Draw(image)
            draw.line(
                [(px(x), py(row["median"])) for x, row in zip(offsets, summaries)],
                fill=color,
                width=6,
                joint="curve",
            )
        draw = ImageDraw.Draw(image)
        draw.line((left, top, left, bottom), fill="#374151", width=3)
        draw.line((left, bottom, right, bottom), fill="#374151", width=3)
        center_text(draw, ((x0 + x1) / 2, y0 + 20), TASK_DISPLAY[task], font=FONTS.get(25, bold=True))
        center_text(draw, ((left + right) / 2, y1 - 20), "Task-relative epoch", font=FONTS.get(19), fill="#4B5563")
    save_image(image, figure_dir / "fig03_task_aligned_continual_vs_single.png")

    # Figure 4: raw endpoint distributions.
    image = new_canvas(
        "ARROW-50 raw return at key endpoints",
        "Dots are individual runs; vertical whisker = IQR; horizontal mark = median",
    )
    add_line_legend(
        image,
        labels=(("Single-task E90 (n=3)", GROUP_COLORS["single"]), ("Continual acquisition (n=5)", GROUP_COLORS["acquisition"]), ("Continual final E540 (n=5)", GROUP_COLORS["final"])),
    )
    categories = ("ST E90", "CL acquire", "CL final")
    for task_index, (task, box) in enumerate(zip(task_order, grid_boxes())):
        values_by_category = (
            [run.by_epoch[90].raw_mean[0] for run in by_single_task[task]],
            [run.by_epoch[BOUNDARY_EPOCHS[task_index]].raw_mean[task_index] for run in continual],
            [run.by_epoch[540].raw_mean[task_index] for run in continual],
        )
        all_values = [value for values in values_by_category for value in values]
        y_low, y_high, y_ticks = nice_limits(all_values, include_zero=True)
        x0, y0, x1, y1 = box
        left, right, top, bottom = x0 + 100, x1 - 25, y0 + 58, y1 - 78
        xs = [left + (index + 0.5) / 3 * (right - left) for index in range(3)]

        def py(value: float) -> float:
            return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

        draw = ImageDraw.Draw(image)
        for tick in y_ticks:
            y = py(tick)
            draw.line((left, y, right, y), fill="#E5E7EB", width=2)
            right_text(draw, (left - 13, y), fmt_tick(tick), font=FONTS.get(18), fill="#6B7280")
        for category_index, (x, values, key) in enumerate(zip(xs, values_by_category, ("single", "acquisition", "final"))):
            summary = stats(values)
            color = GROUP_COLORS[key]
            draw.line((x, py(summary["q25"]), x, py(summary["q75"])), fill=color, width=8)
            draw.line((x - 18, py(summary["median"]), x + 18, py(summary["median"])), fill="#111827", width=6)
            jitters = (-19, 0, 19) if len(values) == 3 else (-28, -14, 0, 14, 28)
            for jitter, value in zip(jitters, values):
                y = py(value)
                draw.ellipse((x + jitter - 7, y - 7, x + jitter + 7, y + 7), fill=color, outline="white", width=2)
            center_text(draw, (x, bottom + 31), categories[category_index], font=FONTS.get(17), fill="#4B5563")
        draw.line((left, top, left, bottom), fill="#374151", width=3)
        draw.line((left, bottom, right, bottom), fill="#374151", width=3)
        center_text(draw, ((x0 + x1) / 2, y0 + 20), TASK_DISPLAY[task], font=FONTS.get(25, bold=True))
    save_image(image, figure_dir / "fig04_raw_endpoint_distributions.png")

    # Figure 5: task-boundary score matrix.
    image = new_canvas(
        "Median local single-task aligned score at continual task boundaries",
        "Rows are completed-task boundaries; columns are evaluation tasks; all cells aggregate our five continual seeds",
        size=(2200, 1350),
    )
    draw = ImageDraw.Draw(image)
    left, top, cell_w, cell_h = 390, 235, 250, 135
    matrix: list[list[float]] = []
    for epoch in BOUNDARY_EPOCHS:
        matrix.append(
            [
                statistics.median(
                    normalize(run.by_epoch[epoch].raw_mean[task_index], anchors[task])
                    for run in continual
                )
                for task_index, task in enumerate(task_order)
            ]
        )
    for column, task in enumerate(task_order):
        center_text(
            draw,
            (left + column * cell_w + cell_w / 2, top - 48),
            TASK_SHORT[task],
            font=FONTS.get(23, bold=True),
        )
    for row, (epoch, values) in enumerate(zip(BOUNDARY_EPOCHS, matrix)):
        label = f"After T{row + 1} · E{epoch}"
        right_text(draw, (left - 24, top + row * cell_h + cell_h / 2), label, font=FONTS.get(23), fill="#374151")
        for column, value in enumerate(values):
            x0 = left + column * cell_w
            y0 = top + row * cell_h
            color = score_color(value)
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline="white", width=4)
            luminance = sum(channel * weight for channel, weight in zip(color, (0.299, 0.587, 0.114)))
            center_text(
                draw,
                (x0 + cell_w / 2, y0 + cell_h / 2),
                f"{value:.2f}",
                font=FONTS.get(28, bold=True),
                fill="white" if luminance < 135 else "#111827",
            )
    legend_x, legend_y, legend_w = left, top + 6 * cell_h + 70, 6 * cell_w
    for pixel in range(legend_w):
        value = -0.5 + pixel / max(1, legend_w - 1) * 2.0
        draw.line((legend_x + pixel, legend_y, legend_x + pixel, legend_y + 28), fill=score_color(value), width=1)
    for value in (-0.5, 0, 0.5, 1.0, 1.5):
        x = legend_x + (value + 0.5) / 2.0 * legend_w
        center_text(draw, (x, legend_y + 52), f"{value:g}", font=FONTS.get(18), fill="#4B5563")
    center_text(draw, (legend_x + legend_w / 2, legend_y + 86), "Local single-task aligned score", font=FONTS.get(20), fill="#374151")
    save_image(image, figure_dir / "fig05_boundary_score_heatmap.png")

    # Figure 6: seed-by-task final score heatmap.
    image = new_canvas(
        "Final local single-task aligned score by seed and Atari task",
        "Each cell is one of our five continual ARROW-50 runs at epoch 540",
        size=(2200, 1180),
    )
    draw = ImageDraw.Draw(image)
    left, top, cell_w, cell_h = 390, 235, 250, 135
    for column, task in enumerate(task_order):
        center_text(draw, (left + column * cell_w + cell_w / 2, top - 48), TASK_SHORT[task], font=FONTS.get(23, bold=True))
    for row, run in enumerate(continual):
        right_text(draw, (left - 24, top + row * cell_h + cell_h / 2), f"seed{run.seed_id} · {run.seed}", font=FONTS.get(22), fill="#374151")
        for task_index, task in enumerate(task_order):
            value = normalize(run.by_epoch[540].raw_mean[task_index], anchors[task])
            x0, y0 = left + task_index * cell_w, top + row * cell_h
            color = score_color(value)
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline="white", width=4)
            luminance = sum(channel * weight for channel, weight in zip(color, (0.299, 0.587, 0.114)))
            center_text(draw, (x0 + cell_w / 2, y0 + cell_h / 2), f"{value:.2f}", font=FONTS.get(28, bold=True), fill="white" if luminance < 135 else "#111827")
    legend_x, legend_y, legend_w = left, top + 5 * cell_h + 70, 6 * cell_w
    for pixel in range(legend_w):
        value = -0.5 + pixel / max(1, legend_w - 1) * 2.0
        draw.line((legend_x + pixel, legend_y, legend_x + pixel, legend_y + 28), fill=score_color(value), width=1)
    for value in (-0.5, 0, 0.5, 1.0, 1.5):
        x = legend_x + (value + 0.5) / 2.0 * legend_w
        center_text(draw, (x, legend_y + 52), f"{value:g}", font=FONTS.get(18), fill="#4B5563")
    save_image(image, figure_dir / "fig06_final_seed_task_heatmap.png")

    # Figure 7: acquisition-to-final dumbbell plot.
    image = new_canvas(
        "Acquisition-to-final change in local single-task aligned score",
        "Median and IQR across our five continual seeds; left/right movement directly shows forgetting or improvement",
        size=(2200, 1350),
    )
    draw = ImageDraw.Draw(image)
    all_scores = []
    for row in task_summary_rows:
        all_scores.extend((row["acquisition_local_score_q25"], row["acquisition_local_score_q75"], row["final_local_score_q25"], row["final_local_score_q75"]))
    low, high, ticks = nice_limits(all_scores, include_zero=True, include_one=True)
    left, right, top, bottom = 470, 2070, 240, 1100

    def px(value: float) -> float:
        return left + (value - low) / (high - low) * (right - left)

    for tick in ticks:
        x = px(tick)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=2)
        center_text(draw, (x, bottom + 38), fmt_tick(tick), font=FONTS.get(20), fill="#6B7280")
    if low <= 1 <= high:
        draw.line((px(1), top, px(1), bottom), fill="#6B7280", width=4)
    for index, row in enumerate(task_summary_rows):
        y = top + (index + 0.5) / 6 * (bottom - top)
        right_text(draw, (left - 28, y), row["task"], font=FONTS.get(25, bold=True), fill="#374151")
        acq = row["acquisition_local_score_median"]
        final = row["final_local_score_median"]
        draw.line((px(acq), y, px(final), y), fill="#9CA3AF", width=5)
        for key, color in (("acquisition", GROUP_COLORS["acquisition"]), ("final", GROUP_COLORS["final"])):
            median = row[f"{key}_local_score_median"]
            q25 = row[f"{key}_local_score_q25"]
            q75 = row[f"{key}_local_score_q75"]
            draw.line((px(q25), y, px(q75), y), fill=color, width=8)
            draw.ellipse((px(median) - 11, y - 11, px(median) + 11, y + 11), fill=color, outline="white", width=3)
        delta = final - acq
        draw.text((right + 20, y - 13), f"Δ {delta:+.2f}", font=FONTS.get(21, bold=True), fill="#166534" if delta >= 0 else "#991B1B")
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    add_line_legend(image, labels=(("At acquisition", GROUP_COLORS["acquisition"]), ("At final E540", GROUP_COLORS["final"])), y=180)
    center_text(draw, ((left + right) / 2, bottom + 84), "Local single-task aligned score", font=FONTS.get(23), fill="#374151")
    save_image(image, figure_dir / "fig07_acquisition_to_final.png")

    # Figure 8: boundary metrics.
    image = new_canvas(
        "Continual metrics as tasks accumulate",
        "Derived only from our runs and local single-task anchors; median and IQR across five seeds",
        size=(2200, 1250),
    )
    boxes = ((80, 185, 1080, 1140), (1120, 185, 2120, 1140))
    draw = ImageDraw.Draw(image)
    # Left: ACC/min-ACC/WC-ACC.
    metric_specs = (("acc", "ACC", "#4C78A8"), ("min_acc", "min-ACC", "#F58518"), ("wc_acc", "WC-ACC", "#54A24B"))
    x_values = list(range(1, 7))
    series: dict[str, list[dict[str, float] | None]] = {}
    left_values: list[float] = []
    for key, _, _ in metric_specs:
        rows: list[dict[str, float] | None] = []
        for boundary_index in range(6):
            values = [metric_by_seed[seed]["boundaries"][boundary_index][key] for seed in range(5)]
            finite = [float(value) for value in values if value is not None]
            row = stats(finite) if finite else None
            rows.append(row)
            if row:
                left_values.extend((row["q25"], row["q75"]))
        series[key] = rows
    x0, y0, x1, y1 = boxes[0]
    left, right, top, bottom = x0 + 115, x1 - 35, y0 + 80, y1 - 85
    low, high, ticks = nice_limits(left_values, include_zero=True, include_one=True)

    def xmap(value: float) -> float:
        return left + (value - 1) / 5 * (right - left)

    def ymap(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    for tick in ticks:
        y = ymap(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=2)
        right_text(draw, (left - 14, y), fmt_tick(tick), font=FONTS.get(19), fill="#6B7280")
    for value in x_values:
        center_text(draw, (xmap(value), bottom + 32), str(value), font=FONTS.get(20), fill="#6B7280")
    for key, label, color in metric_specs:
        valid = [(x, row) for x, row in zip(x_values, series[key]) if row is not None]
        upper = [(xmap(x), ymap(row["q75"])) for x, row in valid]
        lower_points = [(xmap(x), ymap(row["q25"])) for x, row in reversed(valid)]
        alpha_polygon(image, upper + lower_points, color, 42)
        draw = ImageDraw.Draw(image)
        draw.line([(xmap(x), ymap(row["median"])) for x, row in valid], fill=color, width=6, joint="curve")
    draw.line((left, top, left, bottom), fill="#374151", width=3)
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    center_text(draw, ((x0 + x1) / 2, y0 + 28), "Stability–plasticity metrics", font=FONTS.get(28, bold=True))
    center_text(draw, ((left + right) / 2, y1 - 28), "Completed task count", font=FONTS.get(21), fill="#374151")
    legend_x = left + 15
    for key, label, color in metric_specs:
        draw.line((legend_x, top + 16, legend_x + 34, top + 16), fill=color, width=6)
        draw.text((legend_x + 44, top + 5), label, font=FONTS.get(18), fill="#374151")
        legend_x += 180
    # Right: forgetting through boundaries.
    x0, y0, x1, y1 = boxes[1]
    left, right, top, bottom = x0 + 115, x1 - 35, y0 + 80, y1 - 85
    forgetting_rows = [
        stats([metric_by_seed[seed]["boundary_forgetting"][index] for seed in range(5)])
        for index in range(6)
    ]
    values = [value for row in forgetting_rows for value in (row["q25"], row["q75"])]
    low, high, ticks = nice_limits(values, include_zero=True)

    def xmap2(value: float) -> float:
        return left + (value - 1) / 5 * (right - left)

    def ymap2(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    for tick in ticks:
        y = ymap2(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=2)
        right_text(draw, (left - 14, y), fmt_tick(tick), font=FONTS.get(19), fill="#6B7280")
    for value in x_values:
        center_text(draw, (xmap2(value), bottom + 32), str(value), font=FONTS.get(20), fill="#6B7280")
    band = [(xmap2(x), ymap2(row["q75"])) for x, row in zip(x_values, forgetting_rows)]
    band.extend((xmap2(x), ymap2(row["q25"])) for x, row in reversed(list(zip(x_values, forgetting_rows))))
    alpha_polygon(image, band, "#E45756", 50)
    draw = ImageDraw.Draw(image)
    draw.line([(xmap2(x), ymap2(row["median"])) for x, row in zip(x_values, forgetting_rows)], fill="#E45756", width=6, joint="curve")
    draw.line((left, top, left, bottom), fill="#374151", width=3)
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    center_text(draw, ((x0 + x1) / 2, y0 + 28), "Average forgetting (lower is better)", font=FONTS.get(28, bold=True))
    center_text(draw, ((left + right) / 2, y1 - 28), "Completed task count", font=FONTS.get(21), fill="#374151")
    save_image(image, figure_dir / "fig08_boundary_metrics.png")

    # Figure 9: final metric distributions.
    image = new_canvas(
        "Final continual metric distributions across our five seeds",
        "Normalization uses only our local three-seed single-task epoch-0 and epoch-90 medians",
        size=(2200, 1400),
    )
    metric_panels = (("forgetting", "Forgetting ↓", "#E45756"), ("acc", "ACC ↑", "#4C78A8"), ("min_acc", "min-ACC ↑", "#F58518"), ("wc_acc", "WC-ACC ↑", "#54A24B"))
    panel_boxes = grid_boxes(size=(2200, 1400), rows=2, columns=2)
    for (key, label, color), box in zip(metric_panels, panel_boxes):
        values = [float(row[key]) for row in metric_rows]
        summary = stats(values)
        low, high, ticks = nice_limits(values, include_zero=True, include_one=key != "forgetting")
        x0, y0, x1, y1 = box
        left, right, top, bottom = x0 + 120, x1 - 45, y0 + 70, y1 - 70

        def py_metric(value: float) -> float:
            return bottom - (value - low) / (high - low) * (bottom - top)

        draw = ImageDraw.Draw(image)
        for tick in ticks:
            y = py_metric(tick)
            draw.line((left, y, right, y), fill="#E5E7EB", width=2)
            right_text(draw, (left - 14, y), fmt_tick(tick), font=FONTS.get(19), fill="#6B7280")
        center = (left + right) / 2
        draw.line((center, py_metric(summary["q25"]), center, py_metric(summary["q75"])), fill=color, width=13)
        draw.line((center - 30, py_metric(summary["median"]), center + 30, py_metric(summary["median"])), fill="#111827", width=7)
        for seed_index, value in enumerate(values):
            x = center + (-80, -40, 0, 40, 80)[seed_index]
            y = py_metric(value)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=SEED_COLORS[seed_index], outline="white", width=3)
            center_text(draw, (x, bottom + 30), f"s{seed_index}", font=FONTS.get(17), fill="#6B7280")
        draw.line((left, top, left, bottom), fill="#374151", width=3)
        draw.line((left, bottom, right, bottom), fill="#374151", width=3)
        center_text(draw, ((x0 + x1) / 2, y0 + 27), label, font=FONTS.get(29, bold=True))
        center_text(draw, ((x0 + x1) / 2, y1 - 25), f"median {summary['median']:.3f}  [IQR {summary['q25']:.3f}, {summary['q75']:.3f}]", font=FONTS.get(20), fill="#374151")
    save_image(image, figure_dir / "fig09_final_metric_distributions.png")

    # Figure 10: single-task curves.
    image = new_canvas(
        "ARROW-50 Atari single-task learning — raw returns",
        "Our three completed seeds per game; these runs provide local alignment anchors, not the continual headline",
    )
    add_line_legend(image, labels=(("3-seed median", "#111827"), ("IQR + individual seeds", "#6B7280")))
    for task_index, (task, box) in enumerate(zip(task_order, grid_boxes())):
        curves = [[run.by_epoch[epoch].raw_mean[0] for epoch in EXPECTED_SINGLE_TASK_EPOCHS] for run in by_single_task[task]]
        summaries = [stats([curve[index] for curve in curves]) for index in range(len(EXPECTED_SINGLE_TASK_EPOCHS))]
        draw_line_panel(
            image,
            box,
            x_values=EXPECTED_SINGLE_TASK_EPOCHS,
            seed_curves=curves,
            median=[row["median"] for row in summaries],
            q25=[row["q25"] for row in summaries],
            q75=[row["q75"] for row in summaries],
            title=TASK_DISPLAY[task],
            color=TASK_COLORS[task_index],
            x_ticks=(0, 30, 60, 90),
            train_span=(0, 90),
            xlabel="Single-task epoch",
        )
    save_image(image, figure_dir / "fig10_single_task_raw_per_task.png")

    # Human-readable tables.
    final_summary_md_rows = []
    for task_index, task in enumerate(task_order):
        values = [run.by_epoch[540].raw_mean[task_index] for run in continual]
        summary = stats(values)
        final_summary_md_rows.append(
            (
                TASK_DISPLAY[task],
                raw_format(summary["median"]),
                f"{raw_format(summary['q25'])} – {raw_format(summary['q75'])}",
                "5",
            )
        )
    (table_dir / "continual_final_raw_summary.md").write_text(
        markdown_table(("Task", "Median raw return", "IQR", "Seeds"), final_summary_md_rows),
        encoding="utf-8",
    )
    metric_md_rows = [
        (
            row["metric"],
            f"{row['median']:.4f}",
            f"{row['q25']:.4f} – {row['q75']:.4f}",
            "5",
        )
        for row in metric_summary_rows
    ]
    (table_dir / "metric_summary_local.md").write_text(
        markdown_table(("Metric", "Median", "IQR", "Seeds"), metric_md_rows),
        encoding="utf-8",
    )

    source_files = []
    for run in continual + single_runs:
        for name in ("run_status.json", "launch.json", config_path(run.run_dir).name, "train.log"):
            path = run.run_dir / name
            if path.is_file():
                source_files.append({"path": relative(path), "sha256": sha256(path)})
    generated_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "arrow_atari_ours_only_figure_bundle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "benchmark": "Atari",
            "method": "ARROW-50",
            "continual_curriculum": "default order, one cycle",
            "continual_seed_count": 5,
            "single_task_seed_count_per_task": 3,
        },
        "external_result_values_used": False,
        "normalization": {
            "name": "local_single_task_epoch0_to_epoch90_medians",
            "formula": "q_local=(raw-ST_E0_median)/(ST_E90_median-ST_E0_median)",
            "source": "only the 18 local single-task runs listed in source_files",
            "scores_clipped": False,
            "anchors": {TASK_DISPLAY[task]: anchors[task] for task in task_order},
        },
        "inventory": inventory,
        "source_files": sorted(source_files, key=lambda row: row["path"]),
        "generated_files": [relative(path) for path in generated_files],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure_notes = (
        ("fig01_continual_raw_per_task.png", "Primary raw evidence: all five continual seeds, without cross-task scaling."),
        ("fig02_continual_local_score_per_task.png", "Cross-task view under anchors estimated only from our single-task cohort."),
        ("fig03_task_aligned_continual_vs_single.png", "Directly compares acquisition dynamics under continual and isolated training."),
        ("fig04_raw_endpoint_distributions.png", "Shows seed spread at single-task final, continual acquisition, and continual final."),
        ("fig05_boundary_score_heatmap.png", "Tracks every evaluation task after each curriculum boundary."),
        ("fig06_final_seed_task_heatmap.png", "Exposes seed-specific final strengths and failures."),
        ("fig07_acquisition_to_final.png", "Makes per-task forgetting or backward improvement explicit."),
        ("fig08_boundary_metrics.png", "Shows how ACC, min-ACC, WC-ACC, and forgetting evolve as tasks accumulate."),
        ("fig09_final_metric_distributions.png", "Shows the full five-seed distribution of final summary metrics."),
        ("fig10_single_task_raw_per_task.png", "Documents the three-seed alignment cohort."),
    )
    readme = [
        "# ARROW-50 Atari — our results only\n",
        "This bundle contains no result values from the ARROW paper or any other external experiment. "
        "It uses five completed local continual seeds and three completed local single-task seeds per Atari game.\n",
        "## Normalization\n",
        "Raw returns remain the primary measurements. Where a common scale is needed, `q_local` uses only our "
        "single-task runs: epoch-0 median = 0 and epoch-90 median = 1 for each game. Scores are not clipped. "
        "These locally anchored values must not be mixed with externally normalized values.\n",
        "## Figures\n",
    ]
    for filename, note in figure_notes:
        readme.append(f"### {filename}\n\n{note}\n\n![{filename}](figures/{filename})\n")
    readme.extend(
        (
            "## Tables\n",
            "- `tables/continual_final_raw_summary.md`: five-seed final raw returns.\n"
            "- `tables/metric_summary_local.md`: five-seed locally normalized continual metrics.\n"
            "- `tables/continual_final_raw_by_seed.csv`: every final raw task return.\n"
            "- `tables/continual_final_local_score_by_seed.csv`: every final locally aligned score.\n"
            "- `tables/task_acquisition_final_summary.csv`: acquisition/final/forgetting summaries.\n"
            "- `tables/local_normalization_anchors.csv`: exact local anchors.\n"
            "- `tables/continual_evaluations_long.csv` and `single_task_evaluations_long.csv`: plot-ready raw records.\n"
            "- `tables/run_inventory.csv`: provenance and runtime inventory.\n",
            "## Reproduce\n",
            "```bash\npython3 scripts/plot_arrow_atari_ours_only.py\n```\n",
        )
    )
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")

    cards = []
    for filename, note in figure_notes:
        cards.append(
            f'<section><h2>{html.escape(filename)}</h2><p>{html.escape(note)}</p>'
            f'<a href="figures/{html.escape(filename)}"><img src="figures/{html.escape(filename)}" alt="{html.escape(filename)}"></a></section>'
        )
    gallery = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ARROW-50 Atari — our results only</title>
<style>body{{font-family:Arial,sans-serif;background:#f3f4f6;color:#111827;margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:24px 0;border-radius:14px;box-shadow:0 2px 8px #0001}}img{{width:100%;height:auto}}p{{color:#4b5563;line-height:1.5}}</style></head>
<body><main><h1>ARROW-50 Atari — our results only</h1><p>No external result values are used. Click any figure for the full-resolution PNG.</p>{''.join(cards)}</main></body></html>"""
    (output / "index.html").write_text(gallery, encoding="utf-8")

    # Refresh generated-file inventory after README/gallery/manifest creation.
    generated_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest["generated_files"] = [relative(path) for path in generated_files]
    manifest["generated_file_sha256"] = {
        relative(path): sha256(path) for path in generated_files if path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_bundle(args.output.resolve(), validate_only=args.validate_only)
    if args.validate_only:
        return 0
    print(f"wrote {args.output.resolve()}")
    print(json.dumps(manifest["inventory"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
