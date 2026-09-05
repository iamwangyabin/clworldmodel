"""Versioned continual-RL metrics used by ARROW-compatible reports.

The functions in this module operate on normalized task scores. Raw episodic
returns remain the source measurements and must be preserved by the caller.
Normalization is deliberately separate so a report cannot silently mix
random/single-task reference constants from different protocols.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


METRIC_SCHEMA_VERSION = "arrow-paper-v1"


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(
    values: Sequence[Sequence[float]], *, name: str
) -> list[list[float]]:
    if not values:
        raise ValueError(f"{name} must contain at least one checkpoint")
    rows = [
        [
            _finite(
                value,
                name=f"{name}[{row_index}][{column_index}]",
            )
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(values)
    ]
    if not rows[0]:
        raise ValueError(f"{name} must contain at least one task")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def normalize_return_matrix(
    raw_returns: Sequence[Sequence[float]],
    random_returns: Sequence[float],
    single_task_returns: Sequence[float],
) -> list[list[float]]:
    """Apply ARROW Eq. 1 using fixed per-task reference values.

    This helper is exact at points whose single-task reference is represented
    by ``single_task_returns`` (for example, task boundaries with the final
    Table A.15 anchors). Time-aligned learning-curve normalization must call
    this function separately for each aligned reference point or supply the
    already-normalized score matrix to the downstream metric functions.

    Scores are intentionally not clipped: values below 0 and above 1 are valid.
    """

    raw = _matrix(raw_returns, name="raw_returns")
    task_count = len(raw[0])
    if len(random_returns) != task_count or len(single_task_returns) != task_count:
        raise ValueError(
            "random_returns and single_task_returns must match the task count"
        )
    random_values = [
        _finite(value, name=f"random_returns[{index}]")
        for index, value in enumerate(random_returns)
    ]
    single_values = [
        _finite(value, name=f"single_task_returns[{index}]")
        for index, value in enumerate(single_task_returns)
    ]
    denominators = [
        single - random for single, random in zip(single_values, random_values)
    ]
    zero_denominators = [
        index for index, denominator in enumerate(denominators) if denominator == 0
    ]
    if zero_denominators:
        raise ValueError(
            "single-task and random references are identical for task indices "
            f"{zero_denominators}"
        )
    return [
        [
            (value - random_values[task_index]) / denominators[task_index]
            for task_index, value in enumerate(row)
        ]
        for row in raw
    ]


def _task_end_rows(
    task_end_rows: Sequence[int], *, row_count: int, task_count: int
) -> list[int]:
    if not task_end_rows:
        raise ValueError("task_end_rows must contain at least one completed task")
    if len(task_end_rows) > task_count:
        raise ValueError("task_end_rows cannot contain more entries than tasks")
    rows = [int(row) for row in task_end_rows]
    if any(row < 0 or row >= row_count for row in rows):
        raise ValueError("task_end_rows contains an out-of-range row")
    if any(left >= right for left, right in zip(rows, rows[1:])):
        raise ValueError("task_end_rows must be strictly increasing")
    return rows


def single_pass_metrics(
    normalized_scores: Sequence[Sequence[float]],
    task_end_rows: Sequence[int],
) -> dict[str, Any]:
    """Compute ARROW ACC, min-ACC, WC-ACC, and forgetting.

    ``normalized_scores`` has shape ``[evaluation checkpoint, task]``.
    ``task_end_rows[i]`` identifies the evaluation performed immediately after
    task ``i`` finished. The final task-end row is the endpoint used for the
    reported single-pass summary. Partial curricula are supported, but the
    caller must label them as partial rather than compare them with a full-suite
    headline number.
    """

    scores = _matrix(normalized_scores, name="normalized_scores")
    ends = _task_end_rows(
        task_end_rows, row_count=len(scores), task_count=len(scores[0])
    )
    boundary_metrics: list[dict[str, Any]] = []

    for task_index, boundary_row in enumerate(ends):
        completed = task_index + 1
        acc = sum(scores[boundary_row][:completed]) / completed
        current_score = scores[boundary_row][task_index]
        if completed == 1:
            min_acc = None
            wc_acc = current_score
            old_task_minima: list[float] = []
        else:
            old_task_minima = []
            for old_task_index in range(task_index):
                first_row_after_learning = ends[old_task_index] + 1
                values = [
                    scores[row][old_task_index]
                    for row in range(first_row_after_learning, boundary_row + 1)
                ]
                if not values:
                    raise ValueError(
                        "Each old task needs at least one evaluation after its "
                        "completion and before the requested boundary"
                    )
                old_task_minima.append(min(values))
            min_acc = sum(old_task_minima) / len(old_task_minima)
            wc_acc = current_score / completed + (1.0 - 1.0 / completed) * min_acc

        boundary_metrics.append(
            {
                "completed_task_count": completed,
                "completed_task_index": task_index,
                "evaluation_row": boundary_row,
                "acc": acc,
                "min_acc": min_acc,
                "wc_acc": wc_acc,
                "current_task_score": current_score,
                "old_task_minima": old_task_minima,
            }
        )

    final_row = ends[-1]
    completed_task_count = len(ends)
    per_task_forgetting = [
        scores[ends[task_index]][task_index] - scores[final_row][task_index]
        for task_index in range(completed_task_count)
    ]
    forgetting = sum(per_task_forgetting) / completed_task_count
    final_boundary = boundary_metrics[-1]
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "completed_task_count": completed_task_count,
        "final_evaluation_row": final_row,
        "forgetting": forgetting,
        "per_task_forgetting": per_task_forgetting,
        "acc": final_boundary["acc"],
        "min_acc": final_boundary["min_acc"],
        "wc_acc": final_boundary["wc_acc"],
        "boundaries": boundary_metrics,
    }


def forward_transfer(
    continual_task_curves: Sequence[Sequence[float]],
    single_task_curves: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compute ARROW Eq. 3 from aligned normalized acquisition curves.

    Each inner sequence covers one task during that task's acquisition window.
    The continual and single-task sequences for a task must use the same sample
    locations and weighting. Evaluation-checkpoint curves are therefore a
    discrete approximation unless every environment step is represented.
    """

    if not continual_task_curves or len(continual_task_curves) != len(
        single_task_curves
    ):
        raise ValueError(
            "continual_task_curves and single_task_curves must contain the same "
            "non-zero number of tasks"
        )
    per_task: list[float] = []
    continual_areas: list[float] = []
    single_task_areas: list[float] = []
    for task_index, (continual_curve, single_curve) in enumerate(
        zip(continual_task_curves, single_task_curves)
    ):
        if not continual_curve or len(continual_curve) != len(single_curve):
            raise ValueError(
                f"Task {task_index} curves must be non-empty and aligned"
            )
        continual_values = [
            _finite(value, name=f"continual_task_curves[{task_index}]")
            for value in continual_curve
        ]
        single_values = [
            _finite(value, name=f"single_task_curves[{task_index}]")
            for value in single_curve
        ]
        continual_area = sum(continual_values) / len(continual_values)
        single_area = sum(single_values) / len(single_values)
        if single_area == 0:
            raise ValueError(
                f"Task {task_index} single-task normalized area is zero"
            )
        continual_areas.append(continual_area)
        single_task_areas.append(single_area)
        per_task.append((continual_area - single_area) / single_area)
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "forward_transfer": sum(per_task) / len(per_task),
        "per_task_forward_transfer": per_task,
        "continual_areas": continual_areas,
        "single_task_areas": single_task_areas,
    }


def two_cycle_metrics(
    first_exposure_end: Sequence[float],
    before_second_exposure: Sequence[float],
    second_exposure_end: Sequence[float],
) -> dict[str, Any]:
    """Compute ARROW Max-F and Recovery from task-aligned score vectors."""

    task_count = len(first_exposure_end)
    if task_count == 0 or not (
        len(before_second_exposure) == task_count
        and len(second_exposure_end) == task_count
    ):
        raise ValueError("Two-cycle vectors must have the same non-zero length")
    first = [
        _finite(value, name=f"first_exposure_end[{index}]")
        for index, value in enumerate(first_exposure_end)
    ]
    before = [
        _finite(value, name=f"before_second_exposure[{index}]")
        for index, value in enumerate(before_second_exposure)
    ]
    second = [
        _finite(value, name=f"second_exposure_end[{index}]")
        for index, value in enumerate(second_exposure_end)
    ]
    zero_first = [index for index, value in enumerate(first) if value == 0]
    if zero_first:
        raise ValueError(
            "Recovery is undefined when first-exposure score is zero for task "
            f"indices {zero_first}"
        )
    max_forgetting = [
        start - pre_revisit for start, pre_revisit in zip(first, before)
    ]
    recovery = [relearned / start for relearned, start in zip(second, first)]
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "max_forgetting": sum(max_forgetting) / task_count,
        "per_task_max_forgetting": max_forgetting,
        "recovery": sum(recovery) / task_count,
        "per_task_recovery": recovery,
    }


def sample_efficiency(
    median_curves: Mapping[str, Sequence[tuple[int, float]]],
    *,
    threshold_fraction: float = 0.85,
) -> dict[str, Any]:
    """Return first frames reaching a shared fraction of the global maximum."""

    fraction = _finite(threshold_fraction, name="threshold_fraction")
    if not 0 < fraction <= 1:
        raise ValueError("threshold_fraction must be in (0, 1]")
    if not median_curves:
        raise ValueError("median_curves must contain at least one method")
    cleaned: dict[str, list[tuple[int, float]]] = {}
    for method, curve in median_curves.items():
        if not curve:
            raise ValueError(f"Method {method!r} has no curve points")
        points: list[tuple[int, float]] = []
        previous_frame = -1
        for point_index, (frame, value) in enumerate(curve):
            frame = int(frame)
            if frame < 0 or frame <= previous_frame:
                raise ValueError(
                    f"Method {method!r} frames must be non-negative and increasing"
                )
            points.append(
                (frame, _finite(value, name=f"{method}[{point_index}].value"))
            )
            previous_frame = frame
        cleaned[method] = points
    global_max = max(value for curve in cleaned.values() for _, value in curve)
    threshold = fraction * global_max
    reached: dict[str, int | None] = {}
    for method, curve in cleaned.items():
        reached[method] = next(
            (frame for frame, value in curve if value >= threshold), None
        )
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "threshold_fraction": fraction,
        "global_cross_method_maximum": global_max,
        "threshold": threshold,
        "frames_to_threshold": reached,
    }


def median_iqr(values: Sequence[float]) -> dict[str, float]:
    """Return median and linearly interpolated 25/75 percentiles."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(
        _finite(value, name=f"values[{index}]")
        for index, value in enumerate(values)
    )

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "median": quantile(0.5),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
    }


def raw_retention_metrics(
    raw_returns: Sequence[Sequence[float]],
    acquisition_rows: Sequence[int],
    final_row: int,
) -> dict[str, Any]:
    """Return-unit retention, with an explicit final endpoint (including revisits).

    Forgetting is acquisition return minus final return, not a clipped maximum;
    negative values mean improvement. No Atari/random/single-task normalization
    is applied. A caller must disclose unmatched evaluation seed cohorts.
    """
    scores = _matrix(raw_returns, name="raw_returns")
    ends = _task_end_rows(acquisition_rows, row_count=len(scores), task_count=len(scores[0]))
    if len(ends) != len(scores[0]) or not ends[-1] <= final_row < len(scores):
        raise ValueError("Raw retention needs an acquisition row for every task and a later final row")
    forgetting = [scores[end][task] - scores[final_row][task] for task, end in enumerate(ends)]
    mean_forgetting = sum(forgetting) / len(forgetting)
    return {
        "metric_schema_version": "raw-retention-v1",
        "final_average_raw_return": sum(scores[final_row]) / len(scores[final_row]),
        "per_task_raw_forgetting": forgetting,
        "mean_raw_forgetting": mean_forgetting,
        "backward_transfer_raw": -mean_forgetting,
        "final_evaluation_row": final_row,
        "acquisition_rows": ends,
    }
