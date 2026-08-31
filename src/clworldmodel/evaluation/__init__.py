"""Evaluation utilities for continual world-model experiments."""

from __future__ import annotations

from .metrics import (
    METRIC_SCHEMA_VERSION,
    forward_transfer,
    median_iqr,
    normalize_return_matrix,
    sample_efficiency,
    single_pass_metrics,
    two_cycle_metrics,
)

__all__ = [
    "METRIC_SCHEMA_VERSION",
    "analyze_task_regions",
    "forward_transfer",
    "median_iqr",
    "normalize_return_matrix",
    "sample_efficiency",
    "single_pass_metrics",
    "task_support_overlap",
    "two_cycle_metrics",
]


def __getattr__(name: str):
    """Load NumPy-backed latent diagnostics only when they are requested."""
    if name in {"analyze_task_regions", "task_support_overlap"}:
        from .latent_regions import analyze_task_regions, task_support_overlap

        return {
            "analyze_task_regions": analyze_task_regions,
            "task_support_overlap": task_support_overlap,
        }[name]
    raise AttributeError(name)
