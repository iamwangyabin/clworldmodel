"""Replay adapters owned by the continual world-model project."""

from .frozen_feature_cache import (
    ArrowFrozenFeatureCache,
    ArrowOnTheFlyFeatureSource,
)


def __getattr__(name: str):
    # A standalone replay import must not require the optional R2 vendor.
    if name == "ArrowR2ReplayAdapter":
        from .arrow_r2_adapter import ArrowR2ReplayAdapter

        return ArrowR2ReplayAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArrowFrozenFeatureCache",
    "ArrowOnTheFlyFeatureSource",
    "ArrowR2ReplayAdapter",
]
