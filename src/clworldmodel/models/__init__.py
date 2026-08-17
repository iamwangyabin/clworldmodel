"""Research model building blocks."""

from .fast_kan import (
    FastKAN,
    FastKANActor,
    FastKANCritic,
    FastKANLayer,
    FixedGaussianRBF,
    RMSNorm,
)
from .r2 import R2BarlowObjective, R2Projector, barlow_twins_loss
from .relu_kan import (
    AdaptiveReLUKANActor,
    BoundedReLUKANActor,
    FixedGridReLUKAN,
    FixedGridReLUKANLayer,
    ReLUKANActor,
)

__all__ = [
    "AdaptiveReLUKANActor",
    "FastKAN",
    "FastKANActor",
    "FastKANCritic",
    "FastKANLayer",
    "FixedGaussianRBF",
    "FixedGridReLUKAN",
    "FixedGridReLUKANLayer",
    "BoundedReLUKANActor",
    "R2BarlowObjective",
    "R2Projector",
    "RMSNorm",
    "ReLUKANActor",
    "barlow_twins_loss",
]
