"""Research model building blocks."""

from .fast_kan import (
    FastKAN,
    FastKANActor,
    FastKANCritic,
    FastKANLayer,
    FixedGaussianRBF,
    RMSNorm,
)
from .dinov3_adapter import ChannelLayerNorm, DinoPatchConvAdapter
from .frozen_dinov3 import FrozenDinoV3Encoder
from .residual_corrections import (
    LocalRBFKANCore,
    ParameterMatchedMLPCore,
    ResidualCorrection,
    build_residual_correction,
    soft_basis_support_overlap,
)
from .r2 import R2BarlowObjective, R2Projector, barlow_twins_loss
from .prediction_adapters import ZeroEffectFeatureAdapter
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
    "ChannelLayerNorm",
    "DinoPatchConvAdapter",
    "FrozenDinoV3Encoder",
    "FixedGaussianRBF",
    "FixedGridReLUKAN",
    "FixedGridReLUKANLayer",
    "LocalRBFKANCore",
    "ParameterMatchedMLPCore",
    "BoundedReLUKANActor",
    "R2BarlowObjective",
    "R2Projector",
    "RMSNorm",
    "ResidualCorrection",
    "ReLUKANActor",
    "ZeroEffectFeatureAdapter",
    "barlow_twins_loss",
    "build_residual_correction",
    "soft_basis_support_overlap",
]
