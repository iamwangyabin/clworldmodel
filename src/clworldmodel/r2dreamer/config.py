"""Typed configuration for the R2-Dreamer size12M ARROW replay route."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace
from typing import Any


R2_DREAMER_SOURCE_COMMIT = "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"


def _namespace(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


@dataclass(frozen=True)
class R2DreamerConfig:
    """Validated R2-Dreamer size12M configuration.

    These defaults mirror the upstream size12M model and base objective. The
    ARROW integration owns replay, task scheduling, and run accounting only.
    """

    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    action_dim: int = 18

    batch_size: int = 16
    batch_length: int = 64

    stoch: int = 32
    deter: int = 2048
    hidden: int = 256
    discrete: int = 16
    depth: int = 16
    units: int = 256
    encoder_mults: tuple[int, ...] = (2, 3, 4, 4)
    encoder_kernel_size: int = 5
    encoder_minres: int = 4
    rssm_blocks: int = 8
    rssm_img_layers: int = 2
    rssm_obs_layers: int = 1
    rssm_dyn_layers: int = 1
    activation: str = "SiLU"
    norm: bool = True
    unimix_ratio: float = 0.01

    kl_free: float = 1.0
    loss_scale_barlow: float = 0.05
    loss_scale_reward: float = 1.0
    loss_scale_continue: float = 1.0
    loss_scale_dynamics: float = 1.0
    loss_scale_representation: float = 0.1
    loss_scale_policy: float = 1.0
    loss_scale_value: float = 1.0
    loss_scale_replay_value: float = 0.3
    barlow_redundancy_scale: float = 5e-4
    normalization_eps: float = 1e-8

    learning_rate: float = 4e-5
    agc: float = 0.3
    agc_pmin: float = 1e-3
    optimizer_eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_updates: int = 1000

    native_train_ratio: int = 128

    action_entropy: float = 3e-4
    imagination_horizon: int = 15
    discount_horizon: int = 333
    lambda_return: float = 0.95
    slow_target_update: int = 1
    slow_target_fraction: float = 0.02
    reward_bins: int = 255

    device: str = "cuda"
    amp: bool = True

    def __post_init__(self) -> None:
        positive_names = (
            "image_height",
            "image_width",
            "image_channels",
            "action_dim",
            "batch_size",
            "batch_length",
            "stoch",
            "deter",
            "hidden",
            "discrete",
            "depth",
            "units",
            "encoder_kernel_size",
            "encoder_minres",
            "rssm_blocks",
            "rssm_img_layers",
            "rssm_obs_layers",
            "rssm_dyn_layers",
            "warmup_updates",
            "native_train_ratio",
            "imagination_horizon",
            "discount_horizon",
            "slow_target_update",
            "reward_bins",
        )
        for name in positive_names:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.encoder_mults or any(mult < 1 for mult in self.encoder_mults):
            raise ValueError("encoder_mults must contain positive channel multipliers")
        downscale = 2 ** len(self.encoder_mults)
        if self.image_height % downscale or self.image_width % downscale:
            raise ValueError(
                "image dimensions must be divisible by the encoder downscale factor"
            )
        if (self.image_height // downscale, self.image_width // downscale) != (
            self.encoder_minres,
            self.encoder_minres,
        ):
            raise ValueError(
                "R2-Dreamer size12M expects the configured encoder_minres after pooling"
            )
        if self.deter % self.rssm_blocks:
            raise ValueError("deter must be divisible by rssm_blocks")
        if self.reward_bins % 2 != 1:
            raise ValueError("reward_bins must be odd for the symmetric two-hot head")
        if self.sample_count < self.embedding_dim:
            raise ValueError(
                "R2 Barlow correlation needs at least as many flattened samples as "
                f"encoder features, got S={self.sample_count} < E={self.embedding_dim}"
            )
        for name in (
            "loss_scale_barlow",
            "loss_scale_reward",
            "loss_scale_continue",
            "loss_scale_dynamics",
            "loss_scale_representation",
            "loss_scale_policy",
            "loss_scale_value",
            "loss_scale_replay_value",
            "barlow_redundancy_scale",
            "learning_rate",
            "agc",
            "agc_pmin",
            "optimizer_eps",
            "normalization_eps",
            "slow_target_fraction",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.unimix_ratio < 1:
            raise ValueError("unimix_ratio must lie strictly between zero and one")
        if not 0 < self.lambda_return <= 1:
            raise ValueError("lambda_return must lie in (0, 1]")

    @property
    def sample_count(self) -> int:
        return self.batch_size * self.batch_length

    @property
    def embedding_dim(self) -> int:
        return (
            self.depth
            * self.encoder_mults[-1]
            * self.encoder_minres
            * self.encoder_minres
        )

    @property
    def feature_dim(self) -> int:
        return self.stoch * self.discrete + self.deter

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "R2DreamerConfig":
        allowed = {field.name for field in fields(cls)}
        unexpected = sorted(set(values) - allowed)
        if unexpected:
            raise ValueError(f"Unknown R2-Dreamer configuration keys: {unexpected}")
        resolved = values.copy()
        if "encoder_mults" in resolved:
            resolved["encoder_mults"] = tuple(resolved["encoder_mults"])
        return cls(**resolved)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def vendor_encoder_config(self) -> SimpleNamespace:
        return _namespace(
            mlp_keys="$^",
            cnn_keys="image",
            mlp=_namespace(
                shape=None,
                layers=3,
                units=self.units,
                act=self.activation,
                norm=self.norm,
                device=self.device,
                outscale=None,
                symlog_inputs=True,
                name="mlp_encoder",
            ),
            cnn=_namespace(
                act=self.activation,
                norm=self.norm,
                kernel_size=self.encoder_kernel_size,
                minres=self.encoder_minres,
                depth=self.depth,
                mults=list(self.encoder_mults),
            ),
        )

    def vendor_rssm_config(self) -> SimpleNamespace:
        return _namespace(
            stoch=self.stoch,
            deter=self.deter,
            hidden=self.hidden,
            discrete=self.discrete,
            img_layers=self.rssm_img_layers,
            obs_layers=self.rssm_obs_layers,
            dyn_layers=self.rssm_dyn_layers,
            blocks=self.rssm_blocks,
            act=self.activation,
            norm=self.norm,
            unimix_ratio=self.unimix_ratio,
            initial="learned",
            device=self.device,
        )

    def vendor_head_config(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        distribution: str,
        outscale: float,
    ) -> SimpleNamespace:
        dist_values: dict[str, Any] = {"name": distribution}
        if distribution == "onehot":
            dist_values["unimix_ratio"] = self.unimix_ratio
        elif distribution == "symexp_twohot":
            dist_values["bin_num"] = self.reward_bins
        return _namespace(
            shape=shape,
            layers=3 if name in {"actor", "value"} else 1,
            units=self.units,
            act=self.activation,
            norm=self.norm,
            device=self.device,
            dist=_namespace(**dist_values),
            outscale=outscale,
            symlog_inputs=False,
            name=name,
        )
