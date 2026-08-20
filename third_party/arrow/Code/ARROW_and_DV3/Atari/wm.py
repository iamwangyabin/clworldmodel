from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rssm import (
    ActionT,
    ContT,
    HiddenT,
    ImageT,
    LatentShape,
    LatentT,
    ResetT,
    Rssm,
    get_mlp_layers,
)
from vae import Decoder

RewardT = torch.Tensor
RewardSymlogT = torch.Tensor
ObservationObjective = Literal[
    "reconstruction",
    "r2",
    "dinov3_next_feature",
    "dinov3_posterior_feature",
]
# RewardT (real): [ N 1 ]
# RewardSymlogT (symlog(real)): [ N 1 ]


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_p, dim=-1)
    log_q = torch.log_softmax(logits_q, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, 0).sum() / mask.sum()


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1)


def batch_standardized_smooth_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sample-wise feature variation without rewarding a common direction."""
    if predictions.shape != targets.shape:
        raise ValueError("Feature predictions and targets must have equal shapes")
    if predictions.ndim < 2:
        raise ValueError("Feature matching requires sample and feature dimensions")
    if std_floor <= 0:
        raise ValueError("Feature standard-deviation floor must be positive")

    targets = targets.detach().float()
    predictions = predictions.float()
    reduce_dims = tuple(range(targets.ndim - 1))
    target_mean = targets.mean(dim=reduce_dims, keepdim=True)
    prediction_mean = predictions.mean(dim=reduce_dims, keepdim=True)
    target_std = targets.std(dim=reduce_dims, unbiased=False, keepdim=True).clamp_min(
        std_floor
    )
    prediction_std = predictions.std(
        dim=reduce_dims, unbiased=False, keepdim=True
    ).clamp_min(std_floor)
    standardized_targets = (targets - target_mean) / target_std
    standardized_predictions = (predictions - prediction_mean) / prediction_std
    losses = F.smooth_l1_loss(
        standardized_predictions,
        standardized_targets,
        reduction="none",
    ).mean(-1)
    constant_losses = F.smooth_l1_loss(
        torch.zeros_like(standardized_targets),
        standardized_targets,
        reduction="none",
    ).mean(-1)
    return losses, constant_losses


class WorldModel(nn.Module):
    def __init__(
        self,
        img_channels: int,
        ls: LatentShape,
        a_dim: int,
        h_dim: int,
        cnn_depth: int = 32,
        mlp_features: int = 512,
        mlp_layers: int = 2,
        wto: bool = False,
        observation_objective: ObservationObjective = "reconstruction",
        r2_barlow_loss_scale: float = 0.05,
        r2_redundancy_scale: float = 5e-4,
        r2_normalization_eps: float = 1e-8,
        observation_encoder: str = "cnn",
        dinov3_model_path: Optional[str] = None,
        dinov3_input_size: int = 256,
        dinov3_max_batch_size: int = 128,
        dinov3_feature_loss_scale: float = 1.0,
        dinov3_feature_mode: str = "cls",
        dinov3_patch_pool_size: int = 4,
        dinov3_patch_feature_dim: int = 384,
        dinov3_patch_projection: str = "none",
        dinov3_feature_loss_kind: str = "cosine",
        dinov3_feature_std_floor: float = 0.05,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        image_embedder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if observation_objective not in {
            "reconstruction",
            "r2",
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        }:
            raise ValueError(f"Unknown observation objective: {observation_objective!r}")
        if r2_barlow_loss_scale <= 0:
            raise ValueError("R2 Barlow loss scale must be positive")
        if r2_redundancy_scale < 0:
            raise ValueError("R2 redundancy scale must be non-negative")
        if r2_normalization_eps <= 0:
            raise ValueError("R2 normalization epsilon must be positive")
        if dinov3_feature_loss_scale <= 0:
            raise ValueError("DINOv3 feature loss scale must be positive")
        if dinov3_feature_std_floor <= 0:
            raise ValueError("DINOv3 feature standard-deviation floor must be positive")
        if observation_objective == "dinov3_next_feature":
            if dinov3_feature_loss_kind != "cosine":
                raise ValueError("DINOv3 prior-feature prediction requires cosine loss")
        elif observation_objective == "dinov3_posterior_feature":
            if dinov3_feature_loss_kind != "batch_standardized_smooth_l1":
                raise ValueError(
                    "DINOv3 posterior features require batch-standardized SmoothL1"
                )

        self.ls = ls
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.observation_objective = observation_objective
        self.r2_barlow_loss_scale = r2_barlow_loss_scale
        self.r2_redundancy_scale = r2_redundancy_scale
        self.r2_normalization_eps = r2_normalization_eps
        self.dinov3_feature_loss_scale = dinov3_feature_loss_scale
        self.dinov3_feature_loss_kind = dinov3_feature_loss_kind
        self.dinov3_feature_std_floor = dinov3_feature_std_floor
        self.residual_correction = residual_correction

        self.rssm = Rssm(
            img_channels,
            ls,
            a_dim,
            h_dim,
            cnn_depth,
            mlp_features,
            mlp_layers,
            wto,
            observation_encoder=observation_encoder,
            dinov3_model_path=dinov3_model_path,
            dinov3_input_size=dinov3_input_size,
            dinov3_max_batch_size=dinov3_max_batch_size,
            dinov3_feature_mode=dinov3_feature_mode,
            dinov3_patch_pool_size=dinov3_patch_pool_size,
            dinov3_patch_feature_dim=dinov3_patch_feature_dim,
            dinov3_patch_projection=dinov3_patch_projection,
            residual_correction=residual_correction,
            residual_bottleneck_features=residual_bottleneck_features,
            residual_grid_size=residual_grid_size,
            residual_input_min=residual_input_min,
            residual_input_max=residual_input_max,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_alpha=residual_alpha,
            image_embedder=image_embedder,
        )

        # Shared feature consumed by observation, reward, and continuation heads.
        self.zh_transform = ZhToModelState(ls, h_dim)
        self.feature_predictor_residual = None

        if observation_objective == "reconstruction":
            self.decoder = Decoder(img_channels, self.zh_transform.out_features, cnn_depth)
        elif observation_objective == "r2":
            from clworldmodel.models.r2 import R2BarlowObjective, R2Projector

            self.r2_projector = R2Projector(
                self.zh_transform.out_features,
                self.rssm.image_embedder.output_size,
            )
            self.r2_objective = R2BarlowObjective(
                redundancy_scale=r2_redundancy_scale,
                normalization_eps=r2_normalization_eps,
            )
        else:
            self.feature_predictor = nn.Linear(
                self.zh_transform.out_features,
                self.rssm.image_embedder.output_size,
            )
        # NOTE: Weight init here may be 0 init
        self.reward_fc = nn.Sequential(
            *get_mlp_layers(
                self.zh_transform.out_features,
                1,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        self.reward_residual = None
        self.continue_residual = None
        if residual_correction == "none":
            self.continue_fc = nn.Sequential(
                *get_mlp_layers(
                    self.zh_transform.out_features,
                    1,
                    final_activation=nn.Sigmoid,
                    hidden_features=mlp_features,
                    layers=mlp_layers,
                )
            )
        else:
            from clworldmodel.models.residual_corrections import (
                build_residual_correction,
            )

            self.reward_residual = build_residual_correction(
                residual_correction,
                self.zh_transform.out_features,
                1,
                bottleneck_features=residual_bottleneck_features,
                grid_min=residual_input_min,
                grid_max=residual_input_max,
                num_grids=residual_grid_size,
                rms_norm_epsilon=residual_rms_norm_epsilon,
                alpha=residual_alpha,
            )
            self.continue_fc = nn.Sequential(
                *get_mlp_layers(
                    self.zh_transform.out_features,
                    1,
                    final_activation=None,
                    hidden_features=mlp_features,
                    layers=mlp_layers,
                )
            )
            self.continue_residual = build_residual_correction(
                residual_correction,
                self.zh_transform.out_features,
                1,
                bottleneck_features=residual_bottleneck_features,
                grid_min=residual_input_min,
                grid_max=residual_input_max,
                num_grids=residual_grid_size,
                rms_norm_epsilon=residual_rms_norm_epsilon,
                alpha=residual_alpha,
            )
        if (
            observation_objective
            in {"dinov3_next_feature", "dinov3_posterior_feature"}
            and residual_correction != "none"
        ):
            from clworldmodel.models.residual_corrections import build_residual_correction

            self.feature_predictor_residual = build_residual_correction(
                residual_correction,
                self.zh_transform.out_features,
                self.rssm.image_embedder.output_size,
                bottleneck_features=residual_bottleneck_features,
                grid_min=residual_input_min,
                grid_max=residual_input_max,
                num_grids=residual_grid_size,
                rms_norm_epsilon=residual_rms_norm_epsilon,
                alpha=residual_alpha,
            )

    def freeze_shared_core(self) -> None:
        """Freeze base world-model functions while leaving adapters trainable."""
        if self.residual_correction == "none":
            raise ValueError("Frozen shared core requires plastic residual adapters")
        self.rssm.freeze_shared_core()
        self.reward_fc.requires_grad_(False)
        self.continue_fc.requires_grad_(False)
        if hasattr(self, "decoder"):
            self.decoder.requires_grad_(False)
        if hasattr(self, "r2_projector"):
            self.r2_projector.requires_grad_(False)
        if hasattr(self, "feature_predictor"):
            self.feature_predictor.requires_grad_(False)
        if self.feature_predictor_residual is not None:
            self.feature_predictor_residual.requires_grad_(True)
        if self.reward_residual is not None:
            self.reward_residual.requires_grad_(True)
        if self.continue_residual is not None:
            self.continue_residual.requires_grad_(True)

    def compute_loss(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        observation_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Returns (loss, metrics)
        if len(actions.shape) == 2:
            raise ValueError("Time dimension required")
        _, n, _ = actions.shape
        init_z, init_h = self.rssm.initial_state(n)
        # Shift actions and xs, since RSSM takes (prev_action, next_obs)
        embeddings = None
        if self.observation_objective in {
            "r2",
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        }:
            embeddings = (
                self.rssm.embed_observations(xs)
                if observation_features is None
                else observation_features
            )
            if embeddings.shape[:2] != actions.shape[:2]:
                raise ValueError(
                    "Observation features and actions must share time/batch axes"
                )
            if embeddings.shape[-1] != self.rssm.image_embedder.output_size:
                raise ValueError("Observation features have an unexpected width")
            if self.observation_objective in {
                "dinov3_next_feature",
                "dinov3_posterior_feature",
            }:
                embeddings = embeddings.detach()
            z_posts, z_samples, hiddens = self.rssm.observe_embeddings(
                init_z, actions, init_h, embeddings, resets
            )
        else:
            z_posts, z_samples, hiddens = self.rssm(init_z, actions, init_h, xs, resets)
        z_priors = self.rssm.transition(hiddens)

        # Dynamics and representation losses
        dyn_loss_scale = 0.5
        rep_loss_scale = 0.1
        # KL takes shape [ T N n_dis n_cls ]
        # KL divergence results in [ T N n_dis ]
        # See equation (5) on Dreamer v3
        one = torch.tensor(1, device=z_posts.device)
        dyn_loss = (
            categorical_kl(z_posts.detach(), z_priors).sum(-1).maximum(one).mean()
        )
        rep_losses = categorical_kl(z_posts, z_priors.detach()).sum(-1).maximum(one)
        rep_loss = rep_losses.mean()
        z_repr_loss = dyn_loss_scale * dyn_loss + rep_loss_scale * rep_loss

        zhs: torch.Tensor = self.zh_transform(z_samples, hiddens)  # [ T N X ] (X is arbitrary)
        t, n, x = zhs.shape
        zhs_f12 = zhs.view(-1, x)
        if self.observation_objective == "reconstruction":
            recon = self.decoder(zhs_f12).view(t, n, *xs.shape[-3:])
            # Loss shape [ T N C 64 64 ]
            observation_losses = (recon - xs).square().sum([2, 3, 4])
            observation_loss = observation_losses.mean()
            observation_metrics = {
                "Loss/recon": observation_loss,
            }
        elif self.observation_objective == "r2":
            if embeddings is None:
                raise RuntimeError("R2 observation objective requires encoder embeddings")
            projected = self.r2_projector(zhs_f12)
            barlow_loss, invariance_loss, redundancy_loss = self.r2_objective(
                projected,
                embeddings.reshape(-1, embeddings.shape[-1]),
            )
            observation_loss = self.r2_barlow_loss_scale * barlow_loss
            observation_metrics = {
                "Loss/r2_barlow": barlow_loss,
                "Loss/r2_barlow_scaled": observation_loss,
                "Metric/r2_invariance": invariance_loss,
                "Metric/r2_redundancy": redundancy_loss,
            }
        elif self.observation_objective == "dinov3_next_feature":
            if embeddings is None:
                raise RuntimeError("DINOv3 objective requires frozen observation features")
            prior_states = self.zh_transform(z_priors.exp(), hiddens)
            predicted_features = self.feature_predictor(prior_states)
            if self.feature_predictor_residual is not None:
                predicted_features = predicted_features + self.feature_predictor_residual(
                    prior_states
                )
            feature_losses = 1.0 - F.cosine_similarity(
                predicted_features.float(),
                embeddings.detach().float(),
                dim=-1,
                eps=1e-8,
            )
            prediction_mask = resets.squeeze(-1) == 0
            prediction_mask = prediction_mask.clone()
            prediction_mask[0] = False
            valid_predictions = prediction_mask.sum().clamp_min(1)
            feature_loss = torch.where(
                prediction_mask,
                feature_losses,
                torch.zeros_like(feature_losses),
            ).sum() / valid_predictions
            observation_loss = self.dinov3_feature_loss_scale * feature_loss
            observation_metrics = {
                "Loss/dinov3_feature": feature_loss,
                "Loss/dinov3_feature_scaled": observation_loss,
                "Metric/dinov3_feature_valid_fraction": prediction_mask.float().mean(),
            }
        else:
            if embeddings is None:
                raise RuntimeError("DINOv3 objective requires frozen observation features")
            predicted_features = self.feature_predictor(zhs)
            if self.feature_predictor_residual is not None:
                predicted_features = predicted_features + self.feature_predictor_residual(
                    zhs
                )
            feature_losses, constant_losses = batch_standardized_smooth_l1(
                predicted_features,
                embeddings,
                std_floor=self.dinov3_feature_std_floor,
            )
            feature_loss = feature_losses.mean()
            constant_loss = constant_losses.mean()
            observation_loss = self.dinov3_feature_loss_scale * feature_loss
            observation_metrics = {
                "Loss/dinov3_feature": feature_loss,
                "Loss/dinov3_feature_scaled": observation_loss,
                "Metric/dinov3_feature_valid_fraction": torch.ones(
                    (), device=feature_loss.device
                ),
                "Metric/dinov3_constant_feature_loss": constant_loss,
                "Metric/dinov3_model_to_constant_ratio": (
                    feature_loss / constant_loss.clamp_min(1e-8)
                ),
            }

        rews_pred = self.reward_fc(zhs)  # [ T N 1 ]
        if self.reward_residual is not None:
            rews_pred = rews_pred + self.reward_residual(zhs)
        rews_loss = (rews_pred - symlog(rews)).square().mean()

        conts_pred = self.continue_fc(zhs)  # [ T N 1 ]
        if self.continue_residual is not None:
            conts_pred = torch.sigmoid(conts_pred + self.continue_residual(zhs))
        conts_loss = torch.nn.functional.binary_cross_entropy(conts_pred, conts, reduction="mean")

        with torch.no_grad():
            low_kl = rep_losses < 1 + 1e-3
            metrics = {
                "Loss/kl": z_repr_loss,
                **observation_metrics,
                "Loss/rew": rews_loss,
                "Loss/cont": conts_loss,
                "Metric/neg_cont_mean": masked_mean(conts_pred, conts == 0),
                "Metric/low_kl": low_kl.float().mean(),
            }
            if self.observation_objective == "reconstruction":
                metrics["Metric/low_kl_recon_loss"] = masked_mean(
                    observation_losses, low_kl
                )

        return z_repr_loss + observation_loss + rews_loss + conts_loss, metrics


class ZhToModelState(nn.Module):
    def __init__(self, ls: LatentShape, h_dim: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        if out_features is None:
            # No linear projection, only concatenation
            self.out_features = ls[0] * ls[1] + h_dim
            self.linear = None
        else:
            self.out_features = out_features
            self.linear = nn.Linear(ls[0] * ls[1] + h_dim, out_features)

    def forward(self, z: LatentT, h: HiddenT) -> torch.Tensor:
        z = z.flatten(-2)
        zh = torch.cat([z, h], dim=-1)
        if self.linear:
            return self.linear(zh)
        return zh
