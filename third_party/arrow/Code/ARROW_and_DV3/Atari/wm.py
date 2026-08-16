from typing import Literal, Optional

import torch
import torch.nn as nn

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
ObservationObjective = Literal["reconstruction", "r2"]
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
    ) -> None:
        super().__init__()
        if observation_objective not in {"reconstruction", "r2"}:
            raise ValueError(f"Unknown observation objective: {observation_objective!r}")
        if r2_barlow_loss_scale <= 0:
            raise ValueError("R2 Barlow loss scale must be positive")
        if r2_redundancy_scale < 0:
            raise ValueError("R2 redundancy scale must be non-negative")
        if r2_normalization_eps <= 0:
            raise ValueError("R2 normalization epsilon must be positive")

        self.ls = ls
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.observation_objective = observation_objective
        self.r2_barlow_loss_scale = r2_barlow_loss_scale
        self.r2_redundancy_scale = r2_redundancy_scale
        self.r2_normalization_eps = r2_normalization_eps

        self.rssm = Rssm(img_channels, ls, a_dim, h_dim, cnn_depth, mlp_features, mlp_layers, wto)

        # Shared feature consumed by observation, reward, and continuation heads.
        self.zh_transform = ZhToModelState(ls, h_dim)

        if observation_objective == "reconstruction":
            self.decoder = Decoder(img_channels, self.zh_transform.out_features, cnn_depth)
        else:
            from clworldmodel.models.r2 import R2BarlowObjective, R2Projector

            self.r2_projector = R2Projector(
                self.zh_transform.out_features,
                self.rssm.image_embedder.output_size,
            )
            self.r2_objective = R2BarlowObjective(
                redundancy_scale=r2_redundancy_scale,
                normalization_eps=r2_normalization_eps,
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
        self.continue_fc = nn.Sequential(
            *get_mlp_layers(
                self.zh_transform.out_features,
                1,
                final_activation=nn.Sigmoid,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )

    def compute_loss(
        self, actions: ActionT, xs: ImageT, rews: RewardT, conts: ContT, resets: ResetT
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Returns (loss, metrics)
        if len(actions.shape) == 2:
            raise ValueError("Time dimension required")
        _, n, _ = actions.shape
        init_z, init_h = self.rssm.initial_state(n)
        # Shift actions and xs, since RSSM takes (prev_action, next_obs)
        embeddings = None
        if self.observation_objective == "r2":
            embeddings = self.rssm.embed_observations(xs)
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
        else:
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

        rews_pred = self.reward_fc(zhs)  # [ T N 1 ]
        rews_loss = (rews_pred - symlog(rews)).square().mean()

        conts_pred = self.continue_fc(zhs)  # [ T N 1 ]
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
