from typing import Optional, Type

import torch
import torch.nn as nn

from vae import Encoder

LatentShape = tuple[int, int]
EmbedT = torch.Tensor
LatentLogDistT = torch.Tensor
LatentT = torch.Tensor
ActionT = torch.Tensor
HiddenT = torch.Tensor
ImageT = torch.Tensor
ContT = torch.Tensor
ResetT = torch.Tensor

# EmbedT: [ N E1 ]
# LatentLogDistT (logits): [ N n_dis n_cls ] where z_dim = n_dis * n_cls
# LatentT (onehot): [ N n_dis n_cls ]
# ActionT (onehot): [ N n_acts ]
# HiddenT: [ N h_dim ]
# ImageT (0 to 1): [ N C 64 64 ]
# ImageLogT (log(0 to 1)): [ N C 64 64 ]
# ContT (0 to 1): [ N 1 ]
# ResetT (0 or 1, 1-ContT shifted right in T by 1): [ N 1 ]
# Optional [ T ... ] dimension in front where applicable


def straight_through_one_hot(
    logits: torch.Tensor, stochastic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    if stochastic:
        flat_indices = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, replacement=True
        ).squeeze(-1)
        sample = torch.nn.functional.one_hot(
            flat_indices, probs.shape[-1]
        ).reshape_as(probs)
    else:
        sample = torch.nn.functional.one_hot(probs.argmax(-1), probs.shape[-1])
    sample = sample.to(probs.dtype) + probs - probs.detach()
    return log_probs, sample


class LayerNormSiLU(nn.Module):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.fw = nn.Sequential(nn.LayerNorm(units, 1e-3), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fw(x)


def get_mlp_layers(
    in_features: int,
    out_features: int,
    activation: Type[nn.Module] = LayerNormSiLU,
    final_activation: Optional[Type[nn.Module]] = LayerNormSiLU,
    hidden_features: int = 512,
    layers: int = 2,
) -> list[nn.Module]:
    get_act = lambda act, units: act() if act is not LayerNormSiLU else act(units)
    sizes = [in_features] + [hidden_features] * (layers - 1) + [out_features] * (layers > 0)
    res = []
    for i, (ft1, ft2) in enumerate(zip(sizes[:-1], sizes[1:])):
        res.append(nn.Linear(ft1, ft2))
        if i < len(sizes) - 2:
            res.append(get_act(activation, ft2))
        elif final_activation is not None:
            res.append(get_act(final_activation, ft2))
    return res


class Rssm(nn.Module):
    def __init__(
        self,
        img_channels: int,
        ls: LatentShape,
        a_dim: int,
        h_dim: int,
        cnn_depth: int,
        mlp_features: int,
        mlp_layers: int,
        wto: bool = False,
        observation_encoder: str = "cnn",
        dinov3_model_path: Optional[str] = None,
        dinov3_input_size: int = 256,
        dinov3_max_batch_size: int = 128,
        dinov3_feature_mode: str = "cls",
        dinov3_patch_pool_size: int = 4,
        dinov3_patch_feature_dim: int = 384,
        dinov3_patch_projection: str = "none",
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_consolidation: str = "none",
        image_embedder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.ls = ls
        self.h_dim = h_dim

        self.recurrent = Recurrent(
            ls,
            a_dim,
            h_dim,
            mlp_features,
            mlp_layers if not wto else 0,
            residual_correction=residual_correction,
            residual_bottleneck_features=residual_bottleneck_features,
            residual_grid_size=residual_grid_size,
            residual_input_min=residual_input_min,
            residual_input_max=residual_input_max,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_alpha=residual_alpha,
            residual_consolidation=residual_consolidation,
        )
        if image_embedder is not None:
            self.image_embedder = image_embedder
        elif observation_encoder == "cnn":
            self.image_embedder = Encoder(img_channels, cnn_depth)
        elif observation_encoder == "dinov3_vits16":
            from clworldmodel.models.frozen_dinov3 import FrozenDinoV3Encoder

            self.image_embedder = FrozenDinoV3Encoder(
                dinov3_model_path,
                input_size=dinov3_input_size,
                max_batch_size=dinov3_max_batch_size,
                feature_mode=dinov3_feature_mode,
                patch_pool_size=dinov3_patch_pool_size,
                patch_feature_dim=dinov3_patch_feature_dim,
                patch_projection=dinov3_patch_projection,
            )
        else:
            raise ValueError(f"Unknown observation encoder: {observation_encoder!r}")
        if not hasattr(self.image_embedder, "output_size"):
            raise TypeError("Image embedder must declare output_size")
        self.representation = Representation(
            ls,
            self.image_embedder.output_size,
            h_dim,
            mlp_features,
            mlp_layers if not wto else 1,
            uniform=0.01,
            residual_correction=residual_correction,
            residual_bottleneck_features=residual_bottleneck_features,
            residual_grid_size=residual_grid_size,
            residual_input_min=residual_input_min,
            residual_input_max=residual_input_max,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_alpha=residual_alpha,
            residual_consolidation=residual_consolidation,
        )
        self.transition = Transition(
            ls,
            h_dim,
            mlp_features,
            mlp_layers,
            uniform=0.01,
            residual_correction=residual_correction,
            residual_bottleneck_features=residual_bottleneck_features,
            residual_grid_size=residual_grid_size,
            residual_input_min=residual_input_min,
            residual_input_max=residual_input_max,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_alpha=residual_alpha,
            residual_consolidation=residual_consolidation,
        )

    def freeze_shared_core(self) -> None:
        """Freeze base RSSM functions while leaving residual adapters plastic."""
        self.image_embedder.requires_grad_(False)
        self.recurrent.freeze_shared_core()
        self.representation.freeze_shared_core()
        self.transition.freeze_shared_core()

    def __call__(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        x: Optional[ImageT],
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        return super().__call__(
            prev_z, prev_a, prev_h, x, reset, stochastic=stochastic, temperature=temperature
        )

    def forward(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        x: Optional[ImageT],
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        if len(prev_a.shape) == 2:
            # Apply reset flags
            prev_z = prev_z * (1 - reset).unsqueeze(-1)  # Need to multiply againt dim [ N 1 1 ]
            prev_h = prev_h * (1 - reset)
            # No time dimension
            h = self.recurrent(prev_z, prev_a, prev_h)
            if x is not None:
                e = self.image_embedder(x)
                z_log_dist = self.representation(e, h)
            else:
                z_log_dist = self.transition(h)
            z_logits, z_sample = straight_through_one_hot(
                z_log_dist / temperature, stochastic
            )
            return z_logits, z_sample, h
        elif len(prev_a.shape) == 3 and x is None:
            z_sample, h = prev_z, prev_h
            z_log_dists = []
            z_samples = []
            hs = []
            for a, r in zip(prev_a, reset):
                z_log_dist, z_sample, h = self(z_sample, a, h, None, r)
                z_log_dists.append(z_log_dist)
                z_samples.append(z_sample)
                hs.append(h)
            return torch.stack(z_log_dists), torch.stack(z_samples), torch.stack(hs)
        elif len(prev_a.shape) == 3:
            # Special batched impl
            embeddings = self.embed_observations(x)
            return self.observe_embeddings(
                prev_z,
                prev_a,
                prev_h,
                embeddings,
                reset,
                stochastic=stochastic,
                temperature=temperature,
            )
        raise ValueError

    def embed_observations(self, x: ImageT) -> EmbedT:
        """Encode a [T, N, C, H, W] observation sequence exactly once."""
        if len(x.shape) != 5:
            raise ValueError(
                "Observation sequences must have shape [T, N, C, H, W], "
                f"got {tuple(x.shape)}"
            )
        t, n = x.shape[:2]
        return self.image_embedder(x.reshape(-1, *x.shape[-3:])).view(t, n, -1)

    def observe_embeddings(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        embeddings: EmbedT,
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        """Run the posterior recurrence from precomputed [T, N, E] embeddings."""
        if len(prev_a.shape) != 3 or len(embeddings.shape) != 3:
            raise ValueError("Actions and embeddings must include time and batch dimensions")
        if prev_a.shape[:2] != embeddings.shape[:2] or reset.shape[:2] != prev_a.shape[:2]:
            raise ValueError("Actions, embeddings, and resets must share [T, N] dimensions")

        hs = []
        z_log_dists = []
        z_samples = []
        z, h = prev_z, prev_h
        for e, a, r in zip(embeddings, prev_a, reset):
            h = self.recurrent(z * (1 - r).unsqueeze(-1), a, h * (1 - r))
            z_log_dist = self.representation(e, h)
            _, z_sample = straight_through_one_hot(
                z_log_dist / temperature, stochastic
            )
            z = z_sample
            hs.append(h)
            z_log_dists.append(z_log_dist)
            z_samples.append(z_sample)
        return torch.stack(z_log_dists), torch.stack(z_samples), torch.stack(hs)

    def initial_state(self, n: int = 1) -> tuple[LatentT, HiddenT]:
        device = next(self.parameters()).device
        return (
            torch.zeros(n, *self.ls, device=device),
            torch.zeros(n, self.h_dim, device=device),
        )


class Recurrent(nn.Module):
    def __init__(
        self,
        ls: LatentShape,
        a_dim: int,
        h_dim: int,
        mlp_features: int,
        mlp_layers: int,
        *,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        z_dim = ls[0] * ls[1]
        self.za_fcs = nn.Sequential(
            *get_mlp_layers(
                z_dim + a_dim,
                mlp_features,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        self.rnn = nn.GRUCell(mlp_features if mlp_layers > 0 else z_dim + a_dim, h_dim)
        if residual_correction == "none":
            self.residual = None
        else:
            from clworldmodel.models.residual_corrections import (
                build_residual_correction,
            )

            # Keep every downstream ARROW parameter identical under a shared seed.
            with torch.random.fork_rng(devices=[]):
                self.residual = build_residual_correction(
                    residual_correction,
                    h_dim,
                    h_dim,
                    bottleneck_features=residual_bottleneck_features,
                    grid_min=residual_input_min,
                    grid_max=residual_input_max,
                    num_grids=residual_grid_size,
                    rms_norm_epsilon=residual_rms_norm_epsilon,
                    alpha=residual_alpha,
                    consolidation_enabled=residual_consolidation != "none",
                )

    def forward(self, prev_z: LatentT, prev_a: ActionT, prev_h: HiddenT) -> HiddenT:
        assert len(prev_z.shape) == 3  # [ N 32 32 ]
        assert len(prev_a.shape) == 2  # [ N n_acts ]
        za = torch.cat((prev_z.flatten(1), prev_a), dim=1)
        hidden = self.rnn(self.za_fcs(za), prev_h)
        if self.residual is not None:
            hidden = hidden + self.residual(hidden)
        return hidden

    def freeze_shared_core(self) -> None:
        self.za_fcs.requires_grad_(False)
        self.rnn.requires_grad_(False)
        if self.residual is not None:
            self.residual.requires_grad_(True)


class Representation(nn.Module):
    # Stock DV3 does not have skip connection in this part, but it accelerates
    # early learning

    def __init__(
        self,
        ls: LatentShape,
        embed_dim: int,
        h_dim: int,
        mlp_features: int,
        mlp_layers: int,
        uniform: float = 0,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        self.ls = ls
        self.uniform = uniform
        n_dis, n_cls = ls
        self.eh_to_inter = nn.Sequential(
            *get_mlp_layers(
                embed_dim + h_dim,
                n_dis * n_cls,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )
        if mlp_layers > 1:
            self.e_to_inter = nn.Linear(embed_dim, n_dis * n_cls)
        else:
            self.e_to_inter = None
        self.inter_to_z_dist = nn.Sequential(
            # [ N n_dis*n_cls ] -> [ N n_dis n_cls ]
            nn.Unflatten(-1, (n_dis, n_cls)),
            nn.LogSoftmax(-1),
        )
        if residual_correction == "none":
            self.residual = None
        else:
            from clworldmodel.models.residual_corrections import (
                build_residual_correction,
            )

            with torch.random.fork_rng(devices=[]):
                self.residual = build_residual_correction(
                    residual_correction,
                    n_dis * n_cls,
                    n_dis * n_cls,
                    bottleneck_features=residual_bottleneck_features,
                    grid_min=residual_input_min,
                    grid_max=residual_input_max,
                    num_grids=residual_grid_size,
                    rms_norm_epsilon=residual_rms_norm_epsilon,
                    alpha=residual_alpha,
                    consolidation_enabled=residual_consolidation != "none",
                )

    def __call__(self, e: EmbedT, h: HiddenT) -> LatentLogDistT:
        return super().__call__(e, h)

    def forward(self, e: EmbedT, h: HiddenT) -> LatentLogDistT:
        assert len(e.shape) == len(h.shape) == 2
        eh = torch.cat((e, h), dim=-1)
        x1 = self.eh_to_inter(eh)
        if self.e_to_inter is not None:
            x1 = x1 + self.e_to_inter(e)
        if self.residual is not None:
            x1 = x1 + self.residual(x1)
        post_log_probs = self.inter_to_z_dist(x1)
        if self.uniform:
            # Use (1-u) of probs from logits + (u) of uniform
            # Conditions the KL loss better this way
            probs = post_log_probs.exp()
            return ((1 - self.uniform) * probs + self.uniform / self.ls[1]).log()
        return post_log_probs

    def freeze_shared_core(self) -> None:
        self.eh_to_inter.requires_grad_(False)
        if self.e_to_inter is not None:
            self.e_to_inter.requires_grad_(False)
        self.inter_to_z_dist.requires_grad_(False)
        if self.residual is not None:
            self.residual.requires_grad_(True)


class Transition(nn.Module):
    def __init__(
        self,
        ls: LatentShape,
        h_dim: int,
        mlp_features: int,
        mlp_layers: int,
        uniform: float = 0,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        self.ls = ls
        self.uniform = uniform

        n_dis, n_cls = ls
        self.h_to_z_prior = nn.Sequential(
            *get_mlp_layers(
                h_dim,
                n_dis * n_cls,
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            ),
            nn.Unflatten(-1, (n_dis, n_cls)),
            nn.LogSoftmax(-1),
        )
        if residual_correction == "none":
            self.residual = None
        else:
            from clworldmodel.models.residual_corrections import (
                build_residual_correction,
            )

            n_logits = n_dis * n_cls
            with torch.random.fork_rng(devices=[]):
                self.residual = build_residual_correction(
                    residual_correction,
                    n_logits,
                    n_logits,
                    bottleneck_features=residual_bottleneck_features,
                    grid_min=residual_input_min,
                    grid_max=residual_input_max,
                    num_grids=residual_grid_size,
                    rms_norm_epsilon=residual_rms_norm_epsilon,
                    alpha=residual_alpha,
                    consolidation_enabled=residual_consolidation != "none",
                )

    def forward(self, h: HiddenT) -> LatentLogDistT:
        prior_log_probs = self.h_to_z_prior(h)
        if self.residual is not None:
            prior_logits = prior_log_probs.flatten(-2)
            prior_logits = prior_logits + self.residual(prior_logits)
            prior_log_probs = torch.log_softmax(
                prior_logits.unflatten(-1, self.ls), dim=-1
            )
        if self.uniform:
            probs = prior_log_probs.exp()
            return ((1 - self.uniform) * probs + self.uniform / self.ls[1]).log()
        return prior_log_probs

    def freeze_shared_core(self) -> None:
        self.h_to_z_prior.requires_grad_(False)
        if self.residual is not None:
            self.residual.requires_grad_(True)
