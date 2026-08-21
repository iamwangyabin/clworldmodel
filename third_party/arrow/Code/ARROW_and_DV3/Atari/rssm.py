import copy
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
        dinov3_patch_projection_seed: int = 0,
        num_task_experts: int = 1,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_input_mode: str = "base_output",
        residual_consolidation: str = "none",
        image_embedder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_task_experts < 1:
            raise ValueError("num_task_experts must be positive")
        if num_task_experts > 1 and residual_correction != "none":
            raise ValueError(
                "Task-routed RSSM experts do not compose with residual corrections"
            )
        self.ls = ls
        self.h_dim = h_dim
        self.num_task_experts = num_task_experts

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
            residual_input_mode=residual_input_mode,
            residual_consolidation=residual_consolidation,
        )
        self.recurrent_experts = nn.ModuleList(
            copy.deepcopy(self.recurrent) for _ in range(num_task_experts - 1)
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
                patch_projection_seed=dinov3_patch_projection_seed,
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
            residual_input_mode=residual_input_mode,
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
            residual_input_mode=residual_input_mode,
            residual_consolidation=residual_consolidation,
        )
        self.transition_experts = nn.ModuleList(
            copy.deepcopy(self.transition) for _ in range(num_task_experts - 1)
        )

    def freeze_shared_core(self) -> None:
        """Freeze base RSSM functions while leaving residual adapters plastic."""
        self.image_embedder.requires_grad_(False)
        self.recurrent.freeze_shared_core()
        for recurrent in self.recurrent_experts:
            recurrent.freeze_shared_core()
        self.representation.freeze_shared_core()
        self.transition.freeze_shared_core()
        for transition in self.transition_experts:
            transition.freeze_shared_core()

    def _task_index(self, task_id: Optional[int | torch.Tensor]) -> int:
        if task_id is None:
            if self.num_task_experts > 1:
                raise ValueError("task_id is required by a task-routed RSSM")
            return 0
        if isinstance(task_id, torch.Tensor):
            if task_id.numel() != 1:
                raise ValueError(
                    "task_id must be scalar; MoE-ARROW minibatches are task-homogeneous"
                )
            task_id = int(task_id.detach().item())
        if not isinstance(task_id, int):
            raise TypeError("task_id must be an integer or scalar tensor")
        if task_id < 0 or task_id >= self.num_task_experts:
            raise ValueError(
                f"task_id {task_id} is outside [0, {self.num_task_experts})"
            )
        return task_id

    def recurrent_for(self, task_id: Optional[int | torch.Tensor]) -> "Recurrent":
        task_index = self._task_index(task_id)
        return self.recurrent if task_index == 0 else self.recurrent_experts[task_index - 1]

    def transition_for(self, task_id: Optional[int | torch.Tensor]) -> "Transition":
        task_index = self._task_index(task_id)
        return self.transition if task_index == 0 else self.transition_experts[task_index - 1]

    def prior(
        self, hidden: HiddenT, task_id: Optional[int | torch.Tensor] = None
    ) -> LatentLogDistT:
        return self.transition_for(task_id)(hidden)

    def copy_task_expert(self, target_task_id: int, source_task_id: int) -> None:
        if target_task_id == source_task_id:
            return
        self.recurrent_for(target_task_id).load_state_dict(
            self.recurrent_for(source_task_id).state_dict()
        )
        self.transition_for(target_task_id).load_state_dict(
            self.transition_for(source_task_id).state_dict()
        )

    def __call__(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        x: Optional[ImageT],
        reset: ResetT,
        stochastic: bool = True,
        temperature: float = 1.0,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        return super().__call__(
            prev_z,
            prev_a,
            prev_h,
            x,
            reset,
            stochastic=stochastic,
            temperature=temperature,
            task_id=task_id,
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
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        if len(prev_a.shape) == 2:
            # Apply reset flags
            prev_z = prev_z * (1 - reset).unsqueeze(-1)  # Need to multiply againt dim [ N 1 1 ]
            prev_h = prev_h * (1 - reset)
            # No time dimension
            h = self.recurrent_for(task_id)(prev_z, prev_a, prev_h)
            if x is not None:
                e = self.image_embedder(x)
                z_log_dist = self.representation(e, h)
            else:
                z_log_dist = self.prior(h, task_id)
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
                z_log_dist, z_sample, h = self(
                    z_sample,
                    a,
                    h,
                    None,
                    r,
                    stochastic=stochastic,
                    temperature=temperature,
                    task_id=task_id,
                )
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
                task_id=task_id,
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
        task_id: Optional[int | torch.Tensor] = None,
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
            h = self.recurrent_for(task_id)(
                z * (1 - r).unsqueeze(-1), a, h * (1 - r)
            )
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
        residual_input_mode: str = "base_output",
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        if residual_input_mode not in {"base_output", "module_input"}:
            raise ValueError(f"Unknown residual input mode: {residual_input_mode!r}")
        z_dim = ls[0] * ls[1]
        self.residual_input_mode = residual_input_mode
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
                    (
                        h_dim
                        if residual_input_mode == "base_output"
                        else z_dim + a_dim + h_dim
                    ),
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
            residual_input = (
                hidden
                if self.residual_input_mode == "base_output"
                else torch.cat((za, prev_h), dim=-1)
            )
            hidden = hidden + self.residual(residual_input)
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
        residual_input_mode: str = "base_output",
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        if residual_input_mode not in {"base_output", "module_input"}:
            raise ValueError(f"Unknown residual input mode: {residual_input_mode!r}")
        self.ls = ls
        self.uniform = uniform
        self.residual_input_mode = residual_input_mode
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
                    (
                        n_dis * n_cls
                        if residual_input_mode == "base_output"
                        else embed_dim + h_dim
                    ),
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
            residual_input = x1 if self.residual_input_mode == "base_output" else eh
            x1 = x1 + self.residual(residual_input)
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
        residual_input_mode: str = "base_output",
        residual_consolidation: str = "none",
    ) -> None:
        super().__init__()
        if residual_input_mode not in {"base_output", "module_input"}:
            raise ValueError(f"Unknown residual input mode: {residual_input_mode!r}")
        self.ls = ls
        self.uniform = uniform
        self.residual_input_mode = residual_input_mode

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
                    n_logits if residual_input_mode == "base_output" else h_dim,
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
            residual_input = (
                prior_logits if self.residual_input_mode == "base_output" else h
            )
            prior_logits = prior_logits + self.residual(residual_input)
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
