import copy
from contextlib import nullcontext
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
ObservationObjective = Literal["reconstruction"]
# RewardT (real): [ N 1 ]
# RewardSymlogT (symlog(real)): [ N 1 ]


def _full_precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    with _full_precision_context(logits_p.device):
        logits_p = logits_p.float()
        logits_q = logits_q.float()
        log_p = torch.log_softmax(logits_p, dim=-1)
        log_q = torch.log_softmax(logits_q, dim=-1)
        return (log_p.exp() * (log_p - log_q)).sum(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, 0).sum() / mask.sum()


def symlog(x: torch.Tensor) -> torch.Tensor:
    with _full_precision_context(x.device):
        x = x.float()
        return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    with _full_precision_context(x.device):
        x = x.float()
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
        observation_encoder: str = "cnn",
        num_task_experts: int = 1,
        task_shared_prediction_heads: bool = False,
        evolving_shared_core: bool = False,
        task_projected_image_encoder: bool = False,
        task_symmetric_image_projectors: bool = False,
        task_projector_bottleneck_features: int = 64,
        task_mechanism_bank: bool = False,
        task_mechanism_reuse: bool = True,
        task_mechanism_recurrent_width: int = 512,
        task_mechanism_representation_width: int = 512,
        task_mechanism_transition_width: int = 256,
        task_mechanism_residual_scale: float = 0.1,
        task_mechanism_num_atoms: int = 1,
        task_mechanism_parameterization: str = "dense_private",
        task_symmetric_mechanisms: bool = False,
        image_embedder: Optional[nn.Module] = None,
        compute_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if observation_objective != "reconstruction":
            raise ValueError("Only the pixel reconstruction objective is supported")
        if observation_encoder != "cnn":
            raise ValueError("Only the CNN observation encoder is supported")
        if not isinstance(task_shared_prediction_heads, bool):
            raise TypeError("task_shared_prediction_heads must be a boolean")
        if not isinstance(evolving_shared_core, bool):
            raise TypeError("evolving_shared_core must be a boolean")
        if task_shared_prediction_heads and not evolving_shared_core:
            raise ValueError(
                "Task-shared prediction heads require an evolving shared core"
            )
        if compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unknown compute dtype: {compute_dtype!r}")

        self.ls = ls
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.compute_dtype = compute_dtype
        self.observation_objective = 'reconstruction'
        self.task_shared_prediction_heads = task_shared_prediction_heads
        self.evolving_shared_core = evolving_shared_core

        self.rssm = Rssm(
            img_channels,
            ls,
            a_dim,
            h_dim,
            cnn_depth,
            mlp_features,
            mlp_layers,
            wto,
            compute_dtype=compute_dtype,
            observation_encoder='cnn',
            num_task_experts=num_task_experts,
            task_projected_image_encoder=task_projected_image_encoder,
            task_symmetric_image_projectors=task_symmetric_image_projectors,
            task_projector_bottleneck_features=task_projector_bottleneck_features,
            task_mechanism_bank=task_mechanism_bank,
            task_mechanism_reuse=task_mechanism_reuse,
            task_mechanism_recurrent_width=task_mechanism_recurrent_width,
            task_mechanism_representation_width=(
                task_mechanism_representation_width
            ),
            task_mechanism_transition_width=task_mechanism_transition_width,
            task_mechanism_residual_scale=task_mechanism_residual_scale,
            task_mechanism_num_atoms=task_mechanism_num_atoms,
            task_mechanism_parameterization=task_mechanism_parameterization,
            task_symmetric_mechanisms=task_symmetric_mechanisms,
            image_embedder=image_embedder,
        )

        # Shared feature consumed by observation, reward, and continuation heads.
        self.zh_transform = ZhToModelState(ls, h_dim)
        self.feature_predictor_experts = nn.ModuleList()
        self.decoder_experts = nn.ModuleList()
        self.decoder = Decoder(img_channels, self.zh_transform.out_features, cnn_depth)
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
                final_activation=None,
                hidden_features=mlp_features,
                layers=mlp_layers,
            )
        )

        return_head_expert_count = (
            0 if task_shared_prediction_heads else num_task_experts - 1
        )
        self.reward_experts = nn.ModuleList(
            copy.deepcopy(self.reward_fc) for _ in range(return_head_expert_count)
        )
        self.continue_experts = nn.ModuleList(
            copy.deepcopy(self.continue_fc) for _ in range(return_head_expert_count)
        )
        self.prediction_adapters = nn.ModuleDict()
        initialized = None
        if num_task_experts > 1:
            initialized = torch.zeros(num_task_experts, dtype=torch.bool)
            initialized[0] = True
        self.register_buffer("task_expert_initialized", initialized)


    def prediction_features_for(
        self,
        head_name: str,
        model_state: torch.Tensor,
        task_id: Optional[int | torch.Tensor],
    ) -> torch.Tensor:
        self.rssm._task_index(task_id)
        return model_state

    def _head_for(
        self,
        base: nn.Module,
        experts: nn.ModuleList,
        task_id: Optional[int | torch.Tensor],
    ) -> nn.Module:
        task_index = self.rssm._task_index(task_id)
        if self.task_shared_prediction_heads:
            return base
        return base if task_index == 0 else experts[task_index - 1]


    def decoder_for(self, task_id: Optional[int | torch.Tensor]) -> nn.Module:
        if not hasattr(self, "decoder"):
            raise RuntimeError("The configured observation objective has no decoder")
        task_index = self.rssm._task_index(task_id)
        return self.decoder

    def initialize_task_expert(
        self, target_task_id: int, source_task_id: int
    ) -> bool:
        """Warm-start a new expert once while keeping later task states isolated."""
        target_index = self.rssm._task_index(target_task_id)
        source_index = self.rssm._task_index(source_task_id)
        if self.task_expert_initialized is None:
            return False
        if bool(self.task_expert_initialized[target_index].item()):
            return False
        if not bool(self.task_expert_initialized[source_index].item()):
            raise ValueError(
                f"Cannot warm-start task {target_task_id} from uninitialized task "
                f"{source_task_id}"
            )
        self.rssm.copy_task_expert(target_index, source_index)
        if not self.task_shared_prediction_heads:
            self._head_for(
                self.reward_fc, self.reward_experts, target_index
            ).load_state_dict(
                self._head_for(
                    self.reward_fc, self.reward_experts, source_index
                ).state_dict()
            )
            self._head_for(
                self.continue_fc, self.continue_experts, target_index
            ).load_state_dict(
                self._head_for(
                    self.continue_fc, self.continue_experts, source_index
                ).state_dict()
            )
        self.task_expert_initialized[target_index] = True
        return True

    def activate_task_expert(
        self, task_id: int, mechanism_phase: str = "full"
    ) -> None:
        """Make exactly one complete task expert plastic and freeze all others."""
        if not (self.rssm.task_mechanism_bank_enabled):
            raise ValueError("Complete task activation requires task-routed modules")
        task_index = self.rssm._task_index(task_id)
        if self.task_expert_initialized is None or not bool(
            self.task_expert_initialized[task_index].item()
        ):
            raise ValueError(f"Task expert {task_index} has not been initialized")

        if self.rssm.task_projected_image_encoder:
            self.rssm.image_embedder.requires_grad_(
                self.evolving_shared_core or task_index == 0
            )
            projector_start = (
                0 if self.rssm.task_symmetric_image_projectors else 1
            )
            for index, projector in enumerate(
                self.rssm.image_projectors, start=projector_start
            ):
                projector.requires_grad_(index == task_index)
        else:
            for index in range(self.rssm.num_task_experts):
                self.rssm.image_embedder_for(index).requires_grad_(
                    False
                )
        self.rssm.observation_adapter.requires_grad_(
            False
        )
        if self.rssm.task_mechanism_bank_enabled:
            base_is_active = self.evolving_shared_core or task_index == 0
            self.rssm.recurrent.requires_grad_(base_is_active)
            self.rssm.representation.requires_grad_(base_is_active)
            self.rssm.transition.requires_grad_(base_is_active)
            self.zh_transform.requires_grad_(base_is_active)
            self.rssm.recurrent_mechanism_bank.activate_task(
                task_index, phase=mechanism_phase
            )
            self.rssm.representation_mechanism_bank.activate_task(
                task_index, phase=mechanism_phase
            )
            self.rssm.transition_mechanism_bank.activate_task(
                task_index, phase=mechanism_phase
            )
        else:
            for index in range(self.rssm.num_task_experts):
                is_active = index == task_index
                self.rssm.recurrent_for(index).requires_grad_(is_active)
                self.rssm.representation_for(index).requires_grad_(is_active)
                self.rssm.transition_for(index).requires_grad_(is_active)
        if self.task_shared_prediction_heads:
            base_head_is_active = True
            self.reward_fc.requires_grad_(base_head_is_active)
            self.continue_fc.requires_grad_(base_head_is_active)
            self.decoder.requires_grad_(base_head_is_active)
        else:
            for index in range(self.rssm.num_task_experts):
                is_active = index == task_index
                self._head_for(
                    self.reward_fc, self.reward_experts, index
                ).requires_grad_(is_active)
                self._head_for(
                    self.continue_fc, self.continue_experts, index
                ).requires_grad_(is_active)
                self.decoder_for(index).requires_grad_(is_active)
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None

    @staticmethod
    def _deduplicate_parameters(
        parameters: list[nn.Parameter],
    ) -> list[nn.Parameter]:
        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
        return unique

    def shared_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return shared groups used by component gradient projection."""

        groups = self.rssm.shared_parameter_groups()
        groups["latent_interface"] = list(self.zh_transform.parameters())
        if (self.task_shared_prediction_heads):
            observation_head = (self.decoder)
            groups["observation_head"] = list(observation_head.parameters())
            groups["reward_head"] = list(self.reward_fc.parameters())
            groups["continue_head"] = list(self.continue_fc.parameters())
        all_ids = [id(parameter) for values in groups.values() for parameter in values]
        if len(all_ids) != len(set(all_ids)):
            raise RuntimeError("Shared world-model parameter groups overlap")
        return groups

    def private_parameters(self, task_id: int) -> list[nn.Parameter]:
        """Return one task's projector/atoms and any task-private heads."""

        task_index = self.rssm._task_index(task_id)
        parameters = list(self.rssm.private_parameters(task_index))
        if not self.task_shared_prediction_heads:
            parameters.extend(
                self._head_for(
                    self.reward_fc, self.reward_experts, task_index
                ).parameters()
            )
            parameters.extend(
                self._head_for(
                    self.continue_fc, self.continue_experts, task_index
                ).parameters()
            )
        return self._deduplicate_parameters(parameters)

    def route_parameters(self, task_id: int) -> list[nn.Parameter]:
        return self.rssm.route_parameters(task_id)

    def activate_shared_only(self) -> None:
        """Freeze all task-private state and expose only the evolving core."""

        self.requires_grad_(False)
        for parameters in self.shared_parameter_groups().values():
            for parameter in parameters:
                parameter.requires_grad_(True)
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None

    def shared_core_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Capture every consolidation-owned module for safe rollback."""

        state = {
            "encoder": copy.deepcopy(self.rssm.image_embedder.state_dict()),
            "observation_adapter": copy.deepcopy(
                self.rssm.observation_adapter.state_dict()
            ),
            "posterior": copy.deepcopy(self.rssm.representation.state_dict()),
            "recurrent": copy.deepcopy(self.rssm.recurrent.state_dict()),
            "prior": copy.deepcopy(self.rssm.transition.state_dict()),
            "latent_interface": copy.deepcopy(self.zh_transform.state_dict()),
        }
        if (self.task_shared_prediction_heads):
            observation_head = (self.decoder)
            state.update(
                {
                    "observation_head": copy.deepcopy(
                        observation_head.state_dict()
                    ),
                    "reward_head": copy.deepcopy(self.reward_fc.state_dict()),
                    "continue_head": copy.deepcopy(
                        self.continue_fc.state_dict()
                    ),
                }
            )
        return state

    def load_shared_core_state_dict(
        self, state: dict[str, dict[str, torch.Tensor]]
    ) -> None:
        required = {
            "encoder",
            "observation_adapter",
            "posterior",
            "recurrent",
            "prior",
            "latent_interface",
        }
        if (self.task_shared_prediction_heads):
            required.update(
                {"observation_head", "reward_head", "continue_head"}
            )
        if set(state) != required:
            raise ValueError(
                f"Shared-core state keys must be {sorted(required)}, got {sorted(state)}"
            )
        self.rssm.image_embedder.load_state_dict(state["encoder"], strict=True)
        self.rssm.observation_adapter.load_state_dict(
            state["observation_adapter"], strict=True
        )
        self.rssm.representation.load_state_dict(state["posterior"], strict=True)
        self.rssm.recurrent.load_state_dict(state["recurrent"], strict=True)
        self.rssm.transition.load_state_dict(state["prior"], strict=True)
        self.zh_transform.load_state_dict(state["latent_interface"], strict=True)
        if (self.task_shared_prediction_heads):
            observation_head = (self.decoder)
            observation_head.load_state_dict(
                state["observation_head"], strict=True
            )
            self.reward_fc.load_state_dict(state["reward_head"], strict=True)
            self.continue_fc.load_state_dict(
                state["continue_head"], strict=True
            )

    def predict_reward_symlog(
        self, model_state: torch.Tensor, task_id: Optional[int | torch.Tensor] = None
    ) -> torch.Tensor:
        head_features = self.prediction_features_for(
            "reward", model_state, task_id
        )
        prediction = self._head_for(
            self.reward_fc, self.reward_experts, task_id
        )(head_features)
        return prediction

    def predict_continue_logits(
        self, model_state: torch.Tensor, task_id: Optional[int | torch.Tensor] = None
    ) -> torch.Tensor:
        head_features = self.prediction_features_for(
            "continue", model_state, task_id
        )
        logits = self._head_for(
            self.continue_fc, self.continue_experts, task_id
        )(head_features)
        return logits

    def predict_observation(
        self,
        model_state: torch.Tensor,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        head_features = self.prediction_features_for(
            "observation", model_state, task_id
        )
        return self.decoder_for(task_id)(head_features)
        if hasattr(self, "feature_predictor"):
            prediction = self.feature_predictor_for(task_id)(head_features)
            return prediction
        raise RuntimeError(
            "The configured observation objective has no prediction head"
        )

    def predict_continue(
        self, model_state: torch.Tensor, task_id: Optional[int | torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.predict_continue_logits(model_state, task_id)
        with _full_precision_context(logits.device):
            return torch.sigmoid(logits.float())



    def forward(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Expose the training loss through `forward` for native PyTorch DDP."""
        return self.compute_loss(
            actions,
            xs,
            rews,
            conts,
            resets,
            task_id=task_id,
        )

    def compute_loss(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics, _trace = self._compute_loss_and_trace(
            actions,
            xs,
            rews,
            conts,
            resets,
            task_id=task_id,
        )
        return loss, metrics

    def compute_loss_and_trace(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, object],
    ]:
        """Return Dreamer loss plus the Q/F/P/H/Actor interface trace."""

        return self._compute_loss_and_trace(
            actions,
            xs,
            rews,
            conts,
            resets,
            task_id=task_id,
        )

    def _compute_loss_and_trace(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, object],
    ]:
        # Returns (loss, metrics)
        if len(actions.shape) == 2:
            raise ValueError("Time dimension required")
        _, n, _ = actions.shape
        init_z, init_h = self.rssm.initial_state(n)
        mechanism_trace: dict[str, list[torch.Tensor]] = {}
        # Shift actions and xs, since RSSM takes (prev_action, next_obs)
        embeddings = None
        z_posts, z_samples, hiddens = self.rssm(
            init_z,
            actions,
            init_h,
            xs,
            resets,
            task_id=task_id,
            mechanism_trace=mechanism_trace,
        )
        z_priors = self.rssm.prior(
            hiddens, task_id, mechanism_trace=mechanism_trace
        )

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
        recon = self.predict_observation(zhs_f12, task_id).view(
            t, n, *xs.shape[-3:]
        )
        observation_prediction = recon
        # Loss shape [ T N C 64 64 ]
        observation_losses = (recon.float() - xs.float()).square().sum([2, 3, 4])
        observation_loss = observation_losses.mean()
        observation_metrics = {
            "Loss/recon": observation_loss,
        }

        rews_pred = self.predict_reward_symlog(zhs, task_id)  # [ T N 1 ]
        rews_loss = (rews_pred.float() - symlog(rews)).square().mean()

        conts_logits = self.predict_continue_logits(zhs, task_id)  # [ T N 1 ]
        with _full_precision_context(conts_logits.device):
            conts_logits = conts_logits.float()
            conts_pred = torch.sigmoid(conts_logits)
            conts_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                conts_logits, conts.float(), reduction="mean"
            )

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
            metrics["Metric/low_kl_recon_loss"] = masked_mean(
                observation_losses, low_kl
            )

        consolidation_loss = torch.zeros((), device=z_repr_loss.device)

        total_loss = (
            z_repr_loss
            + observation_loss
            + rews_loss
            + conts_loss
            + consolidation_loss
        )
        current_atom_outputs: dict[str, torch.Tensor] = {}
        for component, values in mechanism_trace.items():
            if not values:
                continue
            current_atom_outputs[component] = (
                values[0] if len(values) == 1 else torch.stack(values)
            )
        trace: dict[str, object] = {
            "posterior_log_probs": z_posts,
            "posterior_logits": z_posts,
            "prior_log_probs": z_priors,
            "prior_logits": z_priors,
            "hiddens": hiddens,
            "actor_states": zhs,
            "current_atom_outputs": current_atom_outputs,
        }
        if self.task_shared_prediction_heads:
            trace["prediction_head_outputs"] = {
                "observation": observation_prediction,
                "reward_symlog": rews_pred,
                "continue_logits": conts_logits,
            }
        return total_loss, metrics, trace


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
