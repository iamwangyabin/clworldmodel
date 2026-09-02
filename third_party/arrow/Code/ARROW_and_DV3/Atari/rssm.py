import copy
from contextlib import nullcontext
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


def _full_precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def straight_through_one_hot(
    logits: torch.Tensor, stochastic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    with _full_precision_context(logits.device):
        logits = logits.float()
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
        dinov3_patch_adapter: str = "none",
        num_task_experts: int = 1,
        full_task_experts: bool = False,
        full_task_rssm_experts: Optional[bool] = None,
        task_banked_image_encoder: bool = False,
        task_projected_image_encoder: bool = False,
        task_symmetric_image_projectors: bool = False,
        task_projector_bottleneck_features: int = 64,
        task_lora_recurrent_rank: int = 0,
        task_lora_representation_rank: int = 0,
        task_lora_transition_rank: int = 0,
        task_recurrent_output_adapter_features: int = 0,
        task_mechanism_bank: bool = False,
        task_mechanism_reuse: bool = True,
        task_mechanism_recurrent_width: int = 512,
        task_mechanism_representation_width: int = 512,
        task_mechanism_transition_width: int = 256,
        task_mechanism_residual_scale: float = 0.1,
        task_mechanism_num_atoms: int = 1,
        task_mechanism_parameterization: str = "dense_private",
        task_mechanism_low_rank: int = 0,
        task_symmetric_mechanisms: bool = False,
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
        compute_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if full_task_rssm_experts is None:
            # Legacy callers used one flag for copied RSSMs and private heads.
            full_task_rssm_experts = full_task_experts
        if not isinstance(full_task_rssm_experts, bool):
            raise TypeError("full_task_rssm_experts must be a boolean")
        if num_task_experts < 1:
            raise ValueError("num_task_experts must be positive")
        if (full_task_experts or full_task_rssm_experts) and num_task_experts < 2:
            raise ValueError("Full task experts require at least two task routes")
        if task_banked_image_encoder and not full_task_experts:
            raise ValueError(
                "A task-banked image encoder requires complete task experts"
            )
        if task_banked_image_encoder and dinov3_patch_adapter != "none":
            raise ValueError(
                "A task-banked image encoder does not use a shared observation adapter"
            )
        if task_projected_image_encoder and not (
            full_task_experts or full_task_rssm_experts or task_mechanism_bank
        ):
            raise ValueError(
                "A projected shared image encoder requires task-routed components"
            )
        if task_projected_image_encoder and task_banked_image_encoder:
            raise ValueError(
                "Projected and fully banked image encoders are mutually exclusive"
            )
        task_lora_ranks = (
            task_lora_recurrent_rank,
            task_lora_representation_rank,
            task_lora_transition_rank,
        )
        if any(rank < 0 for rank in task_lora_ranks):
            raise ValueError("Task RSSM LoRA ranks must be non-negative")
        if task_recurrent_output_adapter_features < 0:
            raise ValueError("Task recurrent output adapter size must be non-negative")
        if (
            task_lora_recurrent_rank > 0
            and task_recurrent_output_adapter_features > 0
        ):
            raise ValueError(
                "Recurrent matrix LoRA and recurrent output adapters are mutually exclusive"
            )
        if task_mechanism_bank and (
            any(task_lora_ranks) or task_recurrent_output_adapter_features
        ):
            raise ValueError(
                "RSSM mechanism banks and LoRA/output adapters are mutually exclusive"
            )
        mechanism_widths = (
            task_mechanism_recurrent_width,
            task_mechanism_representation_width,
            task_mechanism_transition_width,
        )
        if task_mechanism_bank and min(mechanism_widths) < 1:
            raise ValueError("RSSM mechanism-bank widths must be positive")
        if task_mechanism_bank and task_mechanism_residual_scale <= 0:
            raise ValueError("RSSM mechanism residual scale must be positive")
        if task_mechanism_bank and task_mechanism_num_atoms < 1:
            raise ValueError("RSSM mechanism atom count must be positive")
        if task_mechanism_bank and any(
            width % task_mechanism_num_atoms for width in mechanism_widths
        ):
            raise ValueError(
                "RSSM mechanism widths must be divisible by the atom count"
            )
        if task_mechanism_parameterization not in {
            "dense_private",
            "shared_frozen_down_film",
            "learned_task0_low_rank",
            "dense_task0_low_rank_atoms",
        }:
            raise ValueError(
                "Unknown RSSM mechanism parameterization: "
                f"{task_mechanism_parameterization!r}"
            )
        if (
            task_mechanism_parameterization != "dense_private"
            and not task_mechanism_bank
        ):
            raise ValueError(
                "A non-default RSSM mechanism parameterization requires mechanism banks"
            )
        if task_mechanism_low_rank < 0:
            raise ValueError("RSSM mechanism low-rank size must be non-negative")
        if task_mechanism_parameterization in {
            "learned_task0_low_rank",
            "dense_task0_low_rank_atoms",
        }:
            if not task_symmetric_mechanisms:
                raise ValueError(
                    "Dense Task-0 low-rank mechanisms require symmetric mechanisms"
                )
            if (
                task_mechanism_parameterization == "learned_task0_low_rank"
                and task_mechanism_reuse
            ):
                raise ValueError(
                    "Learned Task-0 low-rank mechanisms disable old-atom reuse"
                )
            if (
                task_mechanism_parameterization
                == "dense_task0_low_rank_atoms"
                and not task_mechanism_reuse
            ):
                raise ValueError(
                    "Atomic low-rank mechanisms require old-atom reuse"
                )
            if task_mechanism_low_rank < 1:
                raise ValueError(
                    "Dense Task-0 low-rank mechanisms require a positive rank"
                )
            if task_mechanism_low_rank % task_mechanism_num_atoms:
                raise ValueError(
                    "RSSM mechanism low-rank size must be divisible by atoms"
                )
        elif task_mechanism_low_rank:
            raise ValueError(
                "RSSM mechanism low-rank size requires a low-rank parameterization"
            )
        if task_mechanism_bank and not task_projected_image_encoder:
            raise ValueError("RSSM mechanism banks require task image projectors")
        if task_symmetric_image_projectors and not task_projected_image_encoder:
            raise ValueError("Symmetric projectors require task image projection")
        if task_symmetric_mechanisms and not task_mechanism_bank:
            raise ValueError("Symmetric mechanisms require RSSM mechanism banks")
        if task_symmetric_image_projectors != task_symmetric_mechanisms:
            raise ValueError(
                "The evolving atomic topology requires symmetric projectors and mechanisms"
            )
        task_rssm_adaptation_enabled = bool(
            any(task_lora_ranks) or task_recurrent_output_adapter_features
        )
        if task_rssm_adaptation_enabled and not (
            task_lora_representation_rank > 0
            and task_lora_transition_rank > 0
            and (
                task_lora_recurrent_rank > 0
                or task_recurrent_output_adapter_features > 0
            )
        ):
            raise ValueError(
                "Task RSSM adaptation requires posterior/prior LoRA and one "
                "recurrent adaptation mechanism"
            )
        if task_rssm_adaptation_enabled and not full_task_rssm_experts:
            raise ValueError(
                "Task RSSM adaptation requires complete task experts"
            )
        if num_task_experts > 1 and residual_correction != "none":
            raise ValueError(
                "Task-routed RSSM experts do not compose with residual corrections"
            )
        self.ls = ls
        self.h_dim = h_dim
        self.num_task_experts = num_task_experts
        self.full_task_experts = full_task_experts
        self.full_task_rssm_experts = full_task_rssm_experts
        self.task_banked_image_encoder = task_banked_image_encoder
        self.task_projected_image_encoder = task_projected_image_encoder
        self.task_symmetric_image_projectors = bool(
            task_symmetric_image_projectors
        )
        self.task_lora_recurrent_rank = task_lora_recurrent_rank
        self.task_lora_representation_rank = task_lora_representation_rank
        self.task_lora_transition_rank = task_lora_transition_rank
        self.task_lora_enabled = any(rank > 0 for rank in task_lora_ranks)
        self.task_recurrent_output_adapter_features = (
            task_recurrent_output_adapter_features
        )
        self.task_recurrent_output_adapter_enabled = (
            task_recurrent_output_adapter_features > 0
        )
        self.task_mechanism_bank_enabled = bool(task_mechanism_bank)
        self.task_mechanism_reuse = bool(task_mechanism_reuse)
        self.task_mechanism_recurrent_width = task_mechanism_recurrent_width
        self.task_mechanism_representation_width = (
            task_mechanism_representation_width
        )
        self.task_mechanism_transition_width = task_mechanism_transition_width
        self.task_mechanism_residual_scale = task_mechanism_residual_scale
        self.task_mechanism_num_atoms = task_mechanism_num_atoms
        self.task_mechanism_parameterization = task_mechanism_parameterization
        self.task_mechanism_low_rank = int(task_mechanism_low_rank)
        self.task_symmetric_mechanisms = bool(task_symmetric_mechanisms)

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
        if self.task_mechanism_bank_enabled:
            self.recurrent_experts = nn.ModuleList()
        elif self.task_recurrent_output_adapter_enabled:
            from clworldmodel.models.rssm_lora import TaskRecurrentOutputRoute

            self.recurrent_experts = nn.ModuleList(
                TaskRecurrentOutputRoute(
                    self.recurrent,
                    output_features=h_dim,
                    bottleneck_features=task_recurrent_output_adapter_features,
                )
                for _ in range(num_task_experts - 1)
            )
        else:
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
                compute_dtype=compute_dtype,
            )
        else:
            raise ValueError(f"Unknown observation encoder: {observation_encoder!r}")
        if not hasattr(self.image_embedder, "output_size"):
            raise TypeError("Image embedder must declare output_size")
        self.image_embedder_experts = nn.ModuleList(
            copy.deepcopy(self.image_embedder)
            for _ in range(num_task_experts - 1 if task_banked_image_encoder else 0)
        )
        self.image_projectors = nn.ModuleList()
        self.image_projector_identity = nn.Identity()
        if task_projected_image_encoder:
            if observation_encoder != "cnn" or self.image_embedder.output_size != 4096:
                raise ValueError(
                    "The first projected-encoder protocol requires 4096-wide CNN features"
                )
            from clworldmodel.models.rssm_lora import SpatialFeatureProjector

            self.image_projectors.extend(
                SpatialFeatureProjector(
                    bottleneck_channels=task_projector_bottleneck_features
                )
                for _ in range(
                    num_task_experts
                    if self.task_symmetric_image_projectors
                    else num_task_experts - 1
                )
            )
        self.observation_adapter_kind = dinov3_patch_adapter
        self.observation_adapter: nn.Module = nn.Identity()
        self.observation_embedding_size = self.image_embedder.output_size
        if dinov3_patch_adapter == "conv_3x3_stride2":
            from clworldmodel.models.dinov3_adapter import DinoPatchConvAdapter

            adapter = DinoPatchConvAdapter(
                patch_grid_size=dinov3_patch_pool_size,
                in_channels=dinov3_patch_feature_dim,
                out_channels=64,
            )
            if adapter.input_size != self.image_embedder.output_size:
                raise ValueError(
                    "DINO patch adapter input does not match the image embedder: "
                    f"{adapter.input_size} != {self.image_embedder.output_size}"
                )
            self.observation_adapter = adapter
            self.observation_embedding_size = adapter.output_size
        elif dinov3_patch_adapter != "none":
            raise ValueError(
                f"Unknown DINOv3 patch adapter: {dinov3_patch_adapter!r}"
            )
        self.representation = Representation(
            ls,
            self.observation_embedding_size,
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
        representation_expert_count = 0
        if full_task_rssm_experts and not self.task_mechanism_bank_enabled:
            representation_expert_count = num_task_experts - 1
        self.representation_experts = nn.ModuleList(
            copy.deepcopy(self.representation)
            for _ in range(representation_expert_count)
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
        transition_expert_count = (
            0 if self.task_mechanism_bank_enabled else num_task_experts - 1
        )
        self.transition_experts = nn.ModuleList(
            copy.deepcopy(self.transition)
            for _ in range(transition_expert_count)
        )
        self.task_mechanism_reports: dict[str, object] = {}
        if self.task_mechanism_bank_enabled:
            from clworldmodel.models.mechanism_bank import MechanismBank

            z_dim = ls[0] * ls[1]
            # Mechanism allocation must not perturb downstream head initialization.
            with torch.random.fork_rng(devices=[]):
                self.recurrent_mechanism_bank = MechanismBank(
                    num_tasks=num_task_experts,
                    in_features=h_dim,
                    out_features=h_dim,
                    hidden_features=task_mechanism_recurrent_width,
                    residual_scale=task_mechanism_residual_scale,
                    reuse_enabled=task_mechanism_reuse,
                    num_atoms=task_mechanism_num_atoms,
                    include_task0=self.task_symmetric_mechanisms,
                    parameterization=task_mechanism_parameterization,
                    low_rank_rank=task_mechanism_low_rank,
                )
                self.representation_mechanism_bank = MechanismBank(
                    num_tasks=num_task_experts,
                    in_features=self.observation_embedding_size + h_dim,
                    out_features=z_dim,
                    hidden_features=task_mechanism_representation_width,
                    residual_scale=task_mechanism_residual_scale,
                    reuse_enabled=task_mechanism_reuse,
                    num_atoms=task_mechanism_num_atoms,
                    include_task0=self.task_symmetric_mechanisms,
                    parameterization=task_mechanism_parameterization,
                    low_rank_rank=task_mechanism_low_rank,
                )
                self.transition_mechanism_bank = MechanismBank(
                    num_tasks=num_task_experts,
                    in_features=h_dim,
                    out_features=z_dim,
                    hidden_features=task_mechanism_transition_width,
                    residual_scale=task_mechanism_residual_scale,
                    reuse_enabled=task_mechanism_reuse,
                    num_atoms=task_mechanism_num_atoms,
                    include_task0=self.task_symmetric_mechanisms,
                    parameterization=task_mechanism_parameterization,
                    low_rank_rank=task_mechanism_low_rank,
                )
            self.task_mechanism_reports = {
                "recurrent": self.recurrent_mechanism_bank.parameter_report(),
                "representation": (
                    self.representation_mechanism_bank.parameter_report()
                ),
                "transition": self.transition_mechanism_bank.parameter_report(),
            }
        self.task_lora_reports: list[dict[str, object]] = []
        if task_rssm_adaptation_enabled:
            from clworldmodel.models.rssm_lora import (
                install_affine_lora,
                reset_affine_lora_from,
            )

            for task_index in range(1, num_task_experts):
                report: dict[str, object] = {"task_index": task_index}
                if task_lora_recurrent_rank:
                    report["recurrent"] = install_affine_lora(
                        self.recurrent_for(task_index),
                        task_lora_recurrent_rank,
                    )
                    reset_affine_lora_from(
                        self.recurrent_for(task_index), self.recurrent
                    )
                elif self.task_recurrent_output_adapter_enabled:
                    route = self.recurrent_for(task_index)
                    report["recurrent"] = route.parameter_report()
                if task_lora_representation_rank:
                    report["representation"] = install_affine_lora(
                        self.representation_for(task_index),
                        task_lora_representation_rank,
                    )
                    reset_affine_lora_from(
                        self.representation_for(task_index), self.representation
                    )
                if task_lora_transition_rank:
                    report["transition"] = install_affine_lora(
                        self.transition_for(task_index),
                        task_lora_transition_rank,
                    )
                    reset_affine_lora_from(
                        self.transition_for(task_index), self.transition
                    )
                self.task_lora_reports.append(report)

    def freeze_shared_core(self) -> None:
        """Freeze base RSSM functions while leaving residual adapters plastic."""
        self.image_embedder.requires_grad_(False)
        for image_embedder in self.image_embedder_experts:
            image_embedder.requires_grad_(False)
        self.observation_adapter.requires_grad_(False)
        for projector in self.image_projectors:
            projector.requires_grad_(False)
        self.recurrent.freeze_shared_core()
        for recurrent in self.recurrent_experts:
            recurrent.freeze_shared_core()
        self.representation.freeze_shared_core()
        for representation in self.representation_experts:
            representation.freeze_shared_core()
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
                    "task_id must be scalar; task-routed minibatches are task-homogeneous"
                )
            task_id = int(task_id.detach().item())
        if not isinstance(task_id, int):
            raise TypeError("task_id must be an integer or scalar tensor")
        if task_id < 0 or task_id >= self.num_task_experts:
            raise ValueError(
                f"task_id {task_id} is outside [0, {self.num_task_experts})"
            )
        return task_id

    def recurrent_for(self, task_id: Optional[int | torch.Tensor]) -> nn.Module:
        task_index = self._task_index(task_id)
        if self.task_mechanism_bank_enabled:
            return self.recurrent
        return (
            self.recurrent
            if task_index == 0
            else self.recurrent_experts[task_index - 1]
        )

    def image_embedder_for(
        self, task_id: Optional[int | torch.Tensor]
    ) -> nn.Module:
        task_index = self._task_index(task_id)
        if task_index == 0 or not self.task_banked_image_encoder:
            return self.image_embedder
        return self.image_embedder_experts[task_index - 1]

    def image_projector_for(
        self, task_id: Optional[int | torch.Tensor]
    ) -> nn.Module:
        task_index = self._task_index(task_id)
        if not self.task_projected_image_encoder:
            return self.image_projector_identity
        if self.task_symmetric_image_projectors:
            return self.image_projectors[task_index]
        if task_index == 0:
            return self.image_projector_identity
        return self.image_projectors[task_index - 1]

    def transition_for(self, task_id: Optional[int | torch.Tensor]) -> "Transition":
        task_index = self._task_index(task_id)
        if self.task_mechanism_bank_enabled:
            return self.transition
        return (
            self.transition
            if task_index == 0
            else self.transition_experts[task_index - 1]
        )

    def representation_for(
        self, task_id: Optional[int | torch.Tensor]
    ) -> "Representation":
        task_index = self._task_index(task_id)
        if self.task_mechanism_bank_enabled:
            return self.representation
        if task_index == 0 or not self.full_task_experts:
            return self.representation
        return self.representation_experts[task_index - 1]

    def recurrent_step(
        self,
        prev_z: LatentT,
        prev_a: ActionT,
        prev_h: HiddenT,
        task_id: Optional[int | torch.Tensor] = None,
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
    ) -> HiddenT:
        task_index = self._task_index(task_id)
        if not self.task_mechanism_bank_enabled:
            return self.recurrent_for(task_index)(prev_z, prev_a, prev_h)
        base_hidden = self.recurrent(prev_z, prev_a, prev_h)
        correction, current_output = (
            self.recurrent_mechanism_bank.forward_with_current(
                base_hidden, task_index
            )
        )
        if mechanism_trace is not None:
            mechanism_trace.setdefault("recurrent", []).append(current_output)
        return base_hidden + correction

    def posterior_step(
        self,
        embedding: EmbedT,
        hidden: HiddenT,
        task_id: Optional[int | torch.Tensor] = None,
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
    ) -> LatentLogDistT:
        task_index = self._task_index(task_id)
        if not self.task_mechanism_bank_enabled:
            return self.representation_for(task_index)(embedding, hidden)
        mechanism_input = torch.cat((embedding, hidden), dim=-1)
        base_logits = self.representation.logits(embedding, hidden)
        correction, current_output = (
            self.representation_mechanism_bank.forward_with_current(
                mechanism_input, task_index
            )
        )
        if mechanism_trace is not None:
            mechanism_trace.setdefault("posterior", []).append(current_output)
        return self.representation.distribution_from_logits(
            base_logits + correction
        )

    def prior(
        self,
        hidden: HiddenT,
        task_id: Optional[int | torch.Tensor] = None,
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
    ) -> LatentLogDistT:
        task_index = self._task_index(task_id)
        if not self.task_mechanism_bank_enabled:
            return self.transition_for(task_index)(hidden)
        base_logits = self.transition.logits(hidden)
        correction, current_output = (
            self.transition_mechanism_bank.forward_with_current(hidden, task_index)
        )
        if mechanism_trace is not None:
            mechanism_trace.setdefault("prior", []).append(current_output)
        return self.transition.distribution_from_logits(base_logits + correction)

    def copy_task_expert(self, target_task_id: int, source_task_id: int) -> None:
        if target_task_id == source_task_id:
            return
        if self.task_mechanism_bank_enabled:
            if target_task_id == 0:
                raise ValueError(
                    "The shared Task-0 RSSM cannot be reset from a later task"
                )
            self.recurrent_mechanism_bank.reset_task(target_task_id)
            self.representation_mechanism_bank.reset_task(target_task_id)
            self.transition_mechanism_bank.reset_task(target_task_id)
            if self.task_projected_image_encoder:
                projector_index = (
                    target_task_id
                    if self.task_symmetric_image_projectors
                    else target_task_id - 1
                )
                self.image_projectors[projector_index].reset_parameters()
        elif (
            self.task_lora_enabled or self.task_recurrent_output_adapter_enabled
        ) and target_task_id > 0:
            from clworldmodel.models.rssm_lora import reset_affine_lora_from

            if self.task_recurrent_output_adapter_enabled:
                self.recurrent_for(target_task_id).reset_delta()
            elif self.task_lora_recurrent_rank:
                reset_affine_lora_from(
                    self.recurrent_for(target_task_id),
                    self.recurrent_for(source_task_id),
                )
            if self.task_lora_transition_rank:
                reset_affine_lora_from(
                    self.transition_for(target_task_id),
                    self.transition_for(source_task_id),
                )
            if self.task_lora_representation_rank:
                reset_affine_lora_from(
                    self.representation_for(target_task_id),
                    self.representation_for(source_task_id),
                )
        else:
            self.recurrent_for(target_task_id).load_state_dict(
                self.recurrent_for(source_task_id).state_dict()
            )
            self.transition_for(target_task_id).load_state_dict(
                self.transition_for(source_task_id).state_dict()
            )
            if self.full_task_experts:
                self.representation_for(target_task_id).load_state_dict(
                    self.representation_for(source_task_id).state_dict()
                )
        if self.task_banked_image_encoder:
            self.image_embedder_for(target_task_id).load_state_dict(
                self.image_embedder_for(source_task_id).state_dict()
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
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
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
            mechanism_trace=mechanism_trace,
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
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        if len(prev_a.shape) == 2:
            # Apply reset flags
            prev_z = prev_z * (1 - reset).unsqueeze(-1)  # Need to multiply againt dim [ N 1 1 ]
            prev_h = prev_h * (1 - reset)
            # No time dimension
            h = self.recurrent_step(
                prev_z, prev_a, prev_h, task_id, mechanism_trace
            )
            if x is not None:
                e = self.adapt_observation_embeddings(
                    self.image_embedder_for(task_id)(x), task_id=task_id
                )
                z_log_dist = self.posterior_step(
                    e, h, task_id, mechanism_trace
                )
            else:
                z_log_dist = self.prior(h, task_id, mechanism_trace)
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
                    mechanism_trace=mechanism_trace,
                )
                z_log_dists.append(z_log_dist)
                z_samples.append(z_sample)
                hs.append(h)
            return torch.stack(z_log_dists), torch.stack(z_samples), torch.stack(hs)
        elif len(prev_a.shape) == 3:
            # Special batched impl
            embeddings = self.embed_observations(x, task_id=task_id)
            return self.observe_embeddings(
                prev_z,
                prev_a,
                prev_h,
                embeddings,
                reset,
                stochastic=stochastic,
                temperature=temperature,
                task_id=task_id,
                mechanism_trace=mechanism_trace,
            )
        raise ValueError

    def embed_observations(
        self,
        x: ImageT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> EmbedT:
        """Encode a [T, N, C, H, W] observation sequence exactly once."""
        if len(x.shape) != 5:
            raise ValueError(
                "Observation sequences must have shape [T, N, C, H, W], "
                f"got {tuple(x.shape)}"
            )
        t, n = x.shape[:2]
        raw_embeddings = self.image_embedder_for(task_id)(
            x.reshape(-1, *x.shape[-3:])
        )
        return self.adapt_observation_embeddings(
            raw_embeddings, task_id=task_id
        ).view(t, n, -1)

    def adapt_observation_embeddings(
        self,
        embeddings: EmbedT,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> EmbedT:
        """Map frozen encoder outputs to the posterior's embedding interface."""
        if embeddings.ndim < 2:
            raise ValueError("Observation embeddings must include a feature axis")
        if embeddings.shape[-1] != self.image_embedder.output_size:
            raise ValueError(
                "Raw observation embeddings have an unexpected width: "
                f"expected {self.image_embedder.output_size}, "
                f"got {embeddings.shape[-1]}"
            )
        leading_shape = embeddings.shape[:-1]
        flattened = embeddings.reshape(-1, embeddings.shape[-1])
        projected = self.image_projector_for(task_id)(flattened)
        adapted = self.observation_adapter(
            projected
        )
        if adapted.shape[-1] != self.observation_embedding_size:
            raise RuntimeError("Observation adapter returned an unexpected width")
        return adapted.reshape(*leading_shape, self.observation_embedding_size)

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
        mechanism_trace: Optional[dict[str, list[torch.Tensor]]] = None,
    ) -> tuple[LatentLogDistT, LatentT, HiddenT]:
        """Run the posterior recurrence from precomputed [T, N, E] embeddings."""
        if len(prev_a.shape) != 3 or len(embeddings.shape) != 3:
            raise ValueError("Actions and embeddings must include time and batch dimensions")
        if prev_a.shape[:2] != embeddings.shape[:2] or reset.shape[:2] != prev_a.shape[:2]:
            raise ValueError("Actions, embeddings, and resets must share [T, N] dimensions")
        if embeddings.shape[-1] != self.observation_embedding_size:
            raise ValueError(
                "Posterior embeddings have an unexpected width: "
                f"expected {self.observation_embedding_size}, "
                f"got {embeddings.shape[-1]}"
            )

        hs = []
        z_log_dists = []
        z_samples = []
        z, h = prev_z, prev_h
        for e, a, r in zip(embeddings, prev_a, reset):
            h = self.recurrent_step(
                z * (1 - r).unsqueeze(-1),
                a,
                h * (1 - r),
                task_id,
                mechanism_trace,
            )
            z_log_dist = self.posterior_step(
                e, h, task_id, mechanism_trace
            )
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

    @staticmethod
    def _unique_parameters(
        modules: list[nn.Module],
    ) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return parameters

    def shared_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return non-overlapping shared encoder/Q/F/P parameter groups."""

        return {
            "encoder": self._unique_parameters(
                [self.image_embedder, self.observation_adapter]
            ),
            "posterior": list(self.representation.parameters()),
            "recurrent": list(self.recurrent.parameters()),
            "prior": list(self.transition.parameters()),
        }

    def private_parameters(self, task_id: int) -> list[nn.Parameter]:
        """Return the selected task's projector and Q/F/P atom parameters."""

        task_index = self._task_index(task_id)
        modules: list[nn.Module] = []
        projector = self.image_projector_for(task_index)
        if projector is not self.image_projector_identity:
            modules.append(projector)
        if self.task_mechanism_bank_enabled:
            for bank in (
                self.recurrent_mechanism_bank,
                self.representation_mechanism_bank,
                self.transition_mechanism_bank,
            ):
                mechanism = bank.mechanism_for(task_index)
                if mechanism is not None:
                    modules.append(mechanism)
        return self._unique_parameters(modules)

    def route_parameters(self, task_id: int) -> list[nn.Parameter]:
        """Return only the selected task's old-atom gate parameters."""

        if not self.task_mechanism_bank_enabled or not self.task_mechanism_reuse:
            return []
        task_index = self._task_index(task_id)
        modules: list[nn.Module] = []
        for bank in (
            self.recurrent_mechanism_bank,
            self.representation_mechanism_bank,
            self.transition_mechanism_bank,
        ):
            route = bank.route_for(task_index)
            if route is not None:
                modules.append(route)
        return self._unique_parameters(modules)


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
        return self.distribution_from_logits(self.logits(e, h))

    def logits(self, e: EmbedT, h: HiddenT) -> torch.Tensor:
        """Return pre-categorical posterior logits for residual composition."""
        assert len(e.shape) == len(h.shape) == 2
        eh = torch.cat((e, h), dim=-1)
        x1 = self.eh_to_inter(eh)
        if self.e_to_inter is not None:
            x1 = x1 + self.e_to_inter(e)
        if self.residual is not None:
            residual_input = x1 if self.residual_input_mode == "base_output" else eh
            x1 = x1 + self.residual(residual_input)
        return x1

    def distribution_from_logits(self, logits: torch.Tensor) -> LatentLogDistT:
        """Normalize flat posterior logits with the original uniform mixture."""
        post_log_probs = self.inter_to_z_dist(logits)
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
        self._logit_layer_count = len(self.h_to_z_prior) - 2
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
        if self.residual is None:
            return self.distribution_from_logits(self.logits(h))
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

    def logits(self, h: HiddenT) -> torch.Tensor:
        """Return flat pre-categorical prior logits for residual composition."""
        if self.residual is not None:
            raise RuntimeError(
                "Raw prior logits are unavailable with legacy residual corrections"
            )
        logits = h
        for index in range(self._logit_layer_count):
            logits = self.h_to_z_prior[index](logits)
        return logits

    def distribution_from_logits(self, logits: torch.Tensor) -> LatentLogDistT:
        """Normalize flat prior logits with the original uniform mixture."""
        prior_log_probs = self.h_to_z_prior[-2](logits)
        prior_log_probs = self.h_to_z_prior[-1](prior_log_probs)
        if self.uniform:
            probs = prior_log_probs.exp()
            return ((1 - self.uniform) * probs + self.uniform / self.ls[1]).log()
        return prior_log_probs

    def freeze_shared_core(self) -> None:
        self.h_to_z_prior.requires_grad_(False)
        if self.residual is not None:
            self.residual.requires_grad_(True)
