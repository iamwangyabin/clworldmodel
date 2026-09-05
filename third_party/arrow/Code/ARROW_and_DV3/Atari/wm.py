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
ObservationObjective = Literal[
    "reconstruction",
    "r2",
    "dinov3_next_feature",
    "dinov3_posterior_feature",
]
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


@torch.no_grad()
def reward_learning_diagnostics(
    prediction_symlog: torch.Tensor, rewards: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Observe sparse reward learning without changing losses or sampling.

    Conditional means use zero when the subset is empty; always inspect the
    positive fraction/count before interpreting such a mean as model accuracy.
    """
    prediction = symexp(prediction_symlog)
    positive = rewards > 0
    zero = rewards == 0
    positive_count = positive.sum().clamp_min(1)
    zero_count = zero.sum().clamp_min(1)
    return {
        "LearningAudit/replay_positive_reward_fraction": positive.float().mean(),
        "LearningAudit/replay_reward_mean": rewards.float().mean(),
        "LearningAudit/predicted_reward_mean": prediction.mean(),
        "LearningAudit/positive_reward_prediction_mean": torch.where(
            positive, prediction, 0
        ).sum() / positive_count,
        "LearningAudit/positive_reward_target_mean": torch.where(
            positive, rewards.float(), 0
        ).sum() / positive_count,
        "LearningAudit/positive_reward_absolute_error": torch.where(
            positive, (prediction - rewards.float()).abs(), 0
        ).sum() / positive_count,
        "LearningAudit/zero_reward_absolute_error": torch.where(
            zero, prediction.abs(), 0
        ).sum() / zero_count,
    }


def symlog(x: torch.Tensor) -> torch.Tensor:
    with _full_precision_context(x.device):
        x = x.float()
        return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    with _full_precision_context(x.device):
        x = x.float()
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
        dinov3_patch_projection_seed: int = 0,
        dinov3_patch_adapter: str = "none",
        dinov3_feature_loss_kind: str = "cosine",
        dinov3_feature_std_floor: float = 0.05,
        residual_correction: str = "none",
        residual_bottleneck_features: int = 64,
        residual_grid_size: int = 8,
        residual_input_min: float = -2.0,
        residual_input_max: float = 2.0,
        residual_rms_norm_epsilon: float = 1e-4,
        residual_alpha: float = 0.1,
        residual_input_mode: str = "base_output",
        residual_consolidation: str = "none",
        num_task_experts: int = 1,
        full_task_experts: bool = False,
        full_task_rssm_experts: Optional[bool] = None,
        task_private_heads: Optional[bool] = None,
        task_shared_prediction_heads: bool = False,
        task_private_prediction_adapters: bool = False,
        prediction_adapter_rank: int = 0,
        prediction_adapter_residual_scale: float = 0.1,
        freeze_shared_prediction_heads_after_task0: bool = False,
        evolving_shared_core: bool = False,
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
        image_embedder: Optional[nn.Module] = None,
        compute_dtype: str = "float32",
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
        if task_private_heads is None:
            task_private_heads = full_task_experts
        if full_task_rssm_experts is None:
            full_task_rssm_experts = full_task_experts
        if not isinstance(task_private_heads, bool):
            raise TypeError("task_private_heads must be a boolean")
        if not isinstance(task_shared_prediction_heads, bool):
            raise TypeError("task_shared_prediction_heads must be a boolean")
        if not isinstance(task_private_prediction_adapters, bool):
            raise TypeError("task_private_prediction_adapters must be a boolean")
        if not isinstance(freeze_shared_prediction_heads_after_task0, bool):
            raise TypeError(
                "freeze_shared_prediction_heads_after_task0 must be a boolean"
            )
        if not isinstance(evolving_shared_core, bool):
            raise TypeError("evolving_shared_core must be a boolean")
        if task_private_heads and task_shared_prediction_heads:
            raise ValueError(
                "Task-private and task-shared prediction heads are mutually exclusive"
            )
        if task_shared_prediction_heads and not evolving_shared_core:
            raise ValueError(
                "Task-shared prediction heads require an evolving shared core"
            )
        if task_private_prediction_adapters:
            if not task_shared_prediction_heads:
                raise ValueError(
                    "Task-private prediction adapters require shared prediction heads"
                )
            if num_task_experts < 2:
                raise ValueError(
                    "Task-private prediction adapters require multiple task routes"
                )
            if prediction_adapter_rank < 1:
                raise ValueError("Prediction-adapter rank must be positive")
            if prediction_adapter_residual_scale <= 0:
                raise ValueError(
                    "Prediction-adapter residual scale must be positive"
                )
            if not freeze_shared_prediction_heads_after_task0:
                raise ValueError(
                    "Private prediction adapters require frozen Task-0 base heads"
                )
        elif prediction_adapter_rank != 0:
            raise ValueError(
                "Prediction-adapter rank requires task-private prediction adapters"
            )
        if (
            freeze_shared_prediction_heads_after_task0
            and not task_private_prediction_adapters
        ):
            raise ValueError(
                "Frozen Task-0 prediction heads require private prediction adapters"
            )
        if task_private_heads and observation_objective not in {
            "reconstruction",
            "dinov3_posterior_feature",
        }:
            raise ValueError(
                "Full task experts require pixel or DINOv3 feature reconstruction"
            )
        if compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unknown compute dtype: {compute_dtype!r}")

        self.ls = ls
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.compute_dtype = compute_dtype
        self.observation_objective = observation_objective
        self.r2_barlow_loss_scale = r2_barlow_loss_scale
        self.r2_redundancy_scale = r2_redundancy_scale
        self.r2_normalization_eps = r2_normalization_eps
        self.dinov3_feature_loss_scale = dinov3_feature_loss_scale
        self.dinov3_feature_loss_kind = dinov3_feature_loss_kind
        self.dinov3_feature_std_floor = dinov3_feature_std_floor
        self.residual_correction = residual_correction
        self.residual_input_mode = residual_input_mode
        self.full_task_experts = full_task_experts
        self.full_task_rssm_experts = bool(full_task_rssm_experts)
        self.task_private_heads = task_private_heads
        self.task_shared_prediction_heads = task_shared_prediction_heads
        self.task_private_prediction_adapters = (
            task_private_prediction_adapters
        )
        self.prediction_adapter_rank = int(prediction_adapter_rank)
        self.prediction_adapter_residual_scale = float(
            prediction_adapter_residual_scale
        )
        self.freeze_shared_prediction_heads_after_task0 = (
            freeze_shared_prediction_heads_after_task0
        )
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
            observation_encoder=observation_encoder,
            dinov3_model_path=dinov3_model_path,
            dinov3_input_size=dinov3_input_size,
            dinov3_max_batch_size=dinov3_max_batch_size,
            dinov3_feature_mode=dinov3_feature_mode,
            dinov3_patch_pool_size=dinov3_patch_pool_size,
            dinov3_patch_feature_dim=dinov3_patch_feature_dim,
            dinov3_patch_projection=dinov3_patch_projection,
            dinov3_patch_projection_seed=dinov3_patch_projection_seed,
            dinov3_patch_adapter=dinov3_patch_adapter,
            num_task_experts=num_task_experts,
            full_task_experts=full_task_experts,
            full_task_rssm_experts=full_task_rssm_experts,
            task_banked_image_encoder=task_banked_image_encoder,
            task_projected_image_encoder=task_projected_image_encoder,
            task_symmetric_image_projectors=task_symmetric_image_projectors,
            task_projector_bottleneck_features=task_projector_bottleneck_features,
            task_lora_recurrent_rank=task_lora_recurrent_rank,
            task_lora_representation_rank=task_lora_representation_rank,
            task_lora_transition_rank=task_lora_transition_rank,
            task_recurrent_output_adapter_features=(
                task_recurrent_output_adapter_features
            ),
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
            task_mechanism_low_rank=task_mechanism_low_rank,
            task_symmetric_mechanisms=task_symmetric_mechanisms,
            residual_correction=residual_correction,
            residual_bottleneck_features=residual_bottleneck_features,
            residual_grid_size=residual_grid_size,
            residual_input_min=residual_input_min,
            residual_input_max=residual_input_max,
            residual_rms_norm_epsilon=residual_rms_norm_epsilon,
            residual_alpha=residual_alpha,
            residual_input_mode=residual_input_mode,
            residual_consolidation=residual_consolidation,
            image_embedder=image_embedder,
        )

        # Shared feature consumed by observation, reward, and continuation heads.
        self.zh_transform = ZhToModelState(ls, h_dim)
        self.feature_predictor_residual = None
        self.feature_predictor_experts = nn.ModuleList()
        self.decoder_experts = nn.ModuleList()

        if observation_objective == "reconstruction":
            self.decoder = Decoder(img_channels, self.zh_transform.out_features, cnn_depth)
            if task_private_heads:
                self.decoder_experts.extend(
                    copy.deepcopy(self.decoder) for _ in range(num_task_experts - 1)
                )
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
            if task_private_heads:
                self.feature_predictor_experts.extend(
                    copy.deepcopy(self.feature_predictor)
                    for _ in range(num_task_experts - 1)
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
                    final_activation=None,
                    hidden_features=mlp_features,
                    layers=mlp_layers,
                )
            )
        elif residual_input_mode == "module_input":
            from clworldmodel.models.residual_corrections import (
                build_residual_correction,
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
            # Preserve the base initialization and global training RNG while
            # giving the independent residual branches distinct parameters.
            with torch.random.fork_rng(devices=[]):
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
                    consolidation_enabled=residual_consolidation != "none",
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
                    consolidation_enabled=residual_consolidation != "none",
                )
                if observation_objective in {
                    "dinov3_next_feature",
                    "dinov3_posterior_feature",
                }:
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
                        consolidation_enabled=residual_consolidation != "none",
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
                consolidation_enabled=residual_consolidation != "none",
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
                consolidation_enabled=residual_consolidation != "none",
            )
        if (
            observation_objective
            in {"dinov3_next_feature", "dinov3_posterior_feature"}
            and residual_correction != "none"
            and residual_input_mode == "base_output"
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
                consolidation_enabled=residual_consolidation != "none",
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
        if self.task_private_prediction_adapters:
            from clworldmodel.models.prediction_adapters import (
                ZeroEffectFeatureAdapter,
            )

            # Adapter allocation must not perturb the Actor-Critic or any other
            # downstream initialization in the matched Task-0 protocol.
            with torch.random.fork_rng(devices=[]):
                for head_name in ("observation", "reward", "continue"):
                    self.prediction_adapters[head_name] = nn.ModuleList(
                        ZeroEffectFeatureAdapter(
                            self.zh_transform.out_features,
                            self.prediction_adapter_rank,
                            residual_scale=(
                                self.prediction_adapter_residual_scale
                            ),
                        )
                        for _ in range(num_task_experts - 1)
                    )
        initialized = None
        if num_task_experts > 1:
            initialized = torch.zeros(num_task_experts, dtype=torch.bool)
            initialized[0] = True
        self.register_buffer("task_expert_initialized", initialized)

    def prediction_adapter_for(
        self,
        head_name: str,
        task_id: Optional[int | torch.Tensor],
    ) -> Optional[nn.Module]:
        if head_name not in {"observation", "reward", "continue"}:
            raise ValueError(f"Unknown prediction-adapter head: {head_name!r}")
        task_index = self.rssm._task_index(task_id)
        if not self.task_private_prediction_adapters or task_index == 0:
            return None
        return self.prediction_adapters[head_name][task_index - 1]

    def prediction_features_for(
        self,
        head_name: str,
        model_state: torch.Tensor,
        task_id: Optional[int | torch.Tensor],
    ) -> torch.Tensor:
        adapter = self.prediction_adapter_for(head_name, task_id)
        return model_state if adapter is None else adapter(model_state)

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

    def feature_predictor_for(
        self, task_id: Optional[int | torch.Tensor]
    ) -> nn.Module:
        if not hasattr(self, "feature_predictor"):
            raise RuntimeError("The configured observation objective has no feature predictor")
        task_index = self.rssm._task_index(task_id)
        if task_index == 0 or not self.task_private_heads:
            return self.feature_predictor
        return self.feature_predictor_experts[task_index - 1]

    def decoder_for(self, task_id: Optional[int | torch.Tensor]) -> nn.Module:
        if not hasattr(self, "decoder"):
            raise RuntimeError("The configured observation objective has no decoder")
        task_index = self.rssm._task_index(task_id)
        if task_index == 0 or not self.task_private_heads:
            return self.decoder
        return self.decoder_experts[task_index - 1]

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
            if (
                self.task_private_heads
                and self.observation_objective == "reconstruction"
            ):
                self.decoder_for(target_index).load_state_dict(
                    self.decoder_for(source_index).state_dict()
                )
            elif self.task_private_heads:
                self.feature_predictor_for(target_index).load_state_dict(
                    self.feature_predictor_for(source_index).state_dict()
                )
        elif self.task_private_prediction_adapters and target_index > 0:
            for head_name in ("observation", "reward", "continue"):
                adapter = self.prediction_adapter_for(head_name, target_index)
                if adapter is None or not hasattr(adapter, "reset_parameters"):
                    raise RuntimeError(
                        "A private prediction adapter is missing its reset contract"
                    )
                adapter.reset_parameters()
        self.task_expert_initialized[target_index] = True
        return True

    def activate_task_expert(
        self, task_id: int, mechanism_phase: str = "full"
    ) -> None:
        """Make exactly one complete task expert plastic and freeze all others."""
        if not (
            self.task_private_heads
            or self.rssm.full_task_rssm_experts
            or self.rssm.task_mechanism_bank_enabled
        ):
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
                    self.rssm.task_banked_image_encoder and index == task_index
                )
        self.rssm.observation_adapter.requires_grad_(
            self.rssm.observation_adapter_kind != "none"
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
        elif (
            self.rssm.task_lora_enabled
            or self.rssm.task_recurrent_output_adapter_enabled
        ):
            from clworldmodel.models.rssm_lora import (
                set_affine_lora_trainable,
                set_recurrent_output_adapter_trainable,
            )

            for index in range(1, self.rssm.num_task_experts):
                if self.rssm.task_recurrent_output_adapter_enabled:
                    set_recurrent_output_adapter_trainable(
                        self.rssm.recurrent_for(index), index == task_index
                    )
                elif self.rssm.task_lora_recurrent_rank:
                    set_affine_lora_trainable(
                        self.rssm.recurrent_for(index), index == task_index
                    )
                if self.rssm.task_lora_representation_rank:
                    set_affine_lora_trainable(
                        self.rssm.representation_for(index), index == task_index
                    )
                if self.rssm.task_lora_transition_rank:
                    set_affine_lora_trainable(
                        self.rssm.transition_for(index), index == task_index
                    )
            # LoRA routes share these exact Parameters. Set the base last so
            # Task 0 remains trainable in a from-scratch protocol, while later
            # tasks keep it frozen and expose only their selected deltas.
            base_is_active = task_index == 0
            self.rssm.recurrent.requires_grad_(base_is_active)
            self.rssm.representation.requires_grad_(base_is_active)
            self.rssm.transition.requires_grad_(base_is_active)
        else:
            for index in range(self.rssm.num_task_experts):
                is_active = index == task_index
                self.rssm.recurrent_for(index).requires_grad_(is_active)
                self.rssm.representation_for(index).requires_grad_(is_active)
                self.rssm.transition_for(index).requires_grad_(is_active)
        if self.task_shared_prediction_heads:
            base_head_is_active = (
                not self.freeze_shared_prediction_heads_after_task0
                or task_index == 0
            )
            self.reward_fc.requires_grad_(base_head_is_active)
            self.continue_fc.requires_grad_(base_head_is_active)
            if self.observation_objective == "reconstruction":
                self.decoder.requires_grad_(base_head_is_active)
            else:
                self.feature_predictor.requires_grad_(base_head_is_active)
            if self.task_private_prediction_adapters:
                for head_name in ("observation", "reward", "continue"):
                    for adapter_index, adapter in enumerate(
                        self.prediction_adapters[head_name], start=1
                    ):
                        adapter.requires_grad_(adapter_index == task_index)
        else:
            for index in range(self.rssm.num_task_experts):
                is_active = index == task_index
                self._head_for(
                    self.reward_fc, self.reward_experts, index
                ).requires_grad_(is_active)
                self._head_for(
                    self.continue_fc, self.continue_experts, index
                ).requires_grad_(is_active)
                if self.observation_objective == "reconstruction":
                    self.decoder_for(index).requires_grad_(is_active)
                else:
                    self.feature_predictor_for(index).requires_grad_(is_active)
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
        if (
            self.task_shared_prediction_heads
            and not self.freeze_shared_prediction_heads_after_task0
        ):
            observation_head = (
                self.decoder
                if self.observation_objective == "reconstruction"
                else self.feature_predictor
            )
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
        if self.task_private_prediction_adapters:
            if task_index == 0:
                parameters.extend(self.reward_fc.parameters())
                parameters.extend(self.continue_fc.parameters())
                observation_head = (
                    self.decoder
                    if self.observation_objective == "reconstruction"
                    else self.feature_predictor
                )
                parameters.extend(observation_head.parameters())
            else:
                for head_name in ("observation", "reward", "continue"):
                    adapter = self.prediction_adapter_for(head_name, task_index)
                    if adapter is None:
                        raise RuntimeError(
                            "A later task is missing its private prediction adapter"
                        )
                    parameters.extend(adapter.parameters())
        elif not self.task_shared_prediction_heads:
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
            if self.task_private_heads:
                observation_head = (
                    self.decoder_for(task_index)
                    if self.observation_objective == "reconstruction"
                    else self.feature_predictor_for(task_index)
                )
                parameters.extend(observation_head.parameters())
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
        if (
            self.task_shared_prediction_heads
            and not self.freeze_shared_prediction_heads_after_task0
        ):
            observation_head = (
                self.decoder
                if self.observation_objective == "reconstruction"
                else self.feature_predictor
            )
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
        if (
            self.task_shared_prediction_heads
            and not self.freeze_shared_prediction_heads_after_task0
        ):
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
        if (
            self.task_shared_prediction_heads
            and not self.freeze_shared_prediction_heads_after_task0
        ):
            observation_head = (
                self.decoder
                if self.observation_objective == "reconstruction"
                else self.feature_predictor
            )
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
        if self.reward_residual is not None:
            prediction = prediction + self.reward_residual(head_features)
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
        if self.continue_residual is not None:
            logits = logits + self.continue_residual(head_features)
        return logits

    def predict_observation(
        self,
        model_state: torch.Tensor,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        head_features = self.prediction_features_for(
            "observation", model_state, task_id
        )
        if self.observation_objective == "reconstruction":
            return self.decoder_for(task_id)(head_features)
        if hasattr(self, "feature_predictor"):
            prediction = self.feature_predictor_for(task_id)(head_features)
            if self.feature_predictor_residual is not None:
                prediction = prediction + self.feature_predictor_residual(
                    head_features
                )
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

    def freeze_shared_core(self) -> None:
        """Freeze base functions while leaving residual adapters trainable."""
        if self.residual_correction == "none":
            raise ValueError("Frozen shared core requires plastic residual adapters")
        self.rssm.freeze_shared_core()
        self.reward_fc.requires_grad_(False)
        self.continue_fc.requires_grad_(False)
        for reward_head in self.reward_experts:
            reward_head.requires_grad_(False)
        for continue_head in self.continue_experts:
            continue_head.requires_grad_(False)
        if hasattr(self, "decoder"):
            self.decoder.requires_grad_(False)
        for decoder in self.decoder_experts:
            decoder.requires_grad_(False)
        if hasattr(self, "r2_projector"):
            self.r2_projector.requires_grad_(False)
        if hasattr(self, "feature_predictor"):
            self.feature_predictor.requires_grad_(False)
        for feature_predictor in self.feature_predictor_experts:
            feature_predictor.requires_grad_(False)
        if self.feature_predictor_residual is not None:
            self.feature_predictor_residual.requires_grad_(True)
        if self.reward_residual is not None:
            self.reward_residual.requires_grad_(True)
        if self.continue_residual is not None:
            self.continue_residual.requires_grad_(True)

    def consolidation_penalty(self) -> torch.Tensor:
        if self.residual_correction != "kan":
            return torch.zeros((), device=next(self.parameters()).device)
        candidates = (
            self.rssm.recurrent.residual,
            self.rssm.representation.residual,
            self.rssm.transition.residual,
            self.reward_residual,
            self.continue_residual,
            self.feature_predictor_residual,
        )
        residuals = tuple(residual for residual in candidates if residual is not None)
        if not residuals:
            raise RuntimeError("KAN world model is missing residual corrections")
        return torch.stack(
            [residual.consolidation_penalty() for residual in residuals]
        ).sum()

    def forward(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        observation_features: Optional[torch.Tensor] = None,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Expose the training loss through `forward` for native PyTorch DDP."""
        return self.compute_loss(
            actions,
            xs,
            rews,
            conts,
            resets,
            observation_features=observation_features,
            task_id=task_id,
        )

    def compute_loss(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        observation_features: Optional[torch.Tensor] = None,
        task_id: Optional[int | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics, _trace = self._compute_loss_and_trace(
            actions,
            xs,
            rews,
            conts,
            resets,
            observation_features=observation_features,
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
        observation_features: Optional[torch.Tensor] = None,
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
            observation_features=observation_features,
            task_id=task_id,
        )

    def _compute_loss_and_trace(
        self,
        actions: ActionT,
        xs: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        observation_features: Optional[torch.Tensor] = None,
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
        if observation_features is not None or self.observation_objective in {
            "r2",
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        }:
            raw_or_adapted = (
                self.rssm.embed_observations(xs, task_id=task_id)
                if observation_features is None
                else observation_features
            )
            if raw_or_adapted.shape[:2] != actions.shape[:2]:
                raise ValueError(
                    "Observation features and actions must share time/batch axes"
                )
            if observation_features is not None:
                embeddings = self.rssm.adapt_observation_embeddings(
                    observation_features.detach(), task_id=task_id
                )
            else:
                embeddings = raw_or_adapted
            if embeddings.shape[-1] != self.rssm.observation_embedding_size:
                raise ValueError("Adapted observation features have an unexpected width")
            if observation_features is None and self.observation_objective in {
                "dinov3_next_feature",
                "dinov3_posterior_feature",
            }:
                embeddings = embeddings.detach()
            z_posts, z_samples, hiddens = self.rssm.observe_embeddings(
                init_z,
                actions,
                init_h,
                embeddings,
                resets,
                task_id=task_id,
                mechanism_trace=mechanism_trace,
            )
        else:
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
        if self.observation_objective == "reconstruction":
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
        elif self.observation_objective == "r2":
            if embeddings is None:
                raise RuntimeError("R2 observation objective requires encoder embeddings")
            projected = self.r2_projector(zhs_f12)
            observation_prediction = projected
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
            predicted_features = self.predict_observation(prior_states, task_id)
            observation_prediction = predicted_features
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
            predicted_features = self.predict_observation(zhs, task_id)
            observation_prediction = predicted_features
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
                **reward_learning_diagnostics(rews_pred, rews),
            }
            if self.observation_objective == "reconstruction":
                metrics["Metric/low_kl_recon_loss"] = masked_mean(
                    observation_losses, low_kl
                )

        consolidation_loss = torch.zeros((), device=z_repr_loss.device)
        if self.residual_correction == "kan":
            consolidation_loss = self.consolidation_penalty()
            metrics["Loss/kan_consolidation"] = consolidation_loss.detach()

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
