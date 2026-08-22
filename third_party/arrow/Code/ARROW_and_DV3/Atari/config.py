import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Type, TypeVar, Union

import gymnasium as gym
from gymnasium.wrappers import TransformReward

import generate_trajectory
import replay
from generate_trajectory import EnvironmentSchedule, SequentialEnvironments
from replay import FifoReplay, LongTermReplay, MultiTypeReplay, Replay

T = TypeVar("T", bound="Serialisable")

ArrowReplayCapacityRatio = Literal["50-50", "25-75", "75-25"]
ObservationObjective = Literal[
    "reconstruction",
    "r2",
    "dinov3_next_feature",
    "dinov3_posterior_feature",
]
ObservationEncoder = Literal["cnn", "dinov3_vits16"]
DinoV3FeatureMode = Literal["cls", "patch_grid"]
DinoV3FeatureLoss = Literal["cosine", "batch_standardized_smooth_l1"]
DinoV3PatchProjection = Literal["none", "task1_pca", "fixed_orthogonal"]
DinoV3ReplayFeatureMode = Literal["cached", "on_the_fly"]
DinoV3PatchAdapter = Literal["none", "conv_3x3_stride2"]
ContinualMethod = Literal[
    "none",
    "moe_arrow",
    "dino_fullbank_arrow",
    "dino_patchbank_arrow",
    "dino_convbank_arrow",
]
ResidualCorrection = Literal["none", "mlp", "kan"]
ResidualInputMode = Literal["base_output", "module_input"]
ResidualConsolidation = Literal["none", "replay_functional"]
SharedCoreMode = Literal[
    "trainable",
    "freeze_after_first_task",
    "snapshot_adaptation",
    "task_isolated",
    "task_banked_shared_adapter",
]
ActorNetwork = Literal[
    "mlp",
    "relu_kan",
    "relu_kan_bounded",
    "relu_kan_adaptive",
    "fast_kan_ac",
    "fast_kan_ac_param_matched",
    "fast_kan_ac_stable",
]
ActorCriticOptimizer = Literal["adam", "laprop"]


def _arrow_fifo_ltdm_capacity_ns(
    total_slots: int, ratio: ArrowReplayCapacityRatio
) -> tuple[int, int]:
    """Split total trajectory slots between FIFO and LTDM (FIFO share listed first)."""
    if ratio == "50-50":
        n_fifo = total_slots // 2
        n_ltdm = total_slots - n_fifo
    elif ratio == "25-75":
        n_fifo = total_slots // 4
        n_ltdm = total_slots - n_fifo
    elif ratio == "75-25":
        n_ltdm = total_slots // 4
        n_fifo = total_slots - n_ltdm
    else:
        raise AssertionError(ratio)
    return n_fifo, n_ltdm


def _arrow_fifo_ltdm_sampling_weights(
    ratio: ArrowReplayCapacityRatio,
) -> tuple[float, float]:
    """Minibatch sampling weights (FIFO, LTDM) matching --arrow-replay-ratio / capacity split."""
    if ratio == "50-50":
        return 0.5, 0.5
    if ratio == "25-75":
        return 0.25, 0.75
    if ratio == "75-25":
        return 0.75, 0.25
    raise AssertionError(ratio)


@dataclass
class Serialisable:
    @classmethod
    def from_file(cls: Type[T], path: Path) -> T:
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        return cls(**data)

    def save(self, path: Path) -> None:
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvConfig(Serialisable):
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    rew_scale: float = 1

    def get_function(self) -> Callable[[], Any]:
        return lambda: TransformReward(
            gym.make(
                self.name,
                frameskip=1,
                repeat_action_probability=0,
                full_action_space=True,
                **self.kwargs,
            ),
            lambda x: self.rew_scale * x,
        )


@dataclass
class EnvScheduleConfig(Serialisable):
    env_schedule_type: Type[EnvironmentSchedule]
    env_configs: list[EnvConfig]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        data = data.copy()
        data["env_schedule_type"] = getattr(generate_trajectory, data["env_schedule_type"])
        data["env_configs"] = [EnvConfig.from_dict(d) for d in data["env_configs"]]
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.env_schedule_type != EnvironmentSchedule
        assert len(self.env_configs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["env_schedule_type"] = self.env_schedule_type.__name__
        data["env_configs"] = [c.to_dict() for c in self.env_configs]
        return data


@dataclass
class RbConfig(Serialisable):
    rb_type: Union[Type[FifoReplay], Type[LongTermReplay]]
    rb_device: str = "cuda"

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        data = data.copy()
        data["rb_type"] = getattr(replay, data["rb_type"])
        return cls(**data)

    def __post_init__(self) -> None:
        assert self.rb_type in {FifoReplay, LongTermReplay}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rb_type"] = self.rb_type.__name__
        return data


@dataclass
class Config(Serialisable):
    esc: EnvScheduleConfig
    algorithm: Literal["dv3", "arrow", "sac"] = "dv3"

    # Present in every published Atari config, including ARROW and DV3.
    img_size: int = 64
    sac_lr: float = 3e-4
    sac_batch_size: int = 256
    sac_dv3_data_n_max: int = 1024
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    sac_alpha: float = 0.2

    seed: int = 1337

    epochs: int = 10_000
    wm_lr: float = 4e-4
    log_frequency: int = 800
    steps_per_batch: int = 1600
    ac_train_steps: int = 800
    ac_train_sync: int = 128
    # False = do not create fresh ac
    # True = create fresh ac every epoch
    # int = create fresh ac every n epochs
    fresh_ac: Union[bool, int] = False

    continual_method: ContinualMethod = "none"
    rssm_num_experts: int = 1
    moe_arrow_current_task_fraction: float = 0.5
    dino_fullbank_current_task_fraction: float = 1.0

    n_sync: int = 2
    gen_seq_len: int = 4096
    env_repeat: int = 4
    data_n: int = 16
    data_n_max: int = 512
    data_t: int = 512

    mb_t_size: int = 32
    mb_n_size: int = 16

    random_policy: Union[Literal["first"], Literal["new"]] = "first"

    pretrain_enabled: bool = True
    pretrain_data_multiplier: int = 4
    pretrain_mb_t_size: int = 8
    pretrain_mb_n_size: int = 16
    pretrain_steps: int = 32_000

    gru_units: int = 512
    cnn_depth: int = 32
    mlp_features: int = 512
    mlp_layers: int = 2
    wall_time_optimisation: bool = False

    actor_network: ActorNetwork = "mlp"
    actor_kan_hidden_features: int = 64
    actor_kan_grid_size: int = 5
    actor_kan_spline_order: int = 3
    actor_kan_input_min: float = 0.0
    actor_kan_input_max: float = 1.0
    actor_kan_trainable_grid: bool = False
    actor_kan_normalize_recurrent_state: bool = True
    fastkan_hidden_features: int = 34
    fastkan_hidden_layers: int = 3
    fastkan_grid_size: int = 8
    fastkan_input_min: float = -2.0
    fastkan_input_max: float = 2.0
    fastkan_rms_norm_epsilon: float = 1e-4
    fastkan_actor_output_scale: float = 0.01
    fastkan_actor_unimix: float = 0.01

    ac_optimizer: ActorCriticOptimizer = "adam"
    ac_lr: float = 1e-4
    ac_fresh_lr: float = 4e-4
    ac_optimizer_eps: float = 1e-8
    ac_optimizer_beta1: float = 0.9
    ac_optimizer_beta2: float = 0.999
    ac_optimizer_warmup_steps: int = 0
    ac_agc_clip: float = 0.0
    ac_grad_clip: float = 100.0
    ac_dream_steps: int = 16
    ac_discount: float = 0.997
    ac_lambda: float = 0.95
    ac_entropy_scale: float = 3e-4
    ac_return_norm_decay: float = 0.99
    ac_persistent_return_norm: bool = False
    ac_slow_critic_regularizer: float = 0.0
    ac_slow_critic_decay: float = 0.98
    ac_replay_critic_loss_scale: float = 0.0
    ac_use_slow_critic_targets: bool = False
    ac_corrected_imagination_bootstrap: bool = False

    observation_objective: ObservationObjective = "reconstruction"
    r2_barlow_loss_scale: float = 0.05
    r2_redundancy_scale: float = 5e-4
    r2_normalization_eps: float = 1e-8
    observation_encoder: ObservationEncoder = "cnn"
    dinov3_model_path: Optional[str] = None
    dinov3_input_size: int = 256
    dinov3_max_batch_size: int = 128
    dinov3_feature_cache_dtype: Literal["float16", "float32"] = "float16"
    dinov3_replay_feature_mode: DinoV3ReplayFeatureMode = "cached"
    dinov3_feature_loss_scale: float = 1.0
    dinov3_feature_mode: DinoV3FeatureMode = "cls"
    dinov3_patch_pool_size: int = 4
    dinov3_patch_feature_dim: int = 384
    dinov3_patch_projection: DinoV3PatchProjection = "none"
    dinov3_patch_projection_frames: int = 0
    dinov3_patch_projection_seed: int = 0
    dinov3_patch_adapter: DinoV3PatchAdapter = "none"
    dinov3_feature_loss_kind: DinoV3FeatureLoss = "cosine"
    dinov3_feature_std_floor: float = 0.05

    residual_correction: ResidualCorrection = "none"
    residual_bottleneck_features: int = 64
    residual_grid_size: int = 8
    residual_input_min: float = -2.0
    residual_input_max: float = 2.0
    residual_rms_norm_epsilon: float = 1e-4
    residual_alpha: float = 0.1
    residual_input_mode: ResidualInputMode = "base_output"
    residual_consolidation: ResidualConsolidation = "none"
    residual_consolidation_batches: int = 16
    residual_consolidation_imagination_horizon: int = 8
    residual_consolidation_gradient_power: float = 2.0
    residual_consolidation_min_plasticity: float = 0.01
    residual_consolidation_anchor_loss_scale: float = 1.0
    shared_core_mode: SharedCoreMode = "trainable"

    action_space: int = 18
    replay_buffers: list[RbConfig] = field(default_factory=list)
    # ARROW only: split of total capacity 2 * data_n_max between FifoReplay vs LongTermReplay
    arrow_replay_capacity_ratio: ArrowReplayCapacityRatio = "50-50"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data.copy()
        data["esc"] = EnvScheduleConfig.from_dict(data["esc"])
        data["replay_buffers"] = [RbConfig.from_dict(d) for d in data["replay_buffers"]]
        return cls(**data)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        assert self.n_sync * self.gen_seq_len == self.data_n * self.data_t
        assert self.random_policy in {"first", "new"}
        assert self.replay_buffers != []
        if self.continual_method not in {
            "none",
            "moe_arrow",
            "dino_fullbank_arrow",
            "dino_patchbank_arrow",
            "dino_convbank_arrow",
        }:
            raise ValueError(f"Unknown continual method: {self.continual_method!r}")
        is_moe_arrow = self.continual_method == "moe_arrow"
        is_dino_fullbank = self.continual_method == "dino_fullbank_arrow"
        is_dino_patchbank = self.continual_method == "dino_patchbank_arrow"
        is_dino_convbank = self.continual_method == "dino_convbank_arrow"
        is_dino_pixelbank = is_dino_patchbank or is_dino_convbank
        uses_task_experts = is_moe_arrow or is_dino_fullbank or is_dino_pixelbank
        if uses_task_experts:
            if self.algorithm != "arrow":
                raise ValueError("Task-aware expert methods require ARROW mixed replay")
            if self.esc.env_schedule_type is not SequentialEnvironments:
                raise ValueError("Task-aware expert methods require a sequential task schedule")
            if len(self.esc.env_configs) < 2:
                raise ValueError("Task-aware expert methods require at least two scheduled tasks")
            if self.rssm_num_experts != len(self.esc.env_configs):
                raise ValueError(
                    "Task-aware expert methods require one RSSM expert per scheduled task"
                )
            if self.residual_correction != "none":
                raise ValueError(
                    "A task-aware expert method does not use residual corrections"
                )
        else:
            if self.rssm_num_experts != 1:
                raise ValueError(
                    "RSSM experts require a task-aware continual_method"
                )

        if is_moe_arrow:
            if not 0 < self.moe_arrow_current_task_fraction < 1:
                raise ValueError(
                    "MoE-ARROW current-task update fraction must lie in (0, 1)"
                )
            if self.shared_core_mode != "trainable":
                raise ValueError("MoE-ARROW keeps shared modules trainable")
            if self.observation_objective != "dinov3_next_feature":
                raise ValueError(
                    "MoE-ARROW predicts frozen DINOv3 features from the RSSM prior"
                )
        elif is_dino_fullbank or is_dino_pixelbank:
            if self.dino_fullbank_current_task_fraction != 1.0:
                raise ValueError(
                    "DINO task banks assign all updates to the current task"
                )
            expected_shared_core_mode = (
                "task_banked_shared_adapter"
                if is_dino_convbank
                else "task_isolated"
            )
            if self.shared_core_mode != expected_shared_core_mode:
                raise ValueError(
                    "The selected DINO task bank requires shared_core_mode="
                    f"'{expected_shared_core_mode}'"
                )
            expected_objective = (
                "reconstruction"
                if is_dino_pixelbank
                else "dinov3_posterior_feature"
            )
            if self.observation_objective != expected_objective:
                raise ValueError(
                    (
                        "DINO patch task banks keep DreamerV3 pixel reconstruction"
                        if is_dino_pixelbank
                        else "DINO-FullBank-ARROW reconstructs posterior DINOv3 features"
                    )
                )
            if self.random_policy != "new":
                raise ValueError(
                    "DINO task banks require a random collection for each new task"
                )
            if is_dino_pixelbank and any(
                replay_config.rb_device.split(":", 1)[0] != "cpu"
                for replay_config in self.replay_buffers
            ):
                raise ValueError(
                    "DINO patch task banks require CPU-addressable mapped observation replay"
                )
        if self.observation_objective not in {
            "reconstruction",
            "r2",
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        }:
            raise ValueError(
                f"Unknown observation objective: {self.observation_objective!r}"
            )
        if self.r2_barlow_loss_scale <= 0:
            raise ValueError("r2_barlow_loss_scale must be positive")
        if self.r2_redundancy_scale < 0:
            raise ValueError("r2_redundancy_scale must be non-negative")
        if self.r2_normalization_eps <= 0:
            raise ValueError("r2_normalization_eps must be positive")
        if self.observation_encoder not in {"cnn", "dinov3_vits16"}:
            raise ValueError(f"Unknown observation encoder: {self.observation_encoder!r}")
        uses_dinov3_objective = self.observation_objective in {
            "dinov3_next_feature",
            "dinov3_posterior_feature",
        }
        uses_dinov3 = self.observation_encoder == "dinov3_vits16"
        if uses_dinov3:
            if self.algorithm != "arrow":
                raise ValueError("KARROW Frozen-Core requires ARROW mixed replay")
            if self.observation_encoder != "dinov3_vits16":
                raise ValueError("DINOv3 feature prediction requires dinov3_vits16")
            if self.dinov3_model_path is None:
                raise ValueError("DINOv3 feature prediction requires dinov3_model_path")
            if not Path(self.dinov3_model_path).is_absolute():
                raise ValueError("dinov3_model_path must be absolute")
            if self.dinov3_input_size != 256:
                raise ValueError("KARROW Frozen-Core fixes DINOv3 input size at 256")
            if self.dinov3_max_batch_size < 1:
                raise ValueError("dinov3_max_batch_size must be positive")
            if self.dinov3_feature_cache_dtype not in {"float16", "float32"}:
                raise ValueError("Unknown DINOv3 feature cache dtype")
            if self.dinov3_replay_feature_mode not in {"cached", "on_the_fly"}:
                raise ValueError("Unknown DINOv3 replay feature mode")
            if is_dino_pixelbank:
                if self.dinov3_replay_feature_mode != "on_the_fly":
                    raise ValueError(
                        "A DINO patch task bank recomputes frozen DINOv3 patches "
                        "from sampled replay observations"
                    )
            elif self.dinov3_replay_feature_mode != "cached":
                raise ValueError(
                    "Only DINO patch task banks support on-the-fly replay features"
                )
            if self.dinov3_feature_loss_scale <= 0:
                raise ValueError("dinov3_feature_loss_scale must be positive")
            if self.dinov3_feature_mode not in {"cls", "patch_grid"}:
                raise ValueError("Unknown DINOv3 feature mode")
            if self.dinov3_patch_pool_size < 1:
                raise ValueError("dinov3_patch_pool_size must be positive")
            if not 1 <= self.dinov3_patch_feature_dim <= 384:
                raise ValueError("dinov3_patch_feature_dim must be in [1, 384]")
            if self.dinov3_patch_projection not in {
                "none",
                "task1_pca",
                "fixed_orthogonal",
            }:
                raise ValueError("Unknown DINOv3 patch projection")
            if self.dinov3_patch_projection_frames < 0:
                raise ValueError("dinov3_patch_projection_frames must be non-negative")
            if self.dinov3_patch_projection_seed < 0:
                raise ValueError("dinov3_patch_projection_seed must be non-negative")
            if self.dinov3_patch_adapter not in {
                "none",
                "conv_3x3_stride2",
            }:
                raise ValueError("Unknown DINOv3 patch adapter")
            if is_dino_convbank:
                if self.dinov3_patch_adapter != "conv_3x3_stride2":
                    raise ValueError(
                        "DINO-ConvBank-ARROW requires the shared 3x3 stride-2 adapter"
                    )
            elif self.dinov3_patch_adapter != "none":
                raise ValueError(
                    "Only DINO-ConvBank-ARROW uses a trainable patch adapter"
                )
            if self.dinov3_feature_loss_kind not in {
                "cosine",
                "batch_standardized_smooth_l1",
            }:
                raise ValueError("Unknown DINOv3 feature loss")
            if self.dinov3_feature_std_floor <= 0:
                raise ValueError("dinov3_feature_std_floor must be positive")
            if self.observation_objective == "dinov3_next_feature":
                if is_moe_arrow:
                    if self.dinov3_feature_mode != "patch_grid":
                        raise ValueError(
                            "MoE-ARROW requires pooled DINOv3 patch features"
                        )
                    if self.dinov3_patch_pool_size != 4:
                        raise ValueError("MoE-ARROW fixes a 4x4 DINOv3 patch grid")
                    if self.dinov3_patch_feature_dim != 64:
                        raise ValueError(
                            "MoE-ARROW fixes each projected patch at 64 dimensions"
                        )
                    if self.dinov3_patch_projection != "fixed_orthogonal":
                        raise ValueError(
                            "MoE-ARROW requires a task-independent fixed projection"
                        )
                    if self.dinov3_patch_projection_frames != 0:
                        raise ValueError("MoE-ARROW never fits a Task-1 projection")
                    if self.dinov3_feature_loss_kind != "cosine":
                        raise ValueError("MoE-ARROW fixes the prior feature loss to cosine")
                else:
                    if self.dinov3_feature_mode != "cls":
                        raise ValueError("KARROW v1 fixes DINOv3 output to the CLS token")
                    if self.dinov3_feature_loss_kind != "cosine":
                        raise ValueError("KARROW v1 fixes the feature loss to cosine")
                    if self.dinov3_patch_feature_dim != 384:
                        raise ValueError(
                            "KARROW v1 does not use projected patch features"
                        )
                    if (
                        self.dinov3_patch_projection != "none"
                        or self.dinov3_patch_projection_frames != 0
                        or self.dinov3_patch_projection_seed != 0
                    ):
                        raise ValueError("KARROW v1 does not fit a patch projection")
            elif self.observation_objective == "dinov3_posterior_feature":
                if self.dinov3_feature_mode != "patch_grid":
                    if is_dino_fullbank:
                        raise ValueError(
                            "DINO-FullBank-ARROW requires pooled DINOv3 patch tokens"
                        )
                    raise ValueError(
                        "KARROW spatial v2 requires pooled DINOv3 patch tokens"
                    )
                if self.dinov3_patch_pool_size != 4:
                    raise ValueError("Posterior DINOv3 objectives fix a 4x4 patch grid")
                if self.dinov3_patch_feature_dim != 64:
                    raise ValueError(
                        "Posterior DINOv3 objectives fix each patch feature at 64 dimensions"
                    )
                if is_dino_fullbank:
                    if self.dinov3_patch_projection != "fixed_orthogonal":
                        raise ValueError(
                            "DINO-FullBank-ARROW requires a task-independent fixed projection"
                        )
                    if self.dinov3_patch_projection_frames != 0:
                        raise ValueError(
                            "DINO-FullBank-ARROW never fits a task-specific projection"
                        )
                else:
                    if self.dinov3_patch_projection != "task1_pca":
                        raise ValueError(
                            "KARROW spatial v2 learns a Task-1 PCA patch projection"
                        )
                    if self.dinov3_patch_projection_frames != 512:
                        raise ValueError(
                            "KARROW spatial v2 fits PCA on 512 initial Task-1 frames"
                        )
                    if self.random_policy != "first":
                        raise ValueError(
                            "KARROW spatial v2 requires an initial random Task-1 collection"
                        )
                if (
                    self.dinov3_feature_loss_kind
                    != "batch_standardized_smooth_l1"
                ):
                    raise ValueError(
                        "Posterior DINOv3 objectives require batch-standardized SmoothL1"
                    )
            elif self.observation_objective == "reconstruction":
                if not is_dino_pixelbank:
                    raise ValueError(
                        "DINOv3 pixel reconstruction is reserved for "
                        "named DINO patch task banks"
                    )
                if self.dinov3_feature_mode != "patch_grid":
                    raise ValueError(
                        "DINO patch task banks require spatial DINOv3 patch tokens"
                    )
                if self.dinov3_patch_pool_size != 16:
                    raise ValueError(
                        "DINO patch task banks retain the complete 16x16 patch grid"
                    )
                if self.dinov3_patch_feature_dim != 384:
                    raise ValueError(
                        "DINO patch task banks retain all 384 patch channels"
                    )
                if self.dinov3_patch_projection != "none":
                    raise ValueError(
                        "DINO patch task banks do not project frozen patch features"
                    )
                if self.dinov3_patch_projection_frames != 0:
                    raise ValueError(
                        "DINO patch task banks do not fit a patch projection"
                    )
                if self.dinov3_patch_projection_seed != 0:
                    raise ValueError(
                        "DINO patch task banks have no patch-projection RNG"
                    )
            else:
                raise ValueError(
                    "The DINOv3 encoder does not support the configured observation "
                    f"objective: {self.observation_objective!r}"
                )
            if self.actor_network != "mlp":
                raise ValueError("KARROW keeps the original MLP actor and critic")
            if self.fresh_ac is not False:
                if uses_task_experts:
                    raise ValueError(
                        "Task-aware expert methods manage persistent per-task actor-critics"
                    )
                raise ValueError("KARROW requires one persistent actor-critic across tasks")
        elif (
            uses_dinov3_objective
            or self.observation_encoder != "cnn"
            or self.dinov3_model_path is not None
            or self.dinov3_feature_mode != "cls"
            or self.dinov3_patch_feature_dim != 384
            or self.dinov3_patch_projection != "none"
            or self.dinov3_patch_projection_frames != 0
            or self.dinov3_patch_projection_seed != 0
            or self.dinov3_patch_adapter != "none"
            or self.dinov3_feature_loss_kind != "cosine"
        ):
            raise ValueError(
                "Only named DINOv3 protocols may configure the DINOv3 encoder"
            )

        if self.residual_correction not in {"none", "mlp", "kan"}:
            raise ValueError(
                f"Unknown residual correction: {self.residual_correction!r}"
            )
        if self.residual_input_mode not in {"base_output", "module_input"}:
            raise ValueError(
                f"Unknown residual input mode: {self.residual_input_mode!r}"
            )
        if self.residual_consolidation not in {"none", "replay_functional"}:
            raise ValueError(
                f"Unknown residual consolidation: {self.residual_consolidation!r}"
            )
        if self.shared_core_mode not in {
            "trainable",
            "freeze_after_first_task",
            "snapshot_adaptation",
            "task_isolated",
            "task_banked_shared_adapter",
        }:
            raise ValueError(f"Unknown shared core mode: {self.shared_core_mode!r}")
        if self.shared_core_mode == "task_isolated" and not (
            is_dino_fullbank or is_dino_patchbank
        ):
            raise ValueError(
                "shared_core_mode='task_isolated' is reserved for DINO task banks"
            )
        if (
            self.shared_core_mode == "task_banked_shared_adapter"
            and not is_dino_convbank
        ):
            raise ValueError(
                "shared_core_mode='task_banked_shared_adapter' is reserved for "
                "DINO-ConvBank-ARROW"
            )
        if self.residual_correction != "none" and not uses_dinov3:
            raise ValueError("KARROW residuals require the frozen DINOv3 protocol")
        if (
            self.residual_correction != "none"
            and self.shared_core_mode
            not in {"freeze_after_first_task", "snapshot_adaptation"}
        ):
            raise ValueError(
                "KARROW residuals require shared_core_mode="
                "freeze_after_first_task or snapshot_adaptation"
            )
        if self.shared_core_mode == "freeze_after_first_task":
            if self.residual_correction == "none":
                raise ValueError(
                    "Frozen shared core requires a plastic residual correction"
                )
            if not uses_dinov3:
                raise ValueError(
                    "Frozen shared core is only defined for the DINOv3 protocol"
                )
            if self.fresh_ac is not False:
                raise ValueError(
                    "Frozen shared core requires one persistent actor-critic"
                )
            if self.esc.env_schedule_type is not SequentialEnvironments:
                raise ValueError(
                    "Frozen shared core requires a sequential task schedule"
                )
            if len(self.esc.env_configs) < 2:
                raise ValueError(
                    "Frozen shared core requires at least two scheduled tasks"
                )
        if self.shared_core_mode == "snapshot_adaptation":
            if self.residual_correction == "none":
                raise ValueError(
                    "Snapshot adaptation requires a plastic residual correction"
                )
            if not uses_dinov3:
                raise ValueError(
                    "Snapshot adaptation is only defined for the DINOv3 protocol"
                )
            if self.fresh_ac is not False:
                raise ValueError(
                    "Snapshot adaptation requires one persistent actor-critic"
                )
            if self.esc.env_schedule_type is not SequentialEnvironments:
                raise ValueError(
                    "Snapshot adaptation requires a sequential task schedule"
                )
        if self.residual_bottleneck_features != 64:
            raise ValueError("KARROW Frozen-Core fixes the residual bottleneck at 64")
        if self.residual_grid_size != 8:
            raise ValueError("KARROW Frozen-Core fixes eight Gaussian basis centers")
        if self.residual_input_min != -2.0 or self.residual_input_max != 2.0:
            raise ValueError("KARROW Frozen-Core fixes the residual basis range at [-2, 2]")
        if self.residual_rms_norm_epsilon != 1e-4:
            raise ValueError("KARROW Frozen-Core fixes residual RMSNorm epsilon at 1e-4")
        if self.residual_alpha != 0.1:
            raise ValueError("KARROW Frozen-Core fixes residual alpha at 0.1")
        consolidation_defaults = (
            16,
            8,
            2.0,
            0.01,
            1.0,
        )
        consolidation_values = (
            self.residual_consolidation_batches,
            self.residual_consolidation_imagination_horizon,
            self.residual_consolidation_gradient_power,
            self.residual_consolidation_min_plasticity,
            self.residual_consolidation_anchor_loss_scale,
        )
        if self.residual_consolidation == "none":
            if consolidation_values != consolidation_defaults:
                raise ValueError(
                    "Residual consolidation settings require "
                    "residual_consolidation='replay_functional'"
                )
        else:
            if self.residual_correction != "kan":
                raise ValueError("Replay consolidation requires KAN residuals")
            if self.shared_core_mode != "freeze_after_first_task":
                raise ValueError("Replay consolidation requires a frozen shared core")
            if self.residual_consolidation_batches < 1:
                raise ValueError("Residual consolidation batches must be positive")
            if self.residual_consolidation_imagination_horizon < 0:
                raise ValueError(
                    "Residual consolidation imagination horizon must be non-negative"
                )
            if self.residual_consolidation_gradient_power <= 0:
                raise ValueError(
                    "Residual consolidation gradient power must be positive"
                )
            if not 0 <= self.residual_consolidation_min_plasticity <= 1:
                raise ValueError(
                    "Residual consolidation minimum plasticity must lie in [0, 1]"
                )
            if self.residual_consolidation_anchor_loss_scale < 0:
                raise ValueError(
                    "Residual consolidation anchor loss scale must be non-negative"
                )
        if self.actor_network not in {
            "mlp",
            "relu_kan",
            "relu_kan_bounded",
            "relu_kan_adaptive",
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
            "fast_kan_ac_stable",
        }:
            raise ValueError(f"Unknown actor network: {self.actor_network!r}")
        if self.actor_kan_hidden_features < 1:
            raise ValueError("actor_kan_hidden_features must be positive")
        if self.actor_kan_grid_size < 1:
            raise ValueError("actor_kan_grid_size must be positive")
        if self.actor_kan_spline_order < 0:
            raise ValueError("actor_kan_spline_order must be non-negative")
        if (
            self.actor_kan_input_min != 0.0
            or self.actor_kan_input_max != 1.0
        ):
            raise ValueError("The first KAN-Actor protocol requires a [0, 1] grid")
        expected_trainable_grid = self.actor_network == "relu_kan_adaptive"
        if self.actor_kan_trainable_grid != expected_trainable_grid:
            if expected_trainable_grid:
                raise ValueError(
                    "relu_kan_adaptive requires actor_kan_trainable_grid=True"
                )
            raise ValueError(
                "Only relu_kan_adaptive may enable actor_kan_trainable_grid"
            )
        if not self.actor_kan_normalize_recurrent_state:
            raise ValueError(
                "The [0, 1] KAN-Actor input domain requires recurrent-state normalization"
            )
        if self.actor_network in {
            "relu_kan",
            "relu_kan_bounded",
            "relu_kan_adaptive",
        }:
            basis_count = self.actor_kan_grid_size + self.actor_kan_spline_order
            if self.actor_kan_hidden_features * basis_count != self.mlp_features:
                raise ValueError(
                    "KAN-Actor coefficient matching requires "
                    "actor_kan_hidden_features * "
                    "(actor_kan_grid_size + actor_kan_spline_order) == "
                    f"mlp_features, got {self.actor_kan_hidden_features} * "
                    f"{basis_count} != {self.mlp_features}"
                )
        if self.fastkan_hidden_features < 1 or self.fastkan_hidden_layers < 1:
            raise ValueError("FastKAN hidden dimensions must be positive")
        if self.fastkan_grid_size < 2:
            raise ValueError("FastKAN grid size must be at least 2")
        if self.fastkan_input_max <= self.fastkan_input_min:
            raise ValueError("FastKAN input maximum must exceed its minimum")
        if self.fastkan_rms_norm_epsilon <= 0:
            raise ValueError("FastKAN RMSNorm epsilon must be positive")
        if self.fastkan_actor_output_scale < 0:
            raise ValueError("FastKAN actor output scale must be non-negative")
        if not 0 <= self.fastkan_actor_unimix < 1:
            raise ValueError("FastKAN actor unimix must lie in [0, 1)")
        if self.ac_optimizer not in {"adam", "laprop"}:
            raise ValueError(f"Unknown actor-critic optimizer: {self.ac_optimizer!r}")
        if self.ac_lr <= 0 or self.ac_fresh_lr <= 0:
            raise ValueError("Actor-critic learning rates must be positive")
        if self.ac_optimizer_eps <= 0:
            raise ValueError("Actor-critic optimizer epsilon must be positive")
        if not 0 <= self.ac_optimizer_beta1 < 1 or not 0 <= self.ac_optimizer_beta2 < 1:
            raise ValueError("Actor-critic optimizer betas must lie in [0, 1)")
        if self.ac_optimizer_warmup_steps < 0 or self.ac_agc_clip < 0:
            raise ValueError("Actor-critic warmup and AGC clip must be non-negative")
        if self.ac_grad_clip < 0 or self.ac_dream_steps < 1:
            raise ValueError("Actor-critic clipping and dream steps are invalid")
        if not 0 < self.ac_discount <= 1 or not 0 <= self.ac_lambda <= 1:
            raise ValueError("Actor-critic discount and lambda are invalid")
        if self.ac_entropy_scale < 0 or not 0 <= self.ac_return_norm_decay < 1:
            raise ValueError("Actor-critic entropy scale or return decay is invalid")
        if self.ac_slow_critic_regularizer < 0:
            raise ValueError("Slow-critic regularizer must be non-negative")
        if not 0 <= self.ac_slow_critic_decay < 1:
            raise ValueError("Slow-critic decay must lie in [0, 1)")
        if self.ac_replay_critic_loss_scale < 0:
            raise ValueError("Replay critic loss scale must be non-negative")

        if self.actor_network in {
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
            "fast_kan_ac_stable",
        }:
            paper_aligned = {
                "fastkan_hidden_layers": 3,
                "fastkan_grid_size": 8,
                "fastkan_input_min": -2.0,
                "fastkan_input_max": 2.0,
                "fastkan_rms_norm_epsilon": 1e-4,
                "fastkan_actor_output_scale": 0.01,
                "fastkan_actor_unimix": 0.01,
                "ac_optimizer": "laprop",
                "ac_lr": 4e-5,
                "ac_fresh_lr": 4e-5,
                "ac_optimizer_eps": 1e-20,
                "ac_optimizer_beta1": 0.9,
                "ac_optimizer_beta2": 0.999,
                "ac_optimizer_warmup_steps": 1000,
                "ac_agc_clip": 0.3,
                "ac_grad_clip": 0.0,
                "ac_dream_steps": 15,
                "ac_discount": 1.0 - 1.0 / 333.0,
                "ac_lambda": 0.95,
                "ac_entropy_scale": 3e-4,
                "ac_return_norm_decay": 0.99,
                "ac_persistent_return_norm": True,
                "ac_slow_critic_regularizer": 1.0,
                "ac_slow_critic_decay": 0.98,
            }
            method_aligned = {
                "fast_kan_ac": {
                    "fastkan_hidden_features": 34,
                    "ac_replay_critic_loss_scale": 0.0,
                    "ac_use_slow_critic_targets": False,
                    "ac_corrected_imagination_bootstrap": False,
                },
                "fast_kan_ac_param_matched": {
                    "fastkan_hidden_features": 53,
                    "ac_replay_critic_loss_scale": 0.3,
                    "ac_use_slow_critic_targets": False,
                    "ac_corrected_imagination_bootstrap": False,
                },
                "fast_kan_ac_stable": {
                    "fastkan_hidden_features": 53,
                    "ac_replay_critic_loss_scale": 0.3,
                    "ac_use_slow_critic_targets": True,
                    "ac_corrected_imagination_bootstrap": True,
                },
            }
            paper_aligned.update(method_aligned[self.actor_network])
            mismatches = {
                key: (getattr(self, key), expected)
                for key, expected in paper_aligned.items()
                if getattr(self, key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"{self.actor_network} requires its named KAN-Dreamer-aligned "
                    "FastKAN settings: "
                    f"{mismatches}"
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["esc"] = self.esc.to_dict()
        data["replay_buffers"] = [c.to_dict() for c in self.replay_buffers]
        return data

    @property
    def uses_task_experts(self) -> bool:
        return self.continual_method in {
            "moe_arrow",
            "dino_fullbank_arrow",
            "dino_patchbank_arrow",
            "dino_convbank_arrow",
        }

    @property
    def uses_full_task_experts(self) -> bool:
        return self.continual_method in {
            "dino_fullbank_arrow",
            "dino_patchbank_arrow",
            "dino_convbank_arrow",
        }

    @property
    def task_update_fraction(self) -> float:
        if self.continual_method == "moe_arrow":
            return self.moe_arrow_current_task_fraction
        if self.continual_method in {
            "dino_fullbank_arrow",
            "dino_patchbank_arrow",
            "dino_convbank_arrow",
        }:
            return self.dino_fullbank_current_task_fraction
        raise ValueError("Task update fractions require a task-aware expert method")

    def get_env_schedule(self) -> EnvironmentSchedule:
        return self.esc.env_schedule_type(
            self.n_sync, [e.get_function() for e in self.esc.env_configs], **self.esc.kwargs
        )

    def get_replay_buffer(
        self,
        storage_directory: Optional[str | Path] = None,
    ) -> Replay:
        if self.algorithm == "arrow":
            total_slots = 2 * self.data_n_max
            n_fifo, n_ltdm = _arrow_fifo_ltdm_capacity_ns(
                total_slots, self.arrow_replay_capacity_ratio
            )

            def _arrow_n(rb: RbConfig) -> int:
                if rb.rb_type is FifoReplay:
                    return n_fifo
                if rb.rb_type is LongTermReplay:
                    return n_ltdm
                raise AssertionError(f"Unexpected replay type: {rb.rb_type}")

            w_fifo, w_ltdm = _arrow_fifo_ltdm_sampling_weights(
                self.arrow_replay_capacity_ratio
            )
            sampling_weights = tuple(
                w_fifo if rc.rb_type is FifoReplay else w_ltdm
                for rc in self.replay_buffers
            )
            storage_root = (
                None
                if storage_directory is None
                else Path(storage_directory).expanduser().resolve()
            )
            if storage_root is not None:
                storage_root.mkdir(parents=True, exist_ok=True)
            replays = []
            for index, rc in enumerate(self.replay_buffers):
                observation_storage_path = (
                    None
                    if storage_root is None
                    else storage_root
                    / f"{index}_{rc.rb_type.__name__}_observations.float32.mmap"
                )
                replays.append(
                    rc.rb_type(
                        self.data_t,
                        _arrow_n(rc),
                        self.action_space,
                        rc.rb_device,
                        store_task_ids=self.uses_task_experts,
                        observation_storage_path=observation_storage_path,
                    )
                )
            return MultiTypeReplay(*replays, sampling_weights=sampling_weights)
        if self.algorithm == "dv3" or self.algorithm == "sac":
            rc = self.replay_buffers[0]
            return rc.rb_type(self.data_t, self.sac_dv3_data_n_max, self.action_space, rc.rb_device)
