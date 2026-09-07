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
ObservationObjective = Literal["reconstruction"]
ObservationEncoder = Literal["cnn"]
ContinualMethod = Literal[("none", "bounded_dream_rehearsal", "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow")]
SharedCoreMode = Literal["trainable", "evolving_replay_protected"]
ActorNetwork = Literal[("mlp",)]
ActorCriticOptimizer = Literal["adam", "laprop"]
ActorCriticSchedule = Literal["constant", "task_cosine_decay"]
TaskMechanismCapacityProfile = Literal["matched_512"]
TaskMechanismParameterization = Literal["dense_private", "adaptive_dense_width"]
EvaluationSeedProtocol = Literal[
    "advancing",
    "fixed_validation_heldout_final",
]
ComputeDType = Literal["float32", "bfloat16"]
ReplayObservationDType = Literal["float32", "uint8"]
DataParallelWorldSize = Literal[1, 2, 4]
EvolvingTask0Profile = Literal["fixed_v1"]
EvolvingCheckpointRetention = Literal["all_boundaries", "latest_boundary"]


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
    replay_observation_dtype: ReplayObservationDType = "float32"
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
    compute_dtype: ComputeDType = "float32"
    data_parallel_world_size: DataParallelWorldSize = 1
    evaluation_seed_protocol: EvaluationSeedProtocol = "advancing"
    task_route_inference: Literal[
        "oracle", "two_frame_probability_reconstruction"
    ] = "oracle"
    evaluation_episode_count_mode: Literal["legacy", "exact"] = "legacy"
    evaluation_max_agent_decisions_per_episode: int = 32768
    evaluation_task_seed_offset: int = 0

    actor_network: ActorNetwork = "mlp"

    ac_optimizer: ActorCriticOptimizer = "adam"
    ac_lr: float = 1e-4
    ac_schedule: ActorCriticSchedule = "constant"
    ac_decay_start_task_epoch: int = 40
    ac_decay_end_task_epoch: int = 90
    ac_final_lr: float = 2.5e-5
    ac_final_entropy_scale: float = 5e-5
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
    observation_encoder: ObservationEncoder = "cnn"
    task_projected_image_encoder: bool = False
    task_projector_bottleneck_features: int = 64
    task_mechanism_bank: bool = False
    task_mechanism_reuse: bool = True
    task_mechanism_capacity_profile: TaskMechanismCapacityProfile = "matched_512"
    task_mechanism_parameterization: TaskMechanismParameterization = "dense_private"
    task_mechanism_recurrent_width: int = 512
    task_mechanism_representation_width: int = 512
    task_mechanism_transition_width: int = 256
    task_mechanism_residual_scale: float = 0.1
    task_mechanism_num_atoms: int = 1
    # Evolving-Core Atomic RSSM is intentionally configured independently of
    # the frozen-base MB/REC methods above.
    evolving_task0_profile: EvolvingTask0Profile = "fixed_v1"
    evolving_shared_core: bool = False
    evolving_checkpoint_retention: EvolvingCheckpointRetention = "all_boundaries"
    first_task_shared_core_lr: float = 2e-4
    shared_core_lr: float = 1e-4
    task_private_lr: float = 2e-4
    task_route_lr: float = 1e-3
    current_batch_n: int = 12
    memory_batch_n: int = 4
    memory_loss_scale: float = 1.0
    interface_q_scale: float = 0.1
    interface_h_scale: float = 0.05
    interface_actor_scale: float = 0.05
    component_gradient_projection: bool = True
    task_atom_output_regularization: float = 1e-4
    boundary_consolidation_steps: int = 1000
    boundary_consolidation_lr: float = 2e-5
    boundary_max_return_drop: float = 0.05
    adaptive_compression_width_fractions: list[float] = field(default_factory=list)
    adaptive_compression_steps_per_candidate: int = 0
    adaptive_compression_lr: float = 0.0
    adaptive_compression_rollouts: int = 0
    adaptive_compression_max_return_drop: float = 0.0
    adaptive_compression_qfp_distill_scale: float = 0.0
    evolving_shared_behavior_current_task_fraction: float = 1.0
    task_shared_prediction_heads: bool = False
    shared_prediction_distill_scale: float = 0.0
    task_private_actor_critic: bool = False
    task_atomic_routes: bool = False
    dream_rehearsal_interval_agent_decisions: int = 2_000
    dream_rehearsal_updates_per_prior_task: int = 50
    dream_rehearsal_batch_sequences: int = 4
    dream_rehearsal_context_steps: int = 16
    dream_rehearsal_horizon: int = 15
    dream_rehearsal_top_fraction: float = 0.25
    dream_rehearsal_realized_threshold: float = 0.3
    dream_rehearsal_realized_bonus: float = 10.0
    dream_rehearsal_grad_clip: float = 100.0
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
        if self.evolving_task0_profile != "fixed_v1":
            raise ValueError("Only the retained D-family fixed_v1 profile is supported")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unknown compute dtype: {self.compute_dtype!r}")
        if self.replay_observation_dtype not in {"float32", "uint8"}:
            raise ValueError(
                "Unknown replay observation dtype: "
                f"{self.replay_observation_dtype!r}"
            )
        assert self.n_sync * self.gen_seq_len == self.data_n * self.data_t
        assert self.random_policy in {"first", "new"}
        assert self.replay_buffers != []
        sequential_task_durations: tuple[int, ...] | None = None
        if self.esc.env_schedule_type is SequentialEnvironments:
            task_durations = self.esc.kwargs.get("task_durations")
            swap_sched = self.esc.kwargs.get("swap_sched")
            if task_durations is not None and swap_sched is not None:
                raise ValueError(
                    "Sequential scheduling accepts swap_sched or task_durations, "
                    "not both"
                )
            if task_durations is None:
                if not isinstance(swap_sched, int) or swap_sched < 1:
                    raise ValueError(
                        "Sequential scheduling requires a positive swap_sched"
                    )
                sequential_task_durations = (swap_sched,) * len(
                    self.esc.env_configs
                )
            else:
                if not isinstance(task_durations, list) or len(
                    task_durations
                ) != len(self.esc.env_configs):
                    raise ValueError(
                        "task_durations must be a list matching the environment count"
                    )
                if any(
                    not isinstance(duration, int) or duration < 1
                    for duration in task_durations
                ):
                    raise ValueError(
                        "task_durations must contain positive integers"
                    )
                sequential_task_durations = tuple(task_durations)
        if self.continual_method not in {"none", "bounded_dream_rehearsal", "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}:
            raise ValueError(f"Unknown continual method: {self.continual_method!r}")
        is_bounded_dream_rehearsal = (
            self.continual_method == "bounded_dream_rehearsal"
        )
        is_moe_arrow = False
        is_cnn_fullbank = False
        is_cnn_projector_lora = False
        is_cnn_compact_shared_actor = False
        is_cnn_mechanism_bank = False
        is_rec_rssm = False
        is_evolving_autoroute = self.uses_reconstruction_task_inference

        expected_inference = (
            "two_frame_probability_reconstruction" if is_evolving_autoroute else "oracle"
        )
        expected_episode_mode = "legacy"
        if self.task_route_inference != expected_inference:
            raise ValueError("Task route inference must match the separately named protocol")
        if self.evaluation_episode_count_mode != expected_episode_mode:
            raise ValueError("Evaluation episode counting must match the named protocol")
        if (
            type(self.evaluation_max_agent_decisions_per_episode) is not int
            or self.evaluation_max_agent_decisions_per_episode != 32768
        ):
            raise ValueError("The exact evaluation episode safety cap is fixed at 32768 decisions")
        if is_evolving_autoroute and (
            self.evolving_task0_profile != "fixed_v1" or len(self.esc.env_configs) != 6
            or self.epochs != 540 or self.esc.kwargs.get("swap_sched") != 90
            or tuple(task.name for task in self.esc.env_configs) != (
                "ALE/MsPacman-v5", "ALE/Boxing-v5", "ALE/CrazyClimber-v5",
                "ALE/Frostbite-v5", "ALE/Seaquest-v5", "ALE/Enduro-v5",
            )
        ):
            raise ValueError(
                "Autoroute requires the original-six fixed_v1 540-epoch protocol"
            )

        is_evolving_shared_heads = False
        is_evolving_adaptive_compression_shared_heads = is_evolving_autoroute or (
            self.continual_method
            == "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow"
        )
        is_evolving_adaptive_qfp_ac_compression_shared_heads = False
        is_evolving_atomic_lora_shared_heads = False
        is_evolving_learned_base_adapters = False
        uses_evolving_shared_heads = (is_evolving_adaptive_compression_shared_heads)
        is_evolving_atomic = self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}
        if not isinstance(self.task_shared_prediction_heads, bool):
            raise ValueError("task_shared_prediction_heads must be a boolean")
        if self.task_shared_prediction_heads and not uses_evolving_shared_heads:
            raise ValueError(
                "Task-shared prediction heads require the separately named "
                "shared prediction heads Evolving-Core method"
            )
        uses_mechanism_bank = (is_evolving_atomic)
        if self.task_mechanism_capacity_profile not in {
            "matched_512",

        }:
            raise ValueError(
                "Unknown mechanism capacity profile: "
                f"{self.task_mechanism_capacity_profile!r}"
            )
        if self.task_mechanism_parameterization not in {
            "dense_private",
            "adaptive_dense_width",

        }:
            raise ValueError(
                "Unknown mechanism parameterization: "
                f"{self.task_mechanism_parameterization!r}"
            )
        if (
            self.task_mechanism_parameterization == "adaptive_dense_width"
            and not (is_evolving_adaptive_compression_shared_heads)
        ):
            raise ValueError(
                "adaptive_dense_width is validated only for the separately named "
                "adaptive-compression shared-head method"
            )
        if not isinstance(self.task_mechanism_num_atoms, int) or (
            self.task_mechanism_num_atoms < 1
        ):
            raise ValueError("task_mechanism_num_atoms must be a positive integer")
        if (
            (is_evolving_adaptive_compression_shared_heads)
            and self.task_mechanism_parameterization != "adaptive_dense_width"
        ):
            raise ValueError(
                "The adaptive-compression shared-head method requires "
                "mechanism parameterization='adaptive_dense_width'"
            )
        if (
            (is_evolving_adaptive_compression_shared_heads)
            and self.task_mechanism_capacity_profile != "matched_512"
        ):
            raise ValueError(
                "The shared-prediction-head Evolving-Core method requires "
                "dense matched_512 Q/F/P mechanisms"
            )
        if not isinstance(self.task_mechanism_bank, bool) or not isinstance(
            self.task_mechanism_reuse, bool
        ):
            raise ValueError("Mechanism-bank enable/reuse settings must be booleans")
        if not 0 <= 0.01 < 1:
            raise ValueError("task_mechanism_min_contribution must lie in [0, 1)")
        if not 0 <= 0.05 < 1:
            raise ValueError("task_mechanism_max_validation_drop must lie in [0, 1)")
        evolving_defaults = {
            "evolving_task0_profile": "fixed_v1",
            "evolving_shared_core": False,
            "first_task_shared_core_lr": 2e-4,
            "shared_core_lr": 1e-4,
            "task_private_lr": 2e-4,
            "task_route_lr": 1e-3,
            "current_batch_n": 12,
            "memory_batch_n": 4,
            "memory_loss_scale": 1.0,
            "interface_q_scale": 0.1,
            "interface_h_scale": 0.05,
            "interface_actor_scale": 0.05,
            "component_gradient_projection": True,
            "task_atom_output_regularization": 1e-4,
            "boundary_consolidation_steps": 1000,
            "boundary_consolidation_lr": 2e-5,
            "boundary_max_return_drop": 0.05,
            "adaptive_compression_width_fractions": [],
            "adaptive_compression_steps_per_candidate": 0,
            "adaptive_compression_lr": 0.0,
            "adaptive_compression_rollouts": 0,
            "adaptive_compression_max_return_drop": 0.0,
            "adaptive_compression_qfp_distill_scale": 0.0,
            "evolving_shared_behavior_current_task_fraction": 1.0,
            "task_shared_prediction_heads": False,
            "shared_prediction_distill_scale": 0.0,
            "task_private_actor_critic": False,
            "task_atomic_routes": False,

        }
        if is_evolving_atomic:
            if self.evolving_checkpoint_retention not in {
                "all_boundaries",
                "latest_boundary",
            }:
                raise ValueError(
                    "Unknown Evolving-Core checkpoint retention: "
                    f"{self.evolving_checkpoint_retention!r}"
                )
            task0_profile_overrides = {
                "fixed_v1": {},
                "fixed_v2": {
                    "first_task_shared_core_lr": 3e-4,
                },
                "task0_shared_lr_1e4": {
                    "first_task_shared_core_lr": 1e-4,
                },
                "task0_shared_lr_3e4": {
                    "first_task_shared_core_lr": 3e-4,
                },
                "task0_private_lr_3e4": {
                    "task_private_lr": 3e-4,
                },
                "task0_actor_lr_2e4": {
                    "ac_lr": 2e-4,
                },
                "task0_epochs_120": {},
                "task0_epochs_150": {},
                "task0_epochs_180": {},
                "task0_epochs_240": {},
            }
            if self.evolving_task0_profile not in task0_profile_overrides:
                raise ValueError(
                    "Unknown Evolving-Core Task-0 profile: "
                    f"{self.evolving_task0_profile!r}"
                )
            expected_evolving = {
                **evolving_defaults,
                "evolving_task0_profile": self.evolving_task0_profile,
                "evolving_shared_core": True,
                "task_shared_prediction_heads": uses_evolving_shared_heads,
                "shared_prediction_distill_scale": (
                    0.1 if uses_evolving_shared_heads else 0.0
                ),
                "task_private_actor_critic": True,
                "task_atomic_routes": True,
                "ac_lr": (1e-4),
            }
            if (is_evolving_adaptive_compression_shared_heads):
                expected_evolving.update(
                    {
                        "adaptive_compression_width_fractions": [
                            0.75,
                            0.5,
                            0.25,
                            0.125,
                        ],
                        "adaptive_compression_steps_per_candidate": 250,
                        "adaptive_compression_lr": 2e-4,
                        "adaptive_compression_rollouts": 16,
                        "adaptive_compression_max_return_drop": 0.05,
                        "adaptive_compression_qfp_distill_scale": 1.0,
                    }
                )
            expected_evolving.update(
                task0_profile_overrides[self.evolving_task0_profile]
            )
            mismatches = {
                name: (getattr(self, name), expected)
                for name, expected in expected_evolving.items()
                if getattr(self, name) != expected
            }
            if mismatches:
                raise ValueError(
                    "Evolving-Core Atomic RSSM requires its fixed optimizer, "
                    f"replay, interface, and topology settings: {mismatches}"
                )
            if self.evolving_task0_profile not in ("fixed_v1", "fixed_v2"):
                if sequential_task_durations is None:
                    raise ValueError(
                        "Evolving-Core Task-0 sweeps require a sequential schedule"
                    )
                task0_profile_epochs = {
                    "task0_shared_lr_1e4": 90,
                    "task0_shared_lr_3e4": 90,
                    "task0_private_lr_3e4": 90,
                    "task0_actor_lr_2e4": 90,
                    "task0_epochs_120": 120,
                    "task0_epochs_150": 150,
                    "task0_epochs_180": 180,
                    "task0_epochs_240": 240,
                }
                expected_task_durations = (
                    task0_profile_epochs[self.evolving_task0_profile],
                    90,
                    90,
                )
                if sequential_task_durations != expected_task_durations:
                    raise ValueError(
                        "Evolving-Core Task-0 sweep profile requires exact task "
                        f"durations {expected_task_durations}"
                    )
                if self.epochs != expected_task_durations[0]:
                    raise ValueError(
                        "Evolving-Core Task-0 sweep profiles must stop exactly at "
                        "the first task boundary"
                    )
            if self.current_batch_n + self.memory_batch_n != self.mb_n_size:
                raise ValueError(
                    "Evolving-Core current and memory sequence counts must sum "
                    "to mb_n_size"
                )
            if self.pretrain_mb_n_size != self.mb_n_size:
                raise ValueError(
                    "Evolving-Core Task 1 requires the same full sequence batch size"
                )
            if self.evaluation_seed_protocol != "fixed_validation_heldout_final":
                raise ValueError(
                    "Evolving-Core consolidation requires fixed validation and "
                    "held-out final cohorts"
                )
            if self.memory_loss_scale < 0:
                raise ValueError("memory_loss_scale must be non-negative")
            if self.shared_prediction_distill_scale < 0:
                raise ValueError(
                    "shared_prediction_distill_scale must be non-negative"
                )
            if not 0 < self.evolving_shared_behavior_current_task_fraction <= 1:
                raise ValueError(
                    "Evolving-Core shared-behavior current-task fraction must lie "
                    "in (0, 1]"
                )
            if min(
                self.interface_q_scale,
                self.interface_h_scale,
                self.interface_actor_scale,
                self.task_atom_output_regularization,
            ) < 0:
                raise ValueError(
                    "Evolving-Core interface and atom regularization scales "
                    "must be non-negative"
                )
            if (is_evolving_adaptive_compression_shared_heads):
                fractions = self.adaptive_compression_width_fractions
                if (
                    not isinstance(fractions, list)
                    or not fractions
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not 0 < float(value) < 1
                        for value in fractions
                    )
                    or any(
                        float(left) <= float(right)
                        for left, right in zip(fractions, fractions[1:])
                    )
                ):
                    raise ValueError(
                        "Adaptive compression width fractions must be strictly "
                        "decreasing values in (0, 1)"
                    )
                if self.adaptive_compression_steps_per_candidate < 1:
                    raise ValueError(
                        "Adaptive compression steps per candidate must be positive"
                    )
                if self.adaptive_compression_lr <= 0:
                    raise ValueError("Adaptive compression LR must be positive")
                if self.adaptive_compression_rollouts < 1:
                    raise ValueError("Adaptive compression rollouts must be positive")
                if not 0 <= self.adaptive_compression_max_return_drop < 1:
                    raise ValueError(
                        "Adaptive compression maximum return drop must lie in [0, 1)"
                    )
                if self.adaptive_compression_qfp_distill_scale <= 0:
                    raise ValueError(
                        "Adaptive compression Q/F/P distillation scale must be positive"
                    )
                expected_tasks = (
                    "ALE/MsPacman-v5",
                    "ALE/Boxing-v5",
                    "ALE/CrazyClimber-v5",
                    "ALE/Frostbite-v5",
                    "ALE/Seaquest-v5",
                    "ALE/Enduro-v5",
                )
                observed_tasks = tuple(task.name for task in self.esc.env_configs)
                if observed_tasks != expected_tasks:
                    raise ValueError(
                        "Adaptive compression v1 is fixed to the ARROW original-six "
                        f"order, got {observed_tasks}"
                    )
                if sequential_task_durations != (90,) * len(expected_tasks):
                    raise ValueError(
                        "Adaptive compression v1 fixes every original-six task to "
                        "90 epochs"
                    )
                topology = {
                    "task_mechanism_reuse": self.task_mechanism_reuse,
                    "task_mechanism_num_atoms": self.task_mechanism_num_atoms,
                    "task_mechanism_recurrent_width": (
                        self.task_mechanism_recurrent_width
                    ),
                    "task_mechanism_representation_width": (
                        self.task_mechanism_representation_width
                    ),
                    "task_mechanism_transition_width": (
                        self.task_mechanism_transition_width
                    ),
                }
                expected_topology = {
                    "task_mechanism_reuse": True,
                    "task_mechanism_num_atoms": 4,
                    "task_mechanism_recurrent_width": 512,
                    "task_mechanism_representation_width": 512,
                    "task_mechanism_transition_width": 256,
                }
                if topology != expected_topology:
                    raise ValueError(
                        "Adaptive compression v1 requires full-width Dense "
                        f"acquisition and four-atom reuse: {topology}"
                    )
                for full_width in (512, 512, 256):
                    candidate_widths = [
                        int(round(full_width * float(fraction)))
                        for fraction in fractions
                    ]
                    if (
                        len(set(candidate_widths)) != len(candidate_widths)
                        or any(
                            width < self.task_mechanism_num_atoms
                            or width % self.task_mechanism_num_atoms
                            for width in candidate_widths
                        )
                    ):
                        raise ValueError(
                            "Adaptive compression candidates must map to unique "
                            "atom-divisible Q/F/P widths"
                        )
        else:
            if self.evolving_checkpoint_retention != "all_boundaries":
                raise ValueError(
                    "Evolving-Core checkpoint retention requires "
                    "continual_method='evolving_atomic_rssm_arrow'"
                )
            evolving_nondefault = {
                name: (getattr(self, name), expected)
                for name, expected in evolving_defaults.items()
                if getattr(self, name) != expected
            }
            if evolving_nondefault:
                raise ValueError(
                    "Evolving-Core settings require "
                    "a named evolving_atomic_rssm continual method: "
                    f"{evolving_nondefault}"
                )
        is_dino_fullbank = False
        is_dino_patchbank = False
        is_dino_convbank = False
        is_dino_pixelbank = (False)
        uses_task_experts = (uses_mechanism_bank)
        is_independent_expert = False
        if self.data_parallel_world_size not in {1, 2, 4}:
            raise ValueError("data_parallel_world_size must be one of 1, 2, or 4")
        if self.evaluation_seed_protocol not in {
            "advancing",
            "fixed_validation_heldout_final",
        }:
            raise ValueError(
                f"Unknown evaluation seed protocol: {self.evaluation_seed_protocol!r}"
            )
        if self.evaluation_task_seed_offset < 0:
            raise ValueError("evaluation_task_seed_offset must be non-negative")
        if (
            self.evaluation_task_seed_offset
            and self.evaluation_seed_protocol != "fixed_validation_heldout_final"
        ):
            raise ValueError(
                "evaluation_task_seed_offset requires fixed validation seeds"
            )
        if self.data_parallel_world_size > 1:
            raise ValueError(
                "multi-GPU data parallelism is validated only for "
                "DINO-ConvBank-ARROW and CNN-FullBank-ARROW"
            )
            distributed_batch_sizes = {
                "mb_n_size": self.mb_n_size,
                "pretrain_mb_n_size": self.pretrain_mb_n_size,
                "ac_train_sync": self.ac_train_sync,
            }
            indivisible = {
                name: value
                for name, value in distributed_batch_sizes.items()
                if value < self.data_parallel_world_size
                or value % self.data_parallel_world_size
            }
            if indivisible:
                raise ValueError(
                    "fixed global sequence batches must divide equally across "
                    f"data-parallel ranks: {indivisible}"
                )
        if (uses_mechanism_bank):
            if self.compute_dtype != "bfloat16":
                raise ValueError("The CNN task-bank protocol requires bfloat16 compute")
            if self.replay_observation_dtype != "uint8":
                raise ValueError(
                    "The CNN task-bank protocol requires uint8 observation replay"
                )
        elif (
            not is_bounded_dream_rehearsal
            and self.replay_observation_dtype != "float32"
        ):
            raise ValueError(
                "uint8 observation replay is reserved for DINO-ConvBank and "
                "CNN-FullBank optimized protocols or Bounded Dream Rehearsal"
            )
        if uses_task_experts:
            if self.algorithm != "arrow":
                raise ValueError("Task-aware expert methods require ARROW mixed replay")
            if self.esc.env_schedule_type is not SequentialEnvironments:
                raise ValueError("Task-aware expert methods require a sequential task schedule")
            if len(self.esc.env_configs) < 2:
                raise ValueError(
                    "Task-aware expert methods require at least two scheduled tasks"
                )
            if self.rssm_num_experts != len(self.esc.env_configs):
                raise ValueError(
                    "Task-aware expert methods require one RSSM expert per scheduled task"
                )
        else:
            if self.rssm_num_experts != 1:
                raise ValueError(
                    "RSSM experts require a task-aware continual_method"
                )

        dream_rehearsal_defaults = {
            "dream_rehearsal_interval_agent_decisions": 2_000,
            "dream_rehearsal_updates_per_prior_task": 50,
            "dream_rehearsal_batch_sequences": 4,
            "dream_rehearsal_context_steps": 16,
            "dream_rehearsal_horizon": 15,
            "dream_rehearsal_top_fraction": 0.25,
            "dream_rehearsal_realized_threshold": 0.3,
            "dream_rehearsal_realized_bonus": 10.0,
            "dream_rehearsal_grad_clip": 100.0,
        }
        if is_bounded_dream_rehearsal:
            from clworldmodel.continual.dream_rehearsal import (
                DreamRehearsalConfig,
            )

            if self.algorithm != "dv3":
                raise ValueError("Bounded Dream Rehearsal requires DreamerV3")
            if self.esc.env_schedule_type is not SequentialEnvironments:
                raise ValueError(
                    "Bounded Dream Rehearsal requires a sequential task schedule"
                )
            if len(self.esc.env_configs) < 2:
                raise ValueError(
                    "Bounded Dream Rehearsal requires at least two tasks"
                )
            if len(self.replay_buffers) != 1 or (
                self.replay_buffers[0].rb_type is not LongTermReplay
            ):
                raise ValueError(
                    "Bounded Dream Rehearsal requires one fixed-capacity "
                    "LongTermReplay reservoir"
                )
            if self.replay_buffers[0].rb_device.split(":", 1)[0] != "cpu":
                raise ValueError(
                    "Bounded Dream Rehearsal requires CPU-resident replay"
                )
            if self.replay_observation_dtype != "uint8":
                raise ValueError(
                    "Bounded Dream Rehearsal requires uint8 replay observations"
                )
            if self.sac_dv3_data_n_max < 1:
                raise ValueError(
                    "Bounded Dream Rehearsal replay capacity must be positive"
                )
            if self.fresh_ac is not False or self.actor_network != "mlp":
                raise ValueError(
                    "Bounded Dream Rehearsal requires one persistent shared MLP actor"
                )
            if self.data_parallel_world_size != 1:
                raise ValueError(
                    "Bounded Dream Rehearsal is initially validated on one device"
                )
            if self.dream_rehearsal_context_steps > self.data_t:
                raise ValueError(
                    "Dream-rehearsal context cannot exceed replay sequence length"
                )
            DreamRehearsalConfig(
                interval_agent_decisions=(
                    self.dream_rehearsal_interval_agent_decisions
                ),
                updates_per_prior_task=(
                    self.dream_rehearsal_updates_per_prior_task
                ),
                batch_sequences=self.dream_rehearsal_batch_sequences,
                context_steps=self.dream_rehearsal_context_steps,
                horizon=self.dream_rehearsal_horizon,
                top_fraction=self.dream_rehearsal_top_fraction,
                realized_threshold=self.dream_rehearsal_realized_threshold,
                realized_bonus=self.dream_rehearsal_realized_bonus,
            )
            if self.dream_rehearsal_grad_clip < 0:
                raise ValueError(
                    "Dream-rehearsal gradient clipping must be non-negative"
                )
        else:
            nondefault_dream_rehearsal = {
                name: (getattr(self, name), expected)
                for name, expected in dream_rehearsal_defaults.items()
                if getattr(self, name) != expected
            }
            if nondefault_dream_rehearsal:
                raise ValueError(
                    "Dream-rehearsal settings require "
                    "continual_method='bounded_dream_rehearsal': "
                    f"{nondefault_dream_rehearsal}"
                )
        if (is_evolving_atomic):
            expected_shared_core_mode = (
                "evolving_replay_protected"
                if is_evolving_atomic
                else ("task_isolated")
            )
            if self.shared_core_mode != expected_shared_core_mode:
                raise ValueError(
                    "The selected full task bank requires shared_core_mode="
                    f"'{expected_shared_core_mode}'"
                )
            expected_objective = (
                "reconstruction"
                if (uses_mechanism_bank)
                else "dinov3_posterior_feature"
            )
            if self.observation_objective != expected_objective:
                raise ValueError(
                    (
                        "CNN and DINO patch task banks keep DreamerV3 pixel reconstruction"
                        if (is_evolving_atomic)
                        else "DINO-FullBank-ARROW reconstructs posterior DINOv3 features"
                    )
                )
            if self.random_policy != "new":
                raise ValueError(
                    "Full task banks require a random collection for each new task"
                )
            if (is_evolving_atomic) and any(
                replay_config.rb_device.split(":", 1)[0] != "cpu"
                for replay_config in self.replay_buffers
            ):
                raise ValueError(
                    "Pixel task banks require CPU-addressable mapped observation replay"
                )
        uses_cnn_projector = (uses_mechanism_bank)
        if self.task_projected_image_encoder != uses_cnn_projector:
            raise ValueError(
                "task_projected_image_encoder is required only by "
                "named CNN projector methods"
            )
        if uses_cnn_projector:
            if self.data_parallel_world_size != 1:
                raise ValueError(
                    "CNN projector methods are initially validated on one GPU"
                )
            if self.task_projector_bottleneck_features != 64:
                raise ValueError(
                    "CNN projector methods fix the projector bottleneck at 64"
                )
            observed_ranks = (
                0,
                0,
                0,
                0,
            )
            if uses_mechanism_bank:
                if observed_ranks != (0, 0, 0, 0):
                    raise ValueError(
                        "CNN-MechanismBank disables every RSSM LoRA/output-adapter path"
                    )
                expected_mechanism_settings = {
                    "matched_512": (True, 512, 512, 256, 0.1),

                }[self.task_mechanism_capacity_profile]
                observed_mechanism_settings = (
                    self.task_mechanism_bank,
                    self.task_mechanism_recurrent_width,
                    self.task_mechanism_representation_width,
                    self.task_mechanism_transition_width,
                    self.task_mechanism_residual_scale,
                )
                if observed_mechanism_settings != expected_mechanism_settings:
                    raise ValueError(
                        "CNN-MechanismBank fixes bank/recurrent/posterior/prior/scale "
                        f"settings to {expected_mechanism_settings}, got "
                        f"{observed_mechanism_settings}"
                    )
                atom_settings = (
                    self.task_mechanism_num_atoms,
                    0,
                    1.0,
                    8,
                    0.01,
                    0.05,
                )
                expected_atom_settings = ((4, 0, 1.0, 8, 0.01, 0.05)
                    if is_evolving_atomic
                    else (1, 0, 1.0, 8, 0.01, 0.05))
                if atom_settings != expected_atom_settings:
                    raise ValueError(
                        "The named mechanism protocol fixes atom/probe/route-LR/"
                        "consolidation settings to "
                        f"{expected_atom_settings}, got {atom_settings}"
                )
                if (
                    (is_evolving_atomic)
                    and not self.task_mechanism_reuse
                ):
                    raise ValueError("Atomic RSSM requires atom reuse")
                if self.fresh_ac is not False:
                    raise ValueError(
                        "CNN-MechanismBank methods require persistent actor-critics"
                    )
                expected_actor_network = (
                    ("mlp")
                )
                if self.actor_network != expected_actor_network:
                    raise ValueError(
                        "The named mechanism protocol requires actor_network="
                        f"'{expected_actor_network}'"
                    )
            else:
                expected_ranks = ((128, 128, 32, 0), (32, 32, 16, 0))
                if observed_ranks not in expected_ranks:
                    method_description = ("CNN-Projector-LoRA-ARROW fixes recurrent/representation/"
                        "transition/output-adapter sizes")
                    raise ValueError(
                        f"{method_description} to a named profile in "
                        f"{expected_ranks}, got {observed_ranks}"
                    )
        if self.task_mechanism_bank != uses_mechanism_bank:
            raise ValueError(
                "task_mechanism_bank is required only by named mechanism methods"
            )
        if not uses_mechanism_bank and not self.task_mechanism_reuse:
            raise ValueError(
                "Disabling mechanism reuse requires CNN-MechanismBank-ARROW"
            )
        if not uses_mechanism_bank:
            if self.task_mechanism_parameterization != "dense_private":
                raise ValueError(
                    "Mechanism parameterization requires a named mechanism method"
                )
            observed_atom_settings = (
                self.task_mechanism_num_atoms,
                0,
                1.0,
                8,
                0.01,
                0.05,
            )
            default_atom_settings = (1, 0, 1.0, 8, 0.01, 0.05)
            if observed_atom_settings != default_atom_settings:
                raise ValueError(
                    "REC-RSSM atom/probe/consolidation settings require a "
                    "named mechanism method"
                )
        shared_actor_defaults = (False, 0.0, 1, 1, 0, 1)
        shared_actor_values = (
            False,
            0.0,
            1,
            1,
            0,
            1,
        )
        if shared_actor_values != shared_actor_defaults:
            raise ValueError(
                "Shared-actor imagination distillation settings require "
                "CNN-Compact-SharedActor"
            )
        if (is_evolving_atomic) and self.observation_encoder != "cnn":
            raise ValueError("CNN task-bank methods require the CNN observation encoder")
        if self.observation_objective not in {
            "reconstruction",

        }:
            raise ValueError(
                f"Unknown observation objective: {self.observation_objective!r}"
            )
        if self.observation_encoder not in {"cnn", }:
            raise ValueError(f"Unknown observation encoder: {self.observation_encoder!r}")
        uses_dinov3_objective = False
        uses_dinov3 = False
        if (self.observation_encoder != "cnn"):
            raise ValueError(
                "Only named DINOv3 protocols may configure the DINOv3 encoder"
            )
        if self.shared_core_mode not in {
            "trainable",
            "evolving_replay_protected",
        }:
            raise ValueError(f"Unknown shared core mode: {self.shared_core_mode!r}")
        if (
            self.shared_core_mode == "evolving_replay_protected"
            and not is_evolving_atomic
        ):
            raise ValueError(
                "shared_core_mode='evolving_replay_protected' is reserved for "
                "Evolving-Core Atomic RSSM"
            )
        if (-2.0 != -2.0):
            raise ValueError("KARROW Frozen-Core fixes the residual basis range at [-2, 2]")
        consolidation_defaults = (
            16,
            8,
            2.0,
            0.01,
            1.0,
        )
        consolidation_values = (
            16,
            8,
            2.0,
            0.01,
            1.0,
        )
        if consolidation_values != consolidation_defaults:
            raise ValueError(
                "Residual consolidation settings require "
                "residual_consolidation='replay_functional'"
            )
        if self.actor_network not in {"mlp"}:
            raise ValueError(f"Unknown actor network: {self.actor_network!r}")
        expected_trainable_grid = False
        if False != expected_trainable_grid:
            if expected_trainable_grid:
                raise ValueError(
                    "relu_kan_adaptive requires actor_kan_trainable_grid=True"
                )
            raise ValueError(
                "Only relu_kan_adaptive may enable actor_kan_trainable_grid"
            )
        if self.ac_optimizer not in {"adam", "laprop"}:
            raise ValueError(f"Unknown actor-critic optimizer: {self.ac_optimizer!r}")
        if self.ac_lr <= 0 or self.ac_fresh_lr <= 0:
            raise ValueError("Actor-critic learning rates must be positive")
        if self.ac_schedule not in {"constant", "task_cosine_decay"}:
            raise ValueError(f"Unknown actor-critic schedule: {self.ac_schedule!r}")
        if self.ac_schedule == "task_cosine_decay":
            if self.ac_decay_start_task_epoch < 0:
                raise ValueError("Actor-critic decay start must be non-negative")
            if self.ac_decay_end_task_epoch <= self.ac_decay_start_task_epoch:
                raise ValueError("Actor-critic decay end must exceed its start")
            if self.ac_final_lr <= 0 or self.ac_final_lr > self.ac_lr:
                raise ValueError(
                    "Actor-critic final learning rate must lie in (0, ac_lr]"
                )
            if not 0 <= self.ac_final_entropy_scale <= self.ac_entropy_scale:
                raise ValueError(
                    "Actor-critic final entropy scale must lie in "
                    "[0, ac_entropy_scale]"
                )
            current_only_actor_training = (False)
            if (not current_only_actor_training):
                raise ValueError(
                    "task_cosine_decay is validated only for named current-only "
                    "actor training profiles"
                )
            actor_schedule_durations = sequential_task_durations
            if (
                not actor_schedule_durations
                or min(actor_schedule_durations) < self.ac_decay_end_task_epoch
            ):
                raise ValueError(
                    "task_cosine_decay must finish within each sequential task"
                )
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


    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["esc"] = self.esc.to_dict()
        data["replay_buffers"] = [c.to_dict() for c in self.replay_buffers]
        return data

    @property
    def uses_reconstruction_task_inference(self) -> bool:
        """Task-aware training, label-free action selection; not full task agnosticism."""

        return self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}

    @property
    def uses_task_experts(self) -> bool:
        return self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}

    @property
    def uses_bounded_dream_rehearsal(self) -> bool:
        return self.continual_method == "bounded_dream_rehearsal"

    @property
    def uses_task_labelled_replay(self) -> bool:
        """Whether task IDs are scheduler metadata on replay trajectories."""

        return self.uses_task_experts or self.uses_bounded_dream_rehearsal


    @property
    def task_update_fraction(self) -> float:
        if self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}:
            return 1.0
        raise ValueError("Task update fractions require a task-aware expert method")

    @property
    def uses_shared_actor(self) -> bool:
        return False

    @property
    def uses_replay_rehearsed_shared_behavior(self) -> bool:
        """Whether one shared actor-critic rehearses task-routed replay."""

        return False


    @property
    def uses_shared_prediction_heads(self) -> bool:
        """Whether decoder/reward/continue are one replay-protected set."""

        return (
            self.continual_method
            in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}
        )


    @property
    def uses_evolving_atomic_rssm(self) -> bool:
        return self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}

    @property
    def uses_adaptive_qfp_compression(self) -> bool:
        """Whether completed Dense Q/F/P modules are return-gated and compacted."""

        return self.continual_method in {"evolving_atomic_rssm_adaptive_compression_shared_heads_arrow", "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"}


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
            observation_dtype = self.replay_observation_dtype
            for index, rc in enumerate(self.replay_buffers):
                observation_storage_path = (
                    None
                    if storage_root is None
                    else storage_root
                    / (
                        f"{index}_{rc.rb_type.__name__}_observations."
                        f"{observation_dtype}.mmap"
                    )
                )
                replays.append(
                    rc.rb_type(
                        self.data_t,
                        _arrow_n(rc),
                        self.action_space,
                        rc.rb_device,
                        store_task_ids=self.uses_task_experts,
                        observation_storage_path=observation_storage_path,
                        observation_dtype=observation_dtype,
                    )
                )
            return MultiTypeReplay(*replays, sampling_weights=sampling_weights)
        if self.algorithm == "dv3" or self.algorithm == "sac":
            rc = self.replay_buffers[0]
            storage_root = (
                None
                if storage_directory is None
                else Path(storage_directory).expanduser().resolve()
            )
            if storage_root is not None:
                storage_root.mkdir(parents=True, exist_ok=True)
            observation_storage_path = (
                None
                if storage_root is None
                else storage_root
                / (
                    f"0_{rc.rb_type.__name__}_observations."
                    f"{self.replay_observation_dtype}.mmap"
                )
            )
            return rc.rb_type(
                self.data_t,
                self.sac_dv3_data_n_max,
                self.action_space,
                rc.rb_device,
                store_task_ids=self.uses_task_labelled_replay,
                observation_storage_path=observation_storage_path,
                observation_dtype=self.replay_observation_dtype,
            )
