import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Type, TypeVar, Union

import gymnasium as gym
from gymnasium.wrappers import TransformReward

import generate_trajectory
import replay
from generate_trajectory import EnvironmentSchedule
from replay import FifoReplay, LongTermReplay, MultiTypeReplay, Replay

T = TypeVar("T", bound="Serialisable")

ArrowReplayCapacityRatio = Literal["50-50", "25-75", "75-25"]
ObservationObjective = Literal["reconstruction", "r2"]
ActorNetwork = Literal[
    "mlp",
    "relu_kan",
    "relu_kan_bounded",
    "relu_kan_adaptive",
    "fast_kan_ac",
    "fast_kan_ac_param_matched",
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

    observation_objective: ObservationObjective = "reconstruction"
    r2_barlow_loss_scale: float = 0.05
    r2_redundancy_scale: float = 5e-4
    r2_normalization_eps: float = 1e-8

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
        if self.observation_objective not in {"reconstruction", "r2"}:
            raise ValueError(
                f"Unknown observation objective: {self.observation_objective!r}"
            )
        if self.r2_barlow_loss_scale <= 0:
            raise ValueError("r2_barlow_loss_scale must be positive")
        if self.r2_redundancy_scale < 0:
            raise ValueError("r2_redundancy_scale must be non-negative")
        if self.r2_normalization_eps <= 0:
            raise ValueError("r2_normalization_eps must be positive")
        if self.actor_network not in {
            "mlp",
            "relu_kan",
            "relu_kan_bounded",
            "relu_kan_adaptive",
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
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

        if self.actor_network in {"fast_kan_ac", "fast_kan_ac_param_matched"}:
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
                "fastkan_hidden_features": (
                    53 if self.actor_network == "fast_kan_ac_param_matched" else 34
                ),
                "ac_replay_critic_loss_scale": (
                    0.3 if self.actor_network == "fast_kan_ac_param_matched" else 0.0
                ),
            }
            paper_aligned.update(method_aligned)
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

    def get_env_schedule(self) -> EnvironmentSchedule:
        return self.esc.env_schedule_type(
            self.n_sync, [e.get_function() for e in self.esc.env_configs], **self.esc.kwargs
        )

    def get_replay_buffer(self) -> Replay:
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
            replays = [
                rc.rb_type(self.data_t, _arrow_n(rc), self.action_space, rc.rb_device)
                for rc in self.replay_buffers
            ]
            return MultiTypeReplay(*replays, sampling_weights=sampling_weights)
        if self.algorithm == "dv3" or self.algorithm == "sac":
            rc = self.replay_buffers[0]
            return rc.rb_type(self.data_t, self.sac_dv3_data_n_max, self.action_space, rc.rb_device)
