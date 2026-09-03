import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torch.optim import Adam, Optimizer
from tqdm import trange

from replay import Replay
from rssm import ActionT, HiddenT, LatentShape, LatentT, get_mlp_layers
from wm import RewardSymlogT, RewardT, WorldModel, symexp, symlog

ActionLogT = torch.Tensor
AcStateT = torch.Tensor
RewardSymlogCatT = torch.Tensor
ReturnT = torch.Tensor
ValueFunction = Callable[[AcStateT], ReturnT]
# ActionLogT (log probs): [ N n_acts ]
# AcState (concatenation of flattened LatentT and HiddenT): [ N n_dis*n_cls+h_dim ]
# RewardSymlogCatT (probs): [ N 255 ]
# RewardT (real): [ N 1 ]
N_CRITIC_BINS = 255


def _autocast_context(device: torch.device, compute_dtype: str):
    if compute_dtype == "float32":
        return nullcontext()
    from clworldmodel.precision import autocast_context

    return autocast_context(device, compute_dtype)


def _full_precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _assert_all_finite(values: torch.Tensor, label: str) -> None:
    """Reject non-finite training tensors without synchronizing healthy CUDA."""

    finite = torch.isfinite(values).all()
    if values.device.type == "cuda" and hasattr(torch, "_assert_async"):
        # A failed invariant aborts this training process; the healthy path does
        # not force the host to wait for every actor-critic update.
        torch._assert_async(finite)
        return
    if not bool(finite):
        raise FloatingPointError(
            f"Non-finite {label}: {values.detach().cpu().tolist()}"
        )


def zh_to_ac_state(z: LatentT, h: HiddenT) -> AcStateT:
    return torch.cat((z.flatten(-2), h), dim=-1)


@torch.no_grad()
def dream_frozen_actor_policy(
    wm: WorldModel,
    teacher_actor: nn.Module,
    *,
    task_id: int,
    n_sync: int,
    burnin_steps: int,
    dream_steps: int,
    temperature: float = 1.0,
) -> tuple[AcStateT, ActionLogT]:
    """Generate old-route states and policy targets without real old replay."""
    if task_id < 0:
        raise ValueError("Distillation task_id must be non-negative")
    if n_sync < 1 or dream_steps < 1 or burnin_steps < 0:
        raise ValueError(
            "Frozen-policy imagination requires positive batch/dream sizes and "
            "a non-negative burn-in"
        )
    if temperature <= 0:
        raise ValueError("Frozen-policy imagination temperature must be positive")

    z, h = wm.rssm.initial_state(n_sync)
    no_reset = torch.zeros(n_sync, 1, device=z.device)
    compute_dtype = getattr(wm, "compute_dtype", "float32")
    states = []
    teacher_logs = []
    with _autocast_context(z.device, compute_dtype):
        for step in range(burnin_steps + dream_steps):
            state = zh_to_ac_state(z, h)
            teacher_log = teacher_actor(state)
            with _full_precision_context(teacher_log.device):
                action = td.OneHotCategorical(
                    logits=teacher_log.float()
                ).sample()
            if step >= burnin_steps:
                states.append(state.detach())
                teacher_logs.append(teacher_log.detach().float())
            _, z, h = wm.rssm(
                z,
                action,
                h,
                None,
                no_reset,
                temperature=temperature,
                task_id=task_id,
            )
    return torch.stack(states), torch.stack(teacher_logs)


def actor_policy_kl(
    student_actor: nn.Module,
    states: AcStateT,
    teacher_logs: ActionLogT,
) -> torch.Tensor:
    """Return KL(teacher || student) over imagined states in FP32."""
    if states.ndim < 2 or teacher_logs.ndim != states.ndim:
        raise ValueError("Actor distillation tensors must include batch and feature axes")
    student_logs = student_actor(states)
    if student_logs.shape != teacher_logs.shape:
        raise ValueError(
            "Teacher and student action distributions must have identical shapes: "
            f"{tuple(teacher_logs.shape)} != {tuple(student_logs.shape)}"
        )
    with _full_precision_context(student_logs.device):
        teacher_logs_fp32 = teacher_logs.float()
        student_logs_fp32 = student_logs.float()
        teacher_probs = teacher_logs_fp32.exp()
        return (
            teacher_probs * (teacher_logs_fp32 - student_logs_fp32)
        ).sum(dim=-1).mean()


def ac_state_to_zh(state: AcStateT, ls: LatentShape, h_dim: int) -> tuple[LatentT, HiddenT]:
    z, h = state[..., :-h_dim], state[..., -h_dim:]
    return z.unflatten(-1, ls), h


def rew_symlog_to_2hot(x: RewardSymlogT) -> RewardSymlogCatT:
    with _full_precision_context(x.device):
        x = x.float()
        hi = 20
        scale = N_CRITIC_BINS // 2 / hi
        x = x * scale
        b = x - x.floor()
        a = 1 - b
        res = torch.zeros(*x.shape[:-1], N_CRITIC_BINS, device=x.device)
        # If this raises a CUDA assert, the symlog target is outside the bins.
        res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2, a)
        res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2 + 1, b)
        return res


class ResidualCategoricalHead(nn.Module):
    """Preserve an MLP categorical head and add a zero-init residual branch."""

    def __init__(
        self,
        base_layers: list[nn.Module],
        *,
        module_input_features: int,
        residual_correction: str,
        residual_input_mode: str,
        residual_bottleneck_features: int,
        residual_grid_size: int,
        residual_input_min: float,
        residual_input_max: float,
        residual_rms_norm_epsilon: float,
        residual_alpha: float,
        residual_consolidation: str,
    ) -> None:
        super().__init__()
        if not base_layers or not isinstance(base_layers[-1], nn.Linear):
            raise TypeError("Residual categorical head requires a final linear layer")
        if residual_input_mode not in {"base_output", "module_input"}:
            raise ValueError(f"Unknown residual input mode: {residual_input_mode!r}")
        self.trunk = nn.Sequential(*base_layers[:-1])
        self.base_head = base_layers[-1]
        self.residual_input_mode = residual_input_mode
        from clworldmodel.models.residual_corrections import build_residual_correction

        self.residual = build_residual_correction(
            residual_correction,
            (
                self.base_head.in_features
                if residual_input_mode == "base_output"
                else module_input_features
            ),
            self.base_head.out_features,
            bottleneck_features=residual_bottleneck_features,
            grid_min=residual_input_min,
            grid_max=residual_input_max,
            num_grids=residual_grid_size,
            rms_norm_epsilon=residual_rms_norm_epsilon,
            alpha=residual_alpha,
            consolidation_enabled=residual_consolidation != "none",
        )
        if self.residual is None:
            raise ValueError("Residual categorical head requires a correction")

    def forward(self, state: AcStateT) -> torch.Tensor:
        features = self.trunk(state)
        residual_input = (
            features if self.residual_input_mode == "base_output" else state
        )
        logits = self.base_head(features) + self.residual(residual_input)
        return torch.log_softmax(logits, dim=-1)


def build_actor(
    in_dim: int,
    act_space: int,
    *,
    actor_network: str,
    h_dim: int,
    kan_hidden_features: int,
    kan_grid_size: int,
    kan_spline_order: int,
    kan_input_min: float,
    kan_input_max: float,
    kan_normalize_recurrent_state: bool,
    fastkan_hidden_features: int,
    fastkan_hidden_layers: int,
    fastkan_grid_size: int,
    fastkan_input_min: float,
    fastkan_input_max: float,
    fastkan_rms_norm_epsilon: float,
    fastkan_actor_output_scale: float,
    fastkan_actor_unimix: float,
) -> nn.Module:
    if actor_network == "mlp":
        return nn.Sequential(
            *get_mlp_layers(in_dim, act_space, final_activation=None),
            nn.LogSoftmax(-1),
        )
    if actor_network in {"relu_kan", "relu_kan_bounded", "relu_kan_adaptive"}:
        from clworldmodel.models.relu_kan import (
            AdaptiveReLUKANActor,
            BoundedReLUKANActor,
            ReLUKANActor,
        )

        actor_class = {
            "relu_kan": ReLUKANActor,
            "relu_kan_bounded": BoundedReLUKANActor,
            "relu_kan_adaptive": AdaptiveReLUKANActor,
        }[actor_network]
        return actor_class(
            in_dim,
            act_space,
            recurrent_features=h_dim,
            hidden_features=kan_hidden_features,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            input_min=kan_input_min,
            input_max=kan_input_max,
            normalize_recurrent_state=kan_normalize_recurrent_state,
        )
    if actor_network in {
        "fast_kan_ac",
        "fast_kan_ac_param_matched",
        "fast_kan_ac_stable",
    }:
        from clworldmodel.models.fast_kan import FastKANActor

        return FastKANActor(
            in_dim,
            act_space,
            hidden_features=fastkan_hidden_features,
            hidden_layers=fastkan_hidden_layers,
            grid_min=fastkan_input_min,
            grid_max=fastkan_input_max,
            num_grids=fastkan_grid_size,
            rms_norm_epsilon=fastkan_rms_norm_epsilon,
            output_scale=fastkan_actor_output_scale,
            unimix=fastkan_actor_unimix,
        )
    raise ValueError(f"Unknown actor network: {actor_network!r}")


def build_critic(
    in_dim: int,
    *,
    actor_network: str,
    fastkan_hidden_features: int,
    fastkan_hidden_layers: int,
    fastkan_grid_size: int,
    fastkan_input_min: float,
    fastkan_input_max: float,
    fastkan_rms_norm_epsilon: float,
) -> nn.Module:
    if actor_network in {
        "fast_kan_ac",
        "fast_kan_ac_param_matched",
        "fast_kan_ac_stable",
    }:
        from clworldmodel.models.fast_kan import FastKANCritic

        return FastKANCritic(
            in_dim,
            N_CRITIC_BINS,
            hidden_features=fastkan_hidden_features,
            hidden_layers=fastkan_hidden_layers,
            grid_min=fastkan_input_min,
            grid_max=fastkan_input_max,
            num_grids=fastkan_grid_size,
            rms_norm_epsilon=fastkan_rms_norm_epsilon,
        )

    critic_fcs = get_mlp_layers(in_dim, N_CRITIC_BINS, final_activation=None)
    # DreamerV3 initializes the categorical value output to a uniform distribution.
    torch.nn.init.constant_(critic_fcs[-1].weight, 0)
    torch.nn.init.constant_(critic_fcs[-1].bias, 0)
    return nn.Sequential(*critic_fcs, nn.LogSoftmax(-1))


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        act_space: int,
        *,
        actor_network: str = "mlp",
        h_dim: int = 512,
        kan_hidden_features: int = 64,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_input_min: float = 0.0,
        kan_input_max: float = 1.0,
        kan_normalize_recurrent_state: bool = True,
        fastkan_hidden_features: int = 34,
        fastkan_hidden_layers: int = 3,
        fastkan_grid_size: int = 8,
        fastkan_input_min: float = -2.0,
        fastkan_input_max: float = 2.0,
        fastkan_rms_norm_epsilon: float = 1e-4,
        fastkan_actor_output_scale: float = 0.01,
        fastkan_actor_unimix: float = 0.01,
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
        if residual_correction != "none" and actor_network != "mlp":
            raise ValueError("KARROW residuals require the unchanged MLP behavior heads")
        if residual_correction == "none":
            self.actor: Callable[[AcStateT], ActionLogT] = build_actor(
                in_dim,
                act_space,
                actor_network=actor_network,
                h_dim=h_dim,
                kan_hidden_features=kan_hidden_features,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
                kan_input_min=kan_input_min,
                kan_input_max=kan_input_max,
                kan_normalize_recurrent_state=kan_normalize_recurrent_state,
                fastkan_hidden_features=fastkan_hidden_features,
                fastkan_hidden_layers=fastkan_hidden_layers,
                fastkan_grid_size=fastkan_grid_size,
                fastkan_input_min=fastkan_input_min,
                fastkan_input_max=fastkan_input_max,
                fastkan_rms_norm_epsilon=fastkan_rms_norm_epsilon,
                fastkan_actor_output_scale=fastkan_actor_output_scale,
                fastkan_actor_unimix=fastkan_actor_unimix,
            )
            critic_layers = None
            residual_critic = None
        else:
            actor_layers = get_mlp_layers(in_dim, act_space, final_activation=None)
            critic_layers = get_mlp_layers(in_dim, N_CRITIC_BINS, final_activation=None)
            torch.nn.init.constant_(critic_layers[-1].weight, 0)
            torch.nn.init.constant_(critic_layers[-1].bias, 0)
            residual_kwargs = {
                "module_input_features": in_dim,
                "residual_correction": residual_correction,
                "residual_input_mode": residual_input_mode,
                "residual_bottleneck_features": residual_bottleneck_features,
                "residual_grid_size": residual_grid_size,
                "residual_input_min": residual_input_min,
                "residual_input_max": residual_input_max,
                "residual_rms_norm_epsilon": residual_rms_norm_epsilon,
                "residual_alpha": residual_alpha,
                "residual_consolidation": residual_consolidation,
            }
            if residual_input_mode == "module_input":
                # Keep base initialization and later training RNG paired with
                # the no-residual control while initializing both branches
                # independently inside the private stream.
                with torch.random.fork_rng(devices=[]):
                    self.actor = ResidualCategoricalHead(
                        actor_layers, **residual_kwargs
                    )
                    residual_critic = ResidualCategoricalHead(
                        critic_layers, **residual_kwargs
                    )
            else:
                self.actor = ResidualCategoricalHead(actor_layers, **residual_kwargs)
                residual_critic = None
        self.symlog_bins: torch.Tensor
        self.register_buffer(
            "symlog_bins", torch.linspace(-20, 20, N_CRITIC_BINS).float().unsqueeze(1)
        )

        if critic_layers is None:
            self.critic: Callable[[AcStateT], RewardSymlogCatT] = build_critic(
                in_dim,
                actor_network=actor_network,
                fastkan_hidden_features=fastkan_hidden_features,
                fastkan_hidden_layers=fastkan_hidden_layers,
                fastkan_grid_size=fastkan_grid_size,
                fastkan_input_min=fastkan_input_min,
                fastkan_input_max=fastkan_input_max,
                fastkan_rms_norm_epsilon=fastkan_rms_norm_epsilon,
            )
        else:
            self.critic = (
                residual_critic
                if residual_critic is not None
                else ResidualCategoricalHead(critic_layers, **residual_kwargs)
            )

    def freeze_shared_core(self) -> None:
        """Freeze MLP behavior heads while leaving their residual adapters plastic."""
        if not isinstance(self.actor, ResidualCategoricalHead) or not isinstance(
            self.critic, ResidualCategoricalHead
        ):
            raise ValueError("Frozen shared core requires residual actor and critic heads")
        for head in (self.actor, self.critic):
            head.trunk.requires_grad_(False)
            head.base_head.requires_grad_(False)
            head.residual.requires_grad_(True)

    def consolidation_penalty(self) -> torch.Tensor:
        if not isinstance(self.actor, ResidualCategoricalHead):
            return torch.zeros((), device=self.symlog_bins.device)
        if self.actor.residual.kind != "kan":
            return torch.zeros((), device=self.symlog_bins.device)
        if not isinstance(self.critic, ResidualCategoricalHead):
            raise RuntimeError("Residual actor requires a residual critic")
        return (
            self.actor.residual.consolidation_penalty()
            + self.critic.residual.consolidation_penalty()
        )

    def __call__(self, state: AcStateT) -> tuple[ActionLogT, RewardT]:
        return super().__call__(state)

    def forward(self, state: AcStateT) -> tuple[ActionLogT, RewardT]:
        # Supports T dimension
        return self.actor(state), self.value(state)

    def value(
        self,
        state: AcStateT,
        critic: Optional[Callable[[AcStateT], RewardSymlogCatT]] = None,
    ) -> RewardT:
        critic_network = self.critic if critic is None else critic
        return self.value_from_log_probs(critic_network(state))

    def value_from_log_probs(self, critic_preds_log: RewardSymlogCatT) -> RewardT:
        with _full_precision_context(critic_preds_log.device):
            critic_bins = critic_preds_log.float().exp()
            return symexp(critic_bins @ self.symlog_bins)

    def compute_loss(
        self,
        states: AcStateT,
        actions: ActionT,
        lam_returns: ReturnT,
        scale: float,
        actor_baseline_values: Optional[ReturnT] = None,
        slow_critic_preds_log: Optional[RewardSymlogCatT] = None,
        slow_critic_regularizer: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logs = self.actor(states).float()
        critic_preds_log = self.critic(states).float()
        with _full_precision_context(states.device):
            actions = actions.float()
            lam_returns = lam_returns.float()
            scale = torch.as_tensor(scale, device=states.device).float()

            # Actor gradients
            # `action_logs` (log probs): [ T N n_acts ]
            # `action_sample_logs` (log probs): [ T N 1 ]
            action_sample_logs = (action_logs * actions).sum(-1, keepdim=True)
            critic_values = self.value_from_log_probs(critic_preds_log.detach())
            if actor_baseline_values is None:
                actor_baseline_values = critic_values
            elif actor_baseline_values.shape != lam_returns.shape:
                raise ValueError(
                    "Actor baseline values and lambda returns must have equal shapes, "
                    f"got {actor_baseline_values.shape} and {lam_returns.shape}"
                )
            reinforce = (
                -action_sample_logs
                * (lam_returns - actor_baseline_values.detach().float())
                / scale
            ).mean()

            # [ T N n_acts ]
            entropy = td.Categorical(logits=action_logs).entropy().mean()

            # Critic gradients
            critic_targets = rew_symlog_to_2hot(symlog(lam_returns))
            critic_loss = -(critic_preds_log * critic_targets).sum(-1).mean()
            if slow_critic_regularizer:
                if slow_critic_preds_log is None:
                    raise ValueError(
                        "slow critic predictions are required by its regularizer"
                    )
                slow_values = self.value_from_log_probs(
                    slow_critic_preds_log.detach()
                )
                slow_targets = rew_symlog_to_2hot(symlog(slow_values))
                slow_loss = -(critic_preds_log * slow_targets).sum(-1).mean()
                critic_loss = critic_loss + slow_critic_regularizer * slow_loss

            return reinforce, entropy, critic_loss

    def compute_replay_critic_loss(
        self,
        states: AcStateT,
        targets: ReturnT,
        slow_critic_preds_log: Optional[RewardSymlogCatT] = None,
        slow_critic_regularizer: float = 0.0,
    ) -> torch.Tensor:
        critic_preds_log = self.critic(states).float()
        with _full_precision_context(states.device):
            targets = targets.float()
            critic_targets = rew_symlog_to_2hot(symlog(targets))
            critic_loss = -(critic_preds_log * critic_targets).sum(-1).mean()
            if slow_critic_regularizer:
                if slow_critic_preds_log is None:
                    raise ValueError(
                        "slow critic predictions are required by its regularizer"
                    )
                slow_values = self.value_from_log_probs(
                    slow_critic_preds_log.detach()
                )
                slow_targets = rew_symlog_to_2hot(symlog(slow_values))
                slow_loss = -(critic_preds_log * slow_targets).sum(-1).mean()
                critic_loss = critic_loss + slow_critic_regularizer * slow_loss
            return critic_loss


class ActorCriticTrainingStep(nn.Module):
    """Put the complete differentiable behavior loss behind one DDP forward."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        *,
        entropy_scale: float,
        replay_critic_loss_scale: float,
        slow_critic_regularizer: float,
    ) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.entropy_scale = entropy_scale
        self.replay_critic_loss_scale = replay_critic_loss_scale
        self.slow_critic_regularizer = slow_critic_regularizer

    def forward(
        self,
        states: AcStateT,
        actions: ActionT,
        lam_returns: ReturnT,
        scale: torch.Tensor,
        actor_baseline_values: Optional[ReturnT],
        slow_critic_preds_log: Optional[RewardSymlogCatT],
        replay_states: Optional[AcStateT],
        replay_targets: Optional[ReturnT],
        slow_replay_preds_log: Optional[RewardSymlogCatT],
    ) -> tuple[torch.Tensor, ...]:
        reinforce, entropy, critic_loss = self.actor_critic.compute_loss(
            states,
            actions,
            lam_returns,
            scale,
            actor_baseline_values=actor_baseline_values,
            slow_critic_preds_log=slow_critic_preds_log,
            slow_critic_regularizer=self.slow_critic_regularizer,
        )
        replay_critic_loss = torch.zeros((), device=states.device)
        if self.replay_critic_loss_scale:
            if replay_states is None or replay_targets is None:
                raise ValueError(
                    "replay critic inputs are required by its configured loss"
                )
            replay_critic_loss = self.actor_critic.compute_replay_critic_loss(
                replay_states,
                replay_targets,
                slow_critic_preds_log=slow_replay_preds_log,
                slow_critic_regularizer=self.slow_critic_regularizer,
            )
        consolidation_loss = self.actor_critic.consolidation_penalty()
        loss = (
            reinforce
            - self.entropy_scale * entropy
            + critic_loss
            + self.replay_critic_loss_scale * replay_critic_loss
            + consolidation_loss
        )
        return (
            loss,
            reinforce,
            entropy,
            critic_loss,
            replay_critic_loss,
            consolidation_loss,
        )


@dataclass(frozen=True)
class ReplayValueBatch:
    states: AcStateT
    rewards: RewardT
    continues: torch.Tensor


def replay_lambda_returns(
    rewards: RewardT,
    continues: torch.Tensor,
    bootstrap_values: ReturnT,
    *,
    discount: float,
    lam: float,
) -> ReturnT:
    """Build replay value targets under ARROW's same-index reward convention."""
    if rewards.shape != continues.shape or rewards.shape != bootstrap_values.shape:
        raise ValueError(
            "Replay rewards, continues, and bootstrap values must have equal shapes, "
            f"got {rewards.shape}, {continues.shape}, {bootstrap_values.shape}"
        )
    if rewards.shape[0] < 2:
        raise ValueError("Replay value targets require at least two context frames")

    with _full_precision_context(rewards.device):
        rewards = rewards.float()
        continues = continues.float()
        bootstrap_values = bootstrap_values.float()
        targets = torch.empty_like(bootstrap_values[:-1])
        next_return = bootstrap_values[-1]
        for t in reversed(range(rewards.shape[0] - 1)):
            live = discount * continues[t]
            next_return = rewards[t] + live * (
                (1.0 - lam) * bootstrap_values[t + 1] + lam * next_return
            )
            targets[t] = next_return
        return targets


@dataclass
class ActorCriticOpt:
    ac: ActorCritic
    opt: Optimizer
    slow_critic: Optional[nn.Module] = None
    return_scale_ema: Optional[torch.Tensor] = None
    return_mean_ema: Optional[torch.Tensor] = None


def build_actor_critic_opt(
    wm: WorldModel,
    *,
    lr: float,
    actor_network: str = "mlp",
    actor_kan_hidden_features: int = 64,
    actor_kan_grid_size: int = 5,
    actor_kan_spline_order: int = 3,
    actor_kan_input_min: float = 0.0,
    actor_kan_input_max: float = 1.0,
    actor_kan_normalize_recurrent_state: bool = True,
    fastkan_hidden_features: int = 34,
    fastkan_hidden_layers: int = 3,
    fastkan_grid_size: int = 8,
    fastkan_input_min: float = -2.0,
    fastkan_input_max: float = 2.0,
    fastkan_rms_norm_epsilon: float = 1e-4,
    fastkan_actor_output_scale: float = 0.01,
    fastkan_actor_unimix: float = 0.01,
    optimizer_name: str = "adam",
    optimizer_eps: float = 1e-8,
    optimizer_beta1: float = 0.9,
    optimizer_beta2: float = 0.999,
    optimizer_warmup_steps: int = 0,
    agc_clip: float = 0.0,
    slow_critic_regularizer: float = 0.0,
    slow_critic_decay: float = 0.98,
    residual_correction: str = "none",
    residual_bottleneck_features: int = 64,
    residual_grid_size: int = 8,
    residual_input_min: float = -2.0,
    residual_input_max: float = 2.0,
    residual_rms_norm_epsilon: float = 1e-4,
    residual_alpha: float = 0.1,
    residual_input_mode: str = "base_output",
    residual_consolidation: str = "none",
) -> ActorCriticOpt:
    """Construct an actor-critic optimizer before the first update.

    Resume experiments need the loaded actor during their first environment
    collection. Keeping construction here avoids a dummy update just to
    materialize the actor-critic wrapper.
    """
    ac = ActorCritic(
        np.prod(wm.ls) + wm.h_dim,
        wm.a_dim,
        actor_network=actor_network,
        h_dim=wm.h_dim,
        kan_hidden_features=actor_kan_hidden_features,
        kan_grid_size=actor_kan_grid_size,
        kan_spline_order=actor_kan_spline_order,
        kan_input_min=actor_kan_input_min,
        kan_input_max=actor_kan_input_max,
        kan_normalize_recurrent_state=actor_kan_normalize_recurrent_state,
        fastkan_hidden_features=fastkan_hidden_features,
        fastkan_hidden_layers=fastkan_hidden_layers,
        fastkan_grid_size=fastkan_grid_size,
        fastkan_input_min=fastkan_input_min,
        fastkan_input_max=fastkan_input_max,
        fastkan_rms_norm_epsilon=fastkan_rms_norm_epsilon,
        fastkan_actor_output_scale=fastkan_actor_output_scale,
        fastkan_actor_unimix=fastkan_actor_unimix,
        residual_correction=residual_correction,
        residual_bottleneck_features=residual_bottleneck_features,
        residual_grid_size=residual_grid_size,
        residual_input_min=residual_input_min,
        residual_input_max=residual_input_max,
        residual_rms_norm_epsilon=residual_rms_norm_epsilon,
        residual_alpha=residual_alpha,
        residual_input_mode=residual_input_mode,
        residual_consolidation=residual_consolidation,
    ).to(next(wm.parameters()).device)
    if optimizer_name == "adam":
        opt = Adam(
            ac.parameters(),
            lr=lr,
            betas=(optimizer_beta1, optimizer_beta2),
            eps=optimizer_eps,
        )
    elif optimizer_name == "laprop":
        from clworldmodel.optim import LaProp

        opt = LaProp(
            ac.parameters(),
            lr=lr,
            betas=(optimizer_beta1, optimizer_beta2),
            eps=optimizer_eps,
            agc_clip=agc_clip,
            warmup_steps=optimizer_warmup_steps,
        )
    else:
        raise ValueError(f"Unknown actor-critic optimizer: {optimizer_name!r}")
    slow_critic = None
    if slow_critic_regularizer:
        slow_critic = copy.deepcopy(ac.critic).eval()
        slow_critic.requires_grad_(False)
    return ActorCriticOpt(ac, opt, slow_critic=slow_critic)


@torch.no_grad()
def dream_rollout(
    wm: WorldModel,
    ac: ActorCritic,
    data: Replay,
    n_sync: int = 1,
    n_steps: int = 16,
    discount: float = 0.997,
    lam: float = 0.95,
    temperature: float = 1.0,
    n_ctx_frames: int = 4,
    target_value: Optional[ValueFunction] = None,
    corrected_terminal_bootstrap: bool = False,
    feature_cache: Optional[object] = None,
    task_id: Optional[int] = None,
) -> tuple[AcStateT, ActionT, RewardT, ReturnT, ReplayValueBatch]:
    # Returns: (T=n_steps N=n_sync)
    # States [ T N n_dis n_cls ]
    # Actions [ T N 18 ]
    # Rewards [ T N 1 ]
    # Lambda returns: [ T N 1 ]
    z, h = wm.rssm.initial_state(n_sync)
    compute_dtype = getattr(wm, "compute_dtype", "float32")
    no_reset = torch.zeros(n_sync, 1, device=z.device)
    # Arbitrary (n_ctx_frames) context frames
    if feature_cache is None:
        if task_id is None:
            sample = data.minibatch(n_ctx_frames, n_sync, mb_device=z.device)
        else:
            sample = data.minibatch(
                n_ctx_frames, n_sync, mb_device=z.device, task_id=task_id
            )
        ctx_acts, ctx_images, ctx_rewards, ctx_conts, ctx_resets = sample
        assert ctx_images.shape == (n_ctx_frames, n_sync, 3, 64, 64), ctx_images.shape
        rssm_kwargs = {"temperature": temperature}
        if task_id is not None:
            rssm_kwargs["task_id"] = task_id
        with _autocast_context(z.device, compute_dtype):
            _, context_z, context_h = wm.rssm(
                z, ctx_acts, h, ctx_images, ctx_resets, **rssm_kwargs
            )
    else:
        feature_kwargs = {"mb_device": z.device}
        if task_id is not None:
            feature_kwargs["task_id"] = task_id
        feature_sample = feature_cache.minibatch(
            n_ctx_frames, n_sync, **feature_kwargs
        )
        ctx_acts, _, ctx_features, ctx_rewards, ctx_conts, ctx_resets = feature_sample
        observe_kwargs = {"temperature": temperature}
        if task_id is not None:
            observe_kwargs["task_id"] = task_id
        with _autocast_context(z.device, compute_dtype):
            _, context_z, context_h = wm.rssm.observe_embeddings(
                z,
                ctx_acts,
                h,
                wm.rssm.adapt_observation_embeddings(ctx_features),
                ctx_resets,
                **observe_kwargs,
            )
    replay_value_batch = ReplayValueBatch(
        states=zh_to_ac_state(context_z, context_h),
        rewards=ctx_rewards,
        continues=ctx_conts,
    )
    z = context_z[-1]
    h = context_h[-1]

    states = []
    actions = []
    rewards = []
    returns_preds = []
    conts = []
    with _autocast_context(z.device, compute_dtype):
        for _ in range(n_steps):
            state = zh_to_ac_state(z, h)
            zh = wm.zh_transform(z, h)
            if hasattr(wm, "predict_reward_symlog"):
                reward_symlog = wm.predict_reward_symlog(zh, task_id)
            else:
                reward_symlog = wm.reward_fc(zh)
                reward_residual = getattr(wm, "reward_residual", None)
                if reward_residual is not None:
                    reward_symlog = reward_symlog + reward_residual(zh)
            reward = symexp(reward_symlog)
            if hasattr(wm, "predict_continue"):
                cont = wm.predict_continue(zh, task_id).float()
            else:
                cont_logits = wm.continue_fc(zh)
                continue_residual = getattr(wm, "continue_residual", None)
                if continue_residual is not None:
                    cont = torch.sigmoid(
                        cont_logits + continue_residual(zh)
                    ).float()
                else:
                    cont = cont_logits.float()
            if target_value is None:
                action_log, returns_pred = ac(state)
                returns_preds.append(returns_pred.float())
            else:
                action_log = ac.actor(state)
            with _full_precision_context(action_log.device):
                action_dist = td.OneHotCategorical(logits=action_log.float())
                action = action_dist.sample()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            conts.append(cont)

            rssm_kwargs = {"temperature": temperature}
            if task_id is not None:
                rssm_kwargs["task_id"] = task_id
            _, z, h = wm.rssm(z, action, h, None, no_reset, **rssm_kwargs)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).float()
        conts = torch.stack(conts).float()
        # False preserves the recorded ARROW/FastKAN pilot's pre-transition bootstrap.
        bootstrap_state = zh_to_ac_state(z, h) if corrected_terminal_bootstrap else state
        if target_value is None:
            returns_preds = torch.stack(returns_preds).float()
            _, final_returns_pred = ac(bootstrap_state)
            final_returns_pred = final_returns_pred.float()
        else:
            target_states = torch.cat((states, bootstrap_state.unsqueeze(0)), dim=0)
            target_values = target_value(target_states).float()
            returns_preds = target_values[:-1]
            final_returns_pred = target_values[-1]

    # Compute returns
    lam_returns = torch.zeros_like(returns_preds, device=returns_preds.device)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_returns_pred = final_returns_pred
            next_r = final_returns_pred
        else:
            next_returns_pred = returns_preds[t + 1]
            next_r = lam_returns[t + 1]
        lam_returns[t] = rewards[t] + discount * conts[t] * (
            (1 - lam) * next_returns_pred + lam * next_r
        )

    return states, actions, rewards, lam_returns, replay_value_batch


@torch.no_grad()
def _graded_dream_rehearsal_batch(
    wm: WorldModel,
    ac: ActorCritic,
    data: Replay,
    *,
    replay_task_id: int,
    n_sync: int,
    context_steps: int,
    dream_steps: int,
    discount: float,
    top_fraction: float,
    realized_threshold: float,
    realized_bonus: float,
    temperature: float = 1.0,
) -> tuple[AcStateT, ActionT, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate and grade one paper-style dream self-imitation batch.

    ``replay_task_id`` selects old-task start states only.  It is deliberately
    not forwarded to the world model or Actor-Critic: Bounded Dream Rehearsal
    has one shared model and policy, and task metadata belongs solely to the
    scheduler/replay sampler.
    """

    from clworldmodel.continual.dream_rehearsal import (
        realized_first_scores,
        top_fraction_indices,
    )

    if replay_task_id < 0:
        raise ValueError("Dream-rehearsal replay task must be non-negative")
    z, h = wm.rssm.initial_state(n_sync)
    compute_dtype = getattr(wm, "compute_dtype", "float32")
    ctx_acts, ctx_images, _, _, ctx_resets = data.minibatch(
        context_steps,
        n_sync,
        mb_device=z.device,
        task_id=replay_task_id,
    )
    with _autocast_context(z.device, compute_dtype):
        _, context_z, context_h = wm.rssm(
            z,
            ctx_acts,
            h,
            ctx_images,
            ctx_resets,
            temperature=temperature,
        )
        # The reference artifact imagines from every posterior state in its
        # replay batch, rather than keeping only the final context state.
        # Flatten only the public [T, N] axes; latent layouts remain opaque.
        z = context_z.flatten(0, 1)
        h = context_h.flatten(0, 1)
        dream_batch = z.shape[0]
        no_reset = torch.zeros(dream_batch, 1, device=z.device)
        states = []
        actions = []
        rewards = []
        continues = []
        for _ in range(dream_steps):
            state = zh_to_ac_state(z, h)
            model_state = wm.zh_transform(z, h)
            reward = symexp(wm.predict_reward_symlog(model_state, None))
            cont = wm.predict_continue(model_state, None)
            action_logs = ac.actor(state)
            with _full_precision_context(action_logs.device):
                action = td.OneHotCategorical(
                    logits=action_logs.float()
                ).sample()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            continues.append(cont)
            _, z, h = wm.rssm(
                z,
                action,
                h,
                None,
                no_reset,
                temperature=temperature,
            )

        states_tensor = torch.stack(states).detach()
        actions_tensor = torch.stack(actions).detach()
        rewards_tensor = torch.stack(rewards).float()
        continues_tensor = torch.stack(continues).float()
        bootstrap_values = ac.value(zh_to_ac_state(z, h)).float()

    scores, realized, _ = realized_first_scores(
        rewards_tensor,
        continues_tensor,
        bootstrap_values,
        discount=discount,
        realized_threshold=realized_threshold,
        realized_bonus=realized_bonus,
    )
    selected = top_fraction_indices(scores, top_fraction)
    return states_tensor, actions_tensor, selected, scores, realized


def train_bounded_dream_rehearsal(
    wm: WorldModel,
    data: Replay,
    aco: ActorCriticOpt,
    *,
    replay_task_id: int,
    updates: int,
    n_sync: int,
    context_steps: int,
    dream_steps: int,
    discount: float,
    top_fraction: float,
    realized_threshold: float,
    realized_bonus: float,
    grad_clip: float = 100.0,
) -> dict[str, float]:
    """Run actor-only graded dream rehearsal from one bounded task library."""

    from clworldmodel.continual.dream_rehearsal import (
        selected_behavior_cloning_loss,
    )

    if updates < 1:
        raise ValueError("Dream rehearsal requires at least one actor update")
    if grad_clip < 0:
        raise ValueError("Dream-rehearsal gradient clipping must be non-negative")
    if not data.available_task_ids() or replay_task_id not in data.available_task_ids():
        raise ValueError(
            f"Bounded replay contains no rehearsal starts for task {replay_task_id}"
        )
    ac, opt = aco.ac, aco.opt
    compute_dtype = getattr(wm, "compute_dtype", "float32")
    loss_total = torch.zeros((), device=next(wm.parameters()).device, dtype=torch.float64)
    score_total = torch.zeros_like(loss_total)
    success_fraction_total = torch.zeros_like(loss_total)
    selected_total = 0
    dreamed_total = 0
    for _ in range(updates):
        states, actions, selected, scores, realized = (
            _graded_dream_rehearsal_batch(
                wm,
                ac,
                data,
                replay_task_id=replay_task_id,
                n_sync=n_sync,
                context_steps=context_steps,
                dream_steps=dream_steps,
                discount=discount,
                top_fraction=top_fraction,
                realized_threshold=realized_threshold,
                realized_bonus=realized_bonus,
            )
        )
        with _autocast_context(states.device, compute_dtype):
            actor_log_probs = ac.actor(states)
        loss = selected_behavior_cloning_loss(
            actor_log_probs,
            actions,
            selected,
        )
        _assert_all_finite(loss, "dream-rehearsal behavior-cloning loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), grad_clip)
        opt.step()

        loss_total.add_(loss.detach().to(torch.float64))
        score_total.add_(scores.detach().mean().to(torch.float64))
        success_fraction_total.add_(
            (realized.detach() > realized_threshold)
            .float()
            .mean()
            .to(torch.float64)
        )
        selected_total += selected.numel()
        dreamed_total += scores.numel()

    values = torch.stack(
        (loss_total, score_total, success_fraction_total)
    ).detach().cpu().tolist()
    return {
        "actor_bc_loss": values[0] / updates,
        "dream_score_mean": values[1] / updates,
        "realized_success_fraction": values[2] / updates,
        "actor_updates": float(updates),
        "selected_trajectories": float(selected_total),
        "dreamed_trajectories": float(dreamed_total),
    }


def train_ac_from_wm(
    wm: WorldModel,
    data: Replay,
    steps: int,
    n_sync: int = 16,
    dream_steps: int = 16,
    aco: Optional[ActorCriticOpt] = None,
    lr: float = 3e-5,
    actor_network: str = "mlp",
    actor_kan_hidden_features: int = 64,
    actor_kan_grid_size: int = 5,
    actor_kan_spline_order: int = 3,
    actor_kan_input_min: float = 0.0,
    actor_kan_input_max: float = 1.0,
    actor_kan_normalize_recurrent_state: bool = True,
    fastkan_hidden_features: int = 34,
    fastkan_hidden_layers: int = 3,
    fastkan_grid_size: int = 8,
    fastkan_input_min: float = -2.0,
    fastkan_input_max: float = 2.0,
    fastkan_rms_norm_epsilon: float = 1e-4,
    fastkan_actor_output_scale: float = 0.01,
    fastkan_actor_unimix: float = 0.01,
    optimizer_name: str = "adam",
    optimizer_eps: float = 1e-8,
    optimizer_beta1: float = 0.9,
    optimizer_beta2: float = 0.999,
    optimizer_warmup_steps: int = 0,
    agc_clip: float = 0.0,
    grad_clip: float = 100.0,
    discount: float = 0.997,
    lam: float = 0.95,
    entropy_scale: float = 3e-4,
    return_norm_decay: float = 0.99,
    persistent_return_norm: bool = False,
    slow_critic_regularizer: float = 0.0,
    slow_critic_decay: float = 0.98,
    replay_critic_loss_scale: float = 0.0,
    use_slow_critic_targets: bool = False,
    corrected_imagination_bootstrap: bool = False,
    residual_correction: str = "none",
    residual_bottleneck_features: int = 64,
    residual_grid_size: int = 8,
    residual_input_min: float = -2.0,
    residual_input_max: float = 2.0,
    residual_rms_norm_epsilon: float = 1e-4,
    residual_alpha: float = 0.1,
    residual_input_mode: str = "base_output",
    residual_consolidation: str = "none",
    protect_residual_updates: bool = False,
    feature_cache: Optional[object] = None,
    task_id: Optional[int] = None,
    task_id_schedule: Optional[Sequence[int]] = None,
    actor_teacher: Optional[nn.Module] = None,
    actor_distill_task_ids: Sequence[int] = (),
    actor_distill_scale: float = 0.0,
    actor_distill_interval: int = 1,
    actor_distill_n_sync: int = 1,
    actor_distill_burnin_steps: int = 0,
    actor_distill_steps: int = 1,
    distributed_context: Optional[object] = None,
) -> tuple[ActorCriticOpt, torch.Tensor, dict[str, float]]:
    if task_id_schedule is not None:
        if task_id is not None:
            raise ValueError("task_id and task_id_schedule are mutually exclusive")
        if len(task_id_schedule) != steps:
            raise ValueError(
                "Actor-critic task schedule must contain exactly one task id per "
                f"optimizer update: {len(task_id_schedule)} != {steps}"
            )
        if any(scheduled_task_id < 0 for scheduled_task_id in task_id_schedule):
            raise ValueError("Actor-critic task schedule ids must be non-negative")
    if aco is None:
        aco = build_actor_critic_opt(
            wm,
            lr=lr,
            actor_network=actor_network,
            actor_kan_hidden_features=actor_kan_hidden_features,
            actor_kan_grid_size=actor_kan_grid_size,
            actor_kan_spline_order=actor_kan_spline_order,
            actor_kan_input_min=actor_kan_input_min,
            actor_kan_input_max=actor_kan_input_max,
            actor_kan_normalize_recurrent_state=actor_kan_normalize_recurrent_state,
            fastkan_hidden_features=fastkan_hidden_features,
            fastkan_hidden_layers=fastkan_hidden_layers,
            fastkan_grid_size=fastkan_grid_size,
            fastkan_input_min=fastkan_input_min,
            fastkan_input_max=fastkan_input_max,
            fastkan_rms_norm_epsilon=fastkan_rms_norm_epsilon,
            fastkan_actor_output_scale=fastkan_actor_output_scale,
            fastkan_actor_unimix=fastkan_actor_unimix,
            optimizer_name=optimizer_name,
            optimizer_eps=optimizer_eps,
            optimizer_beta1=optimizer_beta1,
            optimizer_beta2=optimizer_beta2,
            optimizer_warmup_steps=optimizer_warmup_steps,
            agc_clip=agc_clip,
            slow_critic_regularizer=(
                slow_critic_regularizer or use_slow_critic_targets
            ),
            slow_critic_decay=slow_critic_decay,
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
    ac, opt = aco.ac, aco.opt
    trainable_parameters = [
        parameter for parameter in ac.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Actor-critic has no trainable parameters")
    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    for parameter in list(opt.state):
        if id(parameter) not in trainable_ids:
            del opt.state[parameter]
    for parameter_group in opt.param_groups:
        parameter_group["params"] = trainable_parameters
    if use_slow_critic_targets and aco.slow_critic is None:
        raise ValueError("Slow critic targets require an initialized slow critic")
    for g in opt.param_groups:
        g["lr"] = lr
    distributed_enabled = bool(
        distributed_context is not None
        and getattr(distributed_context, "enabled", False)
    )
    distillation_enabled = actor_teacher is not None
    if distillation_enabled != bool(actor_distill_task_ids):
        raise ValueError(
            "Actor distillation requires both a frozen teacher and old task routes"
        )
    if distillation_enabled:
        if task_id is None or any(
            old_task_id < 0 or old_task_id >= task_id
            for old_task_id in actor_distill_task_ids
        ):
            raise ValueError(
                "Actor distillation routes must be non-negative tasks older than "
                "the current task"
            )
        if actor_distill_scale <= 0 or actor_distill_interval < 1:
            raise ValueError(
                "Actor distillation requires a positive scale and interval"
            )
        if actor_distill_n_sync < 1 or actor_distill_steps < 1:
            raise ValueError(
                "Actor distillation requires positive batch and rollout sizes"
            )
        if actor_distill_burnin_steps < 0:
            raise ValueError("Actor distillation burn-in must be non-negative")
        if distributed_enabled:
            raise ValueError(
                "Shared-actor imagination distillation is validated only on one GPU"
            )
        actor_teacher.eval()
    elif actor_distill_scale:
        raise ValueError("Actor distillation scale requires a frozen teacher")
    distributed_training_step = None
    if distributed_enabled:
        training_step = ActorCriticTrainingStep(
            ac,
            entropy_scale=entropy_scale,
            replay_critic_loss_scale=replay_critic_loss_scale,
            slow_critic_regularizer=slow_critic_regularizer,
        )
        distributed_training_step = distributed_context.wrap_module(training_step)
    scale_ema = aco.return_scale_ema if persistent_return_norm else None
    lam_returns_mean_ema = aco.return_mean_ema if persistent_return_norm else None

    metric_names = (
        "actor_reinforce_loss",
        "actor_entropy",
        "critic_imagination_loss",
        "critic_replay_loss",
        "kan_consolidation_loss",
        "total_loss",
        "return_mean",
        "return_scale",
        "gradient_norm",
    )
    metric_sums = torch.zeros(
        len(metric_names),
        device=next(wm.parameters()).device,
        dtype=torch.float64,
    )
    actor_old_policy_kl_total = torch.zeros(
        (), device=metric_sums.device, dtype=torch.float64
    )
    actor_distillation_batches = 0
    actor_distillation_states = 0
    actor_distillation_burnin_state_uses = 0
    capture_kan_parameter_values = None
    protect_kan_parameter_updates = None
    if protect_residual_updates:
        if residual_consolidation != "replay_functional":
            raise ValueError(
                "Protected residual updates require replay-functional consolidation"
            )
        from clworldmodel.continual import (
            capture_kan_parameter_values,
            protect_kan_parameter_updates,
        )
    progbar = trange(steps, desc="Train AC from WM",disable=True)
    for step in progbar:
        rollout_task_id = (
            int(task_id_schedule[step])
            if task_id_schedule is not None
            else task_id
        )
        target_value = None
        if use_slow_critic_targets:
            target_value = lambda state: ac.value(state, critic=aco.slow_critic)
        states, actions, _, lam_returns, replay_value_batch = dream_rollout(
            wm,
            ac,
            data,
            n_sync=n_sync,
            n_steps=dream_steps,
            discount=discount,
            lam=lam,
            target_value=target_value,
            corrected_terminal_bootstrap=corrected_imagination_bootstrap,
            feature_cache=feature_cache,
            task_id=rollout_task_id,
        )

        statistics_returns = (
            distributed_context.all_gather_sequence_batch(lam_returns)
            if distributed_enabled
            else lam_returns
        )
        scale = torch.quantile(statistics_returns, 0.95) - torch.quantile(
            statistics_returns, 0.05
        )
        lam_returns_mean = statistics_returns.mean()
        if scale_ema is None:
            scale_ema = scale
            lam_returns_mean_ema = lam_returns_mean
        else:
            scale_ema = return_norm_decay * scale_ema + (1 - return_norm_decay) * scale
            lam_returns_mean_ema = (
                return_norm_decay * lam_returns_mean_ema
                + (1 - return_norm_decay) * lam_returns_mean
            )

        one = torch.tensor(1, device=scale.device)
        compute_dtype = getattr(wm, "compute_dtype", "float32")
        if distributed_enabled:
            slow_critic_preds_log = None
            actor_baseline_values = None
            replay_states = None
            replay_targets = None
            slow_replay_preds_log = None
            with _autocast_context(states.device, compute_dtype):
                if aco.slow_critic is not None:
                    with torch.no_grad():
                        slow_critic_preds_log = aco.slow_critic(states)
                if use_slow_critic_targets:
                    actor_baseline_values = ac.value_from_log_probs(
                        slow_critic_preds_log
                    )
                if replay_critic_loss_scale:
                    with torch.no_grad():
                        replay_bootstrap_values = ac.value(
                            replay_value_batch.states,
                            critic=aco.slow_critic if use_slow_critic_targets else None,
                        )
                        replay_targets = replay_lambda_returns(
                            replay_value_batch.rewards,
                            replay_value_batch.continues,
                            replay_bootstrap_values,
                            discount=discount,
                            lam=lam,
                        )
                        replay_states = replay_value_batch.states[:-1]
                        slow_replay_preds_log = (
                            aco.slow_critic(replay_states)
                            if aco.slow_critic is not None
                            else None
                        )
                (
                    loss,
                    reinforce,
                    entropy,
                    critic_loss,
                    replay_critic_loss,
                    consolidation_loss,
                ) = distributed_training_step(
                    states,
                    actions,
                    lam_returns,
                    torch.max(one, scale_ema),
                    actor_baseline_values,
                    slow_critic_preds_log,
                    replay_states,
                    replay_targets,
                    slow_replay_preds_log,
                )
        else:
            slow_critic_preds_log = None
            with _autocast_context(states.device, compute_dtype):
                if aco.slow_critic is not None:
                    with torch.no_grad():
                        slow_critic_preds_log = aco.slow_critic(states)
                actor_baseline_values = None
                if use_slow_critic_targets:
                    actor_baseline_values = ac.value_from_log_probs(
                        slow_critic_preds_log
                    )
                reinforce, entropy, critic_loss = ac.compute_loss(
                    states,
                    actions,
                    lam_returns,
                    torch.max(one, scale_ema),
                    actor_baseline_values=actor_baseline_values,
                    slow_critic_preds_log=slow_critic_preds_log,
                    slow_critic_regularizer=slow_critic_regularizer,
                )
            replay_critic_loss = torch.zeros((), device=states.device)
            if replay_critic_loss_scale:
                with _autocast_context(states.device, compute_dtype):
                    with torch.no_grad():
                        replay_bootstrap_values = ac.value(
                            replay_value_batch.states,
                            critic=aco.slow_critic if use_slow_critic_targets else None,
                        )
                        replay_targets = replay_lambda_returns(
                            replay_value_batch.rewards,
                            replay_value_batch.continues,
                            replay_bootstrap_values,
                            discount=discount,
                            lam=lam,
                        )
                        slow_replay_preds_log = (
                            aco.slow_critic(replay_value_batch.states[:-1])
                            if aco.slow_critic is not None
                            else None
                        )
                    replay_critic_loss = ac.compute_replay_critic_loss(
                        replay_value_batch.states[:-1],
                        replay_targets,
                        slow_critic_preds_log=slow_replay_preds_log,
                        slow_critic_regularizer=slow_critic_regularizer,
                    )
            consolidation_loss = ac.consolidation_penalty()
            loss = (
                reinforce
                - entropy_scale * entropy
                + critic_loss
                + replay_critic_loss_scale * replay_critic_loss
                + consolidation_loss
            )
        actor_old_policy_kl = torch.zeros((), device=states.device)
        if distillation_enabled and step % actor_distill_interval == 0:
            old_task_id = actor_distill_task_ids[
                actor_distillation_batches % len(actor_distill_task_ids)
            ]
            old_states, old_teacher_logs = dream_frozen_actor_policy(
                wm,
                actor_teacher,
                task_id=old_task_id,
                n_sync=actor_distill_n_sync,
                burnin_steps=actor_distill_burnin_steps,
                dream_steps=actor_distill_steps,
            )
            with _autocast_context(states.device, compute_dtype):
                actor_old_policy_kl = actor_policy_kl(
                    ac.actor, old_states, old_teacher_logs
                )
            loss = loss + actor_distill_scale * actor_old_policy_kl
            actor_distillation_batches += 1
            actor_distillation_states += actor_distill_n_sync * actor_distill_steps
            actor_distillation_burnin_state_uses += (
                actor_distill_n_sync * actor_distill_burnin_steps
            )
            actor_old_policy_kl_total.add_(
                actor_old_policy_kl.detach().to(torch.float64)
            )
        _assert_all_finite(loss, "actor-critic loss")

        protected_values = (
            capture_kan_parameter_values(ac)
            if capture_kan_parameter_values is not None
            else None
        )
        opt.zero_grad()
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in ac.parameters()
                if parameter.grad is not None
            )
        )
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(ac.parameters(), grad_clip)
        opt.step()
        if protected_values is not None:
            protect_kan_parameter_updates(ac, protected_values)
        if aco.slow_critic is not None:
            with torch.no_grad():
                for target, source in zip(
                    aco.slow_critic.parameters(), ac.critic.parameters()
                ):
                    target.mul_(slow_critic_decay).add_(
                        source, alpha=1.0 - slow_critic_decay
                    )

        step_metrics = {
            "actor_reinforce_loss": reinforce,
            "actor_entropy": entropy,
            "critic_imagination_loss": critic_loss,
            "critic_replay_loss": replay_critic_loss,
            "kan_consolidation_loss": consolidation_loss,
            "total_loss": loss,
            "return_mean": lam_returns_mean,
            "return_scale": torch.max(one, scale_ema),
            "gradient_norm": gradient_norm,
        }
        step_metric_values = torch.stack(
            tuple(value.detach().float() for value in step_metrics.values())
        ).to(torch.float64)
        _assert_all_finite(step_metric_values, "actor-critic metric vector")
        metric_sums.add_(step_metric_values)

        if not progbar.disable and step % 50 == 0:
            progbar.set_postfix(
                {
                    "Actor entropy": f"{entropy.item():.3f}",
                    "Lam returns mean EMA": f"{lam_returns_mean_ema.item():.3f}",
                }
            )

    if persistent_return_norm:
        aco.return_scale_ema = scale_ema.detach()
        aco.return_mean_ema = lam_returns_mean_ema.detach()
    summary_values = torch.cat(
        (metric_sums, actor_old_policy_kl_total.unsqueeze(0))
    ).detach().cpu().tolist()
    metric_totals = dict(zip(metric_names, summary_values[:-1]))
    actor_old_policy_kl_total_value = summary_values[-1]
    if distributed_enabled:
        metric_totals = distributed_context.mean_float_mapping(metric_totals)
    metrics = {name: total / steps for name, total in metric_totals.items()}
    metrics["replay_critic_loss_scale"] = replay_critic_loss_scale
    metrics["actor_old_policy_kl"] = (
        actor_old_policy_kl_total_value / actor_distillation_batches
        if actor_distillation_batches
        else 0.0
    )
    metrics["shared_actor_distillation_batches"] = float(
        actor_distillation_batches
    )
    metrics["shared_actor_distillation_states"] = float(
        actor_distillation_states
    )
    metrics["shared_actor_distillation_burnin_state_uses"] = float(
        actor_distillation_burnin_state_uses
    )
    return aco, lam_returns_mean_ema, metrics
