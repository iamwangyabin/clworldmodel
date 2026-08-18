import copy
from dataclasses import dataclass
from typing import Callable, Optional

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


def zh_to_ac_state(z: LatentT, h: HiddenT) -> AcStateT:
    return torch.cat((z.flatten(-2), h), dim=-1)


def ac_state_to_zh(state: AcStateT, ls: LatentShape, h_dim: int) -> tuple[LatentT, HiddenT]:
    z, h = state[..., :-h_dim], state[..., -h_dim:]
    return z.unflatten(-1, ls), h


def rew_symlog_to_2hot(x: RewardSymlogT) -> RewardSymlogCatT:
    hi = 20
    scale = N_CRITIC_BINS // 2 / hi
    x = x * scale
    b = x - x.floor()
    a = 1 - b
    res = torch.zeros(*x.shape[:-1], N_CRITIC_BINS, device=x.device)
    # If you get some weird CUDA assert error, it's because `x` is under/overflowing here
    res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2, a)
    res.scatter_(-1, x.floor().long() + N_CRITIC_BINS // 2 + 1, b)
    return res


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
    ) -> None:
        super().__init__()
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
        self.symlog_bins: torch.Tensor
        self.register_buffer(
            "symlog_bins", torch.linspace(-20, 20, N_CRITIC_BINS).float().unsqueeze(1)
        )

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
        critic_bins = critic_preds_log.exp()
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
        action_logs = self.actor(states)
        critic_preds_log = self.critic(states)

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
            * (lam_returns - actor_baseline_values.detach())
            / scale
        ).mean()

        # [ T N n_acts ]
        entropy = td.Categorical(logits=action_logs).entropy().mean()

        # Critic gradients
        critic_targets = rew_symlog_to_2hot(symlog(lam_returns))
        critic_loss = -(critic_preds_log * critic_targets).sum(-1).mean()
        if slow_critic_regularizer:
            if slow_critic_preds_log is None:
                raise ValueError("slow critic predictions are required by its regularizer")
            slow_values = symexp(
                slow_critic_preds_log.detach().exp() @ self.symlog_bins
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
        critic_preds_log = self.critic(states)
        critic_targets = rew_symlog_to_2hot(symlog(targets))
        critic_loss = -(critic_preds_log * critic_targets).sum(-1).mean()
        if slow_critic_regularizer:
            if slow_critic_preds_log is None:
                raise ValueError("slow critic predictions are required by its regularizer")
            slow_values = symexp(
                slow_critic_preds_log.detach().exp() @ self.symlog_bins
            )
            slow_targets = rew_symlog_to_2hot(symlog(slow_values))
            slow_loss = -(critic_preds_log * slow_targets).sum(-1).mean()
            critic_loss = critic_loss + slow_critic_regularizer * slow_loss
        return critic_loss


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
) -> tuple[AcStateT, ActionT, RewardT, ReturnT, ReplayValueBatch]:
    # Returns: (T=n_steps N=n_sync)
    # States [ T N n_dis n_cls ]
    # Actions [ T N 18 ]
    # Rewards [ T N 1 ]
    # Lambda returns: [ T N 1 ]
    z, h = wm.rssm.initial_state(n_sync)
    no_reset = torch.zeros(n_sync, 1, device=z.device)
    # Arbitrary (n_ctx_frames) context frames
    ctx_acts, ctx_images, ctx_rewards, ctx_conts, ctx_resets = data.minibatch(
        n_ctx_frames, n_sync, mb_device=z.device
    )
    assert ctx_images.shape == (n_ctx_frames, n_sync, 3, 64, 64), ctx_images.shape
    _, context_z, context_h = wm.rssm(
        z, ctx_acts, h, ctx_images, ctx_resets, temperature=temperature
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
    for _ in range(n_steps):
        state = zh_to_ac_state(z, h)
        zh = wm.zh_transform(z, h)
        reward = symexp(wm.reward_fc(zh))
        cont = wm.continue_fc(zh)
        if target_value is None:
            action_log, returns_pred = ac(state)
            returns_preds.append(returns_pred)
        else:
            action_log = ac.actor(state)
        action_dist = td.OneHotCategorical(logits=action_log)
        action = action_dist.sample()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        conts.append(cont)

        _, z, h = wm.rssm(z, action, h, None, no_reset, temperature=temperature)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    # False preserves the recorded ARROW/FastKAN pilot's pre-transition bootstrap.
    bootstrap_state = zh_to_ac_state(z, h) if corrected_terminal_bootstrap else state
    if target_value is None:
        returns_preds = torch.stack(returns_preds)
        _, final_returns_pred = ac(bootstrap_state)
    else:
        target_states = torch.cat((states, bootstrap_state.unsqueeze(0)), dim=0)
        target_values = target_value(target_states)
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
) -> tuple[ActorCriticOpt, torch.Tensor, dict[str, float]]:
    if aco is None:
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
        ).cuda()
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
        if slow_critic_regularizer or use_slow_critic_targets:
            slow_critic = copy.deepcopy(ac.critic).eval()
            slow_critic.requires_grad_(False)
        aco = ActorCriticOpt(ac, opt, slow_critic=slow_critic)
    ac, opt = aco.ac, aco.opt
    if use_slow_critic_targets and aco.slow_critic is None:
        raise ValueError("Slow critic targets require an initialized slow critic")
    for g in opt.param_groups:
        g["lr"] = lr
    scale_ema = aco.return_scale_ema if persistent_return_norm else None
    lam_returns_mean_ema = aco.return_mean_ema if persistent_return_norm else None

    metric_totals = {
        "actor_reinforce_loss": 0.0,
        "actor_entropy": 0.0,
        "critic_imagination_loss": 0.0,
        "critic_replay_loss": 0.0,
        "total_loss": 0.0,
        "return_mean": 0.0,
        "return_scale": 0.0,
        "gradient_norm": 0.0,
    }
    progbar = trange(steps, desc="Train AC from WM",disable=True)
    for step in progbar:
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
        )

        scale = torch.quantile(lam_returns, 0.95) - torch.quantile(lam_returns, 0.05)
        lam_returns_mean = lam_returns.mean()
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
        slow_critic_preds_log = None
        if aco.slow_critic is not None:
            with torch.no_grad():
                slow_critic_preds_log = aco.slow_critic(states)
        actor_baseline_values = None
        if use_slow_critic_targets:
            actor_baseline_values = ac.value_from_log_probs(slow_critic_preds_log)
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
        loss = (
            reinforce
            - entropy_scale * entropy
            + critic_loss
            + replay_critic_loss_scale * replay_critic_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite actor-critic loss: "
                f"reinforce={reinforce.item()} entropy={entropy.item()} "
                f"critic={critic_loss.item()} replay_critic={replay_critic_loss.item()}"
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
            "total_loss": loss,
            "return_mean": lam_returns.mean(),
            "return_scale": torch.max(one, scale_ema),
            "gradient_norm": gradient_norm,
        }
        for name, value in step_metrics.items():
            scalar = float(value.detach().item())
            if not np.isfinite(scalar):
                raise FloatingPointError(f"Non-finite actor-critic metric {name}={scalar}")
            metric_totals[name] += scalar

        if step % 50 == 0:
            progbar.set_postfix(
                {
                    "Actor entropy": f"{entropy.item():.3f}",
                    "Lam returns mean EMA": f"{lam_returns_mean_ema.item():.3f}",
                }
            )

    if persistent_return_norm:
        aco.return_scale_ema = scale_ema.detach()
        aco.return_mean_ema = lam_returns_mean_ema.detach()
    metrics = {name: total / steps for name, total in metric_totals.items()}
    metrics["replay_critic_loss_scale"] = replay_critic_loss_scale
    return aco, lam_returns_mean_ema, metrics
