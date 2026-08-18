"""R2-Dreamer size12M agent driven by a project-owned replay interface.

The model and optimizer primitives are sourced from the pinned R2-Dreamer
vendor. This module deliberately owns the continual-learning boundary: ARROW
supplies trajectory retention and sampling while this agent owns all world
model and controller updates.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from clworldmodel.models.r2 import barlow_twins_loss
from third_party.r2dreamer import networks
from third_party.r2dreamer.optim import LaProp, clip_grad_agc_
from third_party.r2dreamer.rssm import RSSM

from .config import R2DreamerConfig


@dataclass(frozen=True)
class R2ReplayBatch:
    """One R2 training batch with batch-major time axes.

    Images are float32 `[B, T, H, W, C]` values in `[0, 1]`; actions are
    one-hot `[B, T, A]`; `is_first` is a boolean `[B, T]` reset flag and
    `is_last` marks the terminal or truncation observation at `[B, T]`.
    """

    images: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    initial_stoch: torch.Tensor
    initial_deter: torch.Tensor

    def validate(self, config: R2DreamerConfig) -> None:
        if self.images.ndim != 5:
            raise ValueError("images must have [B, T, H, W, C] shape")
        batch, time, height, width, channels = self.images.shape
        if (batch, time) != (config.batch_size, config.batch_length):
            raise ValueError(
                "R2 batch dimensions must match its frozen configuration, got "
                f"B={batch}, T={time}"
            )
        if (height, width, channels) != (
            config.image_height,
            config.image_width,
            config.image_channels,
        ):
            raise ValueError("image shape does not match R2-Dreamer configuration")
        if self.actions.shape != (batch, time, config.action_dim):
            raise ValueError("actions must have [B, T, action_dim] shape")
        if self.rewards.shape != (batch, time, 1):
            raise ValueError("rewards must have [B, T, 1] shape")
        if self.continues.shape != (batch, time, 1):
            raise ValueError("continues must have [B, T, 1] shape")
        if self.is_first.shape not in {(batch, time), (batch, time, 1)}:
            raise ValueError("is_first must have [B, T] or [B, T, 1] shape")
        if self.is_last.shape not in {(batch, time), (batch, time, 1)}:
            raise ValueError("is_last must have [B, T] or [B, T, 1] shape")
        if self.initial_stoch.shape != (
            batch,
            config.stoch,
            config.discrete,
        ):
            raise ValueError("initial_stoch must have [B, stoch, discrete] shape")
        if self.initial_deter.shape != (batch, config.deter):
            raise ValueError("initial_deter must have [B, deter] shape")
        if not torch.isfinite(self.images).all():
            raise ValueError("images contain non-finite values")
        if not torch.isfinite(self.actions).all():
            raise ValueError("actions contain non-finite values")
        if not torch.isfinite(self.rewards).all():
            raise ValueError("rewards contain non-finite values")
        if not torch.isfinite(self.continues).all():
            raise ValueError("continues contain non-finite values")
        if not torch.isfinite(self.initial_stoch).all():
            raise ValueError("initial_stoch contains non-finite values")
        if not torch.isfinite(self.initial_deter).all():
            raise ValueError("initial_deter contains non-finite values")

    def to(self, device: torch.device | str) -> "R2ReplayBatch":
        return R2ReplayBatch(
            images=self.images.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            continues=self.continues.to(device),
            is_first=self.is_first.to(device),
            is_last=self.is_last.to(device),
            initial_stoch=self.initial_stoch.to(device),
            initial_deter=self.initial_deter.to(device),
        )


@dataclass(frozen=True)
class R2PolicyState:
    """RSSM state carried between environment decisions."""

    stoch: torch.Tensor
    deter: torch.Tensor
    previous_action: torch.Tensor


@dataclass(frozen=True)
class R2UpdateResult:
    """Metrics and posterior states for one native R2-Dreamer update.

    The detached posterior states are written back to the ARROW replay sidecar
    at the source positions sampled by the adapter, matching R2-Dreamer's
    latent-state replay semantics.
    """

    metrics: dict[str, float]
    posterior_stoch: torch.Tensor
    posterior_deter: torch.Tensor


class R2DreamerAgent(nn.Module):
    """Decoder-free R2-Dreamer agent using the upstream size12M profile."""

    def __init__(self, config: R2DreamerConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._amp_enabled = config.amp and self.device.type == "cuda"

        image_shape = (
            config.image_height,
            config.image_width,
            config.image_channels,
        )
        self.encoder = networks.MultiEncoder(
            config.vendor_encoder_config(),
            {
                "image": image_shape,
                "is_first": (),
                "is_last": (),
                "is_terminal": (),
                "reward": (1,),
            },
        )
        self.embedding_dim = self.encoder.out_dim
        if self.embedding_dim != config.embedding_dim:
            raise RuntimeError(
                "R2 encoder output does not match the resolved configuration: "
                f"{self.embedding_dim} != {config.embedding_dim}"
            )

        self.rssm = RSSM(
            config.vendor_rssm_config(),
            self.embedding_dim,
            config.action_dim,
        )
        if self.rssm.feat_size != config.feature_dim:
            raise RuntimeError("R2 RSSM feature size disagrees with configuration")

        self.projector = networks.Projector(self.rssm.feat_size, self.embedding_dim)
        self.reward = networks.MLPHead(
            config.vendor_head_config(
                name="reward",
                shape=(config.reward_bins,),
                distribution="symexp_twohot",
                outscale=0.0,
            ),
            self.rssm.feat_size,
        )
        self.continue_head = networks.MLPHead(
            config.vendor_head_config(
                name="cont",
                shape=(1,),
                distribution="binary",
                outscale=1.0,
            ),
            self.rssm.feat_size,
        )
        self.actor = networks.MLPHead(
            config.vendor_head_config(
                name="actor",
                shape=(config.action_dim,),
                distribution="onehot",
                outscale=0.01,
            ),
            self.rssm.feat_size,
        )
        self.value = networks.MLPHead(
            config.vendor_head_config(
                name="value",
                shape=(config.reward_bins,),
                distribution="symexp_twohot",
                outscale=0.0,
            ),
            self.rssm.feat_size,
        )
        self.slow_value = copy.deepcopy(self.value)
        self.slow_value.requires_grad_(False)
        self.slow_value.train(False)
        self._slow_value_updates = 0
        self.return_ema = networks.ReturnEMA(device=self.device)

        self._trainable_parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        self.optimizer = LaProp(
            self._trainable_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.optimizer_eps,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda update: min(
                1.0, (update + 1) / config.warmup_updates
            ),
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self._amp_enabled,
            init_scale=config.amp_initial_scale,
        )
        self.to(self.device)

    def train(self, mode: bool = True) -> "R2DreamerAgent":
        super().train(mode)
        self.slow_value.train(False)
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._trainable_parameters)

    def parameter_accounting(self) -> dict[str, int]:
        modules = {
            "encoder": self.encoder,
            "rssm": self.rssm,
            "projector": self.projector,
            "reward": self.reward,
            "continue": self.continue_head,
            "actor": self.actor,
            "value": self.value,
        }
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}
        counts["trainable_total"] = self.trainable_parameter_count
        counts["trainable_parameter_bytes"] = self.trainable_parameter_count * 4
        return counts

    def initial_policy_state(self, batch_size: int) -> R2PolicyState:
        stoch, deter = self.rssm.initial(batch_size)
        action = torch.zeros(
            batch_size,
            self.config.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        return R2PolicyState(stoch, deter, action)

    def _observation_dict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.shape[-1] == self.config.image_channels:
            channel_last = images
        elif images.ndim >= 4 and images.shape[-3] == self.config.image_channels:
            channel_last = images.movedim(-3, -1)
        else:
            raise ValueError("image tensor must have channels last or channels third from last")
        return {"image": channel_last.to(dtype=torch.float32)}

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        is_first: torch.Tensor,
        state: R2PolicyState,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, R2PolicyState]:
        """Choose one action per image and advance the posterior RSSM state."""
        observations = self._observation_dict(images.to(self.device))
        embedding = self.encoder(observations)
        reset = is_first.to(self.device).to(dtype=torch.bool).reshape(-1)
        stoch, deter, _ = self.rssm.obs_step(
            state.stoch,
            state.deter,
            state.previous_action,
            embedding,
            reset,
        )
        features = self.rssm.get_feat(stoch, deter)
        action_dist = self.actor(features)
        action = action_dist.mode if deterministic else action_dist.rsample()
        return action, R2PolicyState(stoch, deter, action)

    def _autocast(self) -> Iterator[None]:
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _normalized_is_first(self, batch: R2ReplayBatch) -> torch.Tensor:
        is_first = batch.is_first
        if is_first.ndim == 3:
            is_first = is_first.squeeze(-1)
        return is_first.to(dtype=torch.bool)

    def _normalized_is_last(self, batch: R2ReplayBatch) -> torch.Tensor:
        is_last = batch.is_last
        if is_last.ndim == 3:
            is_last = is_last.squeeze(-1)
        return is_last.to(dtype=batch.continues.dtype).unsqueeze(-1)

    def _world_model_terms(
        self,
        batch: R2ReplayBatch,
        initial: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        observations = self._observation_dict(batch.images)
        embedding = self.encoder(observations)
        post_stoch, post_deter, post_logits = self.rssm.observe(
            embedding,
            batch.actions,
            initial,
            self._normalized_is_first(batch),
        )
        _, prior_logits = self.rssm.prior(post_deter)
        dynamics, representation = self.rssm.kl_loss(
            post_logits,
            prior_logits,
            self.config.kl_free,
        )
        features = self.rssm.get_feat(post_stoch, post_deter)
        projected = self.projector(features.reshape(-1, features.shape[-1]))
        barlow, invariance, redundancy = barlow_twins_loss(
            projected,
            embedding.reshape(-1, embedding.shape[-1]),
            redundancy_scale=self.config.barlow_redundancy_scale,
            normalization_eps=self.config.normalization_eps,
        )
        reward_loss = -self.reward(features).log_prob(batch.rewards).mean()
        continue_loss = -self.continue_head(features).log_prob(batch.continues).mean()
        terms = {
            "dynamics": dynamics.mean(),
            "representation": representation.mean(),
            "barlow": barlow,
            "barlow_invariance": invariance,
            "barlow_redundancy": redundancy,
            "reward": reward_loss,
            "continue": continue_loss,
        }
        return terms, (post_stoch, post_deter), features

    @staticmethod
    def _lambda_return(
        last: torch.Tensor,
        terminal: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        lambda_return: float,
    ) -> torch.Tensor:
        if not (
            last.shape
            == terminal.shape
            == reward.shape
            == value.shape
            == bootstrap.shape
        ):
            raise ValueError("lambda return inputs must have matching shapes")
        live = (1 - terminal)[:, 1:] * discount
        mix = (1 - last)[:, 1:] * lambda_return
        intermediate = reward[:, 1:] + (1 - mix) * live * bootstrap[:, 1:]
        returns = [bootstrap[:, -1]]
        for index in reversed(range(live.shape[1])):
            returns.append(intermediate[:, index] + live[:, index] * mix[:, index] * returns[-1])
        return torch.stack(list(reversed(returns))[:-1], dim=1)

    @torch.no_grad()
    def _imagine(
        self,
        initial: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stoch, deter = initial
        features = []
        actions = []
        for _ in range(self.config.imagination_horizon + 1):
            feature = self.rssm.get_feat(stoch, deter)
            action = self.actor(feature).rsample()
            features.append(feature)
            actions.append(action)
            stoch, deter = self.rssm.img_step(stoch, deter, action)
        return torch.stack(features, dim=1), torch.stack(actions, dim=1)

    def _controller_terms(
        self,
        batch: R2ReplayBatch,
        posterior: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        post_stoch, post_deter = posterior
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, post_deter.shape[-1]).detach(),
        )
        imagined_features, imagined_actions = self._imagine(start)
        with torch.no_grad():
            imagined_reward = self.reward(imagined_features).mode()
            imagined_continue = self.continue_head(imagined_features).mean
            imagined_value = self.value(imagined_features).mode()
            imagined_slow_value = self.slow_value(imagined_features).mode()
            zeros = torch.zeros_like(imagined_continue)
            imagined_return = self._lambda_return(
                zeros,
                1 - imagined_continue,
                imagined_reward,
                imagined_value,
                imagined_value,
                discount=1 - 1 / self.config.discount_horizon,
                lambda_return=self.config.lambda_return,
            )
            offset, scale = self.return_ema(imagined_return)
            advantage = (imagined_return - imagined_value[:, :-1]) / scale
            weight = torch.cumprod(
                imagined_continue * (1 - 1 / self.config.discount_horizon), dim=1
            )

        policy = self.actor(imagined_features)
        log_probability = policy.log_prob(imagined_actions)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        policy_loss = torch.mean(
            weight[:, :-1].detach()
            * -(
                log_probability * advantage.detach()
                + self.config.action_entropy * entropy
            )
        )

        imagined_value_dist = self.value(imagined_features)
        padded_return = torch.cat(
            [imagined_return, torch.zeros_like(imagined_return[:, -1:])], dim=1
        )
        value_loss = torch.mean(
            weight[:, :-1].detach()
            * (
                -imagined_value_dist.log_prob(padded_return.detach())
                - imagined_value_dist.log_prob(imagined_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )

        with torch.no_grad():
            replay_value = self.value(features).mode()
            bootstrap = imagined_return[:, 0].reshape(
                self.config.batch_size, self.config.batch_length, 1
            )
            replay_last = self._normalized_is_last(batch)
            replay_return = self._lambda_return(
                replay_last,
                1 - batch.continues,
                batch.rewards,
                replay_value,
                bootstrap,
                discount=1 - 1 / self.config.discount_horizon,
                lambda_return=self.config.lambda_return,
            )
            replay_slow_value = self.slow_value(features).mode()
        replay_value_dist = self.value(features)
        replay_return_padded = torch.cat(
            [replay_return, torch.zeros_like(replay_return[:, -1:])], dim=1
        )
        replay_value_loss = torch.mean(
            (1 - replay_last[:, :-1])
            * (
                -replay_value_dist.log_prob(replay_return_padded.detach())
                - replay_value_dist.log_prob(replay_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )
        return {
            "policy": policy_loss,
            "value": value_loss,
            "replay_value": replay_value_loss,
            "return_offset": offset,
            "return_scale": scale,
        }

    def _total_loss(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        weighted = (
            self.config.loss_scale_dynamics * terms["dynamics"]
            + self.config.loss_scale_representation * terms["representation"]
            + self.config.loss_scale_barlow * terms["barlow"]
            + self.config.loss_scale_reward * terms["reward"]
            + self.config.loss_scale_continue * terms["continue"]
            + self.config.loss_scale_policy * terms["policy"]
            + self.config.loss_scale_value * terms["value"]
            + self.config.loss_scale_replay_value * terms["replay_value"]
        )
        if not torch.isfinite(weighted):
            raise FloatingPointError("R2-Dreamer total loss is non-finite")
        return weighted

    def _update_slow_value(self) -> None:
        if self._slow_value_updates % self.config.slow_target_update == 0:
            with torch.no_grad():
                mix = self.config.slow_target_fraction
                for value, slow_value in zip(
                    self.value.parameters(), self.slow_value.parameters()
                ):
                    slow_value.copy_(mix * value + (1 - mix) * slow_value)
        self._slow_value_updates += 1

    @staticmethod
    def _gradients_are_finite(parameters: tuple[nn.Parameter, ...]) -> bool:
        """Check before AGC so an AMP overflow remains recoverable by GradScaler."""
        return all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )

    def update_batch(self, batch: R2ReplayBatch) -> R2UpdateResult:
        """Apply one full R2-Dreamer world-model and controller update."""
        batch = batch.to(self.device)
        batch.validate(self.config)
        initial = (batch.initial_stoch, batch.initial_deter)
        self._update_slow_value()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            terms, posterior, features = self._world_model_terms(batch, initial)
            terms.update(self._controller_terms(batch, posterior, features))
            total_loss = self._total_loss(terms)
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in self._trainable_parameters
                    if parameter.grad is not None
                ]
            )
        )
        optimizer_step = self._gradients_are_finite(self._trainable_parameters)
        if optimizer_step:
            clip_grad_agc_(
                self._trainable_parameters,
                self.config.agc,
                self.config.agc_pmin,
                foreach=True,
            )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if optimizer_step:
            self.scheduler.step()
        metrics = {
            "loss/total": total_loss.detach(),
            "loss/dynamics": terms["dynamics"].detach(),
            "loss/representation": terms["representation"].detach(),
            "loss/barlow": terms["barlow"].detach(),
            "metric/barlow_invariance": terms["barlow_invariance"].detach(),
            "metric/barlow_redundancy": terms["barlow_redundancy"].detach(),
            "loss/reward": terms["reward"].detach(),
            "loss/continue": terms["continue"].detach(),
            "loss/policy": terms["policy"].detach(),
            "loss/value": terms["value"].detach(),
            "loss/replay_value": terms["replay_value"].detach(),
            "metric/grad_norm": gradient_norm.detach(),
            "metric/return_scale": terms["return_scale"].detach(),
            "opt/lr": torch.tensor(self.scheduler.get_last_lr()[0], device=self.device),
            "opt/grad_scale": torch.tensor(
                self.grad_scaler.get_scale(), device=self.device
            ),
            "opt/optimizer_step": torch.tensor(
                float(optimizer_step), device=self.device
            ),
        }
        return R2UpdateResult(
            metrics={name: float(value.cpu()) for name, value in metrics.items()},
            posterior_stoch=posterior[0].detach(),
            posterior_deter=posterior[1].detach(),
        )
