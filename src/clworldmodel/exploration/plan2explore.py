"""Plan2Explore latent-dynamics disagreement for discrete Dreamer agents."""

from __future__ import annotations

import torch
from torch import nn


class DisagreementEnsemble(nn.Module):
    """Predict the next stochastic latent from state and action.

    Inputs may have any leading axes. The returned model axis is first:
    ``[models, ..., stochastic_features]``.
    """

    def __init__(
        self,
        state_features: int,
        action_features: int,
        target_features: int,
        *,
        models: int,
        hidden_layers: int,
        hidden_features: int,
    ) -> None:
        super().__init__()
        if min(
            state_features,
            action_features,
            target_features,
            models,
            hidden_layers,
            hidden_features,
        ) < 1:
            raise ValueError("Plan2Explore dimensions and model counts must be positive")
        layers = []
        for _ in range(models):
            modules: list[nn.Module] = []
            width = state_features + action_features
            for _ in range(hidden_layers):
                modules.extend((nn.Linear(width, hidden_features), nn.ELU()))
                width = hidden_features
            modules.append(nn.Linear(width, target_features))
            # TF Dense's source default is Glorot uniform with zero bias.
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            layers.append(nn.Sequential(*modules))
        self.models = nn.ModuleList(layers)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if states.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Plan2Explore states and actions must share leading axes")
        inputs = torch.cat((states, actions), dim=-1)
        return torch.stack([model(inputs) for model in self.models])

    def disagreement(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Population standard deviation averaged over target coordinates."""

        predictions = self(states, actions).float()
        return predictions.std(dim=0, correction=0).mean(dim=-1, keepdim=True)

    def prediction_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_stochastic_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Summed unit-variance Gaussian NLL, excluding its constant term."""

        predictions = self(states.detach(), actions.detach()).float()
        targets = next_stochastic_latents.detach().float()
        if predictions.shape[1:] != targets.shape:
            raise ValueError(
                "Plan2Explore predictions and targets must have equal non-model axes: "
                f"{tuple(predictions.shape[1:])} != {tuple(targets.shape)}"
            )
        # Independent Normal.log_prob sums target coordinates; only B/T are
        # averaged, then the ten model losses are summed (not averaged).
        per_item = 0.5 * (predictions - targets).square().sum(dim=-1)
        return per_item.reshape(len(self.models), -1).mean(dim=1).sum()
