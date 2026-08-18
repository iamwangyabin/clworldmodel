"""PyTorch LaProp with DreamerV3's per-tensor adaptive gradient clipping."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    """RMS-normalize gradients before momentum, matching DreamerV3's chain."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 4e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-20,
        agc_clip: float = 0.3,
        agc_parameter_floor: float = 1e-3,
        warmup_steps: int = 1000,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must lie in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if agc_clip < 0 or agc_parameter_floor <= 0:
            raise ValueError("AGC values must be non-negative with a positive floor")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            agc_clip=agc_clip,
            agc_parameter_floor=agc_parameter_floor,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                grad = gradient
                if group["agc_clip"]:
                    grad_norm = torch.linalg.vector_norm(grad)
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    upper = group["agc_clip"] * torch.maximum(
                        parameter_norm,
                        parameter_norm.new_tensor(group["agc_parameter_floor"]),
                    )
                    grad = grad * (1.0 / torch.maximum(grad_norm / upper, upper.new_tensor(1.0)))

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                rms = (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(group["eps"])
                normalized_grad = grad / rms
                exp_avg.mul_(beta1).add_(normalized_grad, alpha=1.0 - beta1)
                direction = exp_avg / (1.0 - beta1**step)

                warmup_steps = group["warmup_steps"]
                warmup = (
                    min((step - 1) / warmup_steps, 1.0)
                    if warmup_steps
                    else 1.0
                )
                parameter.add_(direction, alpha=-group["lr"] * warmup)

        return loss
