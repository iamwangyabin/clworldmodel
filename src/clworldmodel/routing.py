# SPDX-License-Identifier: Apache-2.0
"""Parameter-free, episode-locked reconstruction routing.

The caller supplies acquired route IDs, never an environment label. The shared
world-model adapter owns reconstruction; this module has no vendored dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


class RoutedActorBank(torch.nn.Module):
    """Temporary inference view of existing private actors, without copying weights.

    Inputs are features [B, D] and *inferred* int64 route IDs [B]. The acquired
    registry is supplied independently of evaluation labels. Registered actor
    references let evaluators restore each original module's train/eval mode;
    this view is never installed in a model, checkpoint, or optimizer.
    """

    def __init__(self, actors: Mapping[int, torch.nn.Module]) -> None:
        super().__init__()
        if not actors or any(type(i) is not int or i < 0 for i in actors):
            raise ValueError("Private actor routes must be non-empty non-negative integer IDs")
        if any(not isinstance(actor, torch.nn.Module) for actor in actors.values()):
            raise TypeError("Private actors must be torch modules")
        self.route_ids = tuple(sorted(actors))
        self.actors = torch.nn.ModuleDict({str(i): actors[i] for i in self.route_ids})

    @torch.no_grad()
    def forward(self, features: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] < 1 or not features.is_floating_point():
            raise ValueError("Private actor features must be non-empty floating-point [B,D]")
        if (route_ids.shape != features.shape[:1] or route_ids.dtype != torch.long
                or route_ids.device != features.device):
            raise ValueError("Private actor route IDs must be int64 [B] on the feature device")
        inferred_ids = route_ids.unique(sorted=True).tolist()
        if any(i not in self.route_ids for i in inferred_ids):
            raise ValueError("Inferred private actor is outside the acquired eligibility registry")
        logits = None
        for route_id in inferred_ids:
            rows = torch.where(route_ids == route_id)[0]
            output = self.actors[str(route_id)](features[rows]).float()
            if (output.ndim != 2 or output.shape[0] != len(rows)
                    or output.shape[-1] < 1 or output.device != features.device):
                raise ValueError("Private actor logits must be [route batch, actions] on the feature device")
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("Non-finite private actor logits")
            if logits is None:
                logits = output.new_empty((len(features), output.shape[-1]))
            elif output.shape[-1] != logits.shape[-1]:
                raise ValueError("Private actors must use the same protocol action space")
            logits[rows] = output
        return logits


class EpisodeReconstructionRouter:
    """Select one route per worker using only its first observation.

    Observations and reconstructions: float [B, C, H, W] in the same pixel
    coordinate system. Reset: bool [B]. Ties select the lowest eligible ID.
    Scores are float32 pixel MSE; no reward, policy, or true task ID is accepted.
    Instances are local to a collection/evaluation call, not learned modules.
    """

    def __init__(self, eligible_route_ids: Sequence[int]) -> None:
        ids = tuple(eligible_route_ids)
        if (not ids or any(type(i) is not int or i < 0 for i in ids)
                or tuple(sorted(set(ids))) != ids):
            raise ValueError("Eligible routes must be sorted unique non-negative integers")
        self.eligible_route_ids = ids
        self.routes: torch.Tensor | None = None
        self.events: list[dict[str, Any]] = []

    @torch.no_grad()
    def route(
        self,
        observations: torch.Tensor,
        reset: torch.Tensor,
        reconstruct: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if (observations.ndim != 4 or not observations.is_floating_point()
                or observations.shape[0] < 1):
            raise ValueError("Routing observations must be non-empty float [B,C,H,W]")
        if reset.shape != observations.shape[:1] or reset.dtype != torch.bool:
            raise ValueError("Routing resets must be bool [B]")
        if reset.device != observations.device:
            raise ValueError("Routing resets and observations must share a device")
        if self.routes is None:
            self.routes = torch.full_like(reset, -1, dtype=torch.long)
        if self.routes.shape != reset.shape or self.routes.device != reset.device:
            raise ValueError("Create a new router when worker count or device changes")
        indices = torch.where(reset | (self.routes < 0))[0]
        if not indices.numel():
            return self.routes.clone()
        frames = observations[indices].float()
        scores = []
        for route_id in self.eligible_route_ids:
            decoded = reconstruct(route_id, frames)
            if decoded.shape != frames.shape or decoded.device != frames.device:
                raise ValueError("Route reconstruction must match observation shape/device")
            scores.append((decoded.float() - frames).square().flatten(1).mean(1))
        scores = torch.stack(scores, dim=-1)  # [reset_workers, eligible_routes]
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Non-finite reconstruction routing score")
        winners = scores.argmin(-1)
        ids = torch.tensor(self.eligible_route_ids, device=frames.device)
        selected = ids[winners]
        self.routes[indices] = selected
        sorted_scores = scores.sort(-1).values
        margins = (sorted_scores[:, 1] - sorted_scores[:, 0]
                   if scores.shape[1] > 1 else torch.zeros_like(sorted_scores[:, 0]))
        for worker, route_id, row, margin in zip(
            indices.tolist(), selected.tolist(), scores.cpu().tolist(), margins.tolist()
        ):
            self.events.append({
                "worker_index": worker, "selected_route_id": route_id,
                "eligible_route_ids": list(self.eligible_route_ids),
                "reconstruction_mse": row, "margin": margin,
            })
        return self.routes.clone()


def routing_audit(
    events: Sequence[dict[str, Any]], *, true_task_id: int, task_count: int
) -> dict[str, Any]:
    """Attach labels *after* inference, solely for persisted diagnostics."""
    if not 0 <= true_task_id < task_count:
        raise ValueError("Audit task ID is outside the configured task set")
    confusion = [[0] * task_count for _ in range(task_count)]
    for event in events:
        selected = event["selected_route_id"]
        if not 0 <= selected < task_count:
            raise ValueError("Audit route is outside the configured task set")
        confusion[true_task_id][selected] += 1
    return {
        "true_task_id_for_audit_only": true_task_id,
        "episode_starts": len(events),
        "accuracy": (confusion[true_task_id][true_task_id] / len(events)
                     if events else None),
        "confusion_matrix": confusion,
        "events": list(events),
    }
