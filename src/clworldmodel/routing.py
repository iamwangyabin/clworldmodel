# SPDX-License-Identifier: Apache-2.0
"""Parameter-free reconstruction routing with bounded episode-local probes.

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


class TwoFrameReconstructionRouter:
    """Accumulate FP32 pixel MSE in FP64 over at most two observations/worker.

    Observations/reconstructions are float [B,C,H,W] in the same pixel scale;
    reset/inactive are bool [B]. Ties choose the lowest eligible route ID.
    No reward, policy, or true task label is accepted by the scoring boundary.
    The adapter callback receives (route ID, frames, worker indices, first mask).
    It owns deterministic RSSM histories in ``candidate_states``; only workers
    still awaiting their second observation need those buffers. ``inactive``
    excludes terminal observations preceding NextStep's ignored reset action.
    Later calls return the held route without invoking the callback.
    """

    def __init__(self, eligible_route_ids: Sequence[int]) -> None:
        ids = tuple(eligible_route_ids)
        if (not ids or any(type(i) is not int or i < 0 for i in ids)
                or tuple(sorted(set(ids))) != ids):
            raise ValueError("Eligible routes must be sorted unique non-negative integers")
        self.eligible_route_ids = ids
        self.routes: torch.Tensor | None = None
        self.events: list[dict[str, Any]] = []
        self.counts: torch.Tensor | None = None
        self.score_sums: torch.Tensor | None = None
        self.episodes: torch.Tensor | None = None
        self.candidate_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def route(
        self,
        observations: torch.Tensor,
        reset: torch.Tensor,
        reconstruct: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        inactive: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (observations.ndim != 4 or not observations.is_floating_point()
                or observations.shape[0] < 1):
            raise ValueError("Routing observations must be non-empty float [B,C,H,W]")
        if (reset.shape != observations.shape[:1] or reset.dtype != torch.bool
                or reset.device != observations.device):
            raise ValueError("Routing resets must be bool [B] on the observation device")
        if inactive is None:
            inactive = torch.zeros_like(reset)
        if (inactive.shape != reset.shape or inactive.dtype != torch.bool
                or inactive.device != reset.device):
            raise ValueError("Inactive workers must be bool [B] on the observation device")
        if self.routes is None:
            self.routes = torch.full_like(reset, -1, dtype=torch.long)
            self.counts = torch.zeros_like(self.routes)
            self.episodes = torch.full_like(self.routes, -1)
            self.score_sums = torch.zeros(len(reset), len(self.eligible_route_ids),
                                          device=reset.device, dtype=torch.float64)
        if self.routes.shape != reset.shape or self.routes.device != reset.device:
            raise ValueError("Create a new router when worker count or device changes")
        starts = reset | (self.routes < 0)
        self.counts[starts] = 0
        self.score_sums[starts] = 0
        self.episodes[starts] += 1
        rows = torch.where((self.counts < 2) & (~inactive | starts))[0]
        if not rows.numel():
            return self.routes.clone()
        frames, first = observations[rows].float(), self.counts[rows] == 0
        scores = []
        for route_id in self.eligible_route_ids:
            decoded = reconstruct(route_id, frames, rows, first)
            if decoded.shape != frames.shape or decoded.device != frames.device:
                raise ValueError("Route reconstruction must match observation shape/device")
            scores.append((decoded.float() - frames).square().flatten(1).mean(1))
        scores = torch.stack(scores, -1)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Non-finite reconstruction routing score")
        self.score_sums[rows] += scores.double()
        self.counts[rows] += 1
        means = self.score_sums[rows] / self.counts[rows, None]
        ids = torch.tensor(self.eligible_route_ids, device=reset.device)
        self.routes[rows] = ids[means.argmin(-1)]
        sorted_means = means.sort(-1).values
        margins = (sorted_means[:, 1] - sorted_means[:, 0]
                   if len(ids) > 1 else torch.zeros_like(sorted_means[:, 0]))
        for i, worker in enumerate(rows.tolist()):
            self.events.append({
                "worker_index": worker, "episode_index": int(self.episodes[worker]),
                "observation_count": int(self.counts[worker]),
                "selected_route_id": int(self.routes[worker]),
                "eligible_route_ids": list(self.eligible_route_ids),
                "reconstruction_mse": scores[i].tolist(),
                "cumulative_mean_mse": means[i].tolist(), "margin": float(margins[i]),
            })
        return self.routes.clone()


def routing_audit(
    events: Sequence[dict[str, Any]], *, true_task_id: int, task_count: int
) -> dict[str, Any]:
    """Attach labels *after* inference, solely for persisted diagnostics."""
    if not 0 <= true_task_id < task_count:
        raise ValueError("Audit task ID is outside the configured task set")
    # V3 primary accuracy uses the last available decision per episode, not a
    # mixture of first/second decisions misreported as episode starts.
    decisions = events
    extra = {}
    if events and "observation_count" in events[0]:
        last = {(e["worker_index"], e["episode_index"]): e for e in events}
        decisions = list(last.values())
        extra = {"routing_decisions": len(events), "accuracy_unit": "last_available_route_per_episode"}
        for n in (1, 2):
            subset = [e for e in events if e["observation_count"] == n]
            extra[f"observation_{n}"] = {
                "count": len(subset),
                "accuracy": (sum(e["selected_route_id"] == true_task_id for e in subset) / len(subset)
                             if subset else None),
            }
    confusion = [[0] * task_count for _ in range(task_count)]
    for event in events:
        selected = event["selected_route_id"]
        if not 0 <= selected < task_count:
            raise ValueError("Audit route is outside the configured task set")
    for event in decisions:
        confusion[true_task_id][event["selected_route_id"]] += 1
    return {
        "true_task_id_for_audit_only": true_task_id,
        "episode_starts": len(decisions),
        "accuracy": (confusion[true_task_id][true_task_id] / len(decisions)
                     if decisions else None),
        "confusion_matrix": confusion,
        "events": list(events),
        **extra,
    }
