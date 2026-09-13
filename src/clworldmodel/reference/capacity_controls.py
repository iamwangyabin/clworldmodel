"""Named capacity-organization controls composed with the pinned WM API.

The injected factory owns the Dreamer losses and tensor layout. This adapter
only chooses parameter ownership/routes; it does not implement another RSSM.
Public batches are [time, batch, ...], images are floating RGB in [0, 1].
"""
from __future__ import annotations

import copy
import weakref

import torch
from torch import nn

from ..precision import autocast_context

CONTROLS = ("shared", "wide", "fullbank", "frozen", "independent")


class _RoutedRSSM(nn.Module):
    """Route the two RSSM calls used by collection and imagined rollouts."""
    task_mechanism_bank_enabled = False  # Outer accounting belongs to the adapter.

    def __init__(self, owner):
        super().__init__()
        self._owner = weakref.ref(owner)

    def initial_state(self, batch_size):
        return self._owner().models[0].rssm.initial_state(batch_size)

    def forward(self, *args, task_id=None, **kwargs):
        model, route = self._owner().select(task_id)
        return model.rssm(*args, task_id=route, **kwargs)


class CapacityWorldModel(nn.Module):
    """Single shared WM, independent WMs, or retained dense residual mechanisms."""

    def __init__(self, models, control, num_tasks):
        super().__init__()
        if control not in CONTROLS or num_tasks < 2:
            raise ValueError("Declare a named control with at least two tasks")
        self.models = nn.ModuleList(models)
        self.control = control
        self.num_tasks = num_tasks
        self.rssm = _RoutedRSSM(self)
        self.register_buffer("initialized_tasks", torch.zeros(num_tasks, dtype=torch.bool))
        self.initialized_tasks[0] = True
        self.ls, self.h_dim, self.a_dim = models[0].ls, models[0].h_dim, models[0].a_dim
        self.compute_dtype = models[0].compute_dtype
        # Optimizer construction must include later-task tensors; activation
        # subsequently freezes inactive tensors without dropping their ownership.
        self.requires_grad_(True)

    @property
    def zh_transform(self):
        return self.models[0].zh_transform  # Parameter-free concatenation, common shape.

    def select(self, task_id):
        if task_id is None:
            raise ValueError("Capacity controls require an explicit oracle task route")
        if isinstance(task_id, torch.Tensor):
            task_id = int(task_id.item())
        if not isinstance(task_id, int) or not 0 <= task_id < self.num_tasks:
            raise ValueError(f"Invalid capacity-control task route: {task_id}")
        if self.control == "fullbank":
            return self.models[task_id], None
        return self.models[0], task_id if self.control in {"frozen", "independent"} else None

    def initialize_task_expert(self, task_id, source_task_id):
        self.select(task_id)
        if bool(self.initialized_tasks[task_id]):
            return False
        if self.control in {"frozen", "independent"}:
            self.models[0].initialize_task_expert(task_id, source_task_id)
        # FullBank starts from the untrained common seed initialization, never
        # from the previous task's trained weights.
        self.initialized_tasks[task_id] = True
        return True

    def activate_task_expert(self, task_id, mechanism_phase="full"):
        self.select(task_id)
        if not bool(self.initialized_tasks[task_id]):
            raise ValueError("Initialize the route before activating it")
        if self.control == "fullbank":
            for index, model in enumerate(self.models):
                model.requires_grad_(index == task_id)
        elif self.control in {"shared", "wide"}:
            self.models[0].requires_grad_(True)
        else:
            model = self.models[0]
            model.activate_task_expert(task_id, mechanism_phase)
            if self.control == "frozen" and task_id > 0:
                for parameters in model.shared_parameter_groups().values():
                    for parameter in parameters:
                        parameter.requires_grad_(False)
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None

    def compute_loss(self, *batch, task_id=None, **kwargs):
        model, route = self.select(task_id)
        return model.compute_loss(*batch, task_id=route, **kwargs)

    def predict_reward_symlog(self, state, task_id):
        model, route = self.select(task_id)
        return model.predict_reward_symlog(state, route)

    def predict_continue(self, state, task_id):
        model, route = self.select(task_id)
        return model.predict_continue(state, route)

    def decoder_for(self, task_id):
        model, route = self.select(task_id)
        return model.decoder_for(route)

    def capacity_accounting(self):
        def count(model):
            parameters = list(model.parameters())
            return {"parameters": sum(p.numel() for p in parameters),
                    "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
                    "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters)}
        return {"schema_version": 1, "control": self.control,
                "world_model": count(self), "per_model": [count(m) for m in self.models],
                "initialized_task_ids": self.initialized_tasks.nonzero().flatten().tolist(),
                "allocation": "All configured WM modules allocated upfront; active/trained counts are separate",
                "actor_critic_included": False}


def build_capacity_model(factory, *args, control, num_tasks, residual_atoms=4, **kwargs):
    if control not in CONTROLS:
        raise ValueError(f"Unknown capacity control: {control}")
    residual = control in {"frozen", "independent"}
    kwargs.update(num_task_experts=num_tasks if residual else 1,
                  task_shared_prediction_heads=residual, evolving_shared_core=residual,
                  task_projected_image_encoder=residual,
                  task_symmetric_image_projectors=residual,
                  task_mechanism_bank=residual,
                  task_mechanism_reuse=control != "independent",
                  task_mechanism_num_atoms=residual_atoms if residual else 1,
                  task_mechanism_parameterization="dense_private",
                  task_symmetric_mechanisms=residual)
    base = factory(*args, **kwargs)
    models = [base]
    if control == "fullbank":
        models.extend(copy.deepcopy(base) for _ in range(num_tasks - 1))
    return CapacityWorldModel(models, control, num_tasks)


def capacity_world_model_update(config, wm, replay, optimizer, task_id, rng):
    """One current+LTDM Dreamer update, with no AWM protection or pruning.

The same current/old sequence allocation and loss weight apply to every
control. In isolated/frozen controls an old loss can be constant with respect
to active parameters; it must not unfreeze an old model to create gradients.
"""
    device = next(wm.parameters()).device
    n_current = config.mb_n_size if task_id == 0 else config.current_batch_n
    current = replay.minibatch_for_task(task_id, config.mb_t_size, n_current,
                                       source="mixed", mb_device=str(device))
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config.compute_dtype):
        loss, metrics = wm.compute_loss(*current, task_id=task_id)
        if task_id > 0:
            old_id = int(rng.integers(0, task_id))
            old = replay.minibatch_for_task(old_id, config.mb_t_size, config.memory_batch_n,
                                           source="ltdm", mb_device=str(device))
            old_loss, _ = wm.compute_loss(*old, task_id=old_id)
            loss = loss + config.memory_loss_scale * old_loss
            metrics = {**metrics, "CapacityControl/old_dreamer_loss": old_loss.detach()}
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000)
    optimizer.step()
    return metrics, norm
