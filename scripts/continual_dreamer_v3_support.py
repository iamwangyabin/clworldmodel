"""Single audited adapter to NM512 V3, with source-recipe Plan2Explore.

Only this script-layer boundary imports the vendor's private model APIs. V3
world-model, actor, critic, distributions and optimizers are not monkeypatched.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/nm512_dreamerv3"
NM_PIN = "6ef8646d807cd10ce0c88e10a7e943211e7fc44c"
CD_PIN = "77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6"


def verify_source() -> dict:
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    if provenance["commit"] != NM_PIN or provenance["local_modifications"]:
        raise ValueError("The NM512 V3 no-modification source pin changed")
    manifest = {}
    for line in (VENDOR / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (VENDOR / name).resolve()
        if not path.is_relative_to(VENDOR.resolve()):
            raise ValueError("Source manifest path escapes vendor root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
        manifest[name] = expected
    if manifest != provenance["files"]:
        raise ValueError("Source hash manifests disagree")
    return provenance


def load_runtime():
    verify_source()
    names = ("tools", "networks", "models")
    for name in names:
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(VENDOR.resolve()):
            raise RuntimeError(f"Vendor module collision: {name}. Start a fresh interpreter.")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    modules = [importlib.import_module(name) for name in names]
    if any(not Path(m.__file__).resolve().is_relative_to(VENDOR.resolve()) for m in modules):
        raise RuntimeError("Native model import escaped its verified source root")
    return modules[2], modules[0]


def resolve_native_config(recipe) -> argparse.Namespace:
    from ruamel.yaml import YAML

    verify_source()
    values = copy.deepcopy(YAML(typ="safe", pure=True).load((VENDOR / "configs.yaml").read_text())["defaults"])
    # YAML 1.1 reads scientific literals without an exponent sign as strings.
    # Match the vendor's CLI coercion, without requiring torch for a dry-run.
    for key, value in values.items():
        if isinstance(value, str) and value.replace(".", "", 1).replace("e", "", 1).isdigit() and "e" in value:
            values[key] = float(value)
    values.update(
        seed=recipe.seed, device=recipe.device, compile=False, precision=32,
        envs=1, action_repeat=1, num_actions=7,
        size=[64, 64], time_limit=recipe.episode_limit,
        batch_size=recipe.batch_size, batch_length=recipe.batch_length,
        discount=recipe.discount, imag_gradient="reinforce", eval_state_mean=False,
        prefill=recipe.prefill_decisions, pretrain=recipe.initial_updates,
        # Driver has an explicit action counter; native Dreamer.__call__ and
        # its train_ratio scheduler are deliberately not used by this adapter.
        train_ratio=recipe.batch_size * recipe.batch_length / recipe.train_every_decisions,
        expl_behavior="source_recipe_plan2explore", expl_until=0,
        expl_extr_scale=recipe.extrinsic_scale, expl_intr_scale=recipe.intrinsic_scale,
        disag_action_cond=True, disag_log=False, disag_target="stoch", disag_offset=1,
        disag_models=recipe.disagreement_models, disag_layers=recipe.disagreement_layers,
        disag_units=recipe.disagreement_units, video_pred_log=False,
    )
    values["actor"].update(dist="onehot", std="none", entropy=recipe.actor_entropy)
    return argparse.Namespace(**values)


@contextmanager
def isolated_evaluation_rng(seed: int):
    """Evaluation has reproducible stochastic latents, but cannot perturb training."""
    import numpy as np
    import torch

    py, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


class SourceRecipeAgent:
    """Task-agnostic interface: image, reset/terminal flags, never a task ID."""

    def __init__(self, recipe, native, observation_space, action_space):
        import torch
        from clworldmodel.exploration import DisagreementEnsemble

        models, self.tools = load_runtime()
        self.recipe, self.native = recipe, native
        self.wm = models.WorldModel(observation_space, action_space, 0, native).to(native.device)
        self.task = models.ImagBehavior(native, self.wm).to(native.device)
        self.explore = models.ImagBehavior(native, self.wm).to(native.device)
        stochastic = native.dyn_stoch * native.dyn_discrete
        self.ensemble = DisagreementEnsemble(
            native.dyn_deter + stochastic, native.num_actions, stochastic,
            models=recipe.disagreement_models, hidden_layers=recipe.disagreement_layers,
            hidden_features=recipe.disagreement_units,
        ).to(native.device)
        self.ensemble_opt = self.tools.Optimizer(
            "ensemble", self.ensemble.parameters(), recipe.disagreement_lr,
            recipe.disagreement_eps, recipe.disagreement_clip,
            recipe.disagreement_weight_decay, opt="adam", use_amp=False,
        )
        for module in (self.wm, self.task, self.explore, self.ensemble):
            module.requires_grad_(False)

    def act(self, observation, state=None, *, evaluation=False):
        import numpy as np
        import torch

        if set(observation) != {"image", "is_first", "is_terminal"}:
            raise ValueError("Agent boundary accepts only image/is_first/is_terminal")
        with torch.no_grad():
            obs = self.wm.preprocess({k: np.expand_dims(v, 0) for k, v in observation.items()})
            embed = self.wm.encoder(obs)
            latent, action = (None, None) if state is None else state
            latent, _ = self.wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
            feat = self.wm.dynamics.get_feat(latent)
            distribution = (self.task if evaluation else self.explore).actor(feat)
            action = distribution.mode() if evaluation else distribution.sample()
            return int(action.argmax(-1).item()), (latent, action)

    def task_reward(self, feat, _state, _action):
        return self.wm.heads["reward"](feat).mode()

    def exploration_reward(self, feat, _state, action):
        # Source training pairs (posterior[t], incoming_action[t]) with stoch[t+1].
        # Native V3 imagination returns outgoing actions, unlike CD's sequence.
        # Shift to the incoming convention. The first reward is discarded by
        # native V3 lambda_return, so its unavailable incoming action is zero.
        import torch

        incoming = torch.cat((torch.zeros_like(action[:1]), action[:-1]), dim=0)
        intrinsic = self.ensemble.disagreement(feat, incoming)
        extrinsic = self.task_reward(feat, _state, action)
        return (self.recipe.intrinsic_scale * intrinsic + self.recipe.extrinsic_scale * extrinsic) / (1 + self.recipe.reward_norm_eps)

    def update(self, data):
        import numpy as np
        import torch

        data = {**data, "reward": np.tanh(data["reward"])}
        post, context, metrics = self.wm._train(data)
        task_metrics = self.task._train(post, self.task_reward)[-1]
        states = context["feat"].detach()
        actions = torch.as_tensor(data["action"], device=self.native.device)
        targets = post["stoch"].flatten(-2)
        with self.tools.RequiresGrad(self.ensemble):
            loss = self.ensemble.prediction_loss(states[:, :-1], actions[:, :-1], targets[:, 1:])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite disagreement ensemble loss")
            ensemble_metrics = self.ensemble_opt(loss, self.ensemble.parameters(), retain_graph=False)
        explore_metrics = self.explore._train(post, self.exploration_reward)[-1]
        result = {}
        for prefix, group in (("wm", metrics), ("task", task_metrics), ("p2e", ensemble_metrics), ("explore", explore_metrics)):
            for key, value in group.items():
                scalar = float(np.asarray(value).mean())
                if not np.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite training metric {prefix}/{key}")
                result[f"{prefix}/{key}"] = scalar
        result["sample/positive_reward_fraction"] = float((data["reward"] > 0).mean())
        result["sample/terminal_fraction"] = float(data["is_terminal"].mean())
        return result

    def save_inference_snapshot(self, path: Path, counters: dict, resolved: dict) -> None:
        """Atomic, explicitly NON-resumable. No weights-only training resumes."""
        import os
        import torch

        temporary = path.with_suffix(".tmp")
        torch.save(dict(
            checkpoint_kind="inference_only", resumable=False,
            replay_checkpointed=False, counters=counters, resolved_config=resolved,
            world_model=self.wm.state_dict(), task_actor=self.task.actor.state_dict(),
            task_value=self.task.value.state_dict(), exploration_actor=self.explore.actor.state_dict(),
            exploration_value=self.explore.value.state_dict(), ensemble=self.ensemble.state_dict(),
        ), temporary)
        os.replace(temporary, path)
