#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CUDA recovery smoke on an immutable AWM-AutoRoute boundary checkpoint.

Restores full replay/model/optimizer/RNG state into fresh working storage, then
exercises old/new private actors and heterogeneous BF16 states on fixed pixels.
No simulator data or smoke updates enter the campaign; parent files stay intact.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from smoke_evolving_atomic_rssm import ROOT, Config, torch, np, train, _world_model
from git_provenance import require_synced_training_git_state
from d_autoroute_resume import inspect_resume, sha256
from clworldmodel.continual import ActorCriticBank
from clworldmodel.routing import TwoFrameReconstructionRouter
from generate_trajectory import _routed_policy_step, _autocast_context


def assert_checkpoint_state_equal(actual, expected):
    """Compare every tensor AND scalar/ownership string in a checkpoint tree."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError("Checkpoint tensor type changed")
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("Checkpoint mapping keys changed")
        for key in expected:
            assert_checkpoint_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("Checkpoint sequence topology changed")
        for current, original in zip(actual, expected):
            assert_checkpoint_state_equal(current, original)
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"Checkpoint metadata changed: {actual!r} != {expected!r}")


@torch.no_grad()
def verify_mixed_routes(wm, bank, config, next_task):
    """Forced *test* episode locks cover both routes, never benchmark routing labels."""
    device = next(wm.parameters()).device
    policy = train._autorouted_behavior(config, None, bank, next_task + 1)
    route_ids = torch.tensor([0, next_task, next_task, 0], device=device)
    frames = torch.full((4, 3, config.img_size, config.img_size), .25, device=device)
    z, h = wm.rssm.initial_state(4)
    dummy = 0
    previous = torch.nn.functional.one_hot(torch.full((4,), dummy, device=device), config.action_space)
    reset = torch.zeros(4, 1, device=device)
    results = []
    modes = {module: module.training for module in wm.modules()}
    with train._preserve_training_rng_state():
        for training in (False, True):
            wm.train(training)
            router = TwoFrameReconstructionRouter(tuple(range(next_task + 1)))
            router.route(frames, reset.bool().reshape(-1), lambda _, x, rows, first: x)
            router.routes = route_ids.clone()
            router.counts = torch.full((4,), 2, device=device, dtype=torch.long)
            references = []
            with _autocast_context(device, config.compute_dtype):
                for route in (0, next_task):
                    rows = torch.where(route_ids == route)[0]
                    _, rz, rh = wm.rssm(z[rows], previous[rows], h[rows], frames[rows],
                                       reset[rows], task_id=route, stochastic=False)
                    references.append((rows, rz, rh))
            nz, nh, actions = _routed_policy_step(wm, policy, router, frames, z, h, previous,
                                                reset, stochastic=False)
            for rows, rz, rh in references:
                torch.testing.assert_close(nz[rows], rz.to(nz.dtype), rtol=0, atol=0)
                torch.testing.assert_close(nh[rows], rh.to(nh.dtype), rtol=0, atol=0)
            # Follow recurrent state for several stochastic collection-like steps.
            for _ in range(4):
                nz, nh, actions = _routed_policy_step(wm, policy, router, frames, nz, nh, previous,
                                                    reset, stochastic=True)
            if not torch.isfinite(nz).all() or not torch.isfinite(nh).all():
                raise FloatingPointError("Mixed-route resumed state is not finite")
            results.append({"training_mode": training, "forced_test_routes": route_ids.tolist(),
                            "native_h_dtypes": [str(rh.dtype) for _, _, rh in references],
                            "assembled_h_dtype": str(nh.dtype), "actions": actions.tolist()})
    for module, training in modes.items():
        module.training = training
    if not any(len(set(result["native_h_dtypes"])) > 1 for result in results):
        raise RuntimeError("Recovery smoke did not reproduce heterogeneous native route dtypes")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    provenance = require_synced_training_git_state(ROOT)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the assigned CUDA GPU for recovery smoke")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = args.checkpoint.resolve().parent.parent
    config_data = json.loads((source / "resolved_training_config.json").read_text())
    launch = json.loads((source / "launch.json").read_text())
    lineage = inspect_resume(args.checkpoint, config_data, launch["protocol"])
    config = Config.from_dict(config_data)
    output.mkdir(parents=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    wm = _world_model(config, device)
    teacher = copy.deepcopy(wm).eval()
    optimizer = torch.optim.Adam(train._evolving_shared_optimizer_parameter_groups(
        wm, core_lr=config.first_task_shared_core_lr, prediction_head_lr=config.task_private_lr), fused=True)
    bank = ActorCriticBank(artifact_kind="evolving_atomic_rssm_actor_critic_bank_resumable_state")
    factory = lambda _: train.build_actor_critic_opt(wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config))
    replay = config.get_replay_buffer(output / "working_replay")
    schedule = config.get_env_schedule()
    generators = [np.random.default_rng(i) for i in range(4)]
    restored = train._restore_evolving_resumable_checkpoint(args.checkpoint, config=config, wm=wm,
        boundary_teacher=teacher, shared_optimizer=optimizer, private_optimizers={}, route_optimizers={},
        actor_critic_bank=bank, actor_critic_factory=factory, replay_buffer=replay, environment_schedule=schedule,
        task_update_rng=generators[0], collection_environment_seed_rng=generators[1],
        validation_environment_seed_rng=generators[2], final_environment_seed_rng=generators[3],
        require_post_boundary=True)
    teacher.requires_grad_(False)
    if restored["completed_epochs"] != lineage["completed_epochs"]:
        raise RuntimeError("Launcher/trainer boundary validation disagree")
    acquired = restored["current_task_id"] + 1
    if tuple(bank.task_ids()) != tuple(range(acquired)) or schedule._step != restored["completed_epochs"]:
        raise RuntimeError("Checkpoint actor/scheduler ownership was not restored")
    # Check the full source tensors/optimizer state, not only their checksum.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    torch.testing.assert_close({k: v.cpu() for k, v in wm.state_dict().items()}, payload["world_model"], rtol=0, atol=0)
    torch.testing.assert_close({k: v.cpu() for k, v in teacher.state_dict().items()}, payload["boundary_teacher"], rtol=0, atol=0)
    assert_checkpoint_state_equal(optimizer.state_dict(), payload["optimizers"]["shared"])
    assert_checkpoint_state_equal(bank.resumable_state_dict(), payload["optimizers"]["actor_critic_bank"])
    assert_checkpoint_state_equal(random.getstate(), payload["rng"]["python"])
    assert_checkpoint_state_equal(np.random.get_state(), payload["rng"]["numpy_legacy"])
    assert_checkpoint_state_equal(torch.random.get_rng_state(), payload["rng"]["torch_cpu"])
    assert_checkpoint_state_equal(torch.cuda.get_rng_state_all(), payload["rng"]["torch_cuda"])
    for generator, name in zip(generators, ("task_update", "collection_environment",
                                           "validation_environment", "final_environment")):
        assert_checkpoint_state_equal(generator.bit_generator.state, payload["rng"][name])
    actual_replay = replay.state_dict()
    expected_replay = copy.deepcopy(payload["replay"])
    from clworldmodel.replay.mapped_tensor import open_file_backed_tensor
    for buffer, actual, expected in zip(replay.replays, actual_replay["replays"], expected_replay["replays"]):
        metadata = expected.pop("observations")
        current = actual.pop("observations")
        if Path(current["path"]).resolve() == Path(metadata["path"]).resolve():
            raise AssertionError("Replay working storage aliases immutable checkpoint")
        original = open_file_backed_tensor(Path(metadata["path"]), tuple(metadata["shape"]),
                                          dtype=buffer.obss.dtype)
        for start in range(0, buffer.t, 8):
            if not torch.equal(buffer.obss[start:start + 8, :buffer.n_valid],
                               original[start:start + 8, :buffer.n_valid]):
                raise AssertionError("Restored replay observations changed")
    assert_checkpoint_state_equal(actual_replay, expected_replay)
    del payload
    if acquired >= config.rssm_num_experts:
        raise ValueError("This acquisition-boundary smoke needs a pending new route")
    wm.initialize_task_expert(acquired, acquired - 1)
    wm.activate_task_expert(acquired)
    bank.ensure(acquired, factory)
    bank.activate(acquired)
    mixed = verify_mixed_routes(wm, bank, config, acquired)
    if sha256(args.checkpoint) != lineage["source_checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint changed during smoke")
    artifact = {"classification": "smoke", "project_git": provenance, "source": lineage,
                "restored": restored, "mixed_route_checks": mixed, "full_state_comparison_passed": True,
                "full_replay_and_rng_comparison_passed": True,
                "replay_task_ids": list(replay.available_task_ids()), "real_environment_steps": 0,
                "optimizer_updates": 0, "routing_accuracy_claimed": False}
    (output / "PASSED.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
