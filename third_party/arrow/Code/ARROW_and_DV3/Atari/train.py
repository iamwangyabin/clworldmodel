import argparse
import hashlib
import json
import os
import random
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import ale_py
import replay
from replay import MultiTypeReplay
from ac import ActorCriticOpt, train_ac_from_wm
from config import (
    Config,
    EnvConfig,
    EnvScheduleConfig,
    RbConfig,
    _arrow_fifo_ltdm_capacity_ns,
    _arrow_fifo_ltdm_sampling_weights,
)
from generate_trajectory import (
    SequentialEnvironments,
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)
from wm import WorldModel


def _environment_seed_streams(
    seed: int,
) -> tuple[np.random.Generator, np.random.Generator]:
    collection_seed, evaluation_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(collection_seed), np.random.default_rng(evaluation_seed)


def _next_environment_seed(seed_rng: np.random.Generator) -> int:
    return int(seed_rng.integers(0, 2**32, dtype=np.uint64))


@contextmanager
def _preserve_training_rng_state():
    """Keep stochastic evaluation from changing subsequent training draws."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _print_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[cuda-mem] {tag}: CUDA is not available.")
        return
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    peak_alloc = torch.cuda.max_memory_allocated(dev)
    peak_reserved = torch.cuda.max_memory_reserved(dev)
    # Reserved peak is usually the safer number for scheduler sizing.
    suggested = peak_reserved * 1.15
    print(
        f"[cuda-mem] {tag} | device={dev} ({torch.cuda.get_device_name(dev)}) "
        f"allocated={_bytes_to_gib(allocated):.2f} GiB "
        f"reserved={_bytes_to_gib(reserved):.2f} GiB "
        f"peak_allocated={_bytes_to_gib(peak_alloc):.2f} GiB "
        f"peak_reserved={_bytes_to_gib(peak_reserved):.2f} GiB "
        f"suggested_slurm_gpu_mem={_bytes_to_gib(suggested):.2f} GiB"
    )


def _print_replay_buffer_debug(config: Config, buf) -> None:
    """Log replay capacities and ARROW FIFO vs LTDM minibatch sampling weights."""
    print(
        f"[replay] algorithm={config.algorithm} data_t={config.data_t} "
        f"data_n_max={config.data_n_max}"
    )
    if isinstance(buf, MultiTypeReplay):
        total_slots = 2 * config.data_n_max
        n_fifo, n_ltdm = _arrow_fifo_ltdm_capacity_ns(
            total_slots, config.arrow_replay_capacity_ratio
        )
        w_fifo, w_ltdm = _arrow_fifo_ltdm_sampling_weights(
            config.arrow_replay_capacity_ratio
        )
        print(
            f"[replay] ARROW total_trajectory_slots={total_slots} "
            f"(2 * data_n_max), arrow_replay_capacity_ratio={config.arrow_replay_capacity_ratio}"
        )
        print(
            f"[replay] capacity split: FifoReplay n={n_fifo} ({n_fifo / total_slots:.4f}), "
            f"LongTermReplay n={n_ltdm} ({n_ltdm / total_slots:.4f})"
        )
        print(
            f"[replay] minibatch sampling weights (random.choices): "
            f"Fifo={w_fifo}, LTDM={w_ltdm} (sum={w_fifo + w_ltdm})"
        )
        for i, sub in enumerate(buf.replays):
            sw = buf.sampling_weights[i]
            nv = getattr(sub, "n_valid", None)
            print(
                f"[replay]   [{i}] {type(sub).__name__}: t={sub.t} n={sub.n} "
                f"n_valid={nv} sampling_weight={sw}"
            )
    else:
        dv3_max = getattr(config, "sac_dv3_data_n_max", None)
        print(
            f"[replay] single buffer: {type(buf).__name__} t={buf.t} n={buf.n} "
            f"n_valid={buf.n_valid} (config.sac_dv3_data_n_max={dv3_max})"
        )


def _stage_clock(enabled: bool) -> float:
    if not enabled:
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _stage_elapsed(start: float, enabled: bool) -> float:
    if not enabled:
        return 0.0
    return _stage_clock(True) - start


def _evaluate_policy_tasks(
    config: Config,
    wm: WorldModel,
    aco: Optional[ActorCriticOpt],
    eval_funcs,
    environment_seed_rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    with _preserve_training_rng_state():
        for env_fns in eval_funcs:
            mean, std = evaluate(
                config.n_sync,
                wm=wm,
                ac=aco.ac if aco is not None else None,
                env_fns=env_fns,
                env_repeat=config.env_repeat,
                n_rollouts=16,
                seed=_next_environment_seed(environment_seed_rng),
            )
            means.append(mean)
            stds.append(std)
    return means, stds


def _raw_return_statistics(
    task_configs, scaled_means: list[float], scaled_stds: list[float]
) -> tuple[list[float], list[float]]:
    if not (len(task_configs) == len(scaled_means) == len(scaled_stds)):
        raise ValueError("Evaluation tasks and statistics must have matching lengths")
    raw_means = []
    raw_stds = []
    for task, scaled_mean, scaled_std in zip(
        task_configs, scaled_means, scaled_stds
    ):
        if task.rew_scale == 0:
            raise ValueError(f"Task {task.name!r} has zero reward scale")
        raw_means.append(float(scaled_mean / task.rew_scale))
        raw_stds.append(float(scaled_std / abs(task.rew_scale)))
    return raw_means, raw_stds


def _task_boundary_metadata(config: Config, epoch: int) -> Optional[dict]:
    if config.esc.env_schedule_type is not SequentialEnvironments:
        return None
    swap_sched = config.esc.kwargs.get("swap_sched")
    if not isinstance(swap_sched, int) or swap_sched < 1:
        raise ValueError("Sequential environment schedule requires positive swap_sched")

    completed_epochs = epoch + 1
    if completed_epochs % swap_sched != 0:
        return None

    boundary_index = completed_epochs // swap_sched
    task_index = (boundary_index - 1) % len(config.esc.env_configs)
    task = config.esc.env_configs[task_index]
    return {
        "boundary_index": boundary_index,
        "task_index": task_index,
        "task_name": task.name,
        "task_reward_scale": task.rew_scale,
    }


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _parameter_accounting(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def _module_state_accounting(module: torch.nn.Module) -> dict[str, int]:
    accounting = _parameter_accounting(module)
    buffers = list(module.buffers())
    buffer_values = sum(buffer.numel() for buffer in buffers)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    return {
        **accounting,
        "buffers": buffer_values,
        "buffer_bytes": buffer_bytes,
        "parameter_and_buffer_bytes": accounting["parameter_bytes"] + buffer_bytes,
    }


def _actor_critic_parameter_accounting(aco: ActorCriticOpt) -> dict:
    actor = aco.ac.actor
    critic = aco.ac.critic
    if not isinstance(actor, torch.nn.Module) or not isinstance(critic, torch.nn.Module):
        raise TypeError("Actor and critic must be torch modules for resource accounting")
    return {
        "schema_version": 1,
        "actor_class": type(actor).__name__,
        "actor": _module_state_accounting(actor),
        "critic_class": type(critic).__name__,
        "critic": _module_state_accounting(critic),
        "actor_critic": _module_state_accounting(aco.ac),
        "accounting_scope": (
            "parameters and persistent buffers; excludes gradients, optimizer state, "
            "and activations"
        ),
    }


def _world_model_parameter_accounting(wm: WorldModel) -> dict:
    if wm.observation_objective == "reconstruction":
        observation_head_name = "decoder"
        observation_head = wm.decoder
    else:
        observation_head_name = "r2_projector"
        observation_head = wm.r2_projector
    return {
        "schema_version": 1,
        "observation_objective": wm.observation_objective,
        "world_model": _parameter_accounting(wm),
        "observation_head_name": observation_head_name,
        "observation_head": _parameter_accounting(observation_head),
        "accounting_scope": (
            "parameters only; excludes gradients, optimizer state, and activations"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_analysis_snapshot(
    snapshot_dir: Path,
    *,
    config: Config,
    wm: WorldModel,
    aco: ActorCriticOpt,
    epoch: int,
    world_model_updates: int,
    total_env_steps: int,
    reason: str,
    task_metadata: Optional[dict] = None,
) -> Path:
    """Save portable weights for offline diagnosis, not resumable training state."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if reason == "task_boundary":
        if task_metadata is None:
            raise ValueError("task-boundary snapshot requires task metadata")
        filename = (
            f"boundary_{task_metadata['boundary_index']:02d}_"
            f"task_{task_metadata['task_index']:02d}_epoch_{epoch:04d}.pt"
        )
    elif reason == "final":
        filename = f"final_epoch_{epoch:04d}.pt"
    else:
        raise ValueError(f"Unknown analysis snapshot reason: {reason}")

    path = snapshot_dir / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis snapshot: {path}")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "artifact_kind": "analysis_snapshot",
        "resumable": False,
        "reason": reason,
        "epoch": epoch,
        "completed_epochs": epoch + 1,
        "world_model_updates": world_model_updates,
        "actor_critic_updates": (epoch + 1) * config.ac_train_steps,
        "total_raw_environment_frames": total_env_steps,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "task": task_metadata,
        "config": config.to_dict(),
        "world_model_state_dict": _cpu_state_dict(wm),
        "actor_critic_state_dict": _cpu_state_dict(aco.ac),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

    digest = _sha256(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest_path = digest_path.with_suffix(digest_path.suffix + ".tmp")
    temporary_digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary_digest_path, digest_path)
    print(
        f"[analysis-snapshot] reason={reason} epoch={epoch} "
        f"path={path} sha256={digest}"
    )
    return path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument(
        "--arrow-replay-ratio",
        choices=["50-50", "25-75", "75-25"],
        default=None,
        help="ARROW: optional FIFO/LTDM capacity split override.",
    )
    parser.add_argument(
        "--observation-objective",
        choices=["reconstruction", "r2"],
        default=None,
        help="Optional world-model observation-objective override.",
    )
    parser.add_argument(
        "--actor-network",
        choices=["mlp", "relu_kan"],
        default=None,
        help="Optional actor architecture override; the critic remains unchanged.",
    )
    parser.add_argument("--r2-barlow-loss-scale", type=float, default=None)
    parser.add_argument("--r2-redundancy-scale", type=float, default=None)
    parser.add_argument("--r2-normalization-eps", type=float, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional explicit training-epoch override for a named pilot protocol.",
    )
    parser.add_argument("--compile-world-model", action="store_true")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Synchronize at stage boundaries and print per-epoch wall times.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Explicit run directory for TensorBoard events and resolved config.",
    )
    parser.add_argument(
        "--analysis-snapshot-dir",
        type=Path,
        help=(
            "Save portable world-model and actor-critic weights at task boundaries "
            "and at training end. These snapshots do not contain replay, optimizers, "
            "or RNG state and are not resumable checkpoints."
        ),
    )
    parser.add_argument(
        "--evaluate-final",
        action="store_true",
        help="Evaluate the final frozen policy after all configured training epochs.",
    )
    args = parser.parse_args()

    save_nets = False
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    log_images = False
    analysis_snapshot_dir = (
        args.analysis_snapshot_dir.resolve()
        if args.analysis_snapshot_dir is not None
        else None
    )
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    if args.config is not None:
        config = Config.from_file(Path(args.config))
    else:
        raise ValueError("--config is required")

    config_overrides = config.to_dict()
    if args.arrow_replay_ratio is not None:
        config_overrides["arrow_replay_capacity_ratio"] = args.arrow_replay_ratio
    if args.observation_objective is not None:
        config_overrides["observation_objective"] = args.observation_objective
    if args.actor_network is not None:
        config_overrides["actor_network"] = args.actor_network
    if args.r2_barlow_loss_scale is not None:
        config_overrides["r2_barlow_loss_scale"] = args.r2_barlow_loss_scale
    if args.r2_redundancy_scale is not None:
        config_overrides["r2_redundancy_scale"] = args.r2_redundancy_scale
    if args.r2_normalization_eps is not None:
        config_overrides["r2_normalization_eps"] = args.r2_normalization_eps
    if args.epochs is not None:
        config_overrides["epochs"] = args.epochs
    config = Config.from_dict(config_overrides)

    if config.algorithm == "arrow":
        print(f"ARROW FIFO/LTDM capacity ratio: {config.arrow_replay_capacity_ratio}")
    print(f"World-model observation objective: {config.observation_objective}")
    print(f"Actor network: {config.actor_network}")

    if config.algorithm == "sac":
        exit(0)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    collection_environment_seed_rng, evaluation_environment_seed_rng = (
        _environment_seed_streams(config.seed)
    )
    print("Training with seed: ", config.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _print_cuda_memory("startup")
    wm = WorldModel(
        3,
        (32, 32),
        config.action_space,
        config.gru_units,
        config.cnn_depth,
        config.mlp_features,
        config.mlp_layers,
        config.wall_time_optimisation,
        observation_objective=config.observation_objective,
        r2_barlow_loss_scale=config.r2_barlow_loss_scale,
        r2_redundancy_scale=config.r2_redundancy_scale,
        r2_normalization_eps=config.r2_normalization_eps,
    ).cuda()
    opt = Adam(wm.parameters(), lr=config.wm_lr, fused=args.fused_adam)
    compute_world_model_loss = wm.compute_loss
    if args.compile_world_model:
        compute_world_model_loss = torch.compile(
            compute_world_model_loss, dynamic=False, mode="reduce-overhead"
        )

    envs = config.get_env_schedule()
    replay = config.get_replay_buffer()
    _print_replay_buffer_debug(config, replay)

    # OPTIONAL: Load from existing
    aco: Optional[ActorCriticOpt] = None

    if log_dir is None:
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        job_id = os.getenv("SLURM_JOB_ID")
        run_name = f"{current_time}_{socket.gethostname()}_{config.seed}_{job_id}"
        # One env in the schedule → single-task; multiple → continual (sequential) training

        if len(config.esc.env_configs) == 1: 
            task_kind = "single"
        else:
            if config.esc.env_configs[0].name == "ALE/MsPacman-v5" and config.esc.kwargs["swap_sched"] == 90:
                task_kind = "cl_original"
            elif config.esc.env_configs[0].name == "ALE/Enduro-v5" and config.esc.kwargs["swap_sched"] == 90:
                task_kind = "cl_reversed"
            else:
                task_kind = "cl_two_cycle"
        
        if config.algorithm == "arrow":
            ratio = config.arrow_replay_capacity_ratio.replace("-", "_")
            log_root = Path.cwd() / "runs" / task_kind / config.algorithm / ratio
        else:
            log_root = Path.cwd() / "runs" / task_kind / config.algorithm        

        log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] log_dir={log_dir}")
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] log_dir={log_dir} (explicit)")


    writer = SummaryWriter(log_dir=log_dir)
    log_dir = Path(log_dir)
    config.save(log_dir / "config.json")
    parameter_accounting_path = log_dir / "model_parameter_accounting.json"
    temporary_accounting_path = parameter_accounting_path.with_suffix(".json.tmp")
    temporary_accounting_path.write_text(
        json.dumps(_world_model_parameter_accounting(wm), indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_accounting_path, parameter_accounting_path)
    actor_accounting_path = log_dir / "actor_critic_parameter_accounting.json"

    
    total_env_steps = 0        # number of *real* environment interactions so far

    best_rews_mean = float("-inf")
    global_step = 0            # gradient updates so far  training iterations

    for epoch in range(config.epochs):
        print("Starting Epoch ", epoch)
        epoch_started = _stage_clock(args.profile_stages)
        collect_started = _stage_clock(args.profile_stages)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if config.random_policy == "first":
            random_policy = epoch == 0
        elif config.random_policy == "new":
            random_policy = envs.is_new_env()
        for _ in range(
            config.pretrain_data_multiplier if random_policy and config.pretrain_enabled else 1
        ):
            _acts, _obss, _rews, _conts, _resets = reinterpret_nt_to_t_n(
                *generate_trajectories(
                    config.n_sync * config.gen_seq_len,
                    config.n_sync,
                    wm=wm,
                    ac=None if random_policy else aco.ac,
                    env_fns=envs.funcs(),
                    env_repeat=config.env_repeat,
                    seed=_next_environment_seed(collection_environment_seed_rng),
                ),
                config.data_t,
                config.data_n,
            )
            replay.add(_acts, _obss, _rews, _conts, _resets)
            print(f"{replay.n_valid=}")
            num_new_env_steps = _acts.shape[0] * _acts.shape[1] * config.env_repeat
            total_env_steps += num_new_env_steps
            writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)

        rews_eps_mean = _rews.sum().item() / _resets.sum().item()
        writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
        len_eps_mean = config.gen_seq_len / _resets.sum().item() * config.env_repeat
        writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)
        if rews_eps_mean >= best_rews_mean:
            best_rews_mean = rews_eps_mean
            if save_nets and aco is not None:
                print(f"Saving best rews eps mean {rews_eps_mean=}")
                torch.save(wm.state_dict(), log_dir / "save_wm_best.pt")
                torch.save(aco.ac.state_dict(), log_dir / "save_ac_best.pt")

        collect_seconds = _stage_elapsed(collect_started, args.profile_stages)

        # Evaluation games
        eval_started = _stage_clock(args.profile_stages)
        if epoch % 10 == 0:
            eval_results_mean, eval_results_std = _evaluate_policy_tasks(
                config,
                wm,
                aco,
                envs.eval_funcs(),
                evaluation_environment_seed_rng,
            )
            eval_raw_mean, eval_raw_std = _raw_return_statistics(
                config.esc.env_configs, eval_results_mean, eval_results_std
            )
            writer.add_scalars(
                "Perf/eval_rew_eps_mean",
                {f"{i}": m for i, m in enumerate(eval_results_mean)},
                global_step,
            )
            writer.add_scalars(
                "Perf/eval_rew_eps_std",
                {f"{i}": s for i, s in enumerate(eval_results_std)},
                global_step,
            )
            writer.add_scalars(
                "Perf/eval_raw_return_mean",
                {f"{i}": mean for i, mean in enumerate(eval_raw_mean)},
                global_step,
            )
            writer.add_scalars(
                "Perf/eval_raw_return_std",
                {f"{i}": std for i, std in enumerate(eval_raw_std)},
                global_step,
            )
            print("Eval for epoch: ",epoch)
            print(f"Eval means: {eval_results_mean}")
            print(f"Eval stds: {eval_results_std}")
            print(f"Eval raw means: {eval_raw_mean}")
            print(f"Eval raw stds: {eval_raw_std}")

        eval_seconds = _stage_elapsed(eval_started, args.profile_stages)

        world_model_started = _stage_clock(args.profile_stages)
        progbar = trange(
            config.steps_per_batch
            if epoch > 0 or not config.pretrain_enabled
            else config.pretrain_steps,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            disable = True,
        )
        for _ in progbar:
            if args.compile_world_model:
                torch.compiler.cudagraph_mark_step_begin()
            if epoch > 0 or not config.pretrain_enabled:
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets = replay.minibatch(
                    config.mb_t_size, config.mb_n_size
                )
            else:
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets = replay.minibatch(
                    config.pretrain_mb_t_size, config.pretrain_mb_n_size
                )

            loss, metrics = compute_world_model_loss(
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000)
            opt.step()

            # Optional progress bar logging
            # if global_step % 10 == 0:
            #     progbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.items()})

            if global_step % config.log_frequency == 0:
                writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
                with torch.no_grad():
                    for metric_key, metric_value in metrics.items():
                        writer.add_scalar(metric_key, metric_value, global_step)

                    if log_images and config.observation_objective == "reconstruction":
                        original = _obss[:16, 0:2].cuda()
                        writer.add_images(
                            "original", original.swapaxes(0, 1).flatten(0, 1), global_step
                        )

                        init_z, init_h = wm.rssm.initial_state(original.shape[1])
                        no_resets = torch.zeros(*original.shape[:2], 1, device=init_z.device)
                        z_posts, z, h = wm.rssm(
                            init_z, _acts[:, 0:2].cuda(), init_h, original, no_resets
                        )
                        zhs = wm.zh_transform(z, h)
                        recon = torch.stack([wm.decoder(zh) for zh in zhs])

                        writer.add_images(
                            "reconstructed",
                            recon.clip(0, 1).swapaxes(0, 1).flatten(0, 1),
                            global_step,
                        )
                        writer.add_images(
                            "latent",
                            z_posts.exp().swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
                        writer.add_images(
                            "latent sample",
                            z.swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
            global_step += 1

        world_model_seconds = _stage_elapsed(world_model_started, args.profile_stages)
        actor_started = _stage_clock(args.profile_stages)

        if config.fresh_ac and epoch % config.fresh_ac == 0:
            aco, approx_perf = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                config.ac_train_sync,
                dream_steps=16,
                lr=4e-4,
                actor_network=config.actor_network,
                actor_kan_hidden_features=config.actor_kan_hidden_features,
                actor_kan_grid_size=config.actor_kan_grid_size,
                actor_kan_spline_order=config.actor_kan_spline_order,
                actor_kan_input_min=config.actor_kan_input_min,
                actor_kan_input_max=config.actor_kan_input_max,
                actor_kan_normalize_recurrent_state=(
                    config.actor_kan_normalize_recurrent_state
                ),
            )
        else:
            aco, approx_perf = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                config.ac_train_sync,
                dream_steps=16,
                aco=aco,
                lr=1e-4,
                actor_network=config.actor_network,
                actor_kan_hidden_features=config.actor_kan_hidden_features,
                actor_kan_grid_size=config.actor_kan_grid_size,
                actor_kan_spline_order=config.actor_kan_spline_order,
                actor_kan_input_min=config.actor_kan_input_min,
                actor_kan_input_max=config.actor_kan_input_max,
                actor_kan_normalize_recurrent_state=(
                    config.actor_kan_normalize_recurrent_state
                ),
            )

        actor_seconds = _stage_elapsed(actor_started, args.profile_stages)
        if not actor_accounting_path.exists():
            temporary_actor_accounting_path = actor_accounting_path.with_suffix(
                ".json.tmp"
            )
            temporary_actor_accounting_path.write_text(
                json.dumps(_actor_critic_parameter_accounting(aco), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_actor_accounting_path, actor_accounting_path)
        writer.add_scalar("Perf/approx_perf", approx_perf, global_step)
        _print_cuda_memory(f"epoch_end_{epoch}")

        if save_nets:
            torch.save(wm.state_dict(), log_dir / "save_wm.pt")
            torch.save(aco.ac.state_dict(), log_dir / "save_ac.pt")

        if analysis_snapshot_dir is not None:
            boundary_metadata = _task_boundary_metadata(config, epoch)
            if boundary_metadata is not None:
                _save_analysis_snapshot(
                    analysis_snapshot_dir,
                    config=config,
                    wm=wm,
                    aco=aco,
                    epoch=epoch,
                    world_model_updates=global_step,
                    total_env_steps=total_env_steps,
                    reason="task_boundary",
                    task_metadata=boundary_metadata,
                )
                writer.flush()
            if epoch == config.epochs - 1:
                _save_analysis_snapshot(
                    analysis_snapshot_dir,
                    config=config,
                    wm=wm,
                    aco=aco,
                    epoch=epoch,
                    world_model_updates=global_step,
                    total_env_steps=total_env_steps,
                    reason="final",
                )
                writer.flush()

        envs.step()
        torch.cuda.empty_cache()
        if args.profile_stages:
            epoch_seconds = _stage_elapsed(epoch_started, True)
            measured = (
                collect_seconds + eval_seconds + world_model_seconds + actor_seconds
            )
            print(
                "[stage-time] "
                f"epoch={epoch} collect={collect_seconds:.3f}s "
                f"eval={eval_seconds:.3f}s "
                f"world_model={world_model_seconds:.3f}s "
                f"actor={actor_seconds:.3f}s "
                f"overhead={max(0.0, epoch_seconds - measured):.3f}s "
                f"total={epoch_seconds:.3f}s"
            )

    if args.evaluate_final:
        eval_funcs = envs.eval_funcs()
        task_configs = config.esc.env_configs
        if config.esc.env_schedule_type is SequentialEnvironments:
            swap_sched = config.esc.kwargs["swap_sched"]
            seen_tasks = min(
                len(task_configs), (config.epochs + swap_sched - 1) // swap_sched
            )
            eval_funcs = eval_funcs[:seen_tasks]
            task_configs = task_configs[:seen_tasks]
        final_scaled_means, final_scaled_stds = _evaluate_policy_tasks(
            config, wm, aco, eval_funcs, evaluation_environment_seed_rng
        )
        final_raw_means, final_raw_stds = _raw_return_statistics(
            task_configs, final_scaled_means, final_scaled_stds
        )
        final_evaluation = {
            "schema_version": 1,
            "evaluation_after_completed_epochs": config.epochs,
            "policy": "stochastic",
            "rollouts_per_task": 16,
            "tasks": [
                {
                    "task_index": index,
                    "task_name": task.name,
                    "reward_scale": task.rew_scale,
                    "scaled_return_mean": scaled_mean,
                    "scaled_return_std": scaled_std,
                    "raw_return_mean": raw_mean,
                    "raw_return_std": raw_std,
                }
                for index, (
                    task,
                    scaled_mean,
                    scaled_std,
                    raw_mean,
                    raw_std,
                ) in enumerate(
                    zip(
                        task_configs,
                        final_scaled_means,
                        final_scaled_stds,
                        final_raw_means,
                        final_raw_stds,
                    )
                )
            ],
        }
        final_evaluation_path = log_dir / "final_evaluation.json"
        temporary_final_evaluation_path = final_evaluation_path.with_suffix(".json.tmp")
        temporary_final_evaluation_path.write_text(
            json.dumps(final_evaluation, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_final_evaluation_path, final_evaluation_path)
        writer.add_scalars(
            "Perf/final_eval_rew_eps_mean",
            {f"{i}": mean for i, mean in enumerate(final_scaled_means)},
            global_step,
        )
        writer.add_scalars(
            "Perf/final_eval_rew_eps_std",
            {f"{i}": std for i, std in enumerate(final_scaled_stds)},
            global_step,
        )
        writer.add_scalars(
            "Perf/final_eval_raw_return_mean",
            {f"{i}": mean for i, mean in enumerate(final_raw_means)},
            global_step,
        )
        writer.add_scalars(
            "Perf/final_eval_raw_return_std",
            {f"{i}": std for i, std in enumerate(final_raw_stds)},
            global_step,
        )
        print(f"Final eval scaled means: {final_scaled_means}")
        print(f"Final eval scaled stds: {final_scaled_stds}")
        print(f"Final eval raw means: {final_raw_means}")
        print(f"Final eval raw stds: {final_raw_stds}")
    writer.close()
    _print_cuda_memory("training_end")
