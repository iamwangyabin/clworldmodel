#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import torch

EXPECTED_PACKAGES = {
    "ale-py": "0.11.1",
    "gymnasium": "1.1.1",
    "numpy": "1.26.4",
    "torch": "2.3.0",
    "torchaudio": "2.3.0",
    "torchvision": "0.18.0",
}
GAMES = {
    "MsPacman": "ms_pacman",
    "Boxing": "boxing",
    "CrazyClimber": "crazy_climber",
    "Frostbite": "frostbite",
    "Seaquest": "seaquest",
    "Enduro": "enduro",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the ARROW reference environment")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also require a visible CUDA GPU (use at container runtime, not image build)",
    )
    return parser


def _check_versions() -> dict[str, str]:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"ARROW reference requires Python 3.10, found {sys.version.split()[0]}")
    versions = {package: importlib.metadata.version(package) for package in EXPECTED_PACKAGES}
    mismatches = {
        package: (EXPECTED_PACKAGES[package], actual)
        for package, actual in versions.items()
        if actual.split("+", 1)[0] != EXPECTED_PACKAGES[package]
    }
    if mismatches:
        raise RuntimeError(f"Reference package version mismatch: {mismatches}")
    if torch.version.cuda != "11.8":
        raise RuntimeError(f"Expected CUDA 11.8 PyTorch build, found {torch.version.cuda}")
    return versions


def _check_atari() -> dict[str, str]:
    gym.register_envs(ale_py)
    rom_paths = {}
    for game, rom_name in GAMES.items():
        rom_path = Path(ale_py.roms.get_rom_path(rom_name))
        if not rom_path.is_file():
            raise RuntimeError(f"Bundled ROM is missing: {rom_path}")
        env = gym.make(
            f"ALE/{game}-v5",
            frameskip=1,
            repeat_action_probability=0,
            full_action_space=True,
        )
        try:
            observation, _ = env.reset(seed=0)
            if observation.shape != (210, 160, 3):
                raise RuntimeError(f"Unexpected {game} observation shape: {observation.shape}")
            if env.action_space.n != 18:
                raise RuntimeError(f"Unexpected {game} action count: {env.action_space.n}")
        finally:
            env.close()
        rom_paths[game] = str(rom_path)
    return rom_paths


def main() -> int:
    args = _parser().parse_args()
    versions = _check_versions()
    rom_paths = _check_atari()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not visible; start the container with --gpus all "
            "and NVIDIA Container Toolkit"
        )
    report = {
        "python": sys.version.split()[0],
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "roms": rom_paths,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
