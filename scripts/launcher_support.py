"""Leaf process and manifest helpers for standalone training launchers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def runtime_info(python: Path, env: dict[str, str]) -> dict:
    """Collect the pinned runtime and accelerator fields for a launch manifest."""
    probe_code = """
import json
import os
import platform
import sys
from importlib import metadata

import torch

assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
properties = torch.cuda.get_device_properties(0)
packages = (
    "ale-py",
    "gymnasium",
    "numpy",
    "opencv-python",
    "sortedcontainers",
    "tensorboard",
    "torch",
    "torchaudio",
    "torchvision",
    "tqdm",
)
package_versions = {name: metadata.version(name) for name in packages}
try:
    package_versions["swanlab"] = metadata.version("swanlab")
except metadata.PackageNotFoundError:
    package_versions["swanlab"] = None
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "packages": package_versions,
    "torch_cuda_build": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": properties.name,
    "cuda_total_memory_bytes": properties.total_memory,
}))
"""
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return json.loads(probe.stdout.strip())


def write_json(path: Path, value: dict) -> None:
    """Atomically write the stable launcher JSON representation."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_and_tee(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    """Run one command while mirroring its combined output to console and disk."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        return process.wait()
