"""Validated mixed-precision helpers shared by project and vendored trainers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Literal

import torch


ComputeDType = Literal["float32", "bfloat16"]


def validate_compute_dtype(compute_dtype: str) -> ComputeDType:
    if compute_dtype not in {"float32", "bfloat16"}:
        raise ValueError(f"Unknown compute dtype: {compute_dtype!r}")
    return compute_dtype


def torch_dtype(compute_dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[validate_compute_dtype(compute_dtype)]


def autocast_context(
    device: torch.device | str,
    compute_dtype: str,
) -> ContextManager[None]:
    """Autocast CUDA kernels while keeping CPU tests and FP32 runs unchanged."""
    compute_dtype = validate_compute_dtype(compute_dtype)
    device_type = torch.device(device).type
    if compute_dtype == "bfloat16" and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def require_cuda_compute_support(compute_dtype: str) -> None:
    """Fail before model allocation when a requested CUDA dtype is unsupported."""
    compute_dtype = validate_compute_dtype(compute_dtype)
    if compute_dtype == "float32":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 compute requires an available CUDA accelerator")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA accelerator does not support BF16 compute")
