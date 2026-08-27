"""File-backed CPU tensors for replay stores that exceed RAM budgets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch


def create_file_backed_tensor(
    path: str | Path,
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a new shared mmap tensor without changing its numeric dtype."""
    dimensions = tuple(int(value) for value in shape)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("Mapped tensor dimensions must all be positive")

    storage_path = Path(path).expanduser().resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    if storage_path.exists():
        raise FileExistsError(
            f"Refusing to reuse mapped tensor storage: {storage_path}"
        )

    values = torch.from_file(
        str(storage_path),
        shared=True,
        size=math.prod(dimensions),
        dtype=dtype,
    )
    return values.reshape(dimensions)
