"""Frozen DINOv3 visual encoder used by KARROW."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDinoV3Encoder(nn.Module):
    """Load a local DINOv3 ViT-S/16 artifact and expose its CLS feature."""

    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    EXPECTED_FEATURES = 384

    def __init__(
        self,
        model_path: str | Path | None,
        *,
        input_size: int = 256,
        max_batch_size: int = 128,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if input_size < 16 or input_size % 16:
            raise ValueError("DINOv3 input size must be a positive multiple of 16")
        if max_batch_size < 1:
            raise ValueError("DINOv3 max batch size must be positive")

        if backbone is None:
            if model_path is None:
                raise ValueError("A local DINOv3 model path is required")
            local_path = Path(model_path)
            if not local_path.is_absolute():
                raise ValueError("DINOv3 model path must be absolute")
            if not local_path.is_dir():
                raise FileNotFoundError(
                    f"DINOv3 model directory does not exist: {local_path}"
                )
            try:
                from transformers import AutoModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Frozen DINOv3 requires the optional 'dinov3' dependencies"
                ) from exc
            backbone = AutoModel.from_pretrained(
                local_path,
                local_files_only=True,
            )

        self.backbone = backbone
        self.input_size = input_size
        self.max_batch_size = max_batch_size
        config = getattr(backbone, "config", None)
        self.output_size = int(
            getattr(config, "hidden_size", getattr(backbone, "embed_dim", self.EXPECTED_FEATURES))
        )
        if self.output_size != self.EXPECTED_FEATURES:
            raise ValueError(
                "KARROW v1 requires DINOv3 ViT-S/16 with 384 features, "
                f"got {self.output_size}"
            )

        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.backbone.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenDinoV3Encoder":
        super().train(False)
        self.backbone.eval()
        return self

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "DINOv3 images must have shape [batch, 3, height, width], "
                f"got {tuple(images.shape)}"
            )
        values = images.float()
        if values.shape[-2:] != (self.input_size, self.input_size):
            values = F.interpolate(
                values,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (values - self.pixel_mean) / self.pixel_std

    def _encode_chunk(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=self._prepare(images))
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            features = last_hidden_state[:, 0]
        else:
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                raise RuntimeError("DINOv3 backbone did not return token features")
        if features.shape[-1] != self.output_size:
            raise RuntimeError("DINOv3 returned an unexpected feature dimension")
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, images.shape[0], self.max_batch_size):
                chunks.append(self._encode_chunk(images[start : start + self.max_batch_size]))
        if not chunks:
            return images.new_empty((0, self.output_size), dtype=torch.float32)
        return torch.cat(chunks, dim=0)
