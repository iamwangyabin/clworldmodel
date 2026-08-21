"""Frozen DINOv3 visual encoder used by KARROW."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDinoV3Encoder(nn.Module):
    """Load a local DINOv3 ViT-S/16 artifact and expose frozen visual features."""

    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    EXPECTED_FEATURES = 384

    def __init__(
        self,
        model_path: str | Path | None,
        *,
        input_size: int = 256,
        max_batch_size: int = 128,
        feature_mode: Literal["cls", "patch_grid"] = "cls",
        patch_pool_size: int = 4,
        patch_feature_dim: int = 384,
        patch_projection: Literal[
            "none", "task1_pca", "fixed_orthogonal"
        ] = "none",
        patch_projection_seed: int = 0,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if input_size < 16 or input_size % 16:
            raise ValueError("DINOv3 input size must be a positive multiple of 16")
        if max_batch_size < 1:
            raise ValueError("DINOv3 max batch size must be positive")
        if feature_mode not in {"cls", "patch_grid"}:
            raise ValueError(f"Unknown DINOv3 feature mode: {feature_mode!r}")
        if patch_pool_size < 1:
            raise ValueError("DINOv3 patch pool size must be positive")
        if patch_feature_dim < 1 or patch_feature_dim > self.EXPECTED_FEATURES:
            raise ValueError("DINOv3 projected patch features must be in [1, 384]")
        if patch_projection not in {"none", "task1_pca", "fixed_orthogonal"}:
            raise ValueError(
                f"Unknown DINOv3 patch projection: {patch_projection!r}"
            )
        if feature_mode != "patch_grid" and patch_projection != "none":
            raise ValueError("DINOv3 patch projection requires patch-grid mode")
        if patch_projection_seed < 0:
            raise ValueError("DINOv3 patch projection seed must be non-negative")

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
        self.feature_mode = feature_mode
        self.patch_pool_size = patch_pool_size
        self.patch_feature_dim = patch_feature_dim
        self.patch_projection_kind = patch_projection
        config = getattr(backbone, "config", None)
        self.token_features = int(
            getattr(
                config,
                "hidden_size",
                getattr(backbone, "embed_dim", self.EXPECTED_FEATURES),
            )
        )
        if self.token_features != self.EXPECTED_FEATURES:
            raise ValueError(
                "KARROW requires DINOv3 ViT-S/16 with 384 token features, "
                f"got {self.token_features}"
            )
        self.patch_size = int(getattr(config, "patch_size", 16))
        if self.patch_size != 16:
            raise ValueError("KARROW requires DINOv3 ViT-S/16 patch size 16")
        self.patch_grid_size = self.input_size // self.patch_size
        if self.patch_grid_size * self.patch_size != self.input_size:
            raise ValueError("DINOv3 input size must be divisible by its patch size")
        if self.patch_pool_size > self.patch_grid_size:
            raise ValueError("DINOv3 patch pool cannot exceed the source patch grid")
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))

        if self.feature_mode == "patch_grid":
            if self.patch_projection_kind == "none":
                if self.patch_feature_dim != self.token_features:
                    raise ValueError(
                        "Unprojected DINOv3 patch features must retain 384 channels"
                    )
            elif self.patch_feature_dim >= self.token_features:
                raise ValueError(
                    "DINOv3 patch projection must reduce the patch feature width"
                )
        output_channels = (
            self.patch_feature_dim
            if self.feature_mode == "patch_grid"
            else self.token_features
        )
        self.output_size = output_channels * (
            self.patch_pool_size**2 if self.feature_mode == "patch_grid" else 1
        )

        projection = None
        projection_mean = None
        if self.patch_projection_kind == "task1_pca":
            projection = torch.zeros(
                self.token_features,
                self.patch_feature_dim,
                dtype=torch.float32,
            )
            projection_mean = torch.zeros(self.token_features, dtype=torch.float32)
        elif self.patch_projection_kind == "fixed_orthogonal":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(patch_projection_seed)
            matrix = torch.randn(
                self.token_features,
                self.patch_feature_dim,
                generator=generator,
                dtype=torch.float64,
            )
            projection = torch.linalg.qr(matrix, mode="reduced").Q
            columns = torch.arange(self.patch_feature_dim)
            pivots = projection.abs().argmax(0)
            signs = projection[pivots, columns].sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            projection = (projection * signs).float()
        self.register_buffer("patch_projection", projection)
        self.register_buffer("patch_projection_mean", projection_mean)
        self.register_buffer(
            "patch_projection_fitted",
            torch.tensor(
                self.patch_projection_kind != "task1_pca", dtype=torch.bool
            ),
        )
        self.register_buffer(
            "patch_projection_explained_variance",
            torch.tensor(float("nan"), dtype=torch.float32),
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

    @property
    def requires_projection_fit(self) -> bool:
        return (
            self.patch_projection_kind == "task1_pca"
            and not bool(self.patch_projection_fitted.item())
        )

    def fit_patch_projection(
        self, patch_features: torch.Tensor
    ) -> dict[str, float | int | str]:
        """Fit the optimal linear-autoencoder subspace, then freeze it."""
        if self.patch_projection_kind != "task1_pca":
            raise RuntimeError("This encoder does not use a learned patch projection")
        if not self.requires_projection_fit:
            raise RuntimeError("The DINOv3 patch projection is already fitted")
        if patch_features.shape[-1] != self.token_features:
            raise ValueError(
                "Patch calibration features must end in the DINOv3 channel axis"
            )
        samples = patch_features.detach().reshape(-1, self.token_features)
        if samples.shape[0] < self.patch_feature_dim:
            raise ValueError(
                "Patch projection calibration needs at least as many samples as outputs"
            )

        # A linear autoencoder's optimal subspace is PCA. Solve it once on CPU
        # so the target cannot collapse or drift with later RSSM gradients.
        samples = samples.to(device="cpu", dtype=torch.float64)
        mean = samples.mean(0)
        centered = samples - mean
        covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        nonnegative_values = eigenvalues.clamp_min(0)
        total_variance = nonnegative_values.sum()
        if not torch.isfinite(total_variance) or total_variance <= 0:
            raise ValueError("Patch projection calibration features have no variance")
        basis = eigenvectors[:, -self.patch_feature_dim :].flip(1)

        # Canonical signs remove the arbitrary sign choice of each eigenvector.
        columns = torch.arange(self.patch_feature_dim)
        pivots = basis.abs().argmax(0)
        signs = basis[pivots, columns].sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs
        captured_variance = (
            nonnegative_values[-self.patch_feature_dim :].sum() / total_variance
        )

        self.patch_projection.copy_(basis.to(self.patch_projection))
        self.patch_projection_mean.copy_(mean.to(self.patch_projection_mean))
        self.patch_projection_explained_variance.copy_(
            captured_variance.to(self.patch_projection_explained_variance)
        )
        self.patch_projection_fitted.fill_(True)
        return {
            "kind": self.patch_projection_kind,
            "calibration_patch_samples": samples.shape[0],
            "input_features": self.token_features,
            "output_features": self.patch_feature_dim,
            "explained_variance_ratio": float(captured_variance),
        }

    def project_patch_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.shape[-1] != self.token_features:
            raise ValueError("DINOv3 patch features have an unexpected channel width")
        if self.patch_projection_kind == "none":
            return patch_features
        if self.patch_projection_kind == "fixed_orthogonal":
            return patch_features @ self.patch_projection
        if self.requires_projection_fit:
            raise RuntimeError(
                "Fit the Task-1 DINOv3 patch projection before encoding observations"
            )
        return (patch_features - self.patch_projection_mean) @ self.patch_projection

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

    def _pooled_patch_features(
        self, last_hidden_state: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        patch_start = 1 + self.num_register_tokens
        patch_tokens = last_hidden_state[:, patch_start:]
        expected_patches = self.patch_grid_size**2
        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "DINOv3 returned an unexpected number of patch tokens: "
                f"expected {expected_patches}, got {patch_tokens.shape[1]}"
            )
        patch_grid = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            self.token_features,
            self.patch_grid_size,
            self.patch_grid_size,
        )
        pooled = F.adaptive_avg_pool2d(
            patch_grid,
            (self.patch_pool_size, self.patch_pool_size),
        )
        return pooled.permute(0, 2, 3, 1)

    def _extract_patch_chunk(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=self._prepare(images))
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError(
                "DINOv3 patch-grid mode requires last_hidden_state tokens"
            )
        return self._pooled_patch_features(last_hidden_state, images.shape[0])

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return unprojected pooled patch grids for Task-1 calibration."""
        if self.feature_mode != "patch_grid":
            raise RuntimeError("Raw patch features require patch-grid mode")
        chunks = []
        with torch.no_grad():
            for start in range(0, images.shape[0], self.max_batch_size):
                chunks.append(
                    self._extract_patch_chunk(
                        images[start : start + self.max_batch_size]
                    )
                )
        if not chunks:
            return images.new_empty(
                (0, self.patch_pool_size, self.patch_pool_size, self.token_features),
                dtype=torch.float32,
            )
        return torch.cat(chunks, dim=0).float()

    def _encode_chunk(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=self._prepare(images))
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if self.feature_mode == "patch_grid":
            if last_hidden_state is None:
                raise RuntimeError(
                    "DINOv3 patch-grid mode requires last_hidden_state tokens"
                )
            spatial_features = self._pooled_patch_features(
                last_hidden_state, images.shape[0]
            )
            features = self.project_patch_features(spatial_features).reshape(
                images.shape[0], -1
            )
        elif last_hidden_state is not None:
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
                chunks.append(
                    self._encode_chunk(images[start : start + self.max_batch_size])
                )
        if not chunks:
            return images.new_empty((0, self.output_size), dtype=torch.float32)
        return torch.cat(chunks, dim=0)
