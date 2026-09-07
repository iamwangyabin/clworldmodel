"""Tiny shared-head D-family model; migrated from the historical test fixture."""

import torch
from torch import nn
from wm import WorldModel

def retained_world_model(
        mechanism_parameterization: str = "adaptive_dense_width",
        *,
        shared_prediction_heads: bool = True,
    ) -> WorldModel:
        class WideEmbedder(nn.Module):
            output_size = 4096

            def __init__(self) -> None:
                super().__init__()
                self.offset = nn.Parameter(torch.zeros(self.output_size))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.offset.unsqueeze(0).expand(images.shape[0], -1)

        return WorldModel(
            3,
            (2, 3),
            4,
            8,
            cnn_depth=4,
            mlp_features=8,
            mlp_layers=2,
            observation_objective="reconstruction",
            num_task_experts=3,
            task_shared_prediction_heads=shared_prediction_heads,
            evolving_shared_core=True,
            task_projected_image_encoder=True,
            task_symmetric_image_projectors=True,
            task_projector_bottleneck_features=64,
            task_mechanism_bank=True,
            task_mechanism_reuse=True,
            task_mechanism_recurrent_width=8,
            task_mechanism_representation_width=8,
            task_mechanism_transition_width=8,
            task_mechanism_num_atoms=4,
            task_mechanism_parameterization=mechanism_parameterization,
            task_symmetric_mechanisms=True,
            image_embedder=WideEmbedder(),
        )
