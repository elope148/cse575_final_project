"""
Model registry and factory.
"""

from typing import Dict, Optional
import torch.nn as nn

from .cnn_baseline import CNNBaseline
from .convlstm import ConvLSTMModel
from .dino_enhanced import DINOEnhancedModel, DINOLiteModel


MODEL_REGISTRY = {
    "cnn": CNNBaseline,
    "convlstm": ConvLSTMModel,
    "dino": DINOEnhancedModel,
    "dino_lite": DINOLiteModel,
}


def build_model(
    model_name: str,
    seq_len: int = 6,
    in_channels: int = 2,
    ext_dim: Optional[int] = None,
    grid_size: int = 32,
    config: Optional[Dict] = None,
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: one of 'cnn', 'convlstm', 'dino', 'dino_lite'
        seq_len: number of input time steps
        in_channels: number of flow channels (2 for inflow/outflow)
        ext_dim: dimension of external features (None to disable)
        grid_size: spatial grid dimension
        config: model-specific config dict from config.yaml

    Returns:
        nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = config or {}

    if model_name == "cnn":
        return CNNBaseline(
            seq_len=seq_len,
            in_channels=in_channels,
            conv_channels=config.get("channels", [64, 128, 64]),
            kernel_size=config.get("kernel_size", 3),
            dropout=config.get("dropout", 0.2),
            use_residual=config.get("use_residual", True),
            ext_dim=ext_dim,
            grid_size=grid_size,
        )

    elif model_name == "convlstm":
        return ConvLSTMModel(
            in_channels=in_channels,
            hidden_dims=config.get("hidden_dims", [64, 64]),
            kernel_size=config.get("kernel_size", 3),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            ext_dim=ext_dim,
            grid_size=grid_size,
        )

    elif model_name == "dino":
        return DINOEnhancedModel(
            in_channels=in_channels,
            dino_backbone=config.get("backbone", "dinov2_vits14"),
            freeze_backbone=config.get("freeze_backbone", True),
            projection_dim=config.get("projection_dim", 128),
            temporal_hidden=config.get("temporal_hidden", 64),
            temporal_layers=config.get("temporal_layers", 2),
            dropout=config.get("dropout", 0.2),
            ext_dim=ext_dim,
            grid_size=grid_size,
        )

    elif model_name == "dino_lite":
        return DINOLiteModel(
            seq_len=seq_len,
            in_channels=in_channels,
            dino_backbone=config.get("backbone", "dinov2_vits14"),
            freeze_backbone=config.get("freeze_backbone", True),
            projection_dim=config.get("projection_dim", 128),
            conv_channels=config.get("conv_channels", 64),
            ext_dim=ext_dim,
            grid_size=grid_size,
        )
