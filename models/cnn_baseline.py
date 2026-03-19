"""
CNN Baseline for traffic flow prediction (DeepST-inspired).

Stacks recent time steps as input channels and applies Conv2D layers
to jointly capture short-term temporal and spatial patterns.
Optionally fuses external features via a learned embedding.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class CNNBaseline(nn.Module):
    """
    CNN-based traffic flow prediction model.

    Input: (B, seq_len, 2, H, W) - sequence of inflow/outflow grids
    Output: (B, 2, H, W) - predicted inflow/outflow grid

    The seq_len frames are stacked channel-wise: (B, seq_len*2, H, W)
    then processed through conv layers.
    """

    def __init__(
        self,
        seq_len: int = 6,
        in_channels: int = 2,
        conv_channels: list = [64, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_residual: bool = True,
        ext_dim: Optional[int] = None,
        grid_size: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.ext_dim = ext_dim

        total_in = seq_len * in_channels
        padding = kernel_size // 2

        # Build encoder
        layers = []
        prev_ch = total_in
        for ch in conv_channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            if use_residual:
                layers.append(ResidualBlock(ch, kernel_size))
            layers.append(nn.Dropout2d(dropout))
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)

        # External feature fusion
        if ext_dim is not None and ext_dim > 0:
            # Project external features to a spatial map and add to conv features
            self.ext_fc = nn.Sequential(
                nn.Linear(ext_dim * seq_len, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, conv_channels[-1] * grid_size * grid_size),
            )
            self.ext_gate = nn.Sequential(
                nn.Conv2d(conv_channels[-1] * 2, conv_channels[-1], 1),
                nn.Sigmoid(),
            )
            fuse_in = conv_channels[-1]
        else:
            self.ext_fc = None
            fuse_in = conv_channels[-1]

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(fuse_in, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] since data is min-max normalized
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]  # (B, seq_len, 2, H, W)
        B, S, C, H, W = x.shape

        # Stack time steps as channels: (B, seq_len*2, H, W)
        x = x.reshape(B, S * C, H, W)

        # Encode
        features = self.encoder(x)  # (B, conv_channels[-1], H, W)

        # Fuse external features if available
        if self.ext_fc is not None and "ext" in batch:
            ext = batch["ext"]  # (B, seq_len, D)
            ext_flat = ext.reshape(B, -1)
            ext_spatial = self.ext_fc(ext_flat)  # (B, C*H*W)
            ext_spatial = ext_spatial.reshape(B, features.shape[1], H, W)

            # Gated fusion
            gate = self.ext_gate(torch.cat([features, ext_spatial], dim=1))
            features = features * gate + ext_spatial * (1 - gate)

        # Predict
        out = self.output_head(features)  # (B, 2, H, W)
        return out
