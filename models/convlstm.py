"""
ConvLSTM-based traffic flow prediction model.

Processes a sequence of spatial traffic maps through ConvLSTM layers,
then decodes the final hidden state into a predicted flow map.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .convlstm_cell import ConvLSTM


class ConvLSTMModel(nn.Module):
    """
    ConvLSTM model for spatio-temporal traffic flow prediction.

    Input: (B, seq_len, 2, H, W) - sequence of inflow/outflow grids
    Output: (B, 2, H, W) - predicted inflow/outflow grid
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dims: List[int] = [64, 64],
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.2,
        ext_dim: Optional[int] = None,
        grid_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.ext_dim = ext_dim

        # Input projection: lift 2 channels to a richer representation
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM backbone
        self.convlstm = ConvLSTM(
            input_dim=32,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
            return_all_layers=False,
        )

        # External feature fusion
        decoder_in = hidden_dims[-1]
        if ext_dim is not None and ext_dim > 0:
            self.ext_encoder = nn.Sequential(
                nn.Linear(ext_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, hidden_dims[-1]),
            )
            # Combine last ext encoding with spatial features via FiLM conditioning
            self.film_gamma = nn.Linear(hidden_dims[-1], hidden_dims[-1])
            self.film_beta = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        else:
            self.ext_encoder = None

        # Decoder: map final hidden state to prediction
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]  # (B, T, 2, H, W)
        B, T, C, H, W = x.shape

        # Project each frame
        x_proj = self.input_proj(x.reshape(B * T, C, H, W))  # (B*T, 32, H, W)
        x_proj = x_proj.reshape(B, T, -1, H, W)              # (B, T, 32, H, W)

        # ConvLSTM
        output, last_states = self.convlstm(x_proj)  # output: (B, T, hidden[-1], H, W)

        # Use the last time step's hidden state
        h_last = last_states[-1][0]  # (B, hidden[-1], H, W)

        # External feature conditioning (FiLM)
        if self.ext_encoder is not None and "ext" in batch:
            ext = batch["ext"]  # (B, T, D)
            ext_last = ext[:, -1, :]  # Use last time step's external features
            ext_emb = self.ext_encoder(ext_last)  # (B, hidden[-1])

            gamma = self.film_gamma(ext_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = self.film_beta(ext_emb).unsqueeze(-1).unsqueeze(-1)
            h_last = gamma * h_last + beta

        # Decode
        out = self.decoder(h_last)  # (B, 2, H, W)
        return out
