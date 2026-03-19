"""
DINO-Enhanced traffic flow prediction model.

Uses a pretrained DINOv2 Vision Transformer to extract rich spatial features
from traffic grid maps, then combines with temporal ConvLSTM modeling.

The key insight: DINO learns powerful spatial representations from self-supervised
pretraining on natural images. These features capture structural patterns
(clusters, corridors, hotspots) that may be useful for traffic grids even though
the domain is different. We treat the 32x32 grid as an "image" and extract
patch-level features, then project them back to spatial dimensions for temporal
modeling.

Architecture:
    1. Resize grid (32x32 → 224x224) and replicate channels to 3-channel "image"
    2. Extract DINOv2 patch tokens (spatial features)
    3. Project patch features back to grid spatial dims
    4. Feed projected features through ConvLSTM for temporal modeling
    5. Decode to inflow/outflow prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .convlstm_cell import ConvLSTM


class DINOFeatureExtractor(nn.Module):
    """
    Extract spatial features from traffic grids using pretrained DINOv2.

    Uses timm library (Python 3.9 compatible) instead of torch.hub.

    DINOv2 ViT-S/14 processes 224x224 images and produces patch tokens
    on a 16x16 grid (224/14 = 16). We project these to our target
    spatial resolution.
    """

    # Map our config names to timm model names
    BACKBONE_MAP = {
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    }

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        freeze: bool = True,
        projection_dim: int = 128,
        target_size: int = 32,
    ):
        super().__init__()
        self.target_size = target_size
        self.projection_dim = projection_dim

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for the DINO model. Install with: "
                "pip install timm"
            )

        # Load pretrained DINOv2 via timm
        timm_name = self.BACKBONE_MAP.get(backbone, backbone)
        self.dino = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=0,  # remove classification head
        )
        self.embed_dim = self.dino.embed_dim
        self.patch_size = 14  # DINOv2 uses 14x14 patches

        # Get the model's expected input size (518 for DINOv2 in timm)
        if hasattr(self.dino, 'patch_embed') and hasattr(self.dino.patch_embed, 'img_size'):
            self.img_size = self.dino.patch_embed.img_size[0]
        else:
            self.img_size = 518  # DINOv2 default in timm

        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        # Spatial size of patch token grid: img_size / patch_size
        self.patch_grid = self.img_size // self.patch_size  # 37 for 518/14

        # Project DINO features to desired dimension
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 32, 32) traffic flow grid (inflow/outflow channels)

        Returns:
            features: (B, projection_dim, target_size, target_size)
        """
        B = x.shape[0]

        # Prepare for DINO: resize to model's expected size and convert to 3-channel
        x_resized = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        ch_r = x_resized[:, 0:1]  # inflow
        ch_g = x_resized[:, 1:2]  # outflow
        ch_b = x_resized.mean(dim=1, keepdim=True)  # average
        x_rgb = torch.cat([ch_r, ch_g, ch_b], dim=1)  # (B, 3, 224, 224)

        # Extract patch features via timm's forward_features
        with torch.no_grad() if not any(p.requires_grad for p in self.dino.parameters()) else torch.enable_grad():
            # timm forward_features returns (B, num_tokens, embed_dim)
            # where num_tokens = 1 (CLS) + 256 (patches for 224/14=16, 16*16=256)
            features = self.dino.forward_features(x_rgb)

            # Remove CLS token (first token), keep only patch tokens
            if features.dim() == 3 and features.shape[1] == 1 + self.patch_grid ** 2:
                patch_tokens = features[:, 1:, :]  # (B, 256, embed_dim)
            else:
                # Some timm models may already strip CLS
                patch_tokens = features  # (B, n_patches, embed_dim)

        # Project
        projected = self.projection(patch_tokens)  # (B, 256, projection_dim)

        # Reshape to spatial grid: (B, projection_dim, 16, 16)
        spatial = projected.transpose(1, 2).reshape(
            B, self.projection_dim, self.patch_grid, self.patch_grid
        )

        # Upsample to target grid size (16 -> 32)
        if self.patch_grid != self.target_size:
            spatial = F.interpolate(
                spatial, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False,
            )

        return spatial  # (B, projection_dim, 32, 32)


class DINOEnhancedModel(nn.Module):
    """
    DINO-Enhanced spatio-temporal traffic flow prediction.

    Combines pretrained DINO spatial features with ConvLSTM temporal modeling.

    For each time step:
    1. Extract DINO features from the traffic grid
    2. Concatenate with raw flow data (skip connection)
    3. Feed through ConvLSTM

    Then decode final hidden state to prediction.
    """

    def __init__(
        self,
        in_channels: int = 2,
        dino_backbone: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        projection_dim: int = 128,
        temporal_hidden: int = 64,
        temporal_layers: int = 2,
        dropout: float = 0.2,
        ext_dim: Optional[int] = None,
        grid_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels

        # DINO feature extractor
        self.dino_extractor = DINOFeatureExtractor(
            backbone=dino_backbone,
            freeze=freeze_backbone,
            projection_dim=projection_dim,
            target_size=grid_size,
        )

        # Combine DINO features with raw flow data
        combined_dim = projection_dim + in_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(combined_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Temporal ConvLSTM
        hidden_dims = [temporal_hidden] * temporal_layers
        self.convlstm = ConvLSTM(
            input_dim=64,
            hidden_dims=hidden_dims,
            kernel_size=3,
            dropout=dropout,
            return_all_layers=False,
        )

        # External feature conditioning
        if ext_dim is not None and ext_dim > 0:
            self.ext_encoder = nn.Sequential(
                nn.Linear(ext_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, temporal_hidden),
            )
            self.film_gamma = nn.Linear(temporal_hidden, temporal_hidden)
            self.film_beta = nn.Linear(temporal_hidden, temporal_hidden)
        else:
            self.ext_encoder = None

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(temporal_hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]  # (B, T, 2, H, W)
        B, T, C, H, W = x.shape

        # Extract DINO features for each time step
        fused_seq = []
        for t in range(T):
            frame = x[:, t]  # (B, 2, H, W)

            # DINO spatial features
            dino_feats = self.dino_extractor(frame)  # (B, proj_dim, H, W)

            # Concatenate with raw flow and fuse
            combined = torch.cat([dino_feats, frame], dim=1)  # (B, proj_dim+2, H, W)
            fused = self.fusion_conv(combined)  # (B, 64, H, W)
            fused_seq.append(fused)

        fused_seq = torch.stack(fused_seq, dim=1)  # (B, T, 64, H, W)

        # Temporal modeling
        output, last_states = self.convlstm(fused_seq)
        h_last = last_states[-1][0]  # (B, temporal_hidden, H, W)

        # External features (FiLM conditioning)
        if self.ext_encoder is not None and "ext" in batch:
            ext = batch["ext"][:, -1, :]
            ext_emb = self.ext_encoder(ext)
            gamma = self.film_gamma(ext_emb).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_beta(ext_emb).unsqueeze(-1).unsqueeze(-1)
            h_last = gamma * h_last + beta

        # Decode
        out = self.decoder(h_last)  # (B, 2, H, W)
        return out


class DINOLiteModel(nn.Module):
    """
    Lightweight alternative that uses DINO features without per-frame extraction.

    Instead of running DINO on every time step (expensive), this model:
    1. Runs DINO only on the most recent frame
    2. Uses a simple CNN to process the temporal sequence
    3. Fuses DINO features with temporal features

    Much faster training while still benefiting from DINO representations.
    """

    def __init__(
        self,
        seq_len: int = 6,
        in_channels: int = 2,
        dino_backbone: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        projection_dim: int = 128,
        conv_channels: int = 64,
        ext_dim: Optional[int] = None,
        grid_size: int = 32,
    ):
        super().__init__()

        # DINO for spatial features (last frame only)
        self.dino_extractor = DINOFeatureExtractor(
            backbone=dino_backbone,
            freeze=freeze_backbone,
            projection_dim=projection_dim,
            target_size=grid_size,
        )

        # Temporal CNN branch (processes all frames)
        self.temporal_cnn = nn.Sequential(
            nn.Conv2d(seq_len * in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion
        fusion_in = projection_dim + conv_channels
        if ext_dim is not None and ext_dim > 0:
            self.ext_fc = nn.Sequential(
                nn.Linear(ext_dim * seq_len, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
            )
            fusion_in += 32  # ext features broadcast spatially
        else:
            self.ext_fc = None

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]  # (B, T, 2, H, W)
        B, T, C, H, W = x.shape

        # DINO features from last frame
        dino_feats = self.dino_extractor(x[:, -1])  # (B, proj_dim, H, W)

        # Temporal CNN on stacked frames
        x_stacked = x.reshape(B, T * C, H, W)
        temporal_feats = self.temporal_cnn(x_stacked)  # (B, conv_ch, H, W)

        # Fusion
        parts = [dino_feats, temporal_feats]

        if self.ext_fc is not None and "ext" in batch:
            ext = batch["ext"].reshape(B, -1)
            ext_feats = self.ext_fc(ext)  # (B, 32)
            ext_spatial = ext_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            parts.append(ext_spatial)

        fused = torch.cat(parts, dim=1)
        out = self.fusion(fused)
        return out