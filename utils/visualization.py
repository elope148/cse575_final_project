"""
Visualization utilities for traffic flow prediction.

Includes:
- Prediction vs ground truth heatmaps
- Spatial error maps
- Training curves
- Saliency / occlusion analysis
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch


def setup_plot_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def plot_flow_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    title_prefix: str = "",
):
    """
    Plot predicted vs actual inflow/outflow grids side by side.

    Args:
        pred: (N, 2, H, W) or (2, H, W)
        target: (N, 2, H, W) or (2, H, W)
        sample_idx: which sample to visualize (if batched)
    """
    if pred.ndim == 4:
        pred = pred[sample_idx]
        target = target[sample_idx]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    channel_names = ["Inflow", "Outflow"]

    for i, name in enumerate(channel_names):
        vmin = min(target[i].min(), pred[i].min())
        vmax = max(target[i].max(), pred[i].max())

        # Ground truth
        im = axes[i, 0].imshow(target[i], cmap="YlOrRd", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{name} - Ground Truth")
        plt.colorbar(im, ax=axes[i, 0], fraction=0.046)

        # Prediction
        im = axes[i, 1].imshow(pred[i], cmap="YlOrRd", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{name} - Predicted")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

        # Error
        error = np.abs(pred[i] - target[i])
        im = axes[i, 2].imshow(error, cmap="Reds")
        axes[i, 2].set_title(f"{name} - Absolute Error")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    plt.suptitle(f"{title_prefix}Prediction vs Ground Truth", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_spatial_error_heatmap(
    error_map: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot spatial error distribution as a heatmap.

    Args:
        error_map: (2, H, W) average error per grid cell
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    channel_names = ["Inflow MAE", "Outflow MAE"]

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        im = ax.imshow(error_map[i], cmap="hot", interpolation="nearest")
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Spatial Error Distribution", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    n_plots = 1 + (1 if val_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss", alpha=0.8)
    axes[0].plot(val_losses, label="Val Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics
    if val_metrics:
        for name, values in val_metrics.items():
            axes[1].plot(values, label=name.upper(), alpha=0.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_horizon_comparison(
    results: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "mae",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing models across prediction horizons.

    Args:
        results: {model_name: {horizon: {metric: value}}}
        metric: which metric to plot
    """
    models = list(results.keys())
    horizons = sorted(list(results[models[0]].keys()))

    x = np.arange(len(horizons))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        values = [results[model][h][metric] for h in horizons]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)

    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Model and Horizon")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{h} steps" for h in horizons])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


@torch.no_grad()
def occlusion_saliency(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    patch_size: int = 4,
) -> np.ndarray:
    """
    Occlusion-based saliency analysis.

    Measures how much the prediction error increases when each spatial
    patch is occluded (set to zero) in the most recent input frame.

    Args:
        model: trained model
        sample: single sample dict with 'x', optionally 'ext'
        device: compute device
        patch_size: size of occlusion patch

    Returns:
        saliency_map: (H, W) importance scores
    """
    model.eval()

    # Add batch dimension if needed
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device) if v.dim() < 5 else v.to(device)

    # Baseline prediction
    base_pred = model(batch).cpu().numpy()[0]  # (2, H, W)

    x = batch["x"].clone()  # (1, T, 2, H, W)
    _, _, _, H, W = x.shape
    saliency = np.zeros((H, W))

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Occlude patch in last time step
            x_occ = x.clone()
            i_end = min(i + patch_size, H)
            j_end = min(j + patch_size, W)
            x_occ[0, -1, :, i:i_end, j:j_end] = 0

            batch_occ = {**batch, "x": x_occ}
            occ_pred = model(batch_occ).cpu().numpy()[0]

            # Increase in error = importance
            diff = np.mean(np.abs(occ_pred - base_pred))
            saliency[i:i_end, j:j_end] = diff

    return saliency


def plot_saliency(
    saliency_map: np.ndarray,
    flow_map: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Plot saliency map, optionally overlaid on the flow data."""
    fig, axes = plt.subplots(1, 2 if flow_map is not None else 1, figsize=(12, 5))
    if flow_map is None:
        axes = [axes]

    im = axes[0].imshow(saliency_map, cmap="hot")
    axes[0].set_title("Occlusion Saliency Map")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    if flow_map is not None:
        # Show total flow (inflow + outflow) with saliency overlay
        total_flow = flow_map.sum(axis=0) if flow_map.ndim == 3 else flow_map
        axes[1].imshow(total_flow, cmap="YlOrRd", alpha=0.7)
        axes[1].imshow(saliency_map, cmap="hot", alpha=0.4)
        axes[1].set_title("Saliency Overlaid on Traffic Flow")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
