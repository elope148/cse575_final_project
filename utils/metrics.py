"""
Evaluation metrics for traffic flow prediction.
"""

import numpy as np
import torch
from typing import Dict, Optional

from data.preprocessing import MinMaxScaler


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def mape(pred: np.ndarray, target: np.ndarray, threshold: float = 10.0) -> float:
    """
    Mean Absolute Percentage Error, computed only on cells where
    the target is above a threshold. Many grid cells have zero or
    near-zero traffic, which causes division explosions.
    """
    mask = np.abs(target) > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
) -> Dict[str, float]:
    """
    Compute all metrics. Optionally inverse-transform before computing.

    Args:
        pred: predicted values (normalized or original scale)
        target: ground truth (normalized or original scale)
        scaler: if provided, inverse transform both arrays first

    Returns:
        dict with mae, rmse, mape values
    """
    if scaler is not None:
        pred = scaler.inverse_transform(pred)
        target = scaler.inverse_transform(target)

    return {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "mape": mape(pred, target),
    }


def per_channel_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    channel_names: list = ["inflow", "outflow"],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each channel (inflow/outflow).

    Args:
        pred: (N, 2, H, W)
        target: (N, 2, H, W)
    """
    results = {}
    for i, name in enumerate(channel_names):
        p = pred[:, i]
        t = target[:, i]
        if scaler is not None:
            p = scaler.inverse_transform(p)
            t = scaler.inverse_transform(t)
        results[name] = {
            "mae": mae(p, t),
            "rmse": rmse(p, t),
        }
    return results


def spatial_error_map(
    pred: np.ndarray,
    target: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
) -> np.ndarray:
    """
    Compute per-grid-cell MAE averaged over all samples.

    Args:
        pred: (N, 2, H, W)
        target: (N, 2, H, W)

    Returns:
        error_map: (2, H, W) - average absolute error per cell
    """
    if scaler is not None:
        pred = scaler.inverse_transform(pred)
        target = scaler.inverse_transform(target)

    return np.mean(np.abs(pred - target), axis=0)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: Optional[MinMaxScaler] = None,
) -> Dict[str, float]:
    """
    Run model on a dataloader and compute metrics.

    Returns dict with mae, rmse, mape (on original scale if scaler provided).
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch["y"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return compute_metrics(preds, targets, scaler)