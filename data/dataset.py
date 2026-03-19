"""
PyTorch Dataset classes for traffic flow prediction.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficFlowDataset(Dataset):
    """
    PyTorch Dataset for traffic flow sequences.

    Each sample consists of:
    - x: (seq_len, 2, H, W) - sequence of inflow/outflow grids
    - y: (2, H, W) - target inflow/outflow grid
    - ext: (seq_len, D) - external features (optional)
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_ext: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.X_ext = torch.from_numpy(X_ext).float() if X_ext is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "x": self.X[idx],       # (seq_len, 2, H, W)
            "y": self.Y[idx],       # (2, H, W)
        }
        if self.X_ext is not None:
            sample["ext"] = self.X_ext[idx]  # (seq_len, D)
        return sample


def create_dataloaders(
    data: Dict,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from preprocessed data dict.

    Args:
        data: output from preprocessing.prepare_data()
        batch_size: batch size
        num_workers: number of worker threads

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = TrafficFlowDataset(data["X_train"], data["Y_train"], data["ext_train"])
    val_ds = TrafficFlowDataset(data["X_val"], data["Y_val"], data["ext_val"])
    test_ds = TrafficFlowDataset(data["X_test"], data["Y_test"], data["ext_test"])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
