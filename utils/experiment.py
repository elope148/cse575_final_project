"""
Experiment tracking and logging utilities.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


class ExperimentTracker:
    """
    Simple experiment tracker that logs metrics and saves checkpoints.
    Also supports TensorBoard logging.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        use_tensorboard: bool = True,
    ):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"

        self.log_dir = Path(log_dir) / self.run_name
        self.checkpoint_dir = Path(checkpoint_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "val_metrics": {}}
        self.best_val_loss = float("inf")
        self.start_time = time.time()

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, skipping.")

        print(f"Experiment: {self.run_name}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Checkpoints: {self.checkpoint_dir}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        lr: Optional[float] = None,
    ):
        """Log metrics for one epoch."""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        for k, v in val_metrics.items():
            if k not in self.history["val_metrics"]:
                self.history["val_metrics"][k] = []
            self.history["val_metrics"][k].append(v)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)
            if lr is not None:
                self.writer.add_scalar("LR", lr, epoch)

        # Console
        elapsed = time.time() - self.start_time
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        print(
            f"  Epoch {epoch:3d} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"{metrics_str} | "
            f"Time: {elapsed:.0f}s"
        )

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        extra: Optional[Dict] = None,
    ):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }
        if extra:
            state.update(extra)

        # Save latest
        path = self.checkpoint_dir / "latest.pt"
        torch.save(state, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            print(f"  ** New best model saved (val_loss: {val_loss:.4f})")

    def save_results(self, final_metrics: Dict[str, Any]):
        """Save final results as JSON."""
        def _to_serializable(obj):
            """Recursively convert numpy/torch types to Python natives."""
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results = {
            "experiment": self.run_name,
            "final_metrics": _to_serializable(final_metrics),
            "history": _to_serializable(self.history),
            "total_time_seconds": time.time() - self.start_time,
        }

        path = self.log_dir / "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {path}")

    def close(self):
        if self.writer:
            self.writer.close()