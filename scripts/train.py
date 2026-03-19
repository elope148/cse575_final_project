"""
Main training script for traffic flow prediction models.

Usage:
    python scripts/train.py --model cnn --horizon 1
    python scripts/train.py --model convlstm --horizon 1 --no-weather
    python scripts/train.py --model dino --horizon 1 --epochs 50
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import prepare_data
from data.dataset import create_dataloaders
from models import build_model
from utils.metrics import compute_metrics, evaluate_model
from utils.experiment import ExperimentTracker
from utils.visualization import plot_training_curves


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config_device: str = "auto") -> torch.device:
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config_device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch["y"])
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> tuple:
    """Validate and return (loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch)
        loss = criterion(pred, batch["y"])
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch["y"].cpu().numpy())

    avg_loss = total_loss / n_batches
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(preds, targets, scaler)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train traffic flow prediction model")
    parser.add_argument("--model", type=str, default="convlstm",
                        choices=["cnn", "convlstm", "dino", "dino_lite"])
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction horizon in time steps")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-time", action="store_true", help="Disable time features")
    parser.add_argument("--no-weather", action="store_true", help="Disable weather features")
    parser.add_argument("--no-holidays", action="store_true", help="Disable holiday features")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    seed = args.seed or cfg.get("seed", 42)
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]

    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    # ---- Data ----
    use_time = not args.no_time and cfg["data"].get("use_time_features", True)
    use_weather = not args.no_weather and cfg["data"].get("use_weather", True)
    use_holidays = not args.no_holidays and cfg["data"].get("use_holidays", True)

    data = prepare_data(
        data_dir=cfg["data"]["data_dir"],
        seq_len=cfg["data"]["seq_len"],
        horizon=args.horizon,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        use_time=use_time,
        use_weather=use_weather,
        use_holidays=use_holidays,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        data, batch_size=batch_size, num_workers=2,
    )

    # ---- Model ----
    ext_dim = data["ext_train"].shape[-1] if data["ext_train"] is not None else None
    model_cfg = cfg["model"].get(args.model, {})

    model = build_model(
        model_name=args.model,
        seq_len=cfg["data"]["seq_len"],
        in_channels=cfg["data"]["channels"],
        ext_dim=ext_dim,
        grid_size=cfg["data"]["grid_size"],
        config=model_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model} | Trainable params: {n_params:,}")

    # ---- Training setup ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    scheduler_name = cfg["training"].get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=cfg["training"]["scheduler_params"].get("eta_min", 1e-6),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )
    else:
        scheduler = None

    grad_clip = cfg["training"].get("gradient_clip", 0.0)
    patience = cfg["training"].get("early_stopping_patience", 15)

    # ---- Experiment tracking ----
    exp_name = f"{args.model}_h{args.horizon}"
    if not use_time:
        exp_name += "_notime"
    if not use_weather:
        exp_name += "_noweather"
    if not use_holidays:
        exp_name += "_noholiday"

    tracker = ExperimentTracker(
        experiment_name=exp_name,
        log_dir=cfg["output"]["log_dir"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    # ---- Training loop ----
    print(f"\nStarting training for {epochs} epochs...")
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip,
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, data["scaler"],
        )

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Track
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        tracker.log_epoch(epoch, train_loss, val_loss, val_metrics, current_lr)
        tracker.save_checkpoint(model, optimizer, epoch, val_loss, is_best)

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ---- Final evaluation on test set ----
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    best_ckpt = torch.load(tracker.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, device, data["scaler"])
    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Save results
    tracker.save_results({
        "test": test_metrics,
        "best_val_loss": best_val_loss,
        "model": args.model,
        "horizon": args.horizon,
        "features": {
            "time": use_time,
            "weather": use_weather,
            "holidays": use_holidays,
        },
    })

    # Plot training curves
    plot_training_curves(
        tracker.history["train_loss"],
        tracker.history["val_loss"],
        tracker.history["val_metrics"],
        save_path=str(tracker.log_dir / "training_curves.png"),
    )

    tracker.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()