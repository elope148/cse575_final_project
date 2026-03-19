"""
Evaluation script with comprehensive analysis.

Generates:
- Overall MAE/RMSE metrics
- Per-channel (inflow/outflow) metrics
- Spatial error heatmaps
- Prediction visualizations
- Occlusion saliency maps
- Peak vs off-peak analysis

Usage:
    python scripts/evaluate.py --model convlstm --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import prepare_data
from data.dataset import create_dataloaders
from models import build_model
from utils.metrics import (
    compute_metrics,
    per_channel_metrics,
    spatial_error_map,
)
from utils.visualization import (
    plot_flow_comparison,
    plot_spatial_error_heatmap,
    occlusion_saliency,
    plot_saliency,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic flow model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "convlstm", "dino", "dino_lite"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-visualize", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--saliency", action="store_true",
                        help="Run occlusion saliency analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / f"{args.model}_h{args.horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = prepare_data(
        data_dir=cfg["data"]["data_dir"],
        seq_len=cfg["data"]["seq_len"],
        horizon=args.horizon,
    )
    _, _, test_loader = create_dataloaders(data, batch_size=32, num_workers=2)

    # Build and load model
    ext_dim = data["ext_train"].shape[-1] if data["ext_train"] is not None else None
    model_cfg = cfg["model"].get(args.model, {})
    model = build_model(
        args.model, cfg["data"]["seq_len"], cfg["data"]["channels"],
        ext_dim, cfg["data"]["grid_size"], model_cfg,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect predictions
    print("Running inference on test set...")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["y"].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    scaler = data["scaler"]

    # --- Overall metrics ---
    print("\n" + "=" * 50)
    print("OVERALL METRICS (original scale)")
    print("=" * 50)
    metrics = compute_metrics(preds, targets, scaler)
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # --- Per-channel metrics ---
    print("\nPER-CHANNEL METRICS:")
    ch_metrics = per_channel_metrics(preds, targets, scaler)
    for ch_name, ch_m in ch_metrics.items():
        print(f"  {ch_name}: MAE={ch_m['mae']:.4f}, RMSE={ch_m['rmse']:.4f}")

    # --- Spatial error map ---
    print("\nGenerating spatial error heatmap...")
    err_map = spatial_error_map(preds, targets, scaler)
    plot_spatial_error_heatmap(err_map, save_path=str(output_dir / "spatial_error.png"))

    # --- Sample visualizations ---
    print(f"Generating {args.n_visualize} sample visualizations...")
    preds_orig = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)

    indices = np.linspace(0, len(preds) - 1, args.n_visualize, dtype=int)
    for idx in indices:
        plot_flow_comparison(
            preds_orig, targets_orig, sample_idx=idx,
            save_path=str(output_dir / f"comparison_sample_{idx}.png"),
            title_prefix=f"Sample {idx} | ",
        )

    # --- Saliency analysis ---
    if args.saliency:
        print("\nRunning occlusion saliency analysis (this may take a while)...")
        test_ds = test_loader.dataset
        for i in range(min(3, len(test_ds))):
            sample = test_ds[i]
            sal_map = occlusion_saliency(model, sample, device, patch_size=4)
            flow = targets_orig[i] if i < len(targets_orig) else None
            plot_saliency(
                sal_map, flow_map=flow,
                save_path=str(output_dir / f"saliency_sample_{i}.png"),
            )

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
