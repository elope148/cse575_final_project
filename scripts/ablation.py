"""
Ablation study runner.

Systematically trains models with different feature configurations
to measure the contribution of each external feature type.

Configurations tested:
1. Flow only (no external features)
2. Flow + time features
3. Flow + time + weather
4. Flow + time + weather + holidays (full model)

Usage:
    python scripts/ablation.py --model convlstm --horizon 1
    python scripts/ablation.py --model cnn --horizons 1 3 6
"""

import argparse
import json
import sys
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ABLATION_CONFIGS = [
    {
        "name": "flow_only",
        "flags": ["--no-time", "--no-weather", "--no-holidays"],
        "description": "Traffic flow data only",
    },
    {
        "name": "flow_time",
        "flags": ["--no-weather", "--no-holidays"],
        "description": "Flow + temporal encodings",
    },
    {
        "name": "flow_time_weather",
        "flags": ["--no-holidays"],
        "description": "Flow + temporal + weather",
    },
    {
        "name": "full",
        "flags": [],
        "description": "Full model (flow + temporal + weather + holidays)",
    },
]


def run_training(model: str, horizon: int, extra_flags: list, epochs: int = 50) -> dict:
    """
    Run a single training configuration and return results.

    This calls train.py as a subprocess so each run is isolated.
    Returns the results dict from the experiment tracker.
    """
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model,
        "--horizon", str(horizon),
        "--epochs", str(epochs),
    ] + extra_flags

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Training failed with return code {result.returncode}")
        return None

    # Find the most recent results file
    log_dir = Path("logs")
    if log_dir.exists():
        results_files = sorted(log_dir.rglob("results.json"), key=lambda p: p.stat().st_mtime)
        if results_files:
            with open(results_files[-1]) as f:
                return json.load(f)

    return None


def run_ablation_study(
    model: str,
    horizons: list,
    epochs: int = 50,
    output_dir: str = "results/ablation",
):
    """Run full ablation study across horizons and feature configs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for horizon in horizons:
        print(f"\n{'#'*60}")
        print(f"# HORIZON: {horizon} time steps")
        print(f"{'#'*60}")

        horizon_results = {}

        for config in ABLATION_CONFIGS:
            print(f"\n>> Configuration: {config['name']} - {config['description']}")

            result = run_training(model, horizon, config["flags"], epochs)

            if result is not None:
                horizon_results[config["name"]] = {
                    "description": config["description"],
                    "test_metrics": result.get("final_metrics", {}).get("test", {}),
                }
                print(f"   Result: {horizon_results[config['name']]['test_metrics']}")
            else:
                horizon_results[config["name"]] = {"description": config["description"], "error": True}

        all_results[f"horizon_{horizon}"] = horizon_results

    # Save summary
    summary_path = output_path / f"ablation_{model}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results saved to {summary_path}")

    # Print summary table
    print_ablation_summary(all_results, model, horizons)

    return all_results


def print_ablation_summary(results: dict, model: str, horizons: list):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY SUMMARY — Model: {model}")
    print(f"{'='*70}")

    header = f"{'Configuration':<25}"
    for h in horizons:
        header += f" | {'H=' + str(h) + ' MAE':>10} {'RMSE':>10}"
    print(header)
    print("-" * len(header))

    for config in ABLATION_CONFIGS:
        row = f"{config['name']:<25}"
        for h in horizons:
            key = f"horizon_{h}"
            if key in results and config["name"] in results[key]:
                r = results[key][config["name"]]
                if "error" not in r:
                    test = r.get("test_metrics", {})
                    mae_val = test.get("mae", float("nan"))
                    rmse_val = test.get("rmse", float("nan"))
                    row += f" | {mae_val:>10.4f} {rmse_val:>10.4f}"
                else:
                    row += f" | {'ERROR':>10} {'ERROR':>10}"
            else:
                row += f" | {'N/A':>10} {'N/A':>10}"
        print(row)

    print(f"{'='*70}")


def run_model_comparison(
    models: list,
    horizon: int = 1,
    epochs: int = 50,
    output_dir: str = "results/comparison",
):
    """Compare different model architectures with full features."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*60}")

        result = run_training(model_name, horizon, [], epochs)
        if result is not None:
            results[model_name] = result.get("final_metrics", {}).get("test", {})
        else:
            results[model_name] = {"error": True}

    # Save
    with open(output_path / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<15} | {'MAE':>10} | {'RMSE':>10}")
    print("-" * 42)
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"{name:<15} | {metrics.get('mae', 0):>10.4f} | {metrics.get('rmse', 0):>10.4f}")
        else:
            print(f"{name:<15} | {'ERROR':>10} | {'ERROR':>10}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--model", type=str, default="convlstm",
                        choices=["cnn", "convlstm", "dino", "dino_lite"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--compare-models", action="store_true",
                        help="Also run model architecture comparison")
    args = parser.parse_args()

    # Feature ablation
    run_ablation_study(args.model, args.horizons, args.epochs)

    # Optional model comparison
    if args.compare_models:
        models_to_compare = ["cnn", "convlstm"]
        # Only include DINO if requested model is DINO variant
        if args.model in ["dino", "dino_lite"]:
            models_to_compare.append(args.model)
        run_model_comparison(models_to_compare, horizon=args.horizons[0], epochs=args.epochs)


if __name__ == "__main__":
    main()
