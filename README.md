# Spatio-Temporal Deep Learning for Traffic Flow Prediction

CSE 575: Statistical Machine Learning — Group Project

## Overview

Grid-based traffic flow prediction using deep learning with external factor integration.
Models predict future inflow/outflow matrices across city grid regions at 5, 15, and 30-minute horizons.

## Architecture

| Model | Description |
|-------|-------------|
| **CNN Baseline** | Stacked recent frames → Conv2D layers (inspired by DeepST) |
| **ConvLSTM** | Sequence of spatial frames → ConvLSTM cells for joint spatio-temporal modeling |
| **DINO-Enhanced** | Pretrained DINOv2 ViT extracts spatial features → fused with temporal ConvLSTM |

## Project Structure

```
traffic_flow_prediction/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   ├── download_data.py      # Download TaxiBJ dataset
│   ├── dataset.py            # PyTorch Dataset classes
│   └── preprocessing.py      # Normalization, sliding windows, external features
├── models/
│   ├── __init__.py
│   ├── cnn_baseline.py       # CNN baseline (DeepST-inspired)
│   ├── convlstm.py           # ConvLSTM model
│   ├── convlstm_cell.py      # ConvLSTM cell implementation
│   └── dino_enhanced.py      # DINO feature extractor + temporal model
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # MAE, RMSE, per-region error
│   ├── visualization.py      # Plot predictions, error heatmaps, saliency
│   └── experiment.py         # Experiment tracker / logger
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation + spatial analysis
│   └── ablation.py           # Ablation study runner
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Download data
python data/download_data.py

# 2. Train CNN baseline
python scripts/train.py --model cnn --horizon 5

# 3. Train ConvLSTM
python scripts/train.py --model convlstm --horizon 5

# 4. Train DINO-enhanced model
python scripts/train.py --model dino --horizon 5

# 5. Run ablation study
python scripts/ablation.py

# 6. Evaluate and generate plots
python scripts/evaluate.py --model convlstm --checkpoint checkpoints/best_convlstm.pt
```

## Datasets

- **TaxiBJ**: Beijing taxi GPS trajectory data aggregated into 32×32 grid inflow/outflow matrices
  at 30-minute intervals. Includes meteorological data and holiday metadata.

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- Per-horizon breakdown (5, 15, 30 min)
- Peak vs off-peak analysis
- Spatial saliency / occlusion analysis
