"""
Data preprocessing for TaxiBJ traffic flow prediction.

Handles:
- Loading raw h5 files (with 'data' and 'date' keys)
- Min-max normalization with inverse transform
- Sliding window construction for supervised learning
- External feature extraction using actual date strings from the data
- Train/val/test splitting

Real TaxiBJ format:
  Flow:    h5 with 'data' (T, 2, 32, 32) and 'date' (T,) byte strings b'YYYYMMDDSS'
  Meteo:   h5 with 'date', 'Temperature' (T,), 'WindSpeed' (T,), 'Weather' (T, 17) one-hot
  Holiday: txt with YYYYMMDD per line (no dashes)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


# ============================================================
# Scaler
# ============================================================

class MinMaxScaler:
    """Min-max normalization to [0, 1] with inverse transform."""

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.min_val = data.min()
        self.max_val = data.max()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None:
            raise RuntimeError("Scaler not fitted.")
        denom = self.max_val - self.min_val
        if denom == 0:
            return np.zeros_like(data)
        return (data - self.min_val) / denom

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None:
            raise RuntimeError("Scaler not fitted.")
        return data * (self.max_val - self.min_val) + self.min_val


# ============================================================
# Date parsing
# ============================================================

def _parse_date_slot(date_bytes) -> Tuple[str, int]:
    """
    Parse TaxiBJ date format: b'2016010101' -> ('20160101', 1)

    Format: YYYYMMDDSS where SS is the slot number (01-48).
    Slot 01 = 00:00-00:30, Slot 48 = 23:30-00:00.
    """
    s = date_bytes.decode("utf-8") if isinstance(date_bytes, bytes) else str(date_bytes)
    date_part = s[:8]
    slot = int(s[8:10]) if len(s) >= 10 else 1
    return date_part, slot


# ============================================================
# Data loading
# ============================================================

def load_flow_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all TaxiBJ flow h5 files and concatenate.

    Returns:
        flow_data: (total_T, 2, 32, 32) float32
        dates: (total_T,) byte strings
    """
    data_dir = Path(data_dir)
    flow_files = sorted(data_dir.glob("BJ*_M32x32_T30_InOut.h5"))

    if not flow_files:
        raise FileNotFoundError(
            f"No flow data files found in {data_dir}. "
            "Run `python data/download_data.py` first."
        )

    all_data = []
    all_dates = []
    for fpath in flow_files:
        with h5py.File(fpath, "r") as f:
            data = f["data"][:]
            all_data.append(data)
            if "date" in f:
                all_dates.append(f["date"][:])
            print(f"  Loaded {fpath.name}: shape {data.shape}")

    combined = np.concatenate(all_data, axis=0).astype(np.float32)
    dates = np.concatenate(all_dates) if all_dates else np.array([])
    print(f"  Total flow data: {combined.shape}")
    return combined, dates


def load_meteorology(data_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load weather data. Returns None if file doesn't exist."""
    meteo_path = Path(data_dir) / "BJ_Meteorology.h5"
    if not meteo_path.exists():
        print("  Meteorology file not found, skipping.")
        return None

    with h5py.File(meteo_path, "r") as f:
        meteo = {key: f[key][:] for key in f.keys()}
        for key, val in meteo.items():
            print(f"  Loaded meteorology/{key}: shape {val.shape}")
    return meteo


def load_holidays(data_dir: str) -> Optional[set]:
    """Load holiday dates. Format: YYYYMMDD per line."""
    holiday_path = Path(data_dir) / "BJ_Holiday.txt"
    if not holiday_path.exists():
        print("  Holiday file not found, skipping.")
        return None

    holidays = set()
    with open(holiday_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                holidays.add(line)
    print(f"  Loaded {len(holidays)} holiday dates.")
    return holidays


# ============================================================
# Feature extraction
# ============================================================

def create_time_features(dates: np.ndarray) -> np.ndarray:
    """
    Create temporal features from TaxiBJ date byte strings.

    Features (5 dims):
    - hour_sin, hour_cos: cyclical 24h encoding
    - dow_sin, dow_cos: cyclical day-of-week encoding
    - is_weekend: binary

    Args:
        dates: (T,) byte strings in YYYYMMDDSS format

    Returns:
        (T, 5) float32
    """
    T = len(dates)
    features = np.zeros((T, 5), dtype=np.float32)

    for i, d in enumerate(dates):
        date_str, slot = _parse_date_slot(d)
        hour = (slot - 1) * 0.5  # slot 1 -> 0.0h, slot 48 -> 23.5h

        try:
            dow = pd.Timestamp(date_str).dayofweek
        except Exception:
            dow = 0

        features[i, 0] = np.sin(2 * np.pi * hour / 24.0)
        features[i, 1] = np.cos(2 * np.pi * hour / 24.0)
        features[i, 2] = np.sin(2 * np.pi * dow / 7.0)
        features[i, 3] = np.cos(2 * np.pi * dow / 7.0)
        features[i, 4] = 1.0 if dow >= 5 else 0.0

    return features


def create_weather_features(
    meteo: Dict[str, np.ndarray],
    flow_dates: np.ndarray,
) -> np.ndarray:
    """
    Align meteorology data to flow timestamps and build feature vectors.

    In real TaxiBJ:
    - Temperature: (T,) continuous
    - WindSpeed: (T,) continuous
    - Weather: (T, 17) one-hot matrix (already encoded!)

    We build a date->index lookup from meteo dates for proper alignment.

    Returns:
        (T_flow, D) where D = 2 (temp+wind) + 17 (weather) = 19
    """
    T_flow = len(flow_dates)
    features = []

    # Build date->index lookup from meteorology
    meteo_idx = {}
    if "date" in meteo:
        for idx, d in enumerate(meteo["date"]):
            key = d.decode("utf-8") if isinstance(d, bytes) else str(d)
            meteo_idx[key] = idx

    def _align(meteo_array):
        """Align a meteorology array to flow timestamps."""
        if meteo_idx and len(flow_dates) > 0:
            indices = []
            for d in flow_dates:
                key = d.decode("utf-8") if isinstance(d, bytes) else str(d)
                if key in meteo_idx:
                    indices.append(meteo_idx[key])
                else:
                    indices.append(indices[-1] if indices else 0)
            return meteo_array[np.array(indices)]
        else:
            # Fallback: positional slicing
            return meteo_array[:T_flow]

    # Temperature
    if "Temperature" in meteo:
        temp = _align(meteo["Temperature"][:].astype(np.float32))
        t_min, t_max = temp.min(), temp.max()
        temp = (temp - t_min) / (t_max - t_min + 1e-8)
        features.append(temp.reshape(-1, 1))

    # Wind speed
    if "WindSpeed" in meteo:
        wind = _align(meteo["WindSpeed"][:].astype(np.float32))
        w_min, w_max = wind.min(), wind.max()
        wind = (wind - w_min) / (w_max - w_min + 1e-8)
        features.append(wind.reshape(-1, 1))

    # Weather (one-hot or integer)
    if "Weather" in meteo:
        weather_raw = meteo["Weather"][:].astype(np.float32)
        if weather_raw.ndim == 2:
            # Already one-hot (T, 17) — real TaxiBJ format
            weather = _align(weather_raw)
            features.append(weather)
        elif weather_raw.ndim == 1:
            # Integer labels — convert to one-hot
            weather_int = _align(weather_raw).astype(np.int32)
            n_cat = max(int(weather_int.max()) + 1, 17)
            features.append(np.eye(n_cat, dtype=np.float32)[weather_int])

    if not features:
        return np.zeros((T_flow, 1), dtype=np.float32)

    return np.concatenate(features, axis=-1)


def create_holiday_features(
    flow_dates: np.ndarray,
    holidays: set,
) -> np.ndarray:
    """
    Binary holiday indicator using date strings from the data.

    Matches against YYYYMMDD format (no dashes).

    Returns:
        (T, 1) float32
    """
    T = len(flow_dates)
    is_holiday = np.zeros((T, 1), dtype=np.float32)

    for i, d in enumerate(flow_dates):
        date_str, _ = _parse_date_slot(d)
        if date_str in holidays:
            is_holiday[i] = 1.0

    return is_holiday


# ============================================================
# Sliding windows
# ============================================================

def build_sliding_windows(
    flow_data: np.ndarray,
    seq_len: int,
    horizon: int,
    external_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Build sliding window samples for supervised learning.

    Args:
        flow_data: (T, 2, H, W) normalized traffic flow tensor
        seq_len: number of past time steps as input
        horizon: how many steps ahead to predict
        external_features: (T, D) optional feature matrix

    Returns:
        X: (N, seq_len, 2, H, W) input sequences
        Y: (N, 2, H, W) target frames
        X_ext: (N, seq_len, D) or None
    """
    T = flow_data.shape[0]
    n_samples = T - seq_len - horizon + 1

    if n_samples <= 0:
        raise ValueError(f"Not enough data: T={T}, seq_len={seq_len}, horizon={horizon}")

    X = np.zeros((n_samples, seq_len, *flow_data.shape[1:]), dtype=np.float32)
    Y = np.zeros((n_samples, *flow_data.shape[1:]), dtype=np.float32)

    X_ext = None
    if external_features is not None:
        D = external_features.shape[1]
        X_ext = np.zeros((n_samples, seq_len, D), dtype=np.float32)

    for i in range(n_samples):
        X[i] = flow_data[i : i + seq_len]
        Y[i] = flow_data[i + seq_len + horizon - 1]
        if external_features is not None:
            X_ext[i] = external_features[i : i + seq_len]

    return X, Y, X_ext


# ============================================================
# Main pipeline
# ============================================================

def prepare_data(
    data_dir: str = "data/raw/TaxiBJ",
    seq_len: int = 6,
    horizon: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_time: bool = True,
    use_weather: bool = True,
    use_holidays: bool = True,
) -> Dict:
    """
    Full data preparation pipeline.

    Returns dict with:
        X_train, Y_train, X_val, Y_val, X_test, Y_test  (numpy arrays)
        ext_train, ext_val, ext_test  (numpy or None)
        scaler: fitted MinMaxScaler
    """
    print("=" * 60)
    print("Loading and preprocessing data")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading flow data...")
    flow_data, dates = load_flow_data(data_dir)
    T = flow_data.shape[0]
    has_dates = len(dates) == T

    # 2. Normalize
    print("\n[2/5] Normalizing...")
    scaler = MinMaxScaler()
    flow_normalized = scaler.fit_transform(flow_data)
    print(f"  Range: [{scaler.min_val:.1f}, {scaler.max_val:.1f}] -> [0, 1]")

    # 3. External features
    print("\n[3/5] Building external features...")
    ext_parts = []

    if use_time and has_dates:
        time_feats = create_time_features(dates)
        ext_parts.append(time_feats)
        print(f"  Time features: {time_feats.shape}")
    elif use_time:
        print("  WARNING: No date strings in h5 files, skipping time features")

    if use_weather:
        meteo = load_meteorology(data_dir)
        if meteo is not None:
            weather_feats = create_weather_features(meteo, dates if has_dates else np.array([]))
            # Ensure correct length
            if len(weather_feats) > T:
                weather_feats = weather_feats[:T]
            elif len(weather_feats) < T:
                pad = np.zeros((T - len(weather_feats), weather_feats.shape[1]), dtype=np.float32)
                weather_feats = np.concatenate([weather_feats, pad], axis=0)
            ext_parts.append(weather_feats)
            print(f"  Weather features: {weather_feats.shape}")

    if use_holidays and has_dates:
        holidays = load_holidays(data_dir)
        if holidays is not None:
            holiday_feats = create_holiday_features(dates, holidays)
            ext_parts.append(holiday_feats)
            print(f"  Holiday features: {holiday_feats.shape}")

    external = np.concatenate(ext_parts, axis=-1) if ext_parts else None
    if external is not None:
        print(f"  Total external dim: {external.shape[1]}")

    # 4. Sliding windows
    print(f"\n[4/5] Building sliding windows (seq_len={seq_len}, horizon={horizon})...")
    X, Y, X_ext = build_sliding_windows(flow_normalized, seq_len, horizon, external)
    print(f"  Samples: {X.shape[0]}, Input: {X.shape}, Target: {Y.shape}")

    # 5. Split (chronological — no shuffling)
    print(f"\n[5/5] Splitting ({train_ratio}/{val_ratio}/{1 - train_ratio - val_ratio:.1f})...")
    N = X.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    result = {
        "X_train": X[:n_train],
        "Y_train": Y[:n_train],
        "X_val": X[n_train : n_train + n_val],
        "Y_val": Y[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :],
        "Y_test": Y[n_train + n_val :],
        "scaler": scaler,
    }

    if X_ext is not None:
        result["ext_train"] = X_ext[:n_train]
        result["ext_val"] = X_ext[n_train : n_train + n_val]
        result["ext_test"] = X_ext[n_train + n_val :]
    else:
        result["ext_train"] = result["ext_val"] = result["ext_test"] = None

    for split in ["train", "val", "test"]:
        print(f"  {split}: {result[f'X_{split}'].shape[0]} samples")

    print("\nData preparation complete.")
    return result


if __name__ == "__main__":
    data = prepare_data()
