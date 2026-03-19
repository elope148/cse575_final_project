"""
TaxiBJ Dataset Download & Synthetic Data Generator

Citation:
    Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks
    for Citywide Crowd Flows Prediction. In AAAI 2017.

Download manually from the working OneDrive link:
    https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBa2g2Tjd4djN1Vm1oT2hDdHdhaURSeTVvRFZJdWc&id=66E5DE6FBC377A48%2178914&cid=66E5DE6FBC377A48

Place all 6 files into data/raw/TaxiBJ/:
    BJ13_M32x32_T30_InOut.h5
    BJ14_M32x32_T30_InOut.h5
    BJ15_M32x32_T30_InOut.h5
    BJ16_M32x32_T30_InOut.h5
    BJ_Meteorology.h5
    BJ_Holiday.txt

Data format:
    Flow h5:     'data' -> (T, 2, 32, 32) float,  'date' -> (T,) byte strings b'YYYYMMDDSS'
    Meteorology: 'Temperature' -> (T,), 'WindSpeed' -> (T,), 'Weather' -> (T, 17) one-hot
    Holidays:    YYYYMMDD per line (no dashes)
"""

import datetime
import os
from pathlib import Path

ALL_FILES = [
    "BJ13_M32x32_T30_InOut.h5",
    "BJ14_M32x32_T30_InOut.h5",
    "BJ15_M32x32_T30_InOut.h5",
    "BJ16_M32x32_T30_InOut.h5",
    "BJ_Holiday.txt",
    "BJ_Meteorology.h5",
]


def check_data(data_dir: str = "data/raw/TaxiBJ") -> bool:
    """Check if all data files are present and valid."""
    dest_dir = Path(data_dir)
    all_ok = True

    print(f"Checking data in {dest_dir.resolve()}:\n")
    for fname in ALL_FILES:
        path = dest_dir / fname
        if path.exists() and path.stat().st_size > 1000:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK]      {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if not all_ok:
        print(
            "\nDownload from OneDrive (confirmed working):"
            "\n  https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBa2g2Tjd4djN1Vm1oT2hDdHdhaURSeTVvRFZJdWc"
            "&id=66E5DE6FBC377A48%2178914&cid=66E5DE6FBC377A48"
            f"\n\nPlace files in: {dest_dir.resolve()}/"
            "\n\nOr generate synthetic data: python data/download_data.py --synthetic"
        )
    else:
        # Verify h5 contents
        try:
            import h5py
            print("\nVerifying h5 files...")
            for fname in ALL_FILES:
                path = dest_dir / fname
                if path.suffix != ".h5":
                    continue
                with h5py.File(path, "r") as f:
                    keys = list(f.keys())
                    if "data" in keys:
                        s = f["data"].shape
                        print(f"  {fname}: shape={s}, keys={keys}")
                    else:
                        shapes = {k: f[k].shape for k in keys}
                        print(f"  {fname}: {shapes}")
        except ImportError:
            pass

    return all_ok


if __name__ == "__main__":

    check_data("data/raw/TaxiBJ")