"""
main.py
-------
End-to-end pipeline: load → filter → segment → extract features → train → evaluate.

Usage (option 1 — pass a directory, loads all .mat files inside):
    python main.py --data_dir ./data/

Usage (option 2 — pass specific files):
    python main.py --mat_files data/S1_E1_A1.mat data/S1_E2_A1.mat data/S1_E3_A1.mat

NinaPro DB5 has 3 exercise files per subject:
    S{n}_E1_A1.mat — exercises 1 (gestures 1–12)
    S{n}_E2_A1.mat — exercises 2 (gestures 13–29)
    S{n}_E3_A1.mat — exercises 3 (gestures 30–52 + rest)
"""

import argparse
import numpy as np
from pathlib import Path

from src.preprocessing import (
    load_db5_mat, full_filter, extract_windows,
    compute_normalization_stats, normalize
)
from src.features import extract_features
from src.train import train_and_evaluate


def resolve_mat_paths(data_dir: str = None, mat_files: list = None) -> list:
    """
    Resolve the list of .mat files to load.
    - If data_dir is given, load all .mat files in that directory (sorted).
    - If mat_files is given, use them directly.
    """
    if data_dir:
        folder = Path(data_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder.resolve()}")
        paths = sorted(folder.glob("*.mat"))
        if not paths:
            raise FileNotFoundError(f"No .mat files found in: {folder.resolve()}")
        return [str(p) for p in paths]

    if mat_files:
        missing = [f for f in mat_files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"File(s) not found: {missing}")
        return mat_files

    raise ValueError("Provide either --data_dir or --mat_files.")


def load_and_concat(mat_paths: list) -> tuple:
    """Load multiple .mat files and concatenate along time axis."""
    emg_list, stim_list, rep_list = [], [], []
    for path in mat_paths:
        print(f"      Loading: {Path(path).name}")
        data = load_db5_mat(path)
        emg_list.append(data["emg"])
        stim_list.append(data["stimulus"])
        rep_list.append(data["repetition"])
    return (
        np.concatenate(emg_list, axis=0),
        np.concatenate(stim_list, axis=0),
        np.concatenate(rep_list, axis=0),
    )


def main(mat_paths: list, window_ms: int = 200, step_ms: int = 10):
    print("=" * 60)
    print("EMG Gesture Classification — NinaPro DB5")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/5] Loading {len(mat_paths)} file(s)...")
    emg, stimulus, repetition = load_and_concat(mat_paths)
    print(f"      EMG shape: {emg.shape} | "
          f"Gestures: {np.unique(stimulus[stimulus > 0])}")

    # 2. Filter
    print("\n[2/5] Filtering (bandpass + notch 60 Hz)...")
    emg = full_filter(emg, notch_freq=60.0)

    # 3. Segment
    print(f"\n[3/5] Segmenting windows ({window_ms} ms, step {step_ms} ms)...")
    splits = extract_windows(emg, stimulus, repetition,
                             window_ms=window_ms, step_ms=step_ms)
    for k, (X, y) in splits.items():
        print(f"      {k:6s}: {X.shape[0]:5d} windows | "
              f"classes: {np.unique(y)[:5]}...")

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # 4. Normalize (stats from train only)
    print("\n[4/5] Normalizing...")
    mean, std = compute_normalization_stats(X_train)
    X_train = normalize(X_train, mean, std)
    X_val   = normalize(X_val,   mean, std)
    X_test  = normalize(X_test,  mean, std)

    # 5. Feature extraction
    print("\n[5/5] Extracting features...")
    F_train = extract_features(X_train)
    F_val   = extract_features(X_val)
    F_test  = extract_features(X_test)
    print(f"      Feature vector size: {F_train.shape[1]}")

    # 6. Train & evaluate
    print("\nTraining classifiers...")
    train_and_evaluate(F_train, y_train, F_val, y_val, F_test, y_test)

    print("\nDone. Check the /results folder for metrics and plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG Gesture Classification")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data_dir",  type=str,
                       help="Directory containing NinaPro DB5 .mat files "
                            "(all .mat files will be loaded)")
    group.add_argument("--mat_files", nargs="+",
                       help="Explicit list of .mat file paths")

    parser.add_argument("--window_ms", type=int, default=200)
    parser.add_argument("--step_ms",   type=int, default=10)
    args = parser.parse_args()
    args.data_dir="EMG_data"

    mat_paths = resolve_mat_paths(data_dir=args.data_dir,
                                  mat_files=args.mat_files)
    main(mat_paths, args.window_ms, args.step_ms)
