"""
preprocessing.py
----------------
Preprocessing pipeline for NinaPro DB5 EMG data.

NinaPro DB5 format:
  - emg:          (T, 16) float — raw EMG signals, 200 Hz
  - restimulus:   (T, 1)  int  — gesture label (0 = rest)
  - rerepetition: (T, 1)  int  — repetition number
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat


# ── Constants ────────────────────────────────────────────────────────────────

FS = 200          # NinaPro DB5 sampling frequency (Hz)
N_CHANNELS = 16   # number of EMG electrodes


# ── Data loading ─────────────────────────────────────────────────────────────

def load_db5_mat(filepath: str) -> dict:
    """Load a NinaPro DB5 .mat file and return a clean dict."""
    mat = loadmat(filepath, squeeze_me=True)
    return {
        "emg":          mat["emg"].astype(np.float32),        # (T, 16)
        "stimulus":     mat["restimulus"].astype(np.int32),   # (T,)
        "repetition":   mat["rerepetition"].astype(np.int32), # (T,)
    }


# ── Filtering ─────────────────────────────────────────────────────────────────

def bandpass_filter(emg: np.ndarray, lowcut: float = 20.0,
                    highcut: float = 90.0, fs: int = FS,
                    order: int = 4) -> np.ndarray:
    """
    Bandpass Butterworth filter to remove DC drift and high-frequency noise.
    Typical range for surface EMG: 20–490 Hz.
    """
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, emg, axis=0).astype(np.float32)


def notch_filter(emg: np.ndarray, freq: float = 60.0,
                 q: float = 30.0, fs: int = FS) -> np.ndarray:
    """
    Notch filter to suppress power-line interference (60 Hz Brazil / 50 Hz EU).
    Adjust `freq` to 50.0 for European datasets.
    """
    b, a = signal.iirnotch(freq / (fs / 2.0), q)
    return signal.filtfilt(b, a, emg, axis=0).astype(np.float32)


def full_filter(emg: np.ndarray, fs: int = FS,
                notch_freq: float = 60.0) -> np.ndarray:
    """Apply bandpass then notch — standard pipeline for surface EMG."""
    emg = bandpass_filter(emg, fs=fs)
    emg = notch_filter(emg, freq=notch_freq, fs=fs)
    return emg


# ── Segmentation ──────────────────────────────────────────────────────────────

def extract_windows(emg: np.ndarray, stimulus: np.ndarray,
                    repetition: np.ndarray,
                    window_ms: int = 200, step_ms: int = 10,
                    fs: int = FS,
                    exclude_rest: bool = True,
                    val_reps: tuple = (5,),
                    test_reps: tuple = (6,)):
    """
    Sliding-window segmentation with train/val/test split by repetition.

    NinaPro DB5 has 6 repetitions per gesture.  We follow the common protocol:
      - train: reps 1-4
      - val:   rep  5
      - test:  rep  6

    Parameters
    ----------
    window_ms : window length in milliseconds (default 200 ms → 40 samples)
    step_ms   : step size in milliseconds      (default  10 ms →  2 samples)
    exclude_rest : drop windows labeled as rest (class 0)

    Returns
    -------
    Dict with keys 'train', 'val', 'test', each containing:
        X : (N, window_len, 16)  float32
        y : (N,)                 int32
    """
    win = int(window_ms * fs / 1000)
    step = int(step_ms * fs / 1000)
    T = emg.shape[0]

    splits = {"train": [], "val": [], "test": []}

    i = 0
    while i + win <= T:
        seg_stim = stimulus[i: i + win]
        seg_rep  = repetition[i: i + win]

        label = int(np.median(seg_stim))
        rep   = int(np.median(seg_rep))

        if exclude_rest and label == 0:
            i += step
            continue

        window = emg[i: i + win]  # (win, 16)

        if rep in test_reps:
            splits["test"].append((window, label))
        elif rep in val_reps:
            splits["val"].append((window, label))
        else:
            splits["train"].append((window, label))

        i += step

    def to_arrays(pairs):
        if not pairs:
            return np.empty((0, win, N_CHANNELS), dtype=np.float32), \
                   np.empty((0,), dtype=np.int32)
        X = np.stack([p[0] for p in pairs]).astype(np.float32)
        y = np.array([p[1] for p in pairs], dtype=np.int32)
        return X, y

    return {k: to_arrays(v) for k, v in splits.items()}


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_normalization_stats(X_train: np.ndarray):
    """
    Compute per-channel mean and std from training windows only.
    X_train shape: (N, window_len, 16)
    """
    # Flatten time axis, keep channel axis
    flat = X_train.reshape(-1, X_train.shape[-1])  # (N*win, 16)
    mean = flat.mean(axis=0, keepdims=True)         # (1, 16)
    std  = flat.std(axis=0, keepdims=True) + 1e-8   # (1, 16)
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalisation using pre-computed train statistics."""
    return (X - mean) / std
