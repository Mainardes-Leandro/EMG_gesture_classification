"""
features.py
-----------
Time-domain and frequency-domain feature extraction for EMG windows.

Input shape: (N, window_len, 16)
Output shape: (N, n_features * 16)
"""

import numpy as np
from scipy import signal as sp_signal


# ── Time-domain features ──────────────────────────────────────────────────────

def mean_absolute_value(X: np.ndarray) -> np.ndarray:
    """MAV — average rectified EMG amplitude. Shape: (N, 16)"""
    return np.mean(np.abs(X), axis=1)


def root_mean_square(X: np.ndarray) -> np.ndarray:
    """RMS — related to signal power. Shape: (N, 16)"""
    return np.sqrt(np.mean(X ** 2, axis=1))


def zero_crossing_rate(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    ZCR — number of times the signal crosses zero (above threshold).
    Correlates with frequency content. Shape: (N, 16)
    """
    signs = np.sign(X - threshold)
    crossings = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    return crossings.astype(np.float32)


def slope_sign_changes(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    SSC — number of slope sign changes.
    Related to frequency content of the signal. Shape: (N, 16)
    """
    diff = np.diff(X, axis=1)          # (N, win-1, 16)
    signs = np.sign(diff)
    ssc = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    return ssc.astype(np.float32)


def waveform_length(X: np.ndarray) -> np.ndarray:
    """WL — cumulative length of the waveform. Shape: (N, 16)"""
    return np.sum(np.abs(np.diff(X, axis=1)), axis=1)


def hjorth_parameters(X: np.ndarray):
    """
    Hjorth activity, mobility, complexity.
    Returns three arrays of shape (N, 16).
    """
    activity   = np.var(X, axis=1)
    d1         = np.diff(X, axis=1)
    var_d1     = np.var(d1, axis=1)
    mobility   = np.sqrt(var_d1 / (activity + 1e-8))
    d2         = np.diff(d1, axis=1)
    var_d2     = np.var(d2, axis=1)
    mob_d1     = np.sqrt(var_d2 / (var_d1 + 1e-8))
    complexity = mob_d1 / (mobility + 1e-8)
    return activity, mobility, complexity


# ── Frequency-domain features ─────────────────────────────────────────────────

def mean_frequency(X: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Mean frequency of the power spectrum. Shape: (N, 16)
    Useful for fatigue detection (shifts downward with fatigue).
    """
    freqs, psd = sp_signal.welch(X, fs=fs, axis=1)  # psd: (N, freqs, 16)
    total_power = np.sum(psd, axis=1) + 1e-8
    mean_freq = np.sum(freqs[None, :, None] * psd, axis=1) / total_power
    return mean_freq.astype(np.float32)


def median_frequency(X: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Median frequency of the power spectrum. Shape: (N, 16)
    Classic fatigue indicator.
    """
    freqs, psd = sp_signal.welch(X, fs=fs, axis=1)
    cumulative = np.cumsum(psd, axis=1)
    total = cumulative[:, -1:, :]
    # Find first freq index where cumulative >= 50% of total
    mask = cumulative >= (total / 2)
    idx = np.argmax(mask, axis=1)
    return freqs[idx].astype(np.float32)


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(X: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Extract all features and concatenate into a flat feature vector per window.

    Input:  X of shape (N, window_len, 16)
    Output: feature matrix of shape (N, n_features * 16)

    Features per channel (9 total):
        MAV, RMS, ZCR, SSC, WL,
        Hjorth activity, mobility, complexity,
        Mean frequency
    """
    mav        = mean_absolute_value(X)           # (N, 16)
    rms        = root_mean_square(X)              # (N, 16)
    zcr        = zero_crossing_rate(X)            # (N, 16)
    ssc        = slope_sign_changes(X)            # (N, 16)
    wl         = waveform_length(X)               # (N, 16)
    act, mob, comp = hjorth_parameters(X)         # (N, 16) each
    mfreq      = mean_frequency(X, fs=fs)         # (N, 16)

    feature_list = [mav, rms, zcr, ssc, wl, act, mob, comp, mfreq]
    return np.concatenate(feature_list, axis=1).astype(np.float32)  # (N, 144)


FEATURE_NAMES = [
    "MAV", "RMS", "ZCR", "SSC", "WL",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Mean_Frequency"
]
