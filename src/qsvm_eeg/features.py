import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from joblib import Parallel, delayed
from loguru import logger
from scipy.signal import butter, filtfilt

BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def _bandpass_filter(signal: np.ndarray, fs: int,
                     lowcut: float, highcut: float,
                     order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    # Handle edge case where highcut >= nyquist
    if highcut >= nyq:
        highcut = nyq - 0.1

    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def _permutation_entropy(time_series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Calculates Permutation Entropy (PE).
    Robust to noise, useful for detecting consciousness levels.
    """
    n = len(time_series)
    if n < order * delay:
        return 0.0

    # Create array of indices for sliding windows
    idx = np.arange(n - (order - 1) * delay)

    # Create the matrix of permutations
    # shape: (n_windows, order)
    permutations = np.array([time_series[i: i + order * delay: delay] for i in idx])

    if len(permutations) == 0:
        return 0.0

    try:
        # Get argsort to determine the rank order pattern (e.g., [1, 2, 0])
        sorted_idx = np.argsort(permutations, axis=1)

        # Count unique patterns
        _, counts = np.unique(sorted_idx, axis=0, return_counts=True)

        # Calculate probabilities
        probs = counts / counts.sum()

        # Shannon Entropy
        return -np.sum(probs * np.log2(probs + 1e-10))
    except Exception:
        return 0.0


def _spike_detection(signal: np.ndarray, threshold: float = 3.0) -> int:
    """
    Counts outliers exceeding N standard deviations.
    """
    if len(signal) == 0:
        return 0

    std = np.std(signal)
    if std == 0:
        return 0

    mean = np.mean(signal)
    # Count points where |x - mean| > threshold * std
    return np.sum(np.abs(signal - mean) > threshold * std)


def extract_single_window(start_idx: int, eeg: np.ndarray, win_samples: int, fs: int) -> Optional[List[float]]:
    """
    Worker function: Extracts features for a single time window.

    Returns:
        list: [DE^2 (5 bands), Spikes (5 bands), PE_Delta] (Total 11 features)
        Returns None if window is invalid (all zeros or too short).
    """
    # Safety Check
    if start_idx + win_samples > len(eeg):
        return None

    window = eeg[start_idx: start_idx + win_samples]

    if len(window) < win_samples: return None
    if np.all(window == 0): return None

    de_sq_list = []
    spike_list = []

    for band_name, (low, high) in BANDS.items():
        try:
            # 1. Filter
            filt = _bandpass_filter(window, fs, low, high)

            # 2. Differential Entropy (DE)
            var = np.var(filt)
            if var <= 0: var = 1e-10
            de = 0.5 * np.log(2 * np.pi * np.e * var)

            de_sq_list.append(de ** 2)

            # 3. Spikes
            spike_list.append(_spike_detection(filt))
        except Exception:
            de_sq_list.append(0.0)
            spike_list.append(0)

    # 4. Permutation Entropy (Only on Delta Band as per original paper/code)
    try:
        delta_low, delta_high = BANDS['delta']
        delta = _bandpass_filter(window, fs, delta_low, delta_high)
        pe = _permutation_entropy(delta)
    except Exception:
        pe = 0.0

    # Concatenate all features: 5 DE + 5 Spikes + 1 PE = 11 Features
    return de_sq_list + spike_list + [pe]


def extract_features(eeg: np.ndarray,
                     fs: int = 128,
                     window_sec: float = 56.0,
                     step_sec: float = 1.0,
                     n_jobs: int = -1) -> np.ndarray:
    t0 = time.perf_counter()

    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    # Validate inputs
    if len(eeg) < win_samples:
        logger.warning("EEG signal shorter than window size.")
        return np.array([])

    starts = range(0, len(eeg) - win_samples + 1, step_samples)
    logger.info(f"Extracting features from {len(starts)} windows (Jobs: {n_jobs})...")

    # Parallel Execution
    features = Parallel(n_jobs=n_jobs)(
        delayed(extract_single_window)(s, eeg, win_samples, fs) for s in starts
    )

    # Filter Nones
    valid_features = np.array([f for f in features if f is not None])

    duration = time.perf_counter() - t0
    logger.info(f"Feature Extraction Complete: {valid_features.shape} | Time: {duration:.2f}s")

    return valid_features
