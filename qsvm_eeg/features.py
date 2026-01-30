import numpy as np
import time

from joblib import Parallel, delayed
from loguru import logger
from scipy.signal import butter, filtfilt

# Standard EEG Frequency Bands
BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 45)
}


def _bandpass_filter(signal: np.ndarray, fs: int,
                     lowcut: float, highcut: float,
                     order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter.

    Args:
        signal (np.ndarray): Input 1D signal.
        fs (int): Sampling frequency (Hz).
        lowcut (float): Lower frequency bound (Hz).
        highcut (float): Upper frequency bound (Hz).
        order (int): Order of the filter. Default is 4.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def _permutation_entropy(time_series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Calculates Permutation Entropy (PE), a measure of the complexity of a time series.

    PE analyzes the order relations between values.
    It is robust to noise and useful for detecting consciousness levels.

    Args:
        time_series (np.ndarray): Input signal.
        order (int): Embedding dimension (length of patterns). Default 3.
        delay (int): Time delay between points. Default 1.

    Returns:
        float: Permutation Entropy value (or 0 if calculation fails).
    """
    n = len(time_series)
    if n < order * delay: return 0

    permutations = np.array([time_series[i:i + order * delay:delay] for i in range(n - (order - 1) * delay)])
    if len(permutations) == 0: return 0

    try:
        sorted_idx: np.ndarray = np.argsort(permutations, axis=1)
        # Count unique permutation patterns
        patterns: np.ndarray
        counts: np.ndarray
        patterns, counts = np.unique(sorted_idx, axis=0, return_counts=True)
        probs: np.ndarray = counts / counts.sum()
        # Shannon Entropy formula
        return -np.sum(probs * np.log2(probs + 1e-10))
    except:
        return 0


def _spike_detection(signal: np.ndarray, threshold: float = 3.) -> int:
    """
    Counts the number of "spikes" - values exceeding N standard deviations.

    Args:
        signal (np.ndarray): Input signal.
        threshold (float): Z-score threshold. Default 3 (3 standard deviations).

    Returns:
        int: Number of outliers.
    """
    if np.std(signal) == 0: return 0
    # Count points where $|x - mean| > threshold \times std$
    return np.sum(np.abs(signal - np.mean(signal)) > threshold * np.std(signal))


def _extract_single_window(start_idx: int, eeg: np.ndarray, win_samples: int, fs: int) -> list:
    """
    Worker function: Extracts features for a single time window.

    Args:
        start_idx (int): Index where the window starts.
        eeg (np.ndarray): The full EEG signal.
        win_samples (int): Length of the window in samples.
        fs (int): Sampling frequency.

    Returns:
        list: Feature vector containing [DE^2 (5 bands), Spikes (5 bands), PE_Delta].
        Returns None if window is invalid.
    """
    window = eeg[start_idx: start_idx + win_samples]

    if len(window) < win_samples: return None
    if np.all(window == 0): return None

    de_sq_list = []
    spike_list = []

    for band_name in BANDS:
        low, high = BANDS[band_name]
        try:
            # Isolate the frequency band
            filt = _bandpass_filter(window, fs, low, high)
            # Calculate Differential Entropy (DE) & spike count in this band
            var = np.var(filt)
            if var <= 0: var = 1e-10
            de = 0.5 * np.log(2 * np.pi * np.e * var)
            de_sq_list.append(de ** 2)
            spike_list.append(_spike_detection(filt))
        except:
            de_sq_list.append(0)
            spike_list.append(0)

    # Calculate Permutation Entropy (Only on Delta Band)
    try:
        delta = _bandpass_filter(window, fs, 0.5, 4)
        pe = _permutation_entropy(delta)
    except:
        pe = 0

    return de_sq_list + spike_list + [pe]


def extract_features(eeg: np.ndarray,
                     fs: int = 128, window_sec: float = 56., step_sec: float =1.,
                     n_jobs: int = -1) -> np.ndarray:
    """
    Main function to extract features from the entire EEG recording using sliding windows.

    Args:
        eeg (np.ndarray): Raw EEG signal.
        fs (int): Sampling frequency (Hz). Default 128.
        window_sec (float): Length of sliding window in seconds. Default 56s.
        step_sec (float): Step size for sliding window in seconds. Default 1s (Overlap = 55s).
        n_jobs (int): Number of CPU cores to use. -1 means use all available cores.

    Returns:
        np.ndarray: Feature matrix of shape (N_windows, N_features).
    """
    t0 = time.perf_counter()
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    starts = range(0, len(eeg) - win_samples + 1, step_samples)
    logger.debug(f"Starting feature extraction on {len(starts)} windows")

    features = Parallel(n_jobs=n_jobs)(
        delayed(_extract_single_window)(s, eeg, win_samples, fs) for s in starts
    )
    t1 = time.perf_counter()

    duration = t1 - t0
    logger.info(f"BENCHMARK | Feature Extraction: {duration:.4f}s")

    return np.array([f for f in features if f is not None])
