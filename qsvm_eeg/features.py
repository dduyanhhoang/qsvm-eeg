import numpy as np
import time

from joblib import Parallel, delayed
from loguru import logger
from scipy.signal import butter, filtfilt

BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 45)
}


def _bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def _permutation_entropy(time_series, order=3, delay=1):
    n = len(time_series)
    if n < order * delay: return 0

    permutations = np.array([time_series[i:i + order * delay:delay] for i in range(n - (order - 1) * delay)])
    if len(permutations) == 0: return 0

    try:
        sorted_idx = np.argsort(permutations, axis=1)
        patterns, counts = np.unique(sorted_idx, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    except:
        return 0


def _spike_detection(signal, threshold=3):
    if np.std(signal) == 0: return 0
    return np.sum(np.abs(signal - np.mean(signal)) > threshold * np.std(signal))


def _extract_single_window(start_idx, eeg, win_samples, fs):
    window = eeg[start_idx: start_idx + win_samples]

    if len(window) < win_samples: return None
    if np.all(window == 0): return None

    de_sq_list = []
    spike_list = []

    for band_name in BANDS:
        low, high = BANDS[band_name]
        try:
            filt = _bandpass_filter(window, fs, low, high)
            var = np.var(filt)
            if var <= 0: var = 1e-10
            de = 0.5 * np.log(2 * np.pi * np.e * var)
            de_sq_list.append(de ** 2)
            spike_list.append(_spike_detection(filt))
        except:
            de_sq_list.append(0);
            spike_list.append(0)

    try:
        delta = _bandpass_filter(window, fs, 0.5, 4)
        pe = _permutation_entropy(delta)
    except:
        pe = 0

    return de_sq_list + spike_list + [pe]


def extract_features(eeg, fs=128, window_sec=56, step_sec=1, n_jobs=-1):
    t0 = time.perf_counter()
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    starts = range(0, len(eeg) - win_samples + 1, step_samples)
    logger.debug(f"Starting feature extraction on {len(starts)} windows")

    features = Parallel(n_jobs=n_jobs)(
        delayed(_extract_single_window)(s, eeg, win_samples, fs) for s in starts
    )

    duration = time.perf_counter() - t0
    logger.info(f"BENCHMARK | Feature Extraction: {duration:.4f}s")

    return np.array([f for f in features if f is not None])
