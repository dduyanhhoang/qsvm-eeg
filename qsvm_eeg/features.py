import numpy as np
from scipy.signal import butter, filtfilt
from joblib import Parallel, delayed

BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 45)
}


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def permutation_entropy(time_series, order=3, delay=1):
    n = len(time_series)
    if n < order * delay: return 0
    # Fast PE implementation
    permutations = np.array([time_series[i:i + order * delay:delay] for i in range(n - (order - 1) * delay)])
    # String conversion method is safer for parallel jobs than axis=0 unique
    perm_strings = [str(row) for row in permutations]
    patterns, counts = np.unique(perm_strings, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def spike_detection(signal, threshold=3):
    if np.std(signal) == 0: return 0
    return np.sum(np.abs(signal - np.mean(signal)) > threshold * np.std(signal))


def _extract_single_window(signal, fs):
    """Internal helper for a single window."""
    feats = []
    for band, (low, high) in BANDS.items():
        try:
            filt = bandpass_filter(signal, fs, low, high)
            var = np.var(filt)
            if var <= 0: var = 1e-10
            # Feature 1: DE Squared
            de_sq = (0.5 * np.log(2 * np.pi * np.e * var)) ** 2
            # Feature 2: Spikes
            spike = spike_detection(filt)
            feats.extend([de_sq, spike])
        except:
            feats.extend([0, 0])

    # Feature 11: PE (Delta only)
    try:
        delta = bandpass_filter(signal, fs, 0.5, 4)
        pe = permutation_entropy(delta)
    except:
        pe = 0
    feats.append(pe)
    return feats


def process_features(eeg, fs=128, window_sec=56, step_sec=1, n_jobs=-1):
    """Main function to parallelize extraction."""
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    starts = range(0, len(eeg) - win_samples + 1, step_samples)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_single_window)(eeg[s:s + win_samples], fs)
        for s in starts
    )
    return np.array([f for f in results if f is not None])
