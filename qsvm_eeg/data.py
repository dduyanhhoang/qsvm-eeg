import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean_data(bis_path, eeg_path, fs_eeg=128, fs_bis=1):
    """Loads CSVs and removes zero-padding from both ends."""
    try:
        df_bis = pd.read_csv(bis_path)
        df_eeg = pd.read_csv(eeg_path)
    except FileNotFoundError:
        print(f"Error: Could not find files at {bis_path}")
        return None, None

    eeg_raw = df_eeg['EEG'].interpolate('linear').values
    bis_raw = df_bis['BIS'].interpolate('linear').values

    # Find valid data range (non-zero BIS)
    bis_start = next((i for i, v in enumerate(bis_raw) if v != 0), None)
    bis_end_rev = next((i for i, v in enumerate(bis_raw[::-1]) if v != 0), None)

    if bis_start is None or bis_end_rev is None:
        return np.array([]), np.array([])

    bis_end = len(bis_raw) - bis_end_rev

    # Align EEG indices
    eeg_start = int(bis_start / fs_bis * fs_eeg)
    eeg_end = int(bis_end / fs_bis * fs_eeg)

    return eeg_raw[eeg_start:eeg_end], bis_raw[bis_start:bis_end]
