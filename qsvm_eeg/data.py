import pandas as pd
import numpy as np
from typing import Optional


def load_raw_data(eeg_path: str, bis_path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads raw EEG and BIS data from CSV files.

    Reads the specified CSV files, extracts the 'EEG' and 'BIS' columns,
    fills missing values using linear interpolation, and returns flattened numpy arrays.

    Args:
        eeg_path (str): File path to the EEG CSV file.
        bis_path (str): File path to the BIS CSV file.

    Returns:
        tuple: A tuple containing:
            - eeg (np.ndarray): Flattened 1D array of EEG signal values.
            - bis (np.ndarray): Flattened 1D array of BIS index values.
            Returns (None, None) if files are not found.
    """
    try:
        df_eeg: pd.DataFrame = pd.read_csv(eeg_path)
        df_bis: pd.DataFrame = pd.read_csv(bis_path)

        # Interpolate missing values (NaNs) and flatten to 1D array
        # 'linear' interpolation connects points across small gaps in data
        eeg: np.ndarray = df_eeg['EEG'].interpolate('linear').values.flatten()
        bis: np.ndarray = df_bis['BIS'].interpolate('linear').values.flatten()
        return eeg, bis
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None


def trim_zero_ends(eeg: np.ndarray, bis: np.ndarray,
                   fs_eeg: int = 128, fs_bis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Trims leading and trailing zeros from signals to synchronize recording times.

    BIS monitors often record '0' before the sensors are properly attached or
    after they are removed. This function detects the valid window of BIS data
    (non-zero values) and crops the high-frequency EEG signal to match the
    exact same time window.

    Args:
        eeg (np.ndarray): The raw EEG signal array.
        bis (np.ndarray): The raw BIS signal array (contains zeros to be trimmed).
        fs_eeg (int, optional): Sampling frequency of EEG in Hz. Defaults to 128.
        fs_bis (int, optional): Sampling frequency of BIS in Hz. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - eeg_trimmed (np.ndarray): The EEG signal cropped to the valid BIS window.
            - bis_trimmed (np.ndarray): The BIS signal without leading/trailing zeros.
            Returns (empty array, empty array) if BIS signal is entirely zeros.
    """
    bis: np.ndarray = np.array(bis)
    eeg: np.ndarray = np.array(eeg)

    # Find the index of the first non-zero value (Start of valid recording)
    bis_start_idx: int = next((i for i, val in enumerate(bis) if val != 0), None)
    # Find the index of the last non-zero value (End of valid recording)
    bis_end_idx: int = next((i for i, val in enumerate(bis[::-1]) if val != 0), None)

    # Edge Case: If signals are empty or all zeros
    if bis_start_idx is None or bis_end_idx is None:
        return np.array([]), np.array([])

    # Convert reversed index back to forward index
    bis_end_idx = len(bis) - bis_end_idx

    # Calculate time stamps (in seconds) for the start and end of the valid window
    start_time: float = bis_start_idx / fs_bis
    end_time: float = bis_end_idx / fs_bis

    # Convert seconds to EEG indices (since EEG has a higher sampling rate)
    start_eeg_idx = int(start_time * fs_eeg)
    end_eeg_idx = int(end_time * fs_eeg)

    return eeg[start_eeg_idx:end_eeg_idx], bis[bis_start_idx:bis_end_idx]
