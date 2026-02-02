import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from loguru import logger
from joblib import Parallel, delayed

# Import feature extraction logic
from .features import extract_single_window

DEFAULT_DATA_DIR = Path("data")


def load_raw_data(patient_id: str,
                  data_dir: Path = DEFAULT_DATA_DIR) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads raw EEG and BIS data.
    """
    raw_dir = data_dir / "raw"
    eeg_path = raw_dir / f"patient{patient_id}_eeg.csv"
    bis_path = raw_dir / f"patient{patient_id}_bis.csv"

    if not eeg_path.exists() or not bis_path.exists():
        logger.error(f"Missing data files for patient {patient_id}")
        return None, None

    try:
        df_eeg = pd.read_csv(eeg_path)
        df_bis = pd.read_csv(bis_path)

        if 'EEG' in df_eeg.columns:
            eeg = df_eeg['EEG'].interpolate('linear').values.flatten()
        else:
            # Fallback if no header (unlikely based on finding)
            eeg = df_eeg.values.flatten()

        if 'BIS' in df_bis.columns:
            bis = df_bis['BIS'].interpolate('linear').values.flatten()
        else:
            bis = df_bis.values.flatten()

        return eeg, bis
    except Exception as e:
        logger.error(f"Error loading data for {patient_id}: {e}")
        return None, None


def trim_zero_ends(eeg: np.ndarray, bis: np.ndarray, fs_eeg: int = 128, fs_bis: int = 1) -> Tuple[
    np.ndarray, np.ndarray]:

    # Find first and last non-zero index in BIS
    non_zeros = np.nonzero(bis)[0]

    if len(non_zeros) == 0:
        logger.warning("BIS signal is all zeros!")
        return np.array([]), np.array([])

    bis_start_idx = non_zeros[0]
    bis_end_idx = non_zeros[-1] + 1  # Slice is exclusive

    # Calculate time window (seconds)
    start_time = bis_start_idx / fs_bis
    end_time = bis_end_idx / fs_bis

    # Convert to EEG indices
    eeg_start_idx = int(start_time * fs_eeg)
    eeg_end_idx = int(end_time * fs_eeg)

    # Safety clamp
    eeg_end_idx = min(eeg_end_idx, len(eeg))

    # Trim both
    eeg_trimmed = eeg[eeg_start_idx: eeg_end_idx]
    bis_trimmed = bis[bis_start_idx: bis_end_idx]

    return eeg_trimmed, bis_trimmed


def _process_window_with_label(start_idx: int, eeg: np.ndarray, bis: np.ndarray,
                               win_samples: int, fs_eeg: int, fs_bis: int,
                               advance_steps: int) -> Optional[Tuple[List[float], float]]:
    """
    Worker function.
    """
    # 1. Extract Features (X)
    features = extract_single_window(start_idx, eeg, win_samples, fs_eeg)

    if features is None:
        return None

    # 2. Extract Label (y)
    # Convert EEG index to BIS index
    # start_idx is in 128Hz. bis array is 1Hz.
    # We want to predict future BIS.

    # Current time in seconds = start_idx / fs_eeg
    # BIS index = int(current_time * fs_bis)
    # Or simplified: bis_idx = start_idx // (fs_eeg / fs_bis) if aligned

    # Original Logic Check: "y = bis[advance_steps:]"
    # The original code sliced arrays. Here we are picking by index.
    # Assuming 'bis' and 'eeg' are now trimmed and time-aligned (t=0 is same for both).

    ratio = fs_eeg / fs_bis  # 128
    bis_current_idx = int(start_idx / ratio)

    # Look ahead
    target_idx = bis_current_idx + advance_steps

    if target_idx >= len(bis):
        return None

    label = bis[target_idx]

    return features, label


def subsample_data(X: np.ndarray, y: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= limit:
        return X, y
    indices = np.linspace(0, len(X) - 1, limit).astype(int)
    return X[indices], y[indices]


def make_dataset(patient_ids: List[str], config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main Pipeline.
    """
    X_all = []
    y_all = []

    base_dir = Path(config.get("dirs", {}).get("raw", "data/raw")).parent
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    fs = config.get("fs", 128)
    # Assume BIS fs is 1Hz (Standard) - could be in config but usually fixed
    fs_bis = 1

    win_sec = config.get("window_sec", 56.0)
    step_sec = config.get("step_sec", 1.0)
    n_jobs = config.get("jobs", -1)
    advance_steps = config.get("advance_steps", 60)
    total_samples_limit = config.get("samples", None)

    for pid in patient_ids:
        # Include advance_steps in cache key so changing it invalidates cache
        cache_name = f"features_p{pid}_w{win_sec}_s{step_sec}_adv{advance_steps}_v2.pkl"
        cache_file = processed_dir / cache_name

        if cache_file.exists():
            logger.info(f"Loading cached data for patient {pid}")
            X_p, y_p = joblib.load(cache_file)
        else:
            logger.info(f"Processing patient {pid} (No cache found)...")

            # 1. Load
            eeg_raw, bis_raw = load_raw_data(pid, base_dir)
            if eeg_raw is None or bis_raw is None:
                continue

            # 2. Trim & Sync
            eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=fs, fs_bis=fs_bis)

            if len(eeg) == 0:
                logger.warning(f"Patient {pid}: Empty data after trim.")
                continue

            # 3. Define Windows
            win_samples = int(win_sec * fs)
            step_samples = int(step_sec * fs)
            starts = range(0, len(eeg) - win_samples + 1, step_samples)

            # 4. Parallel Extraction
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_window_with_label)(
                    s, eeg, bis, win_samples, fs, fs_bis, advance_steps
                )
                for s in starts
            )

            valid_results = [r for r in results if r is not None]

            if not valid_results:
                logger.warning(f"No valid windows for patient {pid}")
                continue

            X_p = np.array([r[0] for r in valid_results])
            y_p = np.array([r[1] for r in valid_results])

            joblib.dump((X_p, y_p), cache_file)
            logger.success(f"Cached {pid}: {X_p.shape}")

        if total_samples_limit is not None:
            limit_per_patient = total_samples_limit // len(patient_ids)
            if len(X_p) > limit_per_patient:
                X_p, y_p = subsample_data(X_p, y_p, limit_per_patient)

        X_all.append(X_p)
        y_all.append(y_p)

    if not X_all:
        raise ValueError("No data loaded!")

    X_final = np.vstack(X_all)
    y_final = np.concatenate(y_all)

    logger.info(f"Final Dataset: X={X_final.shape}, y={y_final.shape}")
    return X_final, y_final
