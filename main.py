import sys
import time
import argparse
import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from qsvm_eeg.data import load_raw_data, trim_zero_ends
from qsvm_eeg.features import extract_features
from qsvm_eeg.circuit import compute_kernel_matrix

FS = 128
AVAILABLE_PATIENTS = ["48", "411", "58"]

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
LOGS_DIR = REPORT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>")
logger.add(LOGS_DIR / "benchmark_{time}.log", rotation="50 MB", level="INFO")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Kernel-based QSVM Experiment on EEG Data.")

    parser.add_argument(
        '-p', '--patients',
        nargs='+',
        default=AVAILABLE_PATIENTS,
        help=f"List of Patient IDs to use. Default: {AVAILABLE_PATIENTS}"
    )

    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=None,
        help="Total number of samples to use (distributed equally among patients). Default: Use all data."
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=-1,
        help="Number of CPU cores for Kernel computation. Default: -1 (All cores)."
    )

    return parser.parse_args()


def save_plot(fig, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = FIGURES_DIR / f"{name}_{timestamp}.png"
    fig.savefig(filename, dpi=300)
    logger.info(f"Saved Plot: {filename}")


def log_results(metrics, params, experiment_id):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = LOGS_DIR / "experiment_log.csv"

    header = "Timestamp,Experiment_ID,Sample_N,MSE,RMSE,R2,Pearson_R,CI_95,C,Epsilon,Kernel,Train_Kernel_Sec,Infer_Kernel_Sec\n"

    if not log_file.exists():
        with open(log_file, "w") as f: f.write(header)

    with open(log_file, "a") as f:
        line = (f"{timestamp},{experiment_id},{params['n_samples']},{metrics['mse']:.5f},"
                f"{metrics['rmse']:.5f},{metrics['r2']:.5f},"
                f"{metrics['pearson']:.5f},{metrics['ci']:.5f},"
                f"{params['C']},{params['epsilon']},Quantum,"
                f"{metrics['train_time']:.4f},"
                f"{metrics['infer_time']:.4f}\n")
        f.write(line)
    logger.success(f"Experiment logged to CSV: {log_file}")


def process_single_patient(pid, limit_per_patient):
    logger.info(f"Processing Patient {pid}")

    eeg_path = DATA_DIR / f'patient{pid}_eeg.csv'
    bis_path = DATA_DIR / f'patient{pid}_bis.csv'

    eeg_raw, bis_raw = load_raw_data(eeg_path, bis_path)
    if eeg_raw is None:
        logger.error(f"Failed to load data for Patient {pid}")
        return None, None

    eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=FS)

    X = extract_features(eeg, fs=FS)

    advance_steps = 60
    y = bis[advance_steps:]

    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    if limit_per_patient is not None:
        if len(X) > limit_per_patient:
            logger.info(f"Patient {pid}: Subsampling {len(X)} -> {limit_per_patient} samples")
            indices = np.linspace(0, len(X) - 1, limit_per_patient).astype(int)
            X = X[indices]
            y = y[indices]
        else:
            logger.warning(f"Patient {pid}: Requested {limit_per_patient} samples, but only has {len(X)}. Using all.")
    else:
        logger.info(f"Patient {pid}: Using all {len(X)} samples.")

    return X, y


def main():
    args = parse_arguments()

    if len(args.patients) == 1:
        experiment_id = f"Single_{args.patients[0]}"
    else:
        experiment_id = f"Mix_{'_'.join(args.patients)}"

    logger.info(f"Starting Experiment: {experiment_id}")
    logger.info(f"Config: Samples={args.samples if args.samples else 'ALL'} | Jobs={args.jobs}")

    X_combined = []
    y_combined = []

    if args.samples is not None:
        limit_per_patient = args.samples // len(args.patients)
    else:
        limit_per_patient = None

    for pid in args.patients:
        X_p, y_p = process_single_patient(pid, limit_per_patient)
        if X_p is not None:
            X_combined.append(X_p)
            y_combined.append(y_p)

    if not X_combined:
        logger.error("No data loaded. Exiting.")
        return

    X = np.vstack(X_combined)
    y = np.concatenate(y_combined)

    logger.info("Shuffling combined dataset")
    p = np.random.RandomState(42).permutation(len(X))
    X, y = X[p], y[p]

    logger.info(f"Total Dataset Shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    logger.info("Scaling Data")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Computing Quantum Kernels (Jobs={args.jobs})")

    t0_train = time.perf_counter()
    K_train = compute_kernel_matrix(X_train_scaled, X_train_scaled, n_jobs=args.jobs)
    train_duration = time.perf_counter() - t0_train

    t0_test = time.perf_counter()
    K_test = compute_kernel_matrix(X_test_scaled, X_train_scaled, n_jobs=args.jobs)
    test_duration = time.perf_counter() - t0_test

    logger.info(f"BENCHMARK | Train Kernel: {train_duration:.4f}s | Infer Kernel: {test_duration:.4f}s")

    logger.info("Training SVR (C=20)")
    t_start_fit = time.perf_counter()
    model = SVR(kernel='precomputed', C=20.0, epsilon=0.1)
    model.fit(K_train, y_train)
    t_end_fit = time.perf_counter()

    logger.info(f"BENCHMARK | Fit Time: {t_end_fit - t_start_fit:.4f}s")

    y_pred = model.predict(K_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred)

    n = len(y_pred)
    overall_ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(n)

    logger.success(f"FINAL RESULTS | RMSE: {rmse:.4f} | R2: {r2:.4f} | R: {r_val:.4f}")

    metrics = {
        'mse': mse, 'rmse': rmse, 'r2': r2,
        'pearson': r_val, 'ci': overall_ci,
        'train_time': train_duration,
        'infer_time': test_duration
    }
    params = {'n_samples': len(X), 'C': 20.0, 'epsilon': 0.1}

    log_results(metrics, params, experiment_id)

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual BIS', alpha=0.7)
    plt.plot(y_pred, label='Quantum Prediction', linestyle='--')
    plt.title(f"{experiment_id}: RMSE={rmse:.2f}, R2={r2:.2f} (N={len(X)})")
    plt.legend()
    save_plot(fig1, f"pred_actual_{experiment_id}")

    fig2 = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_test, alpha=0.5, color='purple')
    plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'k--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Correlation: {experiment_id}")
    save_plot(fig2, f"corr_{experiment_id}")

    # plt.show()


if __name__ == "__main__":
    main()
