'''
Kernel-based QSVR Experiment Runner.

DESCRIPTION:
    This script runs the baseline kernel based QSVR pipeline for comparison.

USAGE:
    # Run standalone
    python quantum.py -p 48 411 -j 8

    # Run via main.py
    python main.py --mode quantum -p 48 411

    # Check help
    python quantum.py -h


NOTE: Running Kernel-based QSVR
      Running with arguments:
          -p, --patients :    List of patient IDs (e.g., "48" "411").
                              Default: ["48", "411"] - All available patients
          -n, --samples  :    Total number of samples to use (subsampled evenly).
                              Default: All available data.
          -j, --jobs     :    Number of CPU workers for parallel kernel calculation.
                              Default: 8 cores/processors.

                              * Use -j 1 for small tests or single-GPU laptop runs.
                              * Use -j 8 for NVIDIA A100/HPC runs to maximize throughput.
                              * Consider lower the number of jobs if getting out or memory error

'''

import sys
import time
import tempfile
import argparse
from typing import Optional

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from qsvm_eeg.data import load_raw_data, trim_zero_ends
from qsvm_eeg.features import extract_features
from qsvm_eeg.circuit import compute_kernel_matrix

FS = 128
AVAILABLE_PATIENTS = ["48", "411"]

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
LOGS_DIR = REPORT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_LOG_FILE = LOGS_DIR / f"quantum_{timestamp}.log"


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to configure the experiment

    Run -h, --help to check available options
    """
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
        default=8,
        help="Number of CPU cores for Kernel computation. Default: 8 (8 cores)."
    )

    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=None,
        help="Broadcast Batch Size (e.g., 32768 for GPU, 500 for CPU)."
    )

    parser.add_argument(
        '-be', '--backend',
        type=str,
        default="lightning.gpu",
        choices=["lightning.gpu", "lightning.qubit", "default.qubit"],
        help="PennyLane Backend to use. Default: lightning.gpu"
    )

    return parser.parse_args()


def save_plot(fig: plt.Figure, name: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = FIGURES_DIR / f"{name}_{timestamp}.png"
    fig.savefig(filename, dpi=300)
    logger.info(f"Saved Plot: {filename}")


def process_single_patient(pid: str,
                           limit_per_patient: Optional[int]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads, cleans, and extracts features for a single patient.

    Args:
        pid (str): patient id
        limit_per_patient (Optional[int]): patient's subsamples

    Returns: X (Features), y (BIS Labels)
    """
    logger.info(f"Processing Patient {pid}")

    eeg_path = DATA_DIR / f'patient{pid}_eeg.csv'
    bis_path = DATA_DIR / f'patient{pid}_bis.csv'

    eeg_raw, bis_raw = load_raw_data(eeg_path, bis_path)
    if eeg_raw is None:
        logger.error(f"Failed to load data for Patient {pid}")
        return None, None

    # Trim silent periods
    eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=FS)

    # Extract spectral/entropy features
    X = extract_features(eeg, fs=FS)

    # Align labels
    # BIS monitors have a delay; we predict the value 60 seconds ahead of the window start.
    advance_steps = 60
    y = bis[advance_steps:]

    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # Subsampling
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


def run_quantum(args) -> dict:
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>")
    logger.add(RUN_LOG_FILE, rotation="50 MB", level="INFO")

    mlflow.set_experiment("QSVM_EEG_Comparison")

    # Create experiment id
    if len(args.patients) == 1:
        experiment_id = f"Single_{args.patients[0]}"
    else:
        experiment_id = f"Mix_{'_'.join(args.patients)}"

    run_name = f"Quantum_{experiment_id}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> "
                   "| <level>{level: <8}</level> "
                   "| <cyan>{function}</cyan> - <level>{message}</level>",
        )
        log_file = tmp_path / f"classical_{timestamp}.log"
        log_handler_id = logger.add(log_file, level="INFO")

        mlflow.set_experiment("QSVM_EEG_Comparison")

        try:
            with mlflow.start_run(run_name=run_name) as run:
                logger.info(f"Starting MLflow Run: {run.info.run_id}")
                mlflow.log_params(vars(args))
                mlflow.log_param("model_type", "Kernel-based_QSVR")

                logger.info(f"Starting Kernel-based QSVR Experiment: {experiment_id}")
                logger.info(f"Config: Samples={args.samples if args.samples else 'ALL'} | Jobs={args.jobs}")

                # Start: Load & process data
                X_combined = []
                y_combined = []

                # Calculate subsample
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

                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=42)

                # Start: Scaling
                logger.info("Scaling Data (MinMax 0 to pi)")
                # Using MinMaxScaler, mapping input to range [0, pi]
                # for better angle encoding
                scaler = MinMaxScaler(feature_range=(0, np.pi))
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Start: Compute quantum kernel for SVR training
                logger.info(f"Computing Quantum Kernel Matrix (Train) | Jobs={args.jobs}")
                t0_kernel_train = time.perf_counter()

                K_train = compute_kernel_matrix(X_train_scaled, X_train_scaled,
                                                n_jobs=args.jobs,
                                                batch_size=args.batch_size,
                                                backend=args.backend)

                kernel_train_time = time.perf_counter() - t0_kernel_train
                logger.info(f"Kernel Matrix (Train) Computed in {kernel_train_time:.4f}s")

                # Start: Hyperparameter tuning
                logger.info("Starting Grid Search SVR (Quantum)")
                t0_fit = time.perf_counter()

                param_grid = {
                    'C': [0.1, 1, 10, 50, 100, 500, 1000],
                    'epsilon': [0.1, 0.5, 1.0, 2.0, 4.0]
                }

                grid_search = GridSearchCV(
                    SVR(kernel='precomputed'),
                    param_grid,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=0,
                )

                grid_search.fit(K_train, y_train)

                fit_time = time.perf_counter() - t0_fit

                total_train_time = kernel_train_time + fit_time

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.success(f"Best Params: {best_params} | Best CV RMSE: {-grid_search.best_score_:.4f}")
                logger.info(
                    f"BENCHMARK | Kernel: {kernel_train_time:.2f}s + Tuning: {fit_time:.2f}s = Total: {total_train_time:.2f}s")

                mlflow.log_params(best_params)

                # Start: Compute quantum kernel for SVR testing
                logger.info("Computing Quantum Kernel Matrix (Test)")
                t0_test_kernel = time.perf_counter()

                K_test = compute_kernel_matrix(X_test_scaled, X_train_scaled,
                                               n_jobs=args.jobs,
                                               batch_size=args.batch_size,
                                               backend=args.backend)
                y_pred = best_model.predict(K_test)

                total_infer_time = time.perf_counter() - t0_test_kernel

                # Start: Extract, log metrics for evaluation
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                r_val, _ = pearsonr(y_test, y_pred)

                n = len(y_pred)
                overall_ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(n)

                logger.success(
                    f"QUANTUM RESULTS | RMSE: {rmse:.4f} | R2: {r2:.4f} | R: {r_val:.4f} | 95% CI: {overall_ci:.4f}")

                metrics = {
                    'mse': mse, 'rmse': rmse, 'r2': r2,
                    'pearson': r_val, 'ci': overall_ci,
                    'train_time': total_train_time,
                    'infer_time': total_infer_time
                }
                mlflow.log_metrics(metrics)

                params = {
                    'n_samples': len(X),
                    'C': best_params['C'],
                    'epsilon': best_params['epsilon']
                }

                # Start: Plot
                fig1 = plt.figure(figsize=(10, 5))
                plt.plot(y_test, label='Actual', alpha=0.7)
                plt.plot(y_pred, label='Quantum Prediction', linestyle='--')
                plt.title(f"{experiment_id}: RMSE={rmse:.2f} | Backend={args.backend}")
                plt.legend()

                plot_file1 = tmp_path / f"pred_actual_{experiment_id}.png"
                fig1.savefig(plot_file1, dpi=300)
                mlflow.log_artifact(plot_file1, artifact_path="figures")

                fig2 = plt.figure(figsize=(8, 6))
                plt.scatter(y_pred, y_test, alpha=0.5, color='purple')
                plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'k--')
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"Correlation: {experiment_id}")

                plot_file2 = tmp_path / f"corr_{experiment_id}.png"
                fig2.savefig(plot_file2, dpi=300)
                mlflow.log_artifact(plot_file2, artifact_path="figures")

                mlflow.log_artifact(RUN_LOG_FILE, artifact_path="logs")
                return metrics
        finally:
            logger.remove(log_handler_id)


def main():
    args = parse_arguments()
    run_quantum(args)


if __name__ == "__main__":
    main()
