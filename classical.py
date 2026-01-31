"""
Classical SVR Experiment Runner.

DESCRIPTION:
    This script runs the baseline classical SVR pipeline for comparison.
    It uses a standard Radial Basis Function (RBF) kernel SVM to predict BIS.

USAGE:
    # Run via main.py (Recommended)
    uv run main.py --mode classical -p 48 411
"""

import sys
import time
import argparse
import tempfile
from typing import Optional, Dict, Tuple, List

import mlflow
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from qsvm_eeg.data import load_raw_data, trim_zero_ends
from qsvm_eeg.features import extract_features

FS: int = 128
AVAILABLE_PATIENTS: List[str] = ["48", "411"]

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "raw"


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to configure the classical experiment.
    """
    parser = argparse.ArgumentParser(description="Run Classical SVR (RBF) Experiment.")

    parser.add_argument(
        "-p", "--patients",
        nargs="+",
        default=AVAILABLE_PATIENTS,
        help=f"List of Patients. Default: {AVAILABLE_PATIENTS}",
    )

    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=None,
        help="Total number of samples to use (distributed equally). Default: Use all data.",
    )

    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use. Default: -1 (All cores).",
    )

    return parser.parse_args()


def process_single_patient(pid: str, limit_per_patient: Optional[int]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads, cleans, and extracts features for a single patient.
    """
    logger.info(f"Processing Patient {pid}")
    eeg_path = DATA_DIR / f"patient{pid}_eeg.csv"
    bis_path = DATA_DIR / f"patient{pid}_bis.csv"

    eeg_raw, bis_raw = load_raw_data(eeg_path, bis_path)
    if eeg_raw is None:
        return None, None

    # Trim silent periods
    eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=FS)

    # Extract spectral/entropy features
    X = extract_features(eeg, fs=FS)

    # Align labels (Predict future BIS)
    advance_steps = 60
    y = bis[advance_steps:]
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    # Subsampling
    if limit_per_patient is not None:
        if len(X) > limit_per_patient:
            logger.info(f"Patient {pid}: Subsampling {len(X)} -> {limit_per_patient}")
            indices = np.linspace(0, len(X) - 1, limit_per_patient).astype(int)
            X, y = X[indices], y[indices]

    return X, y


def run_classical(args: argparse.Namespace) -> dict:
    # Create Experiment ID
    if len(args.patients) == 1:
        experiment_id = f"Single_{args.patients[0]}"
    else:
        experiment_id = f"Mix_{'_'.join(args.patients)}"

    run_name = f"Classical_{experiment_id}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Setup Temp Log File
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = tmp_path / f"classical_{timestamp}.log"
        log_handler_id = logger.add(log_file, level="INFO")

        mlflow.set_experiment("QSVM_EEG_Comparison")

        try:
            with mlflow.start_run(run_name=run_name) as run:
                logger.info(f"Starting MLflow Run: {run.info.run_id}")
                mlflow.log_params(vars(args))
                mlflow.log_param("model_type", "Classical_RBF")

                logger.info(f"Starting Classical SVR (RBF) Experiment: {experiment_id}")
                logger.info(f"Config: Samples={args.samples if args.samples else 'ALL'} | Jobs={args.jobs}")

                # Start: Load & Process Data
                X_combined, y_combined = [], []
                limit = args.samples // len(args.patients) if args.samples else None

                for pid in args.patients:
                    X_p, y_p = process_single_patient(pid, limit)
                    if X_p is not None:
                        X_combined.append(X_p)
                        y_combined.append(y_p)

                if not X_combined:
                    return

                X = np.vstack(X_combined)
                y = np.concatenate(y_combined)

                logger.info("Shuffling combined dataset")
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=42)

                # Start: Scaling
                logger.info("Scaling Data (StandardScaler)")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Start: Hyperparameter tunning
                logger.info("Starting Grid Search (Classical)")

                param_grid = {
                    "C": [0.1, 1, 10, 50, 100, 500, 1000],
                    "epsilon": [0.1, 0.5, 1.0, 2.0, 4.0],
                    "gamma": ["scale", "auto"],
                }

                t0_train = time.perf_counter()

                n_jobs = getattr(args, 'jobs', -1)

                grid_search = GridSearchCV(
                    SVR(kernel="rbf"),
                    param_grid,
                    cv=5,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=n_jobs,
                    verbose=0,
                )

                grid_search.fit(X_train_scaled, y_train)
                train_time = time.perf_counter() - t0_train

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.success(
                    f"Best Params: {best_params} | Best CV RMSE: {-grid_search.best_score_:.4f}"
                )
                logger.info(f"BENCHMARK | Total Tuning Time: {train_time:.4f}s")

                mlflow.log_params(best_params)

                # Start: Test
                t0_infer = time.perf_counter()
                y_pred = best_model.predict(X_test_scaled)
                infer_time = time.perf_counter() - t0_infer

                # Start: Evaluation
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                r_val, _ = pearsonr(y_test, y_pred)
                ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(len(y_pred))

                logger.success(f"CLASSICAL RESULTS | RMSE: {rmse:.4f} | R2: {r2:.4f} | 95% CI: {ci:.4f}")

                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "pearson": r_val,
                    "ci": ci,
                    "train_time": train_time,
                    "infer_time": infer_time,
                }

                mlflow.log_metrics(metrics)

                # Start: Plot
                fig1 = plt.figure(figsize=(10, 5))
                plt.plot(y_test, label='Actual', alpha=0.7)
                plt.plot(y_pred, label='Classical RBF', linestyle='--')
                plt.title(f"Classical RBF: {experiment_id} (N={len(X)}) | R2={r2:.2f}")
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

                mlflow.log_artifact(log_file, artifact_path="logs")

                return metrics

        finally:
            logger.remove(log_handler_id)


def main():
    args = parse_arguments()
    run_classical(args)


if __name__ == "__main__":
    main()