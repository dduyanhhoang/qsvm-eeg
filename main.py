import sys
import time
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, t

from qsvm_eeg.data import load_raw_data, trim_zero_ends
from qsvm_eeg.features import extract_features
from qsvm_eeg.circuit import compute_kernel_matrix

SAMPLE_LIMIT = 3000
FS = 128
PATIENT_ID = "411"

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
LOGS_DIR = REPORT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"


def save_plot(fig, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = FIGURES_DIR / f"{name}_{timestamp}.png"
    fig.savefig(filename, dpi=300)
    print(f"   [Saved Plot] {filename}")


def log_results(metrics, params):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = LOGS_DIR / "experiment_log.csv"

    header = "Timestamp,Patient,Sample_N,MSE,RMSE,R2,Pearson_R,CI_95,C,Epsilon,Kernel\n"

    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write(header)

    with open(log_file, "a") as f:
        line = (f"{timestamp},{PATIENT_ID},{params['n_samples']},{metrics['mse']:.5f},"
                f"{metrics['rmse']:.5f},{metrics['r2']:.5f},"
                f"{metrics['pearson']:.5f},{metrics['ci']:.5f},"
                f"{params['C']},{params['epsilon']},Quantum\n")
        f.write(line)
    print(f"   [Saved Log] {log_file}")


def main():
    print(f"--- Starting Quantum Experiment (Patient {PATIENT_ID}) ---")

    eeg_path = DATA_DIR / f'patient{PATIENT_ID}_eeg.csv'
    bis_path = DATA_DIR / f'patient{PATIENT_ID}_bis.csv'

    eeg_raw, bis_raw = load_raw_data(eeg_path, bis_path)
    if eeg_raw is None: return

    print("2. Aligning and Trimming...")
    eeg, bis = trim_zero_ends(eeg_raw, bis_raw, fs_eeg=FS)

    print("3. Extracting Features...")
    X = extract_features(eeg, fs=FS)

    advance_steps = 60
    y = bis[advance_steps:]

    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    if SAMPLE_LIMIT and len(X) > SAMPLE_LIMIT:
        print(f"   [INFO] Subsampling to {SAMPLE_LIMIT} samples.")
        indices = np.linspace(0, len(X) - 1, SAMPLE_LIMIT).astype(int)
        X = X[indices]
        y = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    print("4. Scaling Data...")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("5. Computing Quantum Kernels...")
    K_train = compute_kernel_matrix(X_train_scaled, X_train_scaled)
    K_test = compute_kernel_matrix(X_test_scaled, X_train_scaled)

    print("6. Training SVR (C=20)...")
    model = SVR(kernel='precomputed', C=20.0, epsilon=0.1)
    model.fit(K_train, y_train)

    y_pred = model.predict(K_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred)

    n = len(y_pred)
    overall_ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(n)

    print(f"\n--- FINAL RESULTS (N={len(X)}) ---")
    print(f"MSE:    : {mse:.5f}")
    print(f"RMSE    : {rmse:.5f}")
    print(f"R2      : {r2:.5f}")
    print(f"R       : {r_val:.5f}")
    print(f"95% CI  : {overall_ci:.5f}")

    metrics = {
        'mse': mse, 'rmse': rmse, 'r2': r2,
        'pearson': r_val, 'ci': overall_ci
    }
    params = {'n_samples': len(X), 'C': 20.0, 'epsilon': 0.1}

    log_results(metrics, params)

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual BIS', alpha=0.7)
    plt.plot(y_pred, label='Quantum Prediction', linestyle='--')
    plt.title(f"Quantum SVR: RMSE={rmse:.2f}, R2={r2:.2f} (N={len(X)})")
    plt.legend()
    save_plot(fig1, "prediction_vs_actual")

    fig2 = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_test, alpha=0.5, color='purple')
    plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'k--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Correlation Plot")
    save_plot(fig2, "correlation_plot")

    plt.show()

    # print("7. Saving Model Artifacts...")
    # artifacts = {
    #     "model": model,
    #     "scaler": scaler,
    #     "X_train": X_train,
    #     "support_indices": model.support_
    # }
    #
    # model_filename = f"qsvm_patient48_{SAMPLE_LIMIT}_model_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.pkl"
    # joblib.dump(artifacts, MODEL_DIR / model_filename)
    # print("   [Saved] qsvm_patient48_model.pkl")


if __name__ == "__main__":
    main()
