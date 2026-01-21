import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
from pathlib import Path

# Import modules
from qsvm_eeg.data import load_and_clean_data
from qsvm_eeg.features import process_features
from qsvm_eeg.circuit import compute_kernel_matrix

DATA_DIR = Path("data/raw")
SAMPLE_LIMIT = 1000

def main():
    # 1. Load
    print("1. Loading Data...")
    eeg, bis = load_and_clean_data(
        DATA_DIR / 'patient48_bis.csv',
        DATA_DIR / 'patient48_eeg.csv'
    )
    if eeg is None: return

    print("2. Extracting Features...")
    X = process_features(eeg)

    advance_steps = 60
    y = bis[advance_steps: advance_steps + len(X)]
    X = X[:len(y)]

    if SAMPLE_LIMIT and len(X) > SAMPLE_LIMIT:
        print(f"   [INFO] Subsampling to {SAMPLE_LIMIT} samples.")
        indices = np.linspace(0, len(X) - 1, SAMPLE_LIMIT).astype(int)
        X = X[indices]
        y = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. Computing Quantum Kernels...")
    t0 = time.time()
    K_train = compute_kernel_matrix(X_train_scaled, X_train_scaled)
    K_test = compute_kernel_matrix(X_test_scaled, X_train_scaled)
    print(f"   -> Done in {time.time() - t0:.1f}s")

    print("4. Training SVR...")
    model = SVR(kernel='precomputed')
    model.fit(K_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(K_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Results (N={len(X)}) ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.2f}")

    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Quantum Prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
