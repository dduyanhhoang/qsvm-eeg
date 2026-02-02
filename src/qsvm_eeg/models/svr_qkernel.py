from typing import Dict, Any, Optional
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# CHANGE: Import DEFAULT_BATCH_SIZE
from ..quantum_kernel import compute_kernel_matrix, validate_backend, DEFAULT_BATCH_SIZE
from .base import BaseModel


class QuantumKernelSVR(BaseModel):
    """
    Quantum Kernel Support Vector Regression.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = None
        self.model = None
        self.X_train_scaled: Optional[np.ndarray] = None

        # 1. Resolve Backend
        requested_backend = self.config.get("backend", "lightning.gpu")
        self.backend = validate_backend(requested_backend)

        # 2. Resolve Batch Size
        # If config is None, use Default.
        config_batch = self.config.get("batch_size")
        self.batch_size = config_batch if config_batch is not None else DEFAULT_BATCH_SIZE

    @property
    def name(self) -> str:
        return "Quantum_Kernel_SVR"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Trains SVR using a precomputed Quantum Kernel.
        """
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.X_train_scaled = self.scaler.fit_transform(X_train)

        n_jobs = self.config.get("jobs", -1)

        K_train = compute_kernel_matrix(
            self.X_train_scaled,
            self.X_train_scaled,
            n_jobs=n_jobs,
            backend=self.backend,
            batch_size=self.batch_size
        )

        model_conf = self.config.get("models", {}).get("svr_qkernel", {})
        param_grid = model_conf.get("param_grid", {
            "C": [0.1, 1, 10, 50, 100, 500, 1000],
            "epsilon": [0.1, 0.5, 1.0, 2.0, 4.0]
        })

        grid_search = GridSearchCV(
            SVR(kernel="precomputed"),
            param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
            verbose=0
        )

        grid_search.fit(K_train, y_train)
        self.model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_cv_rmse": -grid_search.best_score_
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None or self.X_train_scaled is None:
            raise ValueError("Model has not been trained yet.")

        X_test_scaled = self.scaler.transform(X_test)
        n_jobs = self.config.get("jobs", -1)

        K_test = compute_kernel_matrix(
            X_test_scaled,
            self.X_train_scaled,
            n_jobs=n_jobs,
            backend=self.backend,
            batch_size=self.batch_size
        )

        return self.model.predict(K_test)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "X_train_scaled": self.X_train_scaled,
            "config": self.config,
            "actual_backend": self.backend,
            "actual_batch_size": self.batch_size  # Save for debugging
        }
        joblib.dump(artifact, path)
