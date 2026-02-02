from typing import Dict, Any
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from .base import BaseModel


class ClassicalSVR(BaseModel):
    """
    Classical Support Vector Regression (RBF Kernel).
    Wrapper around sklearn.svm.SVR.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = None
        self.model = None

    @property
    def name(self) -> str:
        return "Classical_SVR_RBF"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Trains RBF SVR using GridSearchCV and StandardScaler.
        """
        # 1. Scaling (StandardScaler as per classical.py)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 2. Get Configs
        # Look into the 'models' section of the config for specific params
        model_conf = self.config.get("models", {}).get("svr_rbf", {})
        param_grid = model_conf.get("param_grid", {
            "C": [0.1, 1, 10, 50, 100, 500, 1000],
            "epsilon": [0.1, 0.5, 1.0, 2.0, 4.0],
            "gamma": ["scale", "auto"]
        })

        n_jobs = self.config.get("jobs", -1)

        # 3. Grid Search
        grid_search = GridSearchCV(
            SVR(kernel="rbf"),
            param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_cv_rmse": -grid_search.best_score_
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise ValueError("Model has not been trained yet.")

        # Apply SAME scaling
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def save(self, path: str) -> None:
        """Saves model AND scaler bundle."""
        if self.model is None:
            raise ValueError("Model has not been trained.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config
        }
        joblib.dump(artifact, path)
