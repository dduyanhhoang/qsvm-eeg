from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import joblib
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract Base Class for all EEG models.
    Enforces a strict contract: every model must implement train, predict, and save.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: The full configuration dictionary (loaded from default.yaml).
        """
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the model.
        Returns: Metadata about training (e.g., best_params, training_time)
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns a unique name for logging."""
        pass

    def save(self, path: str) -> None:
        """
        Common method to save the underlying model to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self.model, path)
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {e}")
