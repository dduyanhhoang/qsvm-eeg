from typing import Dict, Any, Type
from .base import BaseModel
from .svr_rbf import ClassicalSVR
from .svr_qkernel import QuantumKernelSVR

# Registry mapping model names (str) to Model Classes
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "svr_rbf": ClassicalSVR,
    "svr_qkernel": QuantumKernelSVR,
}


def get_model(name: str, config: Dict[str, Any]) -> BaseModel:
    """
    Factory function to instantiate a model by name.
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")

    ModelClass = MODEL_REGISTRY[name]
    return ModelClass(config)
