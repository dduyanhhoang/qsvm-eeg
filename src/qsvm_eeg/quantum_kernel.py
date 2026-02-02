from typing import Union, Optional
import numpy as np
import pennylane as qml
import time
from functools import lru_cache
from joblib import Parallel, delayed
from loguru import logger
from pennylane.templates import AngleEmbedding

# Default config
DEFAULT_BATCH_SIZE = 32768
DEFAULT_BACKEND = "lightning.gpu"


def validate_backend(backend_name: str) -> str:
    """
    Checks if the requested backend is available (ONCE).
    Returns the backend_name if valid, otherwise returns 'lightning.qubit'.
    """
    try:
        qml.device(backend_name, wires=1)
        return backend_name
    except Exception:
        logger.warning(f"Backend '{backend_name}' not available. Falling back to 'lightning.qubit'.")
        return "lightning.qubit"


@lru_cache(maxsize=4)
def get_cached_qnode(backend_name: str, n_qubits: int) -> qml.QNode:
    """
    Creates a QNode for the specific backend and qubit count.
    """
    try:
        dev = qml.device(backend_name, wires=n_qubits)
    except Exception:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=None, diff_method=None)
    def qnode(X_batch: np.ndarray, x_single: np.ndarray) -> np.ndarray:
        AngleEmbedding(X_batch, wires=range(n_qubits))
        qml.adjoint(AngleEmbedding)(x_single, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    return qnode


def _compute_chunk(chunk_indices: Union[list, range],
                   X_A: np.ndarray,
                   X_B: np.ndarray,
                   batch_size: int,
                   backend_name: str,
                   n_qubits: int) -> np.ndarray:
    """
    Worker function.
    """
    qnode = get_cached_qnode(backend_name, n_qubits)

    n_B = len(X_B)
    results = []
    batch_slices = range(0, n_B, batch_size)

    for i in chunk_indices:
        row_parts = []
        x_A_i = X_A[i]

        for j in batch_slices:
            X_B_batch = X_B[j: j + batch_size]
            probs_batch = qnode(X_B_batch, x_A_i)

            if probs_batch.ndim == 1:
                row_parts.append([probs_batch[0]])
            else:
                row_parts.append(probs_batch[:, 0])

        full_row = np.concatenate(row_parts)
        results.append(full_row)

    return np.array(results)


def compute_kernel_matrix(X_A: np.ndarray,
                          X_B: np.ndarray,
                          n_jobs: int = 1,
                          batch_size: Optional[int] = None,
                          backend: Optional[str] = None) -> np.ndarray:
    """
    Computes the Quantum Kernel Matrix K(X_A, X_B).
    """
    n_A, n_features_A = X_A.shape
    n_B, n_features_B = X_B.shape

    if n_features_A != n_features_B:
        raise ValueError(f"Feature mismatch: X_A has {n_features_A}, X_B has {n_features_B}")

    n_qubits = n_features_A

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if backend is None:
        backend = DEFAULT_BACKEND

    backend = validate_backend(backend)

    if "gpu" in backend and n_jobs == -1:
        n_jobs = 4

    logger.info(f"Computing Kernel ({n_A}x{n_B}) | Backend: {backend} | Qubits: {n_qubits} | Jobs: {n_jobs}")
    start = time.perf_counter()

    if n_jobs == -1:
        import os
        n_workers = os.cpu_count() or 4
        n_chunks = n_workers * 4
    else:
        n_chunks = n_jobs

    n_chunks = min(n_chunks, n_A)
    chunks = np.array_split(range(n_A), n_chunks)

    if n_jobs == 1:
        results = [_compute_chunk(chunk, X_A, X_B, batch_size, backend, n_qubits) for chunk in chunks]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_chunk)(chunk, X_A, X_B, batch_size, backend, n_qubits)
            for chunk in chunks
        )

    matrix = np.vstack(results)

    duration = time.perf_counter() - start
    logger.info(f"BENCHMARK | Kernel Time: {duration:.2f}s")

    return matrix
