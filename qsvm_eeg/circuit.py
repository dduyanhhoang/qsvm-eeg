from typing import Union, Optional

import numpy as np
import pennylane as qml
import time

from functools import lru_cache
from joblib import Parallel, delayed
from loguru import logger
from pennylane.templates import AngleEmbedding

N_QUBITS = 11

# Safe starting batch size for GPUs
DEFAULT_BATCH_SIZE = 32768
DEFAULT_BACKEND = "lightning.gpu"


@lru_cache(maxsize=1)
def get_cached_qnode(backend_name: str) -> qml.QNode:
    """
    Creates a QNode for the specific backend.

    @lru_cache(maxsize=1) ensures cache for qnode initialization
    once per worker
    Args:
        backend_name (str): 'lightning.gpu' (GPU) or 'lightning.qubit' (CPU).

    Returns:
        qml.QNode: The compiled quantum circuit ready for execution.
    """
    try:
        dev = qml.device(backend_name, wires=N_QUBITS)
    except Exception as e:
        logger.error(f"Failed to init backend '{backend_name}': {e}")
        logger.warning("Falling back to CPU ('lightning.qubit')")
        dev = qml.device("lightning.qubit", wires=N_QUBITS)

    # Setting qnode without training overheads
    @qml.qnode(dev, interface=None, diff_method=None)
    def qnode(X_batch: np.ndarray, x_single: np.ndarray) -> np.ndarray:
        """
        Quantum Circuit for Kernel Estimation.

        Args:
            X_batch (np.ndarray): Batch of input vectors [Batch_Size, N_Features].
            x_single (np.ndarray): Single input vector [N_Features].
        """
        AngleEmbedding(X_batch, wires=range(N_QUBITS))
        qml.adjoint(AngleEmbedding)(x_single, wires=range(N_QUBITS))
        return qml.probs(wires=range(N_QUBITS))

    return qnode


def _compute_chunk(chunk_indices: Union[list, range],
                   X_A: np.ndarray,
                   X_B: np.ndarray,
                   batch_size: int,
                   backend_name: str) -> np.ndarray:
    """
    Worker function: Computes a subset (chunk) of rows for the Kernel Matrix.

    Args:
        chunk_indices (list): Indices of rows in X_A this worker is responsible for.
        X_A (np.ndarray): The 'Left' matrix.
        X_B (np.ndarray): The 'Right' matrix.
        batch_size (int): Number of circuits to evaluate in parallel on the GPU.
        backend_name (str): Name of the backend to pass to the cached QNode factory.

    Returns:
        np.ndarray: A sub-matrix of kernel values for the assigned chunk.
    """
    qnode: qml.QNode = get_cached_qnode(backend_name)

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
    Computes the Quantum Kernel Matrix K(X_A, X_B) using parallel workers.

    This is the main entry point for the kernel calculation. It handles:
    1. Configuration defaults and safety checks.
    2. Splitting the workload into chunks for parallel processing.
    3. Aggregating results from multiple workers.

    Args:
        X_A (np.ndarray): Input feature matrix A [N_Samples_A, N_Features].
        X_B (np.ndarray): Input feature matrix B [N_Samples_B, N_Features].
        n_jobs (int): Number of CPU workers.
                      -1 uses all cores (or 4-8 on GPU to prevent context thrashing).
        batch_size (int, optional): Circuits to run in parallel per GPU call.
                                    Defaults to DEFAULT_BATCH_SIZE.
        backend (str, optional): PennyLane backend name. Defaults to 'lightning.gpu'.

    Returns:
        np.ndarray: The computed Kernel Matrix of shape (N_Samples_A, N_Samples_B).
    """
    n_A = len(X_A)
    n_B = len(X_B)

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if backend is None:
        backend = DEFAULT_BACKEND

    if "gpu" in backend and n_jobs == -1:
        n_jobs = 4

    logger.info(f"Computing Kernel ({n_A}x{n_B}) | Backend: {backend} | Batch: {batch_size} | Jobs: {n_jobs}")
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
        results = [_compute_chunk(chunk, X_A, X_B, batch_size, backend) for chunk in chunks]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_compute_chunk)(chunk, X_A, X_B, batch_size, backend)
                                          for chunk in chunks)

    matrix = np.vstack(results)

    duration = time.perf_counter() - start
    logger.info(f"BENCHMARK | Kernel Time: {duration:.2f}s")

    return matrix
