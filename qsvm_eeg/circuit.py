import numpy as np
import pennylane as qml
import time
from joblib import Parallel, delayed
from loguru import logger
from pennylane.templates import AngleEmbedding

N_QUBITS = 11
dev_kernel = qml.device("lightning.qubit", wires=N_QUBITS)
BACKEND = "lightning.qubit"


@qml.qnode(dev_kernel)
def kernel_circuit_broadcast(X_batch, x_single):
    AngleEmbedding(X_batch, wires=range(N_QUBITS))
    qml.adjoint(AngleEmbedding)(x_single, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def _compute_chunk(chunk_indices, X_A, X_B):
    n_B = len(X_B)
    results = []

    for i in chunk_indices:
        probs = kernel_circuit_broadcast(X_B, X_A[i])
        row = probs[:, 0]
        results.append(row)

    return np.array(results)


def compute_kernel_matrix(X_A, X_B, n_jobs=-1):
    n_A = len(X_A)
    logger.info(f"Computing Kernel ({n_A}x{len(X_B)}) with n_jobs={n_jobs}...")

    start = time.perf_counter()

    if n_jobs == -1:
        import os
        n_cores = os.cpu_count() or 4
        n_chunks = n_cores * 4
    else:
        n_chunks = n_jobs

    chunks = np.array_split(range(n_A), n_chunks)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_chunk)(chunk, X_A, X_B) for chunk in chunks
    )

    matrix = np.vstack(results)

    duration = time.perf_counter() - start
    logger.info(f"BENCHMARK | Kernel Time: {duration:.2f}s")

    return matrix
