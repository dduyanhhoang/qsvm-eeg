import numpy as np
import pennylane as qml
import time
from joblib import Parallel, delayed
from loguru import logger
from pennylane.templates import AngleEmbedding

N_QUBITS = 11
# NOTE: Settings for GPU using lightning.gpu device.
#       Consider switching CPU using lightning.qubit
#       if it's too slow, using n_jobs=-1 when run
#       quantum.py:
#
#           python quantum.py -j -1
#
#       and the BROADCAST_BATCH_SIZE=500 for memory
#       safety
BACKEND = "lightning.gpu"
dev_kernel = qml.device(BACKEND, wires=N_QUBITS)

# NOTE: BROADCAST_BATCH_SIZE controls PennyLane's Parameter Broadcasting.
#       Consider lower the size if out of memory
#       BROADCAST_BATCH_SIZE = 32768
BROADCAST_BATCH_SIZE = 65536

@qml.qnode(dev_kernel, interface=None, diff_method=None)
def kernel_circuit_broadcast(X_batch, x_single):
    AngleEmbedding(X_batch, wires=range(N_QUBITS))
    qml.adjoint(AngleEmbedding)(x_single, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def _compute_chunk(chunk_indices, X_A, X_B):
    """
    Worker function: Computes a subset (chunk) of rows for the Kernel Matrix.
    """
    n_B = len(X_B)
    results = []

    for i in chunk_indices:
        row_parts = []

        for j in range(0, n_B, BROADCAST_BATCH_SIZE):
            X_B_batch = X_B[j: j + BROADCAST_BATCH_SIZE]

            probs_batch = kernel_circuit_broadcast(X_B_batch, X_A[i])

            if probs_batch.ndim == 1:
                row_parts.append([probs_batch[0]])
            else:
                row_parts.append(probs_batch[:, 0])

        full_row = np.concatenate(row_parts)
        results.append(full_row)

    return np.array(results)


def compute_kernel_matrix(X_A, X_B, n_jobs=1):
    n_A = len(X_A)
    n_B = len(X_B)

    logger.info(f"Computing Kernel ({n_A}x{n_B}) | Backend: {BACKEND} | Jobs: {n_jobs}")
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
        results = [_compute_chunk(chunk, X_A, X_B) for chunk in chunks]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_chunk)(chunk, X_A, X_B) for chunk in chunks
        )

    matrix = np.vstack(results)

    duration = time.perf_counter() - start
    logger.info(f"BENCHMARK | Kernel Time: {duration:.2f}s")

    return matrix
