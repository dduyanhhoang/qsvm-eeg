import pennylane as qml
from pennylane.templates import AngleEmbedding
import numpy as np
import time
from joblib import Parallel, delayed

N_QUBITS = 11
dev_kernel = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(dev_kernel)
def kernel_circuit(x1, x2):
    AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def _compute_row(i, X_A, X_B, is_symmetric):
    n_B = len(X_B)
    row_res = np.zeros(n_B)

    start_index = i if is_symmetric else 0

    for j in range(start_index, n_B):
        if is_symmetric and i == j:
            row_res[j] = 1.0
        else:
            row_res[j] = kernel_circuit(X_A[i], X_B[j])[0]

    return row_res


def compute_kernel_matrix(X_A, X_B, verbose=True):
    n_A, n_B = len(X_A), len(X_B)
    is_symmetric = np.array_equal(X_A, X_B)

    if verbose:
        print(f"   Computing Quantum Kernel ({n_A} x {n_B})...")
        print(f"   Mode: {'Symmetric + Parallel' if is_symmetric else 'Parallel'}")

    start_t = time.time()

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_compute_row)(i, X_A, X_B, is_symmetric) for i in range(n_A)
    )

    matrix = np.array(results)

    if is_symmetric:
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    if verbose:
        duration = time.time() - start_t
        print(f"   Done in {duration:.1f}s")

    return matrix
