import pennylane as qml
import numpy as np
from joblib import Parallel, delayed

N_QUBITS = 11
dev_kernel = qml.device("lightning.qubit", wires=N_QUBITS)

ZEROS_STATE = [0] * N_QUBITS


@qml.qnode(dev_kernel)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.expval(qml.Projector(ZEROS_STATE, wires=range(N_QUBITS)))


def _compute_row(x_row, X_all):
    """Computes one row of the kernel matrix."""
    return [kernel_circuit(x_row, x2) for x2 in X_all]


def compute_kernel_matrix(X_A, X_B, n_jobs=-1):
    """
    Computes Kernel Matrix K(A, B) in parallel.
    """
    matrix = Parallel(n_jobs=n_jobs)(
        delayed(_compute_row)(x_row, X_B) for x_row in X_A
    )
    return np.array(matrix)
