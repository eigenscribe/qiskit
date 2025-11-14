import numpy as np
from qiskit.quantum_info import Statevector

def is_separable(state: Statevector, a_qubits: int, 
                 tol: float = 1e-12) -> bool:
    """
    Helper function that checks separability of a pure state 
    across the bartition (a_qubits | (total - a_qubits)).

    Parameters
    ----------
    state : Statevector
        Pure-state vector.
    a_qubits : int
        Number of qubits assigned to subsystem A (0 < a_qubits < total).
    tol : float, optional
        Numerical tolerance for counting non-zero singular values.

    Returns
    -------
    bool
        True if Schmidt rank == 1 (seaparable across chosen cut).
    """

    n = int(np.log2(state.dim))     # total number of qubits
    if not (0 < a_qubits < n):
        raise ValueError("a_qubits must be between 1 and total-1.")
    
    # Reshape into a (2**a_qubits) x (2**(n-a_qubits)) matrix.
    side_a = 2 ** a_qubits
    side_b = 2 ** (n - a_qubits)
    mat = np.asarray(state.data, dtype=complex).reshape(side_a, side_b)

    # Singular values = Schmidt coefficients
    svals = np.linalg.svd(mat, compute_uv=False)
    rank = np.sum(svals > tol)

    return print(f"Product state across {a_qubits}|{n-a_qubits}",
          "separable 💕" if rank == 1 else "entangled 💞")