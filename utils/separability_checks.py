import numpy as np
from qiskit.quantum_info import Statevector
from typing import Dict, Tuple

def is_separable(
    state: Statevector,
    a_qubits: int,
    tol: float = 1e-12
) -> bool:
    """
    Determine whether a pure quantum state is separable across the bipartition (a_qubits | total - a_qubits) using Schmidt rank.

    Parameters
    ----------
    state : Statevector
        A pure quantum state. 
    a_qubits : int
        Number of qubits in subsystem A (must satisfy 0 < a_qubits < total_qubits).
    tol : float, optional
        Numerical tolerance for determining non-zero singular values.

    Returns 
    -------
    bool
        True if the state is separable across the specified bipartition.
    """

    n = int(np.log2(state.dim))     # total qubits; Note: state.dim = size of the amplitude vector for a quantum state
    if not (0 < a_qubits < n):
        raise ValueError(f"a_qubits must be in [1, {n-1}]")
    
    # Schmidt decomposition: reshape to (2^a) x (2^(n-a))
    dim_a = 2 ** a_qubits
    dim_b = 2 ** (n - a_qubits)

    matrix = state.data.reshape(dim_a, dim_b)
    singular_vals = np.linalg.svd(matrix, compute_uv=False)

    schmidt_rank = np.sum(singular_vals > tol)
    return schmidt_rank == 1

def print_partition_classification(
    state: Statevector,
    a_qubits: int,
    tol: float = 1e-12
) -> None:
    """
    Print separability classification of a pure quantum statevector for a single bipartition.

    Parameters
    ----------
    state : Statevector
        Quantum state of interest.
    a_qubits : int
        Size of the subsystem A.
    tol : float, optional
        Numerical tolerance.
    """
    n = int(np.log2(state.dim))
    sep = is_separable(state, a_qubits, tol=tol)
    status = "separable 💕" if sep else "entangled 💞"
    print(f"Partition {a_qubits} | {n - a_qubits}: {status}")

def classify_all_bipartitions(
    state: Statevector,
    tol: float = 1e-12,
    pretty_print: bool = True
) -> Dict[Tuple[int, int], bool]:
    """
    Classify separability vs entanglement for every bipartition (k | n-k), for k = 1, ..., n-1.

    Parameters 
    ---------
    state : Statevector
        Quantum state vector of interest. 
    tol : float, optional
        Numerical tolerance for Schmidt rank evaluation.
    pretty_print : bool, optional
        If True, print formatted result

    Returns
    -------
    dict
        Keys are (k, n-k), values are boolean:
            True = separable
            False = entangled
    """
    n = int(np.log2(state.dim))
    results: Dict[Tuple[int, int], bool] = {}

    for k in range(1, n):
        sep = is_separable(state, k, tol=tol)
        results[(k, n-k)] = sep

        if pretty_print:
            status = "separable 💕" if sep else "entangled 💞"
            print(f"Partition {k} | {n-k}: {status}")

    return results