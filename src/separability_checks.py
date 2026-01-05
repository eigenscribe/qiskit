import numpy as np
from qiskit.quantum_info import Statevector
from typing import Dict, Tuple 

def is_separable(
    state: Statevector, 
    a_qubits: int,
    tol: float = 1e-12
) -> bool:
    """"
    Determine whether a pure quantum state is separable across the bipartition (a_qubits | total - a_qubits)
    using Scmidt rank.

    Parameters
    ----------
    state : Statevector
        A pure quantum system.
    a_qubits : int
        Number of qubits in subsystem A (must satisfy 0 < a_qubits < total_qubits).
    tol : float, optional
        Numerical tolerance for determining non-zero singular values.

    Returns
    -------
    bool
        True if the state is separable across the specified biparition.
    """
    n = int(np.log2(state.dim))    # total qubits
    if not (0 < a_qubits < n):
        raise ValueError(f"a_qubits must be in [1, {n-1}]")
    
    dim_a = 2 ** a_qubits
    dim_b = 2 ** (n - a_qubits)

    mat = state.data.reshape(dim_a, dim_b)
    singular_vals = np.linalg.svd(mat, compute_uv=False)
    schmidt_rank = np.sum(singular_vals > tol)

    return schmidt_rank == 1

def print_partition_classification(
    state: Statevector,
    a_qubits: int, 
    tol: float = 1e-12
) -> None:
    """"
    Print separability classification of a pure quantum statevector for a single bipartition.

    Parameters
    ----------
    state : Statevector
        Quantum state of interest.
    a_qubit : int
        Size of the subsystem A.
    tol : float, optimal
        Numerical tolerance.
    """
    sep = is_separable(state, a_qubits, tol=tol)
    status = "separable 💕" if sep else "entangled 💞"
    print(f"Partition {a_qubits} | {int(np.log2(state.dim) - a_qubits)}: {status}")

def print_partition_schmidt(
    state: Statevector,
    a_qubits: int,
    tol: float = 1e-12,
    show_separable: bool = False
) -> None:
    """
    Print the Schmidt coefficients of a pure state for a given bipartition
    and indicate separability vs. entanglement with emojis.

    Parameters
    ----------
    state : Statevector
        Quantum state of interest.
    a_qubits : int
        Size of subsystem A.
    tol : float, optional
        Numerical tolerance for separability check.
    show_separable : bool, optional
        If True, also prints Schmidt coefficients for separable states.
    """
    n = int(np.log2(state.dim))
    dim_a = 2 ** a_qubits
    dim_b = 2 ** (n - a_qubits)
    mat = state.data.reshape(dim_a, dim_b)

    svals = np.linalg.svd(mat, compute_uv=False)
    svals_sorted = np.sort(svals)[::-1]

    sep = np.sum(svals_sorted > tol) == 1
    status = "separable 💕" if sep else "entangled 💞"
    print(f"Partition {a_qubits} | {n - a_qubits}: {status}\n")

    if not sep or show_separable:
        coeffs_str = ", ".join([f"{v:.4f}" for v in svals_sorted])
        print(f"Schmidt coefficients: [{coeffs_str}]\n")

def classify_all_bipartitions(
    state: Statevector,
    tol: float = 1e-12,
    pretty_print: bool = True,
    show_schmidt: bool = True
) -> Dict[Tuple[int, int], bool]:
    """
    Classify separability vs. entanglement for every bipartition (k | n-k), for k = 1, ..., n-1.
    Optionally print Schmidt coefficients for entangled states.

    Parameters
    ----------
    state : Statevector
        Quantum state vector of interest.
    tol : float, optional
        Numerical tolerance for Schmidt rank evaluation.
    pretty_pring : bool, optional
        If True, print formatted separability results.
    show_schmidt : bool, optional
        If True, print Schmidt coefficients for entangled partitions.

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
            print(f"Partition {k} | {n-k}: {status}\n")

            if show_schmidt and not sep:
                dim_a = 2 ** k
                dim_b = 2 ** (n - k)
                mat = state.data.reshape(dim_a, dim_b)
                svals = np.linalg.svd(mat, compute_uv=False)
                svals_sorted = np.sort(svals)[::-1]
                coeffs_str = ", ".join([f"{v:.4f}" for v in svals_sorted])
                print(f"Schmidt coefficients: [{coeffs_str}]\n")

    return results

def visualize_all_bipartitions(
    state: Statevector,
    tol: float = 1e-12,
    show_separable_coeffs: bool = False
) -> None:
    """"
    Display all bipartitions of a quantum state in a table format with separability
    status and Schmidt coefficients.

    Parameters
    ----------
    state : Statevector
        Quantum state vector. 
    tol : float, optional
        Numerical tolerance for Schmidt rank evaluation.
    show_separable_coeffs : bool, optional
        If True, also display Schmidt coefficients for separable partitions.
    """
    from IPython.display import display, HTML
    
    n = int(np.log2(state.dim))
    rows = []

    for k in range(1, n):
        dim_a = 2 ** k
        dim_b = 2 ** (n - k)
        mat = state.data.reshape(dim_a, dim_b)
        svals = np.linalg.svd(mat, compute_uv=False)
        svals_sorted = np.sort(svals)[::-1]

        sep = np.sum(svals_sorted > tol) == 1
        status = "separable 💕" if sep else "entangled 💞"

        if not sep or show_separable_coeffs:
            coeffs_str = ", ".join([f"{v:.4f}" for v in svals_sorted])
        else: 
            coeffs_str = "" 

        rows.append(f"<tr><td>{k}</td><td>{n-k}</td><td>{status}</td><td>{coeffs_str}</td></tr>")

    table_html = f"""
    <table border="1" style="border-collapse: collapse; text-align: left;">
        <tr>
            <th>Subsystem A (qubits)</th>
            <th>Subsystem B (qubits)</th>
            <th>Status</th>
            <th>Schmidt Coefficients</th>
        </tr>
        {''.join(rows)}
    </table>
    """
    display(HTML(table_html))
            