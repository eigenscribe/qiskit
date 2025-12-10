"""
3-Qubit Conditional Probabiliity Calculation using Qiskit StateVector

Usage
-----
$ python conditional_probs.py               # P(q1=0 | q0, q2)
$ python conditional_probs.py --q1 1        # P(q1=1 | q0, q2)

Features
--------
1. Builds a user-defined 3-qubit superposition circuit.
2. Computes P(q1= k | q0, q2) for k in {0,1}, where k is defined by the user via the command-line argument --q1.
3. Illustrates Bayesian updating after partial measurement.

Good Python Habits
------------------
✔️ `if __name__ == '__main__'`
✔️ `def main() -> None:`
✔️ Functions split into small “big_func” style helpers
✔️ Full type annotations
✔️ List comprehensions
"""

from __future__ import annotations
import argparse 
from typing import Dict, Tuple 

import numpy as np
from qiskit import QuantumCircuit 
from qiskit.quantum_info import Statevector 
from IPython.display import display, Latex, Math
import matplotlib.pyplot as plt


def create_custom_state() -> QuantumCircuit:
    """
    Return a 3-qubit circuit that prepares a non-trivial superposition.
    
    Tweaking this function while self-studying is encouraged!
    """
    qc = QuantumCircuit(3)
    qc.h(0)     # put q0 in |+>
    qc.ry(np.pi / 4, 1)     # Rotate q1
    qc.cx(1, 2)             # Entangle q1 -> q2
    qc.barrier(label="custom superposititon")
    return qc 

def compute_conditional_probabilities(
    state: Statevector,
    target_q1: int = 0,
) -> Dict[Tuple[int, int], float]:
    """
    Compute P(q1 = target_q1 | q0, q2).
    
    Parameters
    ----------
    state : Statevector
        Full 3-qubit state. 
    target_q1 : int, optional
        The value (0 or 1) to condition on for q1. 
        
    Return
    ------
    dict[(q0, q2), float]
        Mapping (q0, q2) -> conditional probability.
    """
    accumulator: Dict[Tuple[int, int], Dict[str, float]] = {}
    
    # Probability of each computational basis result
    probs_dict = state.probabilities_dict()
    
    # Qiskit bit ordering in the string is q2 q1 q0 (big-endian)
    for bitstring in [f"{i:03b}" for i in range(8)]:    # List comprehension
        q0, q1, q2 = int(bitstring[2]), int(bitstring[1]), int(bitstring[0])
        key = (q0, q2)
        prob = probs_dict.get(bitstring, 0.0)
        
        if key not in accumulator: 
            accumulator[key] = {"numer": 0.0, "denom": 0.0}
            
        accumulator[key]["denom"] += prob
        if q1 == target_q1:
            accumulator[key]["numer"] += prob 
            
    # Normalize -> conditional probabilities
    cond_probs: Dict[Tuple[int, int], float] = {
        key: (val["numer"] / val["denom"]) if val["denom"] > 0 else 0.0
        for key, val in accumulator.items()
    }
    return cond_probs 

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute P(q1 = k | q0, q2) for a 3-qubit state."
    )    
    parser.add_argument(
        "--q1",
        type=int,
        default=0,
        choices=(0, 1),
        help="Value of q1 to condition on (0 or 1). Default: 0.",
    )
    return parser.parse_args()

def main() -> None:
    """Entry point: build circuit, display state information, and print conditoinal probabilities."""
    args = parse_args()
    target_q1: int = args.q1
    
    # Build circuit and statevectors
    circuit: QuantumCircuit = create_custom_state()
    initial_state: Statevector = Statevector.from_label("000")
    final_state: Statevector = Statevector.from_instruction(circuit)
    
    # Display statevectors
    print("Initial statevector:")
    print(initial_state)
    print("\nFinal statevecotr (after circuit):")
    print(final_state)
    
    # Draw circuit (Latex to consol, MPL to window)
    print("\nCircuit diagram (LaTeX):")
    print(circuit.draw("latex"))
    
    mpl_fig = circuit.draw("mpl")
    mpl_fig.suptitle("Quantum Circuit (Matplotlib)")
    plt.show(block=False)       # non-blocking so script can continue
    
    
    # Conditional probabilities
    cond_probs = compute_conditional_probabilities(final_state, target_q1)
    print(f"\nConditional probabilities P(q1 = {target_q1} | q0, q2):")
    for (q0, q2), prob in sorted(cond_prob.items()):
        print(f"P(q1={target_q1} |  q0={q0}, q2={q2}) = {prob:.4f}")
    
if __name__ == "__main__":
    main()
    