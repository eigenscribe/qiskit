#!/usr/bin/env python3
"""
Generic conditional-probability helper for an n-qubit QuantumCircuit.
---------------------------------------------------------------------

Library functions
-----------------
analyze_circuit(circuit: QuantumCircuit, target_q1: int = 0) -> Dict[(int,int), float]

Library usage
-------------
from qiskit import QuantumCircuit
from utils.qubit_cond_probabilities import analyze_circuit

qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.cs(1,2)
final_sv, cond = analyze_circuit(
    qc,
    target_qubit=1,
    target_bit=0,
    iniit_state="000",                      # optional - default is |00..0>
)
display(final_sv.draw("latex"))
print(cond)

CLI usage
---------
$ qubit-cond-prob                           # default circuit, q0=0
$ qubit-cond-prob --qubit 2 --bit 1         # P(q2=1 | rest)
$ qubit-cond-prob --circuit my.qpy --init 101
"""

# --------------------------------------------------------
# 0️⃣ Import Libraries
# --------------------------------------------------------

from __future__ import annotations

import argparse
import pathlib 
from typing import Dict, Tuple 

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector 

# ---------------------------------------------------------
# 1️⃣ Helper Functions
# ---------------------------------------------------------

def _default_circuit() -> QuantumCircuit:
    """Return a simple 3-qubit demo circuit."""
    qc = QuantumCircuit(3)
    qc.h(0)     # put q0 in |+> state
    qc.ry(np.pi / 4, 1)
    qc.cx(1, 2)
    qc.barrier(label="demo")
    return qc

def _load_qpy(path: pathlib.Path) -> QuantumCircuit:
    """Load the first circuit in a QPY file."""
    with path.open("rb") as f:
        circuits = qpy.load(f)
    if not circuits:
        raise ValueError(f"No circuits found in {path}")
    if len(circuits) > 1:
        print("⚠️ Multiple circuits in QPY; using the first")
    return circuits[0]

def _compute_conditional_probs(
    state: Statevector,
    target_qubit: int,
    target_bit: int,
) -> Dict[str, float]:
    """
    Compute P(q_target = target_bit | rest) for *state*.
    
    The returned dict maps    rest_bitstring -> probability.
    Bit-strings are big-endian (Qiskit's ordering), e.g., "10" means 
    q_(n-1)=1, q_(n-2)=0, ... and *exclude* the target qubit.
    """
    n = int(np.log2(state.dim))
    probs_dict = state.probabilities_dict()
    
    acc: Dict[str, Dict[str, float]] = {}
    # iterate over all computational basis states (big-endian strings)
    for bitstring, prob in probs_dict.items():
        little_endian = bitstring[::-1]             # index == qubit index
        tgt = int(little_endian[target_qubit])
        rest_bits_le = (
            little_endian[:target_qubit] + 
            little_endian[target_qubit + 1:]
        ) 
    rest_key = rest_bits_le[::-1]                   # convert back to big-endian
    
    bucket = acc.setdefault(rest_key, {"num": 0.0, "den": 0.0})
    bucket["den"] += prob
    if tgt == target_bit:
        bucket["num"] += prob
        
    return {
        key: b["num"] / b["den"] if b["den"] else 0.0
        for key, b in acc.items()
    }

# --------------------------------------------------------
# 2️⃣ Library "big_funcs"
# --------------------------------------------------------
def analyze_circuit(
    circuit: QuantumCircuit,
    *,
    target_qubit: int = 0,
    target_bit: int = 0,
    init_state: str | None = None,
) -> Tuple[Statevector, Dict[str, float]]:
    """
    Apply *circuit* to an initial basis state and return:
        (final_statevector, conditional_probabilities).
    
    Parameters
    ----------
    circuit         : QuantumCircuit
    target_qubit    : int           qubit intex to condition on
    target_bit      : {0,1}         target value of that qubit
    init_state      : str | None    e.g. "000"; default is all-zero string
    
    Returns
    -------
    (Statevector, dict[str, float])
    """
    nq = circuit.num_qubits 
    if init_state is None:
        init_state = "0" * nq
    if len(init_state) != nq or set(init_state) - {"0", "1"}:
        raise ValueError(
            f"init_state must be a {nq}-bit string of 0/1, got '{init_state}'."
        )
    if not (0 <= target_qubit < nq):
        raise ValueError(f"target_qubit must be in [0,{nq-1}].")
    if target_bit not in (0, 1):
        raise ValueError("target_bit must be 0 or 1.")
    
    initial_statevector = Statevector.from_label(init_state)
    final_statevector = initial_statevector.evolve(circuit)
    
    cond_probs = _compute_conditional_probs(final_statevector, target_qubit, target_bit)
    return final_statevector, cond_probs

# --------------------------------------------------------
# 3️⃣ CLI
# --------------------------------------------------------   
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return final Statevector and conditional probabilities "
                    "for a QuantumCircuit"
    )
    parser.add_argument(
        "--circuit", type=pathlib.Path,
        help="Path to a .qpy file holding the circuit (default: demo circuit)."
    )
    parser.add_argument(
        "--init", type=str,
        help="Initial computational-basis state, e.g., 101 (default: all zeros)."
    )
    parser.add_argument(
        "--qubit", type=int, default=0, 
        help="Target qubit index (default: 0)."
    )
    parser.add_argument(
        "--bit", type=int, choices=(0, 1), default=0,
        help="Target bit value of that qubit (default: 0)."
    )
    return parser.parse_args()

# --------------------------------------------------------
# 4️⃣ Good Habits 1 & 2
# --------------------------------------------------------   

def main() -> None:
    """Console entry-point (installed by pyproject.toml)."""
    args = _parse_args()
    
    circuit = (
        _load_qpy(args.circuit) if args.circuit is not None else _default_circuit()
    )
    
    final_statevector, cond = analyze_circuit(
        circuit,
        target_qubit=args.qubit,
        target_bit=args.bit,
        init_state=args.init,
    )
    
    print("Final statevector:")
    print(final_statevector)
    print(f"\nConditional probabilities "
          f"P(q{args.qubit} = {args.bit} | rest):")
    for rest, p in sorted(cond.items()):
        print(f"P = {p:.4f}    given rest = {rest}")
        
if __name__ == "__main__":
    main()        