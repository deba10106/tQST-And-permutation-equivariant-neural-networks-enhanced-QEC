"""
Adaptive Truncated Quantum State Tomography (tQST) implementation for IBM Quantum Hardware.
"""

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MeasurementResult:
    """Container for measurement results."""
    qubit: int
    pauli: str
    expectation: float
    counts: dict

class AdaptiveTQST:
    def __init__(
        self,
        n_qubits: int,
        threshold: float = 0.1,
        shots: int = 1024,
        use_real_device: bool = False,
        backend_name: str = 'ibmq_jakarta'
    ):
        """
        Initialize Adaptive tQST for IBM Quantum.
        
        Args:
            n_qubits: Number of qubits
            threshold: Threshold for adaptive measurement selection
            shots: Number of measurement shots
            use_real_device: Whether to use real IBM Quantum device
            backend_name: Name of the IBM Quantum backend to use
        """
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.shots = shots
        self.use_real_device = use_real_device
        self.backend_name = backend_name
        self.pauli_operators = ["I", "X", "Y", "Z"]
        
        # Initialize backend
        if use_real_device:
            try:
                IBMQ.load_account()
                self.provider = IBMQ.get_provider(hub='ibm-q')
                self.backend = self.provider.get_backend(backend_name)
            except Exception as e:
                print(f"Error connecting to IBM Quantum: {e}")
                print("Falling back to simulator...")
                self.use_real_device = False
                
        if not self.use_real_device:
            self.backend = Aer.get_backend('qasm_simulator')
    
    def _create_measurement_circuit(
        self,
        base_circuit: QuantumCircuit,
        qubit: int,
        pauli: str
    ) -> QuantumCircuit:
        """Create measurement circuit with appropriate basis rotation."""
        measure_qc = base_circuit.copy()
        measure_qc.barrier()
        
        # Apply basis rotation before measurement
        if pauli == "X":
            measure_qc.h(qubit)  # Hadamard gate rotates X-basis to Z-basis
        elif pauli == "Y":
            measure_qc.sdg(qubit)  # S† gate
            measure_qc.h(qubit)   # Rotate Y-basis to Z-basis
            
        measure_qc.measure_all()
        return measure_qc
    
    def _compute_expectation(self, counts: dict) -> float:
        """Compute expectation value from measurement counts."""
        total_shots = sum(counts.values())
        return (counts.get('0', 0) - counts.get('1', 0)) / total_shots
    
    def perform_adaptive_measurements(
        self,
        circuit: QuantumCircuit
    ) -> List[MeasurementResult]:
        """
        Perform adaptive measurements using tQST protocol.
        
        Args:
            circuit: Quantum circuit to measure
            
        Returns:
            List of MeasurementResult containing selected measurements
        """
        selected_measurements = []
        
        for qubit in range(self.n_qubits):
            for pauli in self.pauli_operators:
                if pauli == "I":
                    continue  # Skip identity measurements
                
                # Create measurement circuit
                measure_qc = self._create_measurement_circuit(circuit, qubit, pauli)
                
                # Execute circuit
                job = execute(measure_qc, self.backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                
                # Compute expectation value
                expectation = self._compute_expectation(counts)
                
                # Adaptive thresholding
                if abs(expectation) > self.threshold:
                    measurement = MeasurementResult(
                        qubit=qubit,
                        pauli=pauli,
                        expectation=expectation,
                        counts=counts
                    )
                    selected_measurements.append(measurement)
        
        return selected_measurements
    
    def measurements_to_syndrome(
        self,
        measurements: List[MeasurementResult]
    ) -> torch.Tensor:
        """
        Convert measurement results to syndrome vector for ML model.
        
        Args:
            measurements: List of measurement results
            
        Returns:
            Tensor containing syndrome vector
        """
        # Create syndrome vector with dimension 128 for compatibility with PEMLP model
        syndrome = torch.zeros(128)
        
        # Map measurements to syndrome vector
        for result in measurements:
            # Map Pauli operators to indices
            pauli_idx = {"X": 0, "Y": 1, "Z": 2}
            base_idx = result.qubit * 3 + pauli_idx[result.pauli]
            
            # Store expectation value
            syndrome[base_idx] = result.expectation
            
            # Store additional measurement information
            syndrome[base_idx + 3 * self.n_qubits] = result.expectation**2  # Square of expectation
            syndrome[base_idx + 6 * self.n_qubits] = abs(result.expectation)  # Absolute value
            
            # Store measurement counts information
            total_shots = sum(result.counts.values())
            syndrome[base_idx + 9 * self.n_qubits] = result.counts.get('0', 0) / total_shots  # P(0)
            syndrome[base_idx + 12 * self.n_qubits] = result.counts.get('1', 0) / total_shots  # P(1)
            
            # Store measurement uncertainty
            uncertainty = np.sqrt((1 - result.expectation**2) / total_shots)  # Statistical uncertainty
            syndrome[base_idx + 15 * self.n_qubits] = uncertainty
            
        return syndrome
