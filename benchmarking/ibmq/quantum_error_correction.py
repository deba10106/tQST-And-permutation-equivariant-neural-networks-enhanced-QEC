"""
Quantum Error Correction implementation for IBM Quantum Hardware.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from qiskit import QuantumCircuit
from .adaptive_tqst import AdaptiveTQST, MeasurementResult

class IBMQuantumErrorCorrection:
    def __init__(
        self,
        n_qubits: int,
        model: torch.nn.Module,
        measurement_dim: int,
        use_real_device: bool = False,
        backend_name: str = 'ibmq_jakarta'
    ):
        """
        Initialize Quantum Error Correction for IBM Quantum.
        
        Args:
            n_qubits: Number of qubits
            model: Trained PEMLP model
            measurement_dim: Dimension of measurement space
            use_real_device: Whether to use real IBM Quantum device
            backend_name: Name of the IBM Quantum backend to use
        """
        self.n_qubits = n_qubits
        self.model = model
        self.measurement_dim = measurement_dim
        
        # Initialize adaptive tQST
        self.tqst = AdaptiveTQST(
            n_qubits=n_qubits,
            use_real_device=use_real_device,
            backend_name=backend_name
        )
    
    def correct_quantum_state(
        self,
        circuit: QuantumCircuit,
        noise_level: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[MeasurementResult]]:
        """
        Perform quantum error correction on a quantum state.
        
        Args:
            circuit: Quantum circuit containing the state to correct
            noise_level: Optional noise level parameter
            
        Returns:
            Tuple containing:
                - Corrected quantum state
                - List of measurement results
        """
        # Perform adaptive measurements
        measurements = self.tqst.perform_adaptive_measurements(circuit)
        
        # Convert measurements to syndrome vector
        syndrome = self.tqst.measurements_to_syndrome(measurements)
        
        # Add noise level if provided
        if noise_level is not None:
            syndrome = torch.cat([syndrome, torch.tensor([noise_level])])
        
        # Use ML model to predict corrected state
        with torch.no_grad():
            corrected_state = self.model(syndrome.unsqueeze(0))
        
        return corrected_state.squeeze(0), measurements
    
    def print_correction_results(
        self,
        measurements: List[MeasurementResult],
        corrected_state: torch.Tensor
    ):
        """Print the results of quantum error correction."""
        print("\nAdaptive Measurement Results:")
        print("-" * 40)
        for result in measurements:
            print(f"Qubit {result.qubit} - Pauli {result.pauli}:")
            print(f"  Expectation: {result.expectation:.4f}")
            print(f"  Counts: {result.counts}")
        
        print("\nCorrected Quantum State:")
        print("-" * 40)
        print(corrected_state.detach().numpy())
