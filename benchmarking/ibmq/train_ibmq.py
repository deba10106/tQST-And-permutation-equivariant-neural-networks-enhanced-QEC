"""
Training script for QEC model using IBM Quantum data and comparing with IBM's built-in QEC/QEM.
"""

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Dict
import numpy as np

from qiskit import QuantumCircuit, execute, ClassicalRegister
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

from ..models.pemlp import QECModel
from .adaptive_tqst import AdaptiveTQST
from .. import config

class IBMQTrainer:
    def __init__(
        self,
        n_qubits: int,
        measurement_dim: int,
        backend,
        device: str = "cuda:0",
        hidden_dims: List[int] = [128, 128, 128],
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        log_dir: str = None
    ):
        """Initialize trainer for IBM Quantum data with comparison to built-in QEC/QEM."""
        self.n_qubits = n_qubits
        self.measurement_dim = measurement_dim
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        
        # Initialize model
        self.model = QECModel(
            n_qubits=n_qubits,
            measurement_dim=measurement_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize tQST
        self.tqst = AdaptiveTQST(n_qubits=n_qubits)
        
        # Initialize measurement error mitigation
        meas_calibs = []
        state_labels = []
        
        # Generate calibration circuits for all basis states
        for state in range(2 ** n_qubits):
            # Create circuit for this basis state
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Convert state to binary and apply X gates where needed
            binary = format(state, f'0{n_qubits}b')  # No need to reverse, Qiskit handles bit order
            for qubit, bit in enumerate(binary[::-1]):  # Reverse for gate application
                if bit == '1':
                    qc.x(qubit)
            
            # Add measurements
            qc.measure_all()
            meas_calibs.append(qc)
            
            # Add state label with spaces every 3 bits
            label = ''
            for i, bit in enumerate(binary):
                if i > 0 and i % 3 == 0:
                    label += ' '
                label += bit
            state_labels.append(label)
        
        # Execute calibration circuits
        job = execute(meas_calibs, backend=backend, shots=8192)
        cal_results = job.result()
        
        # Create measurement filter
        self.meas_fitter = CompleteMeasFitter(cal_results, state_labels=state_labels)
        self.meas_filter = self.meas_fitter.filter
        
        # Initialize tensorboard writer
        if log_dir is None:
            log_dir = os.path.join(config.RESULTS_DIR, f"logs/QEC_IBMQ_{n_qubits}qubits_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Initialize metrics storage
        self.metrics_file = os.path.join(config.RESULTS_DIR, 'metrics_ibmq_comparison.json')
    
    def _build_calibration_circuits(self):
        """Build calibration circuits for measurement error mitigation."""
        # Generate calibration circuits for all measurement qubits
        qr = QuantumCircuit(self.n_qubits)
        meas_calibs, state_labels = [], []
        
        # Generate all 2^n basis states
        state_labels = [format(i, f'0{self.n_qubits}b') for i in range(2**self.n_qubits)]
        
        for label in state_labels:
            circ = QuantumCircuit(self.n_qubits, self.n_qubits)
            for qubit, bit in enumerate(label):
                if bit == '1':
                    circ.x(qubit)
            circ.measure_all()
            meas_calibs.append(circ)
        
        return meas_calibs, state_labels
    
    def generate_training_data(
        self,
        n_samples: int,
        test_circuits: Tuple[List[QuantumCircuit], List[QuantumCircuit]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data using both our method and IBM's built-in QEC/QEM."""
        measured_circuits, unmeasured_circuits = test_circuits
        measurements_list = []
        ideal_states_list = []
        
        for circuit, unmeasured_circuit in zip(measured_circuits, unmeasured_circuits):
            # Get ideal state from unmeasured circuit
            ideal_statevector = Statevector.from_instruction(unmeasured_circuit)
            ideal_state = torch.tensor(ideal_statevector.data, dtype=torch.complex64)
            
            # Perform our adaptive measurements on measured circuit
            measurements = self.tqst.perform_adaptive_measurements(circuit)
            syndrome = self.tqst.measurements_to_syndrome(measurements)
            
            measurements_list.append(syndrome)
            ideal_states_list.append(ideal_state)
        
        return torch.stack(measurements_list), torch.stack(ideal_states_list)
    
    def evaluate_methods(
        self,
        test_circuits: Tuple[List[QuantumCircuit], List[QuantumCircuit]]
    ) -> Dict[str, float]:
        """Compare our method with IBM's built-in QEC/QEM."""
        measured_circuits, unmeasured_circuits = test_circuits
        results = {
            'raw_fidelity': [],
            'ibm_qem_fidelity': [],
            'our_method_fidelity': []
        }
        
        for circuit, unmeasured_circuit in zip(measured_circuits, unmeasured_circuits):
            # Get ideal state from unmeasured circuit
            ideal_statevector = Statevector.from_instruction(unmeasured_circuit)
            
            # 1. Raw execution (no error mitigation)
            job = execute(circuit, self.backend, shots=8192)
            raw_counts = job.result().get_counts()
            
            # Format counts to match calibration labels
            formatted_counts = {}
            for bitstring, count in raw_counts.items():
                # Remove spaces and format to match calibration labels
                bits = bitstring.replace(' ', '')
                formatted = ''
                for i, bit in enumerate(bits):
                    if i > 0 and i % 3 == 0:
                        formatted += ' '
                    formatted += bit
                formatted_counts[formatted] = count
            
            raw_fidelity = self._compute_fidelity_from_counts(raw_counts, ideal_statevector)
            results['raw_fidelity'].append(raw_fidelity)
            
            # 2. IBM's measurement error mitigation
            mitigated_counts = self.meas_filter.apply(formatted_counts)
            ibm_fidelity = self._compute_fidelity_from_counts(mitigated_counts, ideal_statevector)
            results['ibm_qem_fidelity'].append(ibm_fidelity)
            
            # 3. Our tQST + PEMLP method
            measurements = self.tqst.perform_adaptive_measurements(circuit)
            syndrome = self.tqst.measurements_to_syndrome(measurements)
            with torch.no_grad():
                corrected_state, _ = self.model(syndrome.unsqueeze(0))  # Ignore error syndrome
            our_fidelity = self._compute_fidelity(corrected_state.squeeze(0), torch.tensor(ideal_statevector.data))
            results['our_method_fidelity'].append(our_fidelity)
        
        # Compute averages
        return {k: np.mean(v) for k, v in results.items()}
    
    def _compute_fidelity_from_counts(self, counts: Dict[str, int], ideal_statevector: Statevector) -> float:
        """Compute fidelity between counts and ideal state."""
        # Convert counts to statevector
        total_shots = sum(counts.values())
        state_vector = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        
        for bitstring, count in counts.items():
            # Remove spaces and use Qiskit's bit order convention
            bitstring = bitstring.replace(' ', '')
            index = int(bitstring, 2)
            state_vector[index] = np.sqrt(count / total_shots)
        
        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector /= norm
        
        # Compute fidelity
        return float(state_fidelity(state_vector, ideal_statevector))
    
    def _compute_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Compute fidelity between two quantum states."""
        return abs(torch.dot(state1.conj(), state2))**2
    
    def _train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_recon_loss = 0
        total_syndrome_loss = 0
        total_fidelity = 0
        n_batches = 0
        
        for measurements, target_states in train_loader:
            measurements = measurements.to(self.device)
            target_states = target_states.to(self.device)
            
            # Forward pass
            reconstructed_states, _ = self.model(measurements)  # Ignore error syndrome
            
            # Compute losses
            recon_loss = self._compute_reconstruction_loss(reconstructed_states, target_states)
            
            # Compute fidelity
            fidelity = self._compute_batch_fidelity(reconstructed_states, target_states)
            
            # Total loss
            loss = recon_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_recon_loss += recon_loss.item()
            total_fidelity += fidelity.item()
            n_batches += 1
        
        # Compute average metrics
        avg_recon_loss = total_recon_loss / n_batches
        avg_fidelity = total_fidelity / n_batches
        
        return {
            'reconstruction_loss': avg_recon_loss,
            'syndrome_loss': 0.0,  # Not using syndrome loss
            'fidelity': avg_fidelity
        }
    
    def _validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_recon_loss = 0
        total_syndrome_loss = 0
        total_fidelity = 0
        n_batches = 0
        
        with torch.no_grad():
            for measurements, target_states in val_loader:
                measurements = measurements.to(self.device)
                target_states = target_states.to(self.device)
                
                # Forward pass
                reconstructed_states, _ = self.model(measurements)  # Ignore error syndrome
                
                # Compute losses
                recon_loss = self._compute_reconstruction_loss(reconstructed_states, target_states)
                
                # Compute fidelity
                fidelity = self._compute_batch_fidelity(reconstructed_states, target_states)
                
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_fidelity += fidelity.item()
                n_batches += 1
        
        # Compute average metrics
        avg_recon_loss = total_recon_loss / n_batches
        avg_fidelity = total_fidelity / n_batches
        
        return {
            'reconstruction_loss': avg_recon_loss,
            'syndrome_loss': 0.0,  # Not using syndrome loss
            'fidelity': avg_fidelity
        }

    def _compute_reconstruction_loss(self, reconstructed_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss between quantum states."""
        # Compute L2 loss between the quantum states
        return torch.mean(torch.sum(torch.abs(reconstructed_states - target_states)**2, dim=1))

    def _compute_syndrome_loss(self, reconstructed_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
        """Compute syndrome loss between quantum states."""
        # Compute cross-entropy loss between the syndrome measurements
        reconstructed_probs = torch.abs(reconstructed_states)**2
        target_probs = torch.abs(target_states)**2
        return -torch.mean(torch.sum(target_probs * torch.log(reconstructed_probs + 1e-10), dim=1))

    def _compute_batch_fidelity(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """Compute average fidelity for a batch of quantum states."""
        # Compute fidelity between each pair of states in the batch
        fidelities = torch.abs(torch.sum(states1.conj() * states2, dim=1))**2
        return torch.mean(fidelities)
    
    def train(
        self,
        test_circuits: Tuple[List[QuantumCircuit], List[QuantumCircuit]],
        n_epochs: int = 100,
        checkpoint_dir: str = "checkpoints_ibmq"
    ):
        """Train the model and compare with IBM's built-in QEC/QEM."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate dataset
        print("Generating training data from IBM Quantum device...")
        measured_circuits, unmeasured_circuits = test_circuits
        measurements, ideal_states = self.generate_training_data(
            len(measured_circuits),
            (measured_circuits, unmeasured_circuits)
        )
        
        # Split into train and validation
        split_idx = int(0.8 * len(measured_circuits))
        train_data = TensorDataset(
            measurements[:split_idx],
            ideal_states[:split_idx]
        )
        val_data = TensorDataset(
            measurements[split_idx:],
            ideal_states[split_idx:]
        )
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        
        # Training loop
        best_fidelity = 0.0
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train and validate
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            
            # Compare with IBM's method
            comparison_metrics = self.evaluate_methods((measured_circuits[split_idx:], unmeasured_circuits[split_idx:]))
            
            # Print results
            print("\nPerformance Comparison:")
            print(f"Raw Fidelity: {comparison_metrics['raw_fidelity']:.4f}")
            print(f"IBM QEM Fidelity: {comparison_metrics['ibm_qem_fidelity']:.4f}")
            print(f"Our Method Fidelity: {comparison_metrics['our_method_fidelity']:.4f}")
            
            # Save metrics
            self.save_metrics({
                'train': train_metrics,
                'val': val_metrics,
                'comparison': comparison_metrics
            })
            
            # Save best model
            if comparison_metrics['our_method_fidelity'] > best_fidelity:
                best_fidelity = comparison_metrics['our_method_fidelity']
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_fidelity': best_fidelity,
                    },
                    os.path.join(checkpoint_dir, 'best_model_ibmq.pt')
                )
        
        self.writer.close()

def generate_test_circuits(n_qubits: int, n_circuits: int) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
    """
    Generate test circuits for training and evaluation.
    Returns:
        Tuple of (measured_circuits, unmeasured_circuits)
    """
    measured_circuits = []
    unmeasured_circuits = []
    
    for _ in range(n_circuits):
        # Create circuit without measurements first
        qc = QuantumCircuit(n_qubits)
        
        # Create different types of quantum states
        # 1. Superposition states
        for i in range(n_qubits):
            qc.h(i)
        
        # 2. Entangled states (reverse order to match Qiskit's convention)
        for i in range(n_qubits-1, 0, -1):
            qc.cx(i-1, i)
        
        # 3. Random rotations
        for i in range(n_qubits):
            qc.rz(np.random.random() * 2 * np.pi, i)
            qc.rx(np.random.random() * 2 * np.pi, i)
            qc.ry(np.random.random() * 2 * np.pi, i)
        
        # Save unmeasured circuit
        unmeasured_circuits.append(qc)
        
        # Create measured version
        qc_measured = qc.copy()
        qc_measured.add_register(ClassicalRegister(n_qubits))
        qc_measured.measure_all()
        measured_circuits.append(qc_measured)
    
    return measured_circuits, unmeasured_circuits

def main():
    """Main training function."""
    # Parameters
    n_qubits = 3
    measurement_dim = 9  # 3 * n_qubits for X, Y, Z measurements
    n_circuits = 100
    n_epochs = 100
    
    # Generate test circuits
    measured_circuits, unmeasured_circuits = generate_test_circuits(n_qubits, n_circuits)
    
    # Initialize trainer
    trainer = IBMQTrainer(
        n_qubits=n_qubits,
        measurement_dim=measurement_dim,
        backend=Aer.get_backend('qasm_simulator'),
        backend_name=config.IBMQ_BACKEND
    )
    
    # Train model and compare with IBM's QEC/QEM
    trainer.train(
        test_circuits=(measured_circuits, unmeasured_circuits),
        n_epochs=n_epochs
    )

if __name__ == "__main__":
    main()
