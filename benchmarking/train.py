"""
Training script for ML-QEC with adaptive tQST and IBM hardware integration.
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
from datetime import datetime

from pemlp_qec import PEMLP_QEC
from qiskit import QuantumCircuit, IBMQ, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_statevector, Operator, state_fidelity, Statevector
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error

class AdaptiveTrainer:
    def __init__(
        self,
        n_qubits: int = 2,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_real_device: bool = False,
        backend_name: str = 'ibmq_jakarta'
    ):
        self.n_qubits = n_qubits
        self.device = device
        self.use_real_device = use_real_device
        
        # Initialize model
        input_dim = output_dim = 2 * (2 ** n_qubits)  # Real + Imaginary components
        self.model = PEMLP_QEC(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        
        # Setup quantum backend
        if use_real_device:
            try:
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q')
                self.backend = provider.get_backend(backend_name)
                self.noise_model = NoiseModel.from_backend(self.backend)
            except:
                print("Could not connect to IBMQ. Falling back to simulator.")
                self.use_real_device = False
                self.backend = Aer.get_backend('qasm_simulator')
                self.noise_model = self._create_custom_noise_model()
        else:
            self.backend = Aer.get_backend('qasm_simulator')
            self.noise_model = self._create_custom_noise_model()
            
    def _create_custom_noise_model(self) -> NoiseModel:
        """Create a custom noise model similar to IBM devices."""
        noise_model = NoiseModel()
        
        # Single-qubit gate errors
        p1 = 0.001  # 1-qubit gate error rate
        single_qubit_error = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, ['u1', 'u2', 'u3']
        )
        
        # Two-qubit gate errors (higher error rate)
        p2 = 0.01  # 2-qubit gate error rate
        two_qubit_error = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(
            two_qubit_error, ['cx']
        )
        
        return noise_model

    def needs_tqst(self, gate: str, qubits: List[int]) -> bool:
        """Determine if tQST is needed based on gate type and error rates."""
        if not qubits:  # Skip if no qubits provided
            return False
            
        if gate in ['cx', 'cz', 'swap']:  # Multi-qubit gates
            return True
            
        # For single-qubit gates, check if the gate is error-prone
        if gate in ['u3', 'rx', 'ry', 'rz']:
            return np.random.random() < 0.3  # 30% chance for measurement
            
        return False

    def apply_adaptive_tqst(self, circuit: QuantumCircuit) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adaptive tQST only after error-prone gates."""
        # Get ideal state using statevector simulator
        sim = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(circuit, sim)
        ideal_state = Statevector.from_instruction(transpiled_circuit)
        
        # Apply noise and selective tQST
        noisy_circuit = circuit.copy()
        noisy_state = None
        
        for inst, qargs, _ in circuit.data:
            gate_name = inst.name
            qubit_indices = [noisy_circuit.find_bit(q).index for q in qargs]
            
            if self.needs_tqst(gate_name, qubit_indices):
                # Perform tQST measurement after error-prone gate
                transpiled_noisy = transpile(noisy_circuit, self.backend)
                result = self.backend.run(
                    transpiled_noisy,
                    noise_model=self.noise_model,
                    shots=1024
                ).result()
                noisy_state = Statevector.from_instruction(transpiled_noisy)
        
        if noisy_state is None:
            noisy_state = ideal_state
            
        # Convert to real representation
        ideal_real = np.concatenate([ideal_state.data.real, ideal_state.data.imag])
        noisy_real = np.concatenate([noisy_state.data.real, noisy_state.data.imag])
        
        return torch.tensor(noisy_real, dtype=torch.float32), torch.tensor(ideal_real, dtype=torch.float32)

    def generate_training_circuits(self, num_circuits: int = 1000) -> List[QuantumCircuit]:
        """Generate training circuits with varying complexity."""
        circuits = []
        for _ in range(num_circuits):
            qc = QuantumCircuit(self.n_qubits)
            
            # Add random single-qubit gates
            for q in range(self.n_qubits):
                if np.random.random() < 0.7:  # 70% chance of adding gate
                    gate_type = np.random.choice(['h', 'x', 'y', 'z'])
                    if gate_type == 'h':
                        qc.h(q)
                    elif gate_type == 'x':
                        qc.rx(np.random.random() * np.pi, q)
                    elif gate_type == 'y':
                        qc.ry(np.random.random() * np.pi, q)
                    else:
                        qc.rz(np.random.random() * np.pi, q)
            
            # Add CNOT gates (error-prone)
            if self.n_qubits > 1:
                for _ in range(np.random.randint(1, 4)):
                    control = np.random.randint(0, self.n_qubits)
                    target = np.random.randint(0, self.n_qubits)
                    if control != target:
                        qc.cx(control, target)
            
            circuits.append(qc)
        return circuits

    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 32,
        n_circuits: int = 1000
    ):
        """Train the model using adaptive tQST and hardware-specific noise."""
        print("Generating training circuits...")
        circuits = self.generate_training_circuits(n_circuits)
        
        print("Training model...")
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            n_batches = len(circuits) // batch_size
            
            for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}"):
                batch_circuits = circuits[i * batch_size:(i + 1) * batch_size]
                
                # Generate batch data using adaptive tQST
                batch_X = []
                batch_Y = []
                for qc in batch_circuits:
                    x, y = self.apply_adaptive_tqst(qc)
                    batch_X.append(x)
                    batch_Y.append(y)
                
                batch_X = torch.stack(batch_X).to(self.device)
                batch_Y = torch.stack(batch_Y).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.model.permutation_equivariant_loss(output, batch_Y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Update learning rate
            avg_loss = total_loss / n_batches
            self.scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'models/best_pemlp_qec.pth')
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

def main():
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = AdaptiveTrainer(
        n_qubits=2,
        hidden_dim=64,
        use_real_device=False  # Set to True to use IBM Quantum device
    )
    
    # Train model
    trainer.train()
    print("Training complete! Model saved to models/best_pemlp_qec.pth")

if __name__ == "__main__":
    main()
