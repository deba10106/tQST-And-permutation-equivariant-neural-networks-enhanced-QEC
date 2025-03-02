"""
Quantum state generation and measurement simulation for tQST.
"""

import numpy as np
import torch
from scipy.linalg import sqrtm
from typing import Tuple, List
import config

class QuantumStateGenerator:
    """
    Generator class for creating quantum states and simulating measurements.
    """
    def __init__(self, n_qubits: int = config.N_QUBITS):
        """
        Initialize the quantum state generator.
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
    def generate_pure_state(self) -> torch.Tensor:
        """
        Generate a random pure quantum state.
        Returns:
            Density matrix of the pure state
        """
        # Create random complex state vector
        psi = torch.randn(self.dim, dtype=torch.cfloat)
        psi = psi / torch.norm(psi)
        
        # Create density matrix
        rho = torch.outer(psi, psi.conj())
        return rho
    
    def generate_mixed_state(self, rank: int) -> torch.Tensor:
        """
        Generate a random mixed state of given rank.
        Args:
            rank: Rank of the mixed state
        Returns:
            Density matrix of the mixed state
        """
        # Generate rank pure states and mix them
        states = [self.generate_pure_state() for _ in range(rank)]
        weights = torch.rand(rank)
        weights = weights / weights.sum()  # Normalize weights
        
        rho = sum(w * state for w, state in zip(weights, states))
        return rho
    
    def apply_noise(self, state: torch.Tensor, noise_strength: float) -> torch.Tensor:
        """
        Apply depolarizing noise to the quantum state.
        Args:
            state: Input quantum state
            noise_strength: Strength of the noise (0 to 1)
        Returns:
            Noisy quantum state
        """
        # Depolarizing channel: ρ → (1-p)ρ + p I/d
        identity = torch.eye(self.dim, dtype=torch.cfloat)
        noisy_state = (1 - noise_strength) * state + (noise_strength/self.dim) * identity
        return noisy_state
    
    def compute_gini_index(self, diag: torch.Tensor) -> float:
        """
        Compute the Gini index of the diagonal elements.
        Args:
            diag: Diagonal elements of density matrix
        Returns:
            Gini index value
        """
        sorted_diag = torch.sort(diag)[0]
        n = len(diag)
        index = torch.arange(1, n + 1, dtype=torch.float32)
        return (2 * torch.sum(index * sorted_diag) / (n * torch.sum(sorted_diag))) - (n + 1)/n
    
    def tqst_measurements(self, state: torch.Tensor, target_len: int = config.MEASUREMENT_DIM) -> torch.Tensor:
        """
        Perform tQST measurements on the quantum state.
        Args:
            state: Quantum state to measure
            target_len: Target length of measurement vector
        Returns:
            Measurement outcomes
        """
        diag = torch.real(torch.diag(state))
        gini = self.compute_gini_index(diag)
        threshold = gini / (self.dim - 1)
        
        measurements = []
        # Measure diagonal elements
        measurements.extend(diag.tolist())
        
        # Selective measurement of off-diagonal elements
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                if np.sqrt(diag[i] * diag[j]) >= threshold:
                    measurements.append(torch.real(state[i,j]).item())
                    measurements.append(torch.imag(state[i,j]).item())
                else:
                    measurements.extend([2.0, 2.0])  # Placeholder for unmeasured elements
                    
        # Adjust to target length
        if len(measurements) > target_len:
            measurements = measurements[:target_len]
        else:
            measurements.extend([2.0] * (target_len - len(measurements)))
            
        return torch.tensor(measurements)
    
    def generate_dataset(
        self,
        n_samples: int,
        target_len: int = config.MEASUREMENT_DIM,
        noise_range: Tuple[float, float] = config.NOISE_LEVELS
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of quantum states and their measurements.
        Args:
            n_samples: Number of samples to generate
            target_len: Length of measurement vector
            noise_range: Range of noise levels to sample from
        Returns:
            measurements: tQST measurement outcomes
            states: Pure quantum states
            noise_levels: Applied noise levels
        """
        measurements_list = []
        states_list = []
        noise_levels_list = []
        
        for _ in range(n_samples):
            # Generate random pure state
            state = self.generate_pure_state()
            
            # Apply random noise
            noise = np.random.uniform(*noise_range)
            noisy_state = self.apply_noise(state, noise)
            
            # Convert numpy array to complex tensor
            if isinstance(noisy_state, np.ndarray):
                noisy_state_tensor = torch.from_numpy(noisy_state).to(torch.cfloat)
            else:
                # If already a tensor, just ensure it's complex
                noisy_state_tensor = noisy_state.to(torch.cfloat)
            
            # Perform measurements
            measurements = self.tqst_measurements(noisy_state_tensor, target_len)
            
            measurements_list.append(measurements)
            states_list.append(noisy_state_tensor)
            noise_levels_list.append(noise)
        
        # Stack tensors
        measurements = torch.stack(measurements_list)
        states = torch.stack(states_list)
        noise_levels = torch.tensor(noise_levels_list)
        
        return measurements, states, noise_levels

# Example usage
if __name__ == "__main__":
    # Create generator for 2-qubit system
    generator = QuantumStateGenerator()
    
    # Generate small dataset
    measurements, states, noise_levels = generator.generate_dataset(
        n_samples=100
    )
    
    print(f"Generated dataset shapes:")
    print(f"Measurements: {measurements.shape}")
    print(f"States: {states.shape}")
    print(f"Noise levels: {noise_levels.shape}")
