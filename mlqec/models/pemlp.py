"""
Permutation-Equivariant Multi-Layer Perceptron (PEMLP) implementation for Quantum Error Correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComplexReLU(nn.Module):
    """ReLU activation for complex tensors by applying ReLU to real and imaginary parts separately."""
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexDropout(nn.Module):
    """Dropout for complex tensors by applying the same mask to both real and imaginary parts."""
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:
            return x
            
        # Generate dropout mask for real part
        mask = torch.bernoulli(torch.full_like(x.real, 1 - self.p))
        mask = mask / (1 - self.p)  # Scale to maintain expected value
        
        # Apply same mask to both real and imaginary parts
        return torch.complex(x.real * mask, x.imag * mask)

class PELinear(nn.Module):
    def __init__(self, in_features):
        super(PELinear, self).__init__()
        self.in_features = in_features
        
        # Initialize weights as complex tensors
        real_weight = torch.randn(in_features) / np.sqrt(in_features)
        imag_weight = torch.randn(in_features) / np.sqrt(in_features)
        self.weight = nn.Parameter(torch.complex(real_weight, imag_weight))
        
        real_bias = torch.randn(1) / np.sqrt(in_features)
        imag_bias = torch.randn(1) / np.sqrt(in_features)
        self.bias = nn.Parameter(torch.complex(real_bias, imag_bias))
    
    def forward(self, x):
        """
        Forward pass implementing permutation-equivariant linear transformation.
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, in_features)
        """
        # Ensure input is complex
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Create symmetric weight matrix
        sym_weight = torch.outer(self.weight, torch.ones_like(self.weight))
        sym_weight = sym_weight + sym_weight.T.conj()
        
        # Apply linear transformation
        out = torch.matmul(x, sym_weight)
        out = out + self.bias
        
        return out

class PEMLP(nn.Module):
    """
    Permutation-Equivariant Multi-Layer Perceptron for quantum state processing.
    """
    def __init__(self, input_dim, hidden_dims=[128, 128, 128], output_dim=1, dropout_rate=0.1):
        super(PEMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network with consistent dimensions
        layers = []
        
        # Input layer (input_dim -> first hidden dim)
        layers.append(PELinear(input_dim))
        layers.append(ComplexReLU())
        layers.append(ComplexDropout(dropout_rate))
        
        # Hidden layers
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(PELinear(current_dim))
            layers.append(ComplexReLU())
            layers.append(ComplexDropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer with complex weights
        real_weight = torch.randn(current_dim, output_dim) / np.sqrt(current_dim)
        imag_weight = torch.randn(current_dim, output_dim) / np.sqrt(current_dim)
        self.final_weight = nn.Parameter(torch.complex(real_weight, imag_weight))
        
        real_bias = torch.randn(output_dim) / np.sqrt(current_dim)
        imag_bias = torch.randn(output_dim) / np.sqrt(current_dim)
        self.final_bias = nn.Parameter(torch.complex(real_bias, imag_bias))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the PEMLP.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Convert input to complex if needed
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        
        # Pass through PE layers
        x = self.network(x)
        
        # Final linear layer with complex weights
        out = torch.matmul(x, self.final_weight) + self.final_bias
        return out

class RealPEMLP(nn.Module):
    """PEMLP variant that outputs real values, used for error syndrome prediction."""
    def __init__(self, input_dim, hidden_dims=[128, 128, 128], output_dim=1, dropout_rate=0.1):
        super(RealPEMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Complex layers for feature extraction
        self.complex_layers = nn.ModuleList()
        current_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            self.complex_layers.append(PELinear(current_dim))
            self.complex_layers.append(ComplexReLU())
            self.complex_layers.append(ComplexDropout(dropout_rate))
            current_dim = hidden_dim
        
        # Final real-valued layers
        self.final_layers = nn.Sequential(
            nn.Linear(2 * current_dim, hidden_dims[-1]),  # 2x for real and imag parts
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], output_dim)
        )
    
    def forward(self, x):
        # Process through complex layers
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        
        for layer in self.complex_layers:
            x = layer(x)
        
        # Convert to real representation
        x_real = torch.cat([x.real, x.imag], dim=-1)
        
        # Final real-valued layers
        return self.final_layers(x_real)

class QECModel(nn.Module):
    """
    Complete model for Quantum Error Correction combining tQST and PEMLP.
    """
    def __init__(self, n_qubits, measurement_dim, hidden_dims=[128, 128, 128]):
        super(QECModel, self).__init__()
        self.n_qubits = n_qubits
        self.measurement_dim = measurement_dim
        
        # Dimension of the quantum state space
        self.state_dim = 2 ** n_qubits
        
        # PEMLP for state reconstruction (complex-valued)
        self.reconstruction_network = PEMLP(
            input_dim=measurement_dim,
            hidden_dims=hidden_dims,
            output_dim=self.state_dim  # For state vector
        )
        
        # RealPEMLP for error detection (real-valued)
        self.error_detection_network = RealPEMLP(
            input_dim=measurement_dim,
            hidden_dims=hidden_dims,
            output_dim=n_qubits  # One syndrome bit per qubit
        )
    
    def forward(self, measurements):
        """
        Forward pass performing both state reconstruction and error detection.
        Args:
            measurements: tQST measurement outcomes of shape (batch_size, measurement_dim)
        Returns:
            reconstructed_state: Reconstructed state vector
            error_syndrome: Predicted error locations (one per qubit)
        """
        # Add dimension checks
        if measurements.shape[1] != self.measurement_dim:
            raise ValueError(f"Expected measurements with dimension {self.measurement_dim}, got {measurements.shape[1]}")
        
        # Get reconstructed state (complex-valued)
        reconstructed_state = self.reconstruction_network(measurements)
        
        # Get error syndromes (real-valued)
        error_syndrome = self.error_detection_network(measurements)
        
        return reconstructed_state, error_syndrome
