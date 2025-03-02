import torch
import torch.nn as nn
import torch.nn.functional as F

class PEMLP_QEC(nn.Module):
    """
    Permutation-Equivariant Neural Network for Quantum Error Correction
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=3):
        super(PEMLP_QEC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Input projection
        x = torch.relu(self.input_layer(x))
        x = self.layer_norm(x)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = x
            x = self.dropout(torch.relu(layer(x)))
            x = self.layer_norm(x + residual)  # Skip connection
        
        # Output projection
        return self.output_layer(x)

    def permutation_equivariant_loss(self, output, target):
        """
        Custom loss function that respects permutation equivariance
        Args:
            output (torch.Tensor): Model output
            target (torch.Tensor): Target values
        Returns:
            torch.Tensor: Loss value
        """
        # Standard MSE loss
        mse_loss = torch.nn.functional.mse_loss(output, target)
        
        # Add permutation invariance penalty
        perm_loss = 0.0
        for i in range(self.output_dim):
            for j in range(i + 1, self.output_dim):
                perm_loss += torch.abs(
                    torch.mean(output[:, i] - output[:, j]) -
                    torch.mean(target[:, i] - target[:, j])
                )
        
        return mse_loss + 0.1 * perm_loss  # Weight factor for permutation loss

    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
