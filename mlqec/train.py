"""
Training script for the QEC model using tQST and PEMLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Tuple, Dict
import os
from datetime import datetime
import config
import torch.nn.functional as F
import json

from models.pemlp import QECModel
from quantum_state_generator import QuantumStateGenerator

class QECTrainer:
    """
    Trainer class for Quantum Error Correction model.
    """
    def __init__(
        self,
        n_qubits: int,
        measurement_dim: int,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.n_qubits = n_qubits
        self.measurement_dim = measurement_dim
        self.device = device
        self.batch_size = batch_size
        
        # Initialize model
        self.model = QECModel(
            n_qubits=n_qubits,
            measurement_dim=measurement_dim
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.reconstruction_loss = self.reconstruction_loss_fn
        self.syndrome_loss = nn.BCEWithLogitsLoss()
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(
            log_dir=f"logs/QEC_{n_qubits}qubits_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize metrics storage
        self.metrics_file = 'metrics.json'
    
    def compute_fidelity(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fidelity between reconstructed and target quantum states.
        F = Tr(sqrt(sqrt(rho)sigma sqrt(rho)))^2
        """
        batch_size = state1.shape[0]
        fidelities = []
        
        for i in range(batch_size):
            recon = state1[i]
            target = state2[i]
            
            # Ensure states are properly normalized
            recon = recon / torch.trace(recon)
            target = target / torch.trace(target)
            
            # Compute fidelity using the formula: F = Tr(sqrt(sqrt(rho)sigma sqrt(rho)))^2
            sqrt_recon = self.matrix_sqrt(recon)
            inner_term = sqrt_recon @ target @ sqrt_recon
            fidelity = torch.abs(torch.trace(self.matrix_sqrt(inner_term))) ** 2
            fidelities.append(fidelity.real)  # Take real part since fidelity is always real
        
        return torch.tensor(fidelities, device=self.device).mean()
    
    def reconstruction_loss_fn(self, reconstructed_states, target_states):
        """
        Compute reconstruction loss for quantum states.
        Handles both complex and real tensors.
        """
        # Convert to complex if not already
        if not reconstructed_states.is_complex():
            reconstructed_states = torch.complex(reconstructed_states, torch.zeros_like(reconstructed_states))
        if not target_states.is_complex():
            target_states = torch.complex(target_states, torch.zeros_like(target_states))
        
        # Compute MSE for real and imaginary parts
        loss_real = F.mse_loss(reconstructed_states.real, target_states.real)
        loss_imag = F.mse_loss(reconstructed_states.imag, target_states.imag)
        
        return loss_real + loss_imag
    
    def matrix_sqrt(self, matrix):
        """Helper function to compute matrix square root of a complex matrix."""
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        # Convert eigenvalues to complex before sqrt
        sqrt_eigenvalues = torch.sqrt(eigenvalues.to(torch.cfloat))
        # Create diagonal matrix of sqrt eigenvalues
        diag_sqrt = torch.diag_embed(sqrt_eigenvalues)
        # Compute matrix square root
        return eigenvectors @ diag_sqrt @ eigenvectors.T.conj()
    
    def log_progress(self, epoch, batch_idx, total_batches, loss, fidelity):
        """
        Log training progress to console and tensorboard.
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            total_batches: Total number of batches
            loss: Current loss value
            fidelity: Current fidelity value
        """
        # Calculate progress percentage
        progress = 100. * batch_idx / total_batches
        
        # Log to console
        print(f'Train Epoch: {epoch} [{batch_idx}/{total_batches} ({progress:.0f}%)]'
              f'\tLoss: {loss:.6f}'
              f'\tFidelity: {fidelity:.6f}')
        
        # Log to tensorboard
        step = epoch * total_batches + batch_idx
        self.writer.add_scalar('Batch/Loss', loss, step)
        self.writer.add_scalar('Batch/Fidelity', fidelity, step)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_recon_loss = 0
        total_syndrome_loss = 0
        total_loss = 0
        total_fidelity = 0
        n_batches = len(dataloader)
        
        for batch_idx, (measurements, states, noise_levels) in enumerate(dataloader):
            # Move data to device
            measurements = measurements.to(self.device)
            states = states.to(self.device)
            noise_levels = noise_levels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed_states, predicted_syndromes = self.model(measurements)
            
            # Generate target syndromes (assuming binary error detection per qubit)
            # For now, we'll use noise levels > 0.05 as indicator of error
            target_syndromes = (noise_levels > 0.05).float()
            # Expand target_syndromes to match predicted_syndromes shape
            target_syndromes = target_syndromes.unsqueeze(1).expand(-1, self.model.n_qubits)
            
            # Compute losses
            recon_loss = self.reconstruction_loss(reconstructed_states, states)
            syndrome_loss = self.syndrome_loss(predicted_syndromes, target_syndromes)
            
            # Total loss
            loss = recon_loss + syndrome_loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Compute fidelity
            fidelity = self.compute_fidelity(reconstructed_states, states)
            
            # Update metrics
            total_recon_loss += recon_loss.item()
            total_syndrome_loss += syndrome_loss.item()
            total_loss += loss.item()
            total_fidelity += fidelity.item()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                self.log_progress(epoch, batch_idx, n_batches, loss.item(), fidelity.item())
        
        # Average metrics
        metrics = {
            'reconstruction_loss': total_recon_loss / n_batches,
            'syndrome_loss': total_syndrome_loss / n_batches,
            'total_loss': total_loss / n_batches,
            'fidelity': total_fidelity / n_batches
        }
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        Args:
            dataloader: Validation data loader
            epoch: Current epoch number
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_recon_loss = 0
        total_syndrome_loss = 0
        total_fidelity = 0
        n_batches = len(dataloader)
        
        with torch.no_grad():
            for measurements, states, noise_levels in dataloader:
                # Move data to device
                measurements = measurements.to(self.device)
                states = states.to(self.device)
                noise_levels = noise_levels.to(self.device)
                
                # Forward pass
                reconstructed_states, predicted_syndromes = self.model(measurements)
                
                # Generate target syndromes (same as in training)
                target_syndromes = (noise_levels > 0.05).float()
                target_syndromes = target_syndromes.unsqueeze(1).expand(-1, self.model.n_qubits)
                
                # Compute losses
                recon_loss = self.reconstruction_loss(reconstructed_states, states)
                syndrome_loss = self.syndrome_loss(predicted_syndromes, target_syndromes)
                
                # Compute fidelity
                fidelity = self.compute_fidelity(reconstructed_states, states)
                
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_syndrome_loss += syndrome_loss.item()
                total_fidelity += fidelity.item()
        
        # Average metrics
        avg_recon_loss = total_recon_loss / n_batches
        avg_syndrome_loss = total_syndrome_loss / n_batches
        avg_fidelity = total_fidelity / n_batches
        
        # Log to tensorboard
        self.writer.add_scalar('Val/ReconstructionLoss', avg_recon_loss, epoch)
        self.writer.add_scalar('Val/SyndromeLoss', avg_syndrome_loss, epoch)
        self.writer.add_scalar('Val/Fidelity', avg_fidelity, epoch)
        
        return {
            'val_recon_loss': avg_recon_loss,
            'val_syndrome_loss': avg_syndrome_loss,
            'val_fidelity': avg_fidelity
        }
    
    def save_metrics(self, metrics):
        # Initialize the structure if file doesn't exist
        if not os.path.exists(self.metrics_file):
            metrics_data = {
                'train': {
                    'reconstruction_loss': [],
                    'syndrome_loss': [],
                    'fidelity': []
                },
                'val': {
                    'val_recon_loss': [],
                    'val_syndrome_loss': [],
                    'val_fidelity': []
                }
            }
        else:
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)

        # Append new metrics to respective lists
        metrics_data['train']['reconstruction_loss'].append(metrics['train']['reconstruction_loss'])
        metrics_data['train']['syndrome_loss'].append(metrics['train']['syndrome_loss'])
        metrics_data['train']['fidelity'].append(metrics['train']['fidelity'])
        
        metrics_data['val']['val_recon_loss'].append(metrics['val']['val_recon_loss'])
        metrics_data['val']['val_syndrome_loss'].append(metrics['val']['val_syndrome_loss'])
        metrics_data['val']['val_fidelity'].append(metrics['val']['val_fidelity'])

        # Save updated metrics back to the file
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Train the model for multiple epochs.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_fidelity = 0.0
        
        for epoch in range(n_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Train - Recon Loss: {train_metrics['reconstruction_loss']:.4f}, "
                  f"Syndrome Loss: {train_metrics['syndrome_loss']:.4f}, "
                  f"Fidelity: {train_metrics['fidelity']:.4f}")
            print(f"Val - Recon Loss: {val_metrics['val_recon_loss']:.4f}, "
                  f"Syndrome Loss: {val_metrics['val_syndrome_loss']:.4f}, "
                  f"Fidelity: {val_metrics['val_fidelity']:.4f}")
            
            # Save metrics
            metrics = {
                'train': {
                    'reconstruction_loss': train_metrics['reconstruction_loss'],
                    'syndrome_loss': train_metrics['syndrome_loss'],
                    'fidelity': train_metrics['fidelity']
                },
                'val': {
                    'val_recon_loss': val_metrics['val_recon_loss'],
                    'val_syndrome_loss': val_metrics['val_syndrome_loss'],
                    'val_fidelity': val_metrics['val_fidelity']
                }
            }
            self.save_metrics(metrics)
            
            # Save best model
            if val_metrics['val_fidelity'] > best_fidelity:
                best_fidelity = val_metrics['val_fidelity']
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_fidelity': best_fidelity,
                    },
                    os.path.join(checkpoint_dir, 'best_model.pt')
                )
        
        self.writer.close()

def main():
    # Parameters
    n_qubits = config.N_QUBITS
    measurement_dim = config.MEASUREMENT_DIM
    n_samples = config.N_SAMPLES
    batch_size = config.BATCH_SIZE
    n_epochs = config.N_EPOCHS
    
    # Generate dataset
    generator = QuantumStateGenerator(n_qubits)
    measurements, states, noise_levels = generator.generate_dataset(
        n_samples=n_samples,
        target_len=measurement_dim
    )
    
    # Split into train and validation
    split_idx = int(0.8 * n_samples)
    train_data = TensorDataset(
        measurements[:split_idx],
        states[:split_idx],
        noise_levels[:split_idx]
    )
    val_data = TensorDataset(
        measurements[split_idx:],
        states[split_idx:],
        noise_levels[split_idx:]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize trainer
    trainer = QECTrainer(
        n_qubits=n_qubits,
        measurement_dim=measurement_dim
    )
    
    # Train model
    trainer.train(train_loader, val_loader, n_epochs)

if __name__ == "__main__":
    main()
