import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pemlp_qec import PEMLP_QEC
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, Operator
from qiskit.primitives import Sampler

def generate_training_data(num_samples=1000, noise_strength=0.1):
    """Generate synthetic training data with various quantum states and noise"""
    X_train = []
    Y_train = []
    
    # Quantum simulator
    simulator = Aer.get_backend('statevector_simulator')
    
    for _ in range(num_samples):
        # Generate random pure state
        qc = QuantumCircuit(2)
        init_state = random_statevector(2**2)
        qc.initialize(init_state, [0, 1])
        
        # Get ideal state
        job = simulator.run(qc)
        ideal_state = np.array(job.result().get_statevector())
        
        # Add noise
        noise_matrix = np.eye(4) + noise_strength * np.random.randn(4, 4)
        noise_matrix = (noise_matrix + noise_matrix.conj().T) / 2  # Make Hermitian
        noisy_state = noise_matrix @ ideal_state
        
        # Convert to real representation
        X_train.append(np.concatenate([noisy_state.real, noisy_state.imag]))
        Y_train.append(np.concatenate([ideal_state.real, ideal_state.imag]))
    
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)

def train_model(model, X_train, Y_train, epochs=100, batch_size=32, lr=0.001):
    """Train the PEMLP model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    n_batches = len(X_train) // batch_size
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_Y = Y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = model.permutation_equivariant_loss(output, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            torch.save(model.state_dict(), 'models/best_pemlp_qec.pth')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

def main():
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    X_train, Y_train = generate_training_data()
    
    # Initialize model
    model = PEMLP_QEC(input_dim=8, hidden_dim=64, output_dim=8)  # 8 = 4 real + 4 imag components
    
    # Train model
    print("Training model...")
    train_model(model, X_train, Y_train)
    print("Training complete! Model saved to models/best_pemlp_qec.pth")

if __name__ == "__main__":
    main()
