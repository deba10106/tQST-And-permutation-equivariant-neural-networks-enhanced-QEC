# File: replicate.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Helper Functions for Density Matrices and tQST Measurements
# -------------------------------
def random_pure_density(D):
    """Generate a random pure density matrix of dimension D."""
    real = torch.randn(D)
    imag = torch.randn(D)
    psi = torch.complex(real, imag)
    psi = psi / torch.norm(psi)
    rho = torch.outer(psi, psi.conj())
    rho = (rho + rho.conj().t()) / 2  # ensure Hermiticity
    return rho

def random_mixed_density(D, rank):
    """Generate a random mixed density matrix of dimension D with given rank."""
    rhos = [random_pure_density(D) for _ in range(rank)]
    rho = sum(rhos) / rank
    return rho

def impose_sparsity(rho, z):
    """Set z randomly chosen diagonal elements (and associated rows/cols) to zero and renormalize."""
    D = rho.shape[0]
    indices = np.random.choice(D, size=z, replace=False)
    rho_new = rho.clone()
    for i in indices:
        rho_new[i, :] = 0.
        rho_new[:, i] = 0.
    tr = torch.real(torch.trace(rho_new))
    if tr > 0:
        rho_new = rho_new / tr
    return rho_new

def generate_density_matrix(n_qubits, M=10):
    """
    Generate one density matrix for an n-qubit system.
    For half the samples, use a pure state; for the other half, generate a mixed state with
    a random number of vanishing diagonal elements.
    """
    D = 2 ** n_qubits
    if np.random.rand() < 0.5:
        rho = random_pure_density(D)
    else:
        z = np.random.randint(0, D-1)  # between 0 and D-2
        r_min = 2
        r_max = max(r_min, D - z)
        r = np.random.randint(r_min, r_max + 1)
        rho = random_mixed_density(D, rank=r)
        if z > 0:
            rho = impose_sparsity(rho, z)
    return rho

def compute_gini(diag):
    """Compute the Gini index of a non-negative vector."""
    diag_sorted, _ = torch.sort(diag)
    N = diag_sorted.numel()
    s = torch.sum(diag_sorted) + 1e-8
    idx = torch.arange(1, N+1, dtype=diag_sorted.dtype, device=diag_sorted.device)
    gini = (2 * torch.sum(idx * diag_sorted) / (N * s)) - ((N + 1) / N)
    return gini

def compute_threshold(rho):
    """Compute the threshold t using the Gini index of the diagonal."""
    diag = torch.real(torch.diag(rho))
    D = diag.numel()
    gi = compute_gini(diag)
    t = gi / (D - 1)
    return t.item()

def tQST_measurements(rho, target_len):
    """
    Produce a 1D measurement vector from rho according to the tQST protocol:
      - Always record the diagonal entries.
      - For each off-diagonal ρ_ij (i<j), if sqrt(ρ_ii*ρ_jj) >= t record [Re, Im];
        otherwise record a mock value 2.0.
    Then pad or truncate to target_len.
    """
    D = rho.shape[0]
    t = compute_threshold(rho)
    meas = []
    diag = torch.real(torch.diag(rho))
    meas.extend(diag.tolist())
    for i in range(D):
        for j in range(i+1, D):
            thresh_val = np.sqrt(diag[i].item() * diag[j].item())
            if thresh_val >= t:
                meas.append(torch.real(rho[i,j]).item())
                meas.append(torch.imag(rho[i,j]).item())
            else:
                meas.extend([2.0, 2.0])
    if len(meas) >= target_len:
        meas = meas[:target_len]
    else:
        meas.extend([2.0] * (target_len - len(meas)))
    return torch.tensor(meas, dtype=torch.float32)

def density_to_flat(rho):
    """
    Flatten the density matrix: first the D diagonal elements, then each off-diagonal element
    (store real and imaginary parts for i<j).
    """
    D = rho.shape[0]
    flat = list(torch.real(torch.diag(rho)).tolist())
    for i in range(D):
        for j in range(i+1, D):
            flat.append(torch.real(rho[i,j]).item())
            flat.append(torch.imag(rho[i,j]).item())
    return torch.tensor(flat, dtype=torch.float32)

def compute_purity(rho):
    """Compute the purity P = Tr(rho^2)."""
    return torch.real(torch.trace(rho @ rho)).item()

# -------------------------------
# 2. PyTorch Dataset
# -------------------------------
class QuantumStateDataset(Dataset):
    def __init__(self, n_qubits, num_samples, target_meas_len):
        self.n_qubits = n_qubits
        self.num_samples = num_samples
        self.target_meas_len = target_meas_len
        self.data = []
        for _ in range(num_samples):
            rho = generate_density_matrix(n_qubits, M=10)
            x = tQST_measurements(rho, target_meas_len)
            y_tomo = density_to_flat(rho)
            y_purity = torch.tensor([compute_purity(rho)], dtype=torch.float32)
            self.data.append((x, y_tomo, y_purity))
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx]

# -------------------------------
# 3. Model Architectures
# -------------------------------
# (a) MLP for density matrix reconstruction (tomography)
class MLP_Tomography(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32]):
        super(MLP_Tomography, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# (b) MLP for purity estimation (scalar output)
class MLP_Purity(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32]):
        super(MLP_Purity, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# (c) A simple permutation-equivariant linear layer
class PELinear(nn.Module):
    def __init__(self, features):
        super(PELinear, self).__init__()
        self.features = features
        self.weight = nn.Parameter(torch.randn(features, features))
        self.bias = nn.Parameter(torch.zeros(features))
    def forward(self, x):
        # Symmetrize the weight matrix to enforce permutation equivariance.
        sym_weight = (self.weight + self.weight.t()) / 2
        return torch.matmul(x, sym_weight) + self.bias

# (d) PEMLP for purity estimation using PELinear layers
class PEMLP_Purity(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super(PEMLP_Purity, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.pe_layers = nn.ModuleList([PELinear(hidden_dim) for _ in range(num_layers)])
        self.output_linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out = self.relu(self.input_linear(x))
        for layer in self.pe_layers:
            out = self.relu(layer(out))
        return self.output_linear(out)

# -------------------------------
# 4. Training and Evaluation Functions
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch in dataloader:
            x, y, _ = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"Tomography Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def train_model_scalar(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch in dataloader:
            x, _, purity = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, purity)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"Purity Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# For evaluation, we compute a proxy fidelity (using normalized MSE)
def compute_proxy_fidelity(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2).item()
    return np.exp(-mse)

# -------------------------------
# 5. Plotting Functions
# -------------------------------
def plot_density_matrix(flat_rho, D, filename):
    """
    Plot a density matrix as a heatmap matching the paper's style.
    The flat vector is reshaped to (D, D).
    """
    # Convert flat vector to matrix form
    rho = torch.zeros((D, D), dtype=torch.complex64)
    idx = 0
    for i in range(D):
        rho[i,i] = flat_rho[idx]
        idx += 1
    for i in range(D):
        for j in range(i+1, D):
            rho[i,j] = flat_rho[idx] + 1j*flat_rho[idx+1]
            rho[j,i] = flat_rho[idx] - 1j*flat_rho[idx+1]
            idx += 2

    # Plot absolute values
    plt.figure(figsize=(6, 5))
    abs_vals = torch.abs(rho).numpy()
    plt.imshow(abs_vals, cmap='viridis', aspect='equal')
    plt.colorbar(label='Magnitude')
    
    # Add grid lines
    for i in range(D+1):
        plt.axhline(y=i-0.5, color='white', linewidth=0.5)
        plt.axvline(x=i-0.5, color='white', linewidth=0.5)
    
    plt.title('Absolute Values')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.tight_layout()
    plt.savefig(os.path.join('plots_rep_2502.15305v1', filename + '_abs.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_phase_density_matrix(flat_rho, D, filename):
    """
    Plot the phase (angle) of the density matrix elements matching the paper's style.
    """
    # Convert flat vector to matrix form
    rho = torch.zeros((D, D), dtype=torch.complex64)
    idx = 0
    for i in range(D):
        rho[i,i] = flat_rho[idx]
        idx += 1
    for i in range(D):
        for j in range(i+1, D):
            rho[i,j] = flat_rho[idx] + 1j*flat_rho[idx+1]
            rho[j,i] = flat_rho[idx] - 1j*flat_rho[idx+1]
            idx += 2

    # Plot phases
    plt.figure(figsize=(6, 5))
    phases = torch.angle(rho).numpy()
    plt.imshow(phases, cmap='twilight', aspect='equal', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase (radians)')
    
    # Add grid lines
    for i in range(D+1):
        plt.axhline(y=i-0.5, color='white', linewidth=0.5)
        plt.axvline(x=i-0.5, color='white', linewidth=0.5)
    
    plt.title('Phase')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.tight_layout()
    plt.savefig(os.path.join('plots_rep_2502.15305v1', filename + '_phase.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fidelity_vs_noise(noise_levels, fidelities, filename):
    """
    Plot a line graph of fidelity vs noise strength matching the paper's style.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, fidelities, 'bo-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Noise Strength', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Reconstruction Fidelity vs. Noise Strength', fontsize=14)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join('plots_rep_2502.15305v1', filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_purity_scatter(true_purities, pred_purities, filename):
    """
    Scatter plot comparing true and predicted purity values matching the paper's style.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Create scatter plot
    plt.scatter(true_purities, pred_purities, c='blue', alpha=0.6, s=50)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('True Purity', fontsize=12)
    plt.ylabel('Predicted Purity', fontsize=12)
    plt.title('Purity Estimation Performance', fontsize=14)
    
    # Set equal aspect ratio and limits
    plt.axis('square')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots_rep_2502.15305v1', filename), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------
# 6. Main Experiment Routine
# -------------------------------
if __name__ == '__main__':
    # Create output directory for figures if not exists
    os.makedirs("plots_rep_2502.15305v1", exist_ok=True)
    
    # Clean up old figures
    for file in os.listdir("plots_rep_2502.15305v1"):
        if file.endswith(".png"):
            os.remove(os.path.join("plots_rep_2502.15305v1", file))
    
    print("Cleaned up old figures. Starting experiments...")
    
    # Settings
    n_qubits_2 = 2
    n_qubits_4 = 4
    target_meas_len_2 = 4 * n_qubits_2   # e.g., 8 for 2 qubits
    target_meas_len_4 = 4 * n_qubits_4   # e.g., 16 for 4 qubits

    num_samples = 1000  # For demonstration; use larger numbers for full replication

    # Create datasets
    dataset2 = QuantumStateDataset(n_qubits_2, num_samples, target_meas_len_2)
    dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)

    # For tomography on 2 qubits: output dimension = D^2, D=2^n_qubits_2
    D2 = 2 ** n_qubits_2
    target_tomo_dim_2 = D2 * D2

    print("Training MLP for 2-qubit tomography...")
    mlp_tomo_2 = MLP_Tomography(input_dim=target_meas_len_2, output_dim=target_tomo_dim_2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_tomo_2.parameters(), lr=1e-4)
    train_model(mlp_tomo_2, dataloader2, criterion, optimizer, num_epochs=50)

    # Save reconstruction plot for a Bell state.
    # Here we manually construct the Bell state density matrix:
    bell_state = (1/np.sqrt(2)) * torch.tensor([1,0,0,-1], dtype=torch.complex64)
    rho_bell = torch.outer(bell_state, bell_state.conj())
    flat_bell = density_to_flat(rho_bell)
    plot_density_matrix(flat_bell, D2, "Fig1_BellState_Reconstruction")
    plot_phase_density_matrix(flat_bell, D2, "Fig1_BellState_Reconstruction")
    print("Saved Bell state reconstruction plots as Fig1_BellState_Reconstruction_abs.png and phase.")

    # For purity estimation (2 qubits) using both standard MLP and PEMLP
    print("Training MLP for 2-qubit purity estimation...")
    mlp_purity_2 = MLP_Purity(input_dim=target_meas_len_2)
    criterion_scalar = nn.MSELoss()
    optimizer_scalar = optim.Adam(mlp_purity_2.parameters(), lr=1e-4)
    train_model_scalar(mlp_purity_2, dataloader2, criterion_scalar, optimizer_scalar, num_epochs=50)

    print("Training PEMLP for 2-qubit purity estimation...")
    pemlp_purity_2 = PEMLP_Purity(input_dim=target_meas_len_2, hidden_dim=32, num_layers=2)
    optimizer_pe = optim.Adam(pemlp_purity_2.parameters(), lr=1e-4)
    train_model_scalar(pemlp_purity_2, dataloader2, criterion_scalar, optimizer_pe, num_epochs=50)

    # Evaluate purity estimation on the 2-qubit dataset
    true_purities = []
    pred_purities = []
    mlp_purity_2.eval()
    with torch.no_grad():
        for (x, _, purity) in dataset2:
            pred = mlp_purity_2(x.unsqueeze(0))
            true_purities.append(purity.item())
            pred_purities.append(pred.item())
    plot_purity_scatter(true_purities, pred_purities, "Fig4_Purity_Estimation")
    print("Saved purity estimation scatter plot as Fig4_Purity_Estimation.png")

    # Simulate noise: For a set of noise strengths, add noise to the measurement outcomes and compute a proxy fidelity.
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    fidelities = []
    mlp_tomo_2.eval()
    for p in noise_levels:
        fid_list = []
        for (x, y_true, _) in dataset2:
            # Apply depolarizing noise: add Gaussian noise scaled by p.
            noise = torch.randn_like(x) * p
            x_noisy = x + noise
            with torch.no_grad():
                y_pred = mlp_tomo_2(x_noisy.unsqueeze(0)).squeeze(0)
            fid_list.append(compute_proxy_fidelity(y_true, y_pred))
        fidelities.append(np.mean(fid_list))
    plot_fidelity_vs_noise(noise_levels, fidelities, "Fig3_Fidelity_vs_Noise")
    print("Saved fidelity vs noise plot as Fig3_Fidelity_vs_Noise.png")

    # For 4-qubit tomography: similar training but with a larger output dimension.
    dataset4 = QuantumStateDataset(n_qubits_4, num_samples, target_meas_len_4)
    dataloader4 = DataLoader(dataset4, batch_size=32, shuffle=True)
    D4 = 2 ** n_qubits_4
    target_tomo_dim_4 = D4 * D4

    print("Training MLP for 4-qubit tomography...")
    mlp_tomo_4 = MLP_Tomography(input_dim=target_meas_len_4, output_dim=target_tomo_dim_4, hidden_dims=[64, 512])
    optimizer4 = optim.Adam(mlp_tomo_4.parameters(), lr=1e-4)
    train_model(mlp_tomo_4, dataloader4, criterion, optimizer4, num_epochs=50)

    # Save a 4-qubit density matrix reconstruction plot.
    # For demonstration, take the first sample from dataset4.
    sample = dataset4[0]
    x_sample, y_sample, _ = sample
    with torch.no_grad():
        y_recon = mlp_tomo_4(x_sample.unsqueeze(0)).squeeze(0)
    plot_density_matrix(y_recon, D4, "Fig2_4Qubits_Reconstruction")
    plot_phase_density_matrix(y_recon, D4, "Fig2_4Qubits_Reconstruction")
    print("Saved 4-qubit reconstruction plots as Fig2_4Qubits_Reconstruction_abs.png and phase.")

    print("Replication complete. All plots are saved in the 'plots' directory.")
