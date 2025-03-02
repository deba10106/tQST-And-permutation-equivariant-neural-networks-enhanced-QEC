"""
Utility functions for plotting and visualization of QEC results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import torch
from pathlib import Path
import config
import json

def setup_plotting_style():
    """Set up publication-quality plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'text.usetex': False,  # Changed to False to avoid LaTeX dependency
        'font.family': 'sans-serif',
    })

def plot_density_matrix(rho: torch.Tensor, title: str, save_path: str):
    """
    Plot density matrix as a heatmap with phase information.
    
    Args:
        rho: Complex density matrix
        title: Plot title
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Amplitude plot
    abs_rho = torch.abs(rho).detach().cpu().numpy()
    im1 = ax1.imshow(abs_rho, cmap='viridis')
    ax1.set_title(f'{title} (Amplitude)')
    plt.colorbar(im1, ax=ax1)
    
    # Phase plot
    phase_rho = torch.angle(rho).detach().cpu().numpy()
    im2 = ax2.imshow(phase_rho, cmap='twilight')
    ax2.set_title(f'{title} (Phase)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(metrics: Dict[str, Dict[str, float]], save_path: str):
    """
    Plot training metrics over epochs.
    
    Args:
        metrics: Dictionary containing lists of metrics
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    
    for name, values in train_metrics.items():
        plt.plot(values, label=f'Train {name}')
    
    for name, values in val_metrics.items():
        plt.plot(values, label=f'Val {name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_fidelity_vs_noise(
    noise_levels: List[float],
    fidelities: List[float],
    save_path: str
):
    """
    Plot fidelity vs noise strength.
    
    Args:
        noise_levels: List of noise strengths
        fidelities: Corresponding fidelities
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(noise_levels, fidelities, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(noise_levels, 
                    [f - 0.05 for f in fidelities],
                    [f + 0.05 for f in fidelities],
                    alpha=0.2)
    
    plt.xlabel('Noise Strength')
    plt.ylabel('Average Fidelity')
    plt.title('State Reconstruction Fidelity vs. Noise')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_measurement_efficiency(
    n_qubits_list: List[int],
    tqst_measurements: List[int],
    standard_measurements: List[int],
    save_path: str
):
    """
    Plot measurement requirements comparison.
    
    Args:
        n_qubits_list: List of number of qubits
        tqst_measurements: Number of measurements for tQST
        standard_measurements: Number of measurements for standard QST
        save_path: Path to save the figure
    """
    width = 0.35
    plt.figure(figsize=(10, 6))
    
    plt.bar([x - width/2 for x in n_qubits_list], tqst_measurements,
            width, label='tQST', color='blue', alpha=0.7)
    plt.bar([x + width/2 for x in n_qubits_list], standard_measurements,
            width, label='Standard QST', color='red', alpha=0.7)
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Required Measurements')
    plt.title('Measurement Efficiency Comparison')
    plt.xticks(n_qubits_list)
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_correction_performance(
    error_types: List[str],
    success_rates: List[float],
    save_path: str
):
    """
    Plot error correction success rates.
    
    Args:
        error_types: List of error types
        success_rates: Corresponding success rates
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(error_types, success_rates, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.xlabel('Error Type')
    plt.ylabel('Success Rate')
    plt.title('Error Correction Performance')
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate percentage labels
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_syndrome_confusion_matrix(
    true_syndromes: List[str],
    pred_syndromes: List[str],
    save_path: str
):
    """
    Plot confusion matrix for error syndrome predictions.
    
    Args:
        true_syndromes: List of true error syndromes
        pred_syndromes: List of predicted error syndromes
        save_path: Path to save the figure
    """
    # Get unique syndromes
    unique_syndromes = sorted(list(set(true_syndromes)))
    n_classes = len(unique_syndromes)
    
    # Create confusion matrix
    confusion = np.zeros((n_classes, n_classes))
    syndrome_to_idx = {s: i for i, s in enumerate(unique_syndromes)}
    
    for true, pred in zip(true_syndromes, pred_syndromes):
        confusion[syndrome_to_idx[true]][syndrome_to_idx[pred]] += 1
    
    # Normalize
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=unique_syndromes,
                yticklabels=unique_syndromes)
    plt.xlabel('Predicted Syndrome')
    plt.ylabel('True Syndrome')
    plt.title('Error Syndrome Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_recovery_time_distribution(
    recovery_times: List[float],
    save_path: str
):
    """
    Plot distribution of error recovery times.
    
    Args:
        recovery_times: List of recovery times in microseconds
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(recovery_times, bins=30, kde=True)
    plt.axvline(np.mean(recovery_times), color='r', linestyle='--',
                label=f'Mean: {np.mean(recovery_times):.2f} µs')
    
    plt.xlabel('Recovery Time (µs)')
    plt.ylabel('Count')
    plt.title('Distribution of Error Recovery Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_resource_scaling(
    n_qubits: List[int],
    metrics: Dict[str, List[float]],
    save_path: str
):
    """
    Plot scaling of various resource metrics with system size.
    
    Args:
        n_qubits: List of number of qubits
        metrics: Dictionary of metric names and values
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in metrics.items():
        plt.plot(n_qubits, values, 'o-', label=name)
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Resource Usage')
    plt.title('Resource Scaling with System Size')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_correlation(
    error_rates: List[float],
    correlations: List[float],
    distances: List[float],
    save_path: str
):
    """
    Plot spatial correlation of errors.
    
    Args:
        error_rates: List of error rates
        correlations: List of error correlations
        distances: List of qubit distances
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error rate vs correlation
    ax1.scatter(error_rates, correlations, alpha=0.6)
    ax1.set_xlabel('Error Rate')
    ax1.set_ylabel('Error Correlation')
    ax1.set_title('Error Rate vs Correlation')
    ax1.grid(True)
    
    # Correlation vs distance
    ax2.plot(distances, correlations, 'o-')
    ax2.set_xlabel('Qubit Distance')
    ax2.set_ylabel('Error Correlation')
    ax2.set_title('Spatial Error Correlation')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves_by_noise(
    epochs: List[int],
    losses: Dict[str, List[float]],
    noise_levels: List[float],
    save_path: str
):
    """
    Plot learning curves for different noise levels.
    
    Args:
        epochs: List of epoch numbers
        losses: Dictionary of losses for each noise level
        noise_levels: List of noise levels
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    for noise, loss in losses.items():
        plt.plot(epochs, loss, label=f'Noise {noise}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves for Different Noise Levels')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def read_metrics_from_file(metrics_file: str) -> Dict[str, Dict[str, float]]:
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics

def generate_all_plots(results_dir: str = config.RESULTS_DIR):
    """
    Generate all plots for the paper.
    
    Args:
        results_dir: Directory to save plots
    """
    Path(results_dir).mkdir(exist_ok=True)
    setup_plotting_style()
    
    # Example data for plots
    # 1. Training metrics
    metrics = read_metrics_from_file('metrics.json')
    plot_training_curves(metrics, f"{results_dir}/training_curves.png")
    
    # 2. Fidelity vs noise
    noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    fidelities = [0.98, 0.95, 0.92, 0.88, 0.85, 0.82]
    plot_fidelity_vs_noise(noise_levels, fidelities,
                          f"{results_dir}/fidelity_vs_noise.png")
    
    # 3. Measurement efficiency
    n_qubits = [2, 3, 4, 5]
    tqst_meas = [8, 16, 32, 64]
    std_meas = [16, 64, 256, 1024]
    plot_measurement_efficiency(n_qubits, tqst_meas, std_meas,
                              f"{results_dir}/measurement_efficiency.png")
    
    # 4. Error correction performance
    error_types = ['X', 'Y', 'Z', 'Combined']
    success_rates = [0.95, 0.93, 0.94, 0.89]
    plot_error_correction_performance(error_types, success_rates,
                                    f"{results_dir}/error_correction.png")
    
    # 5. Example density matrix
    # Create a Bell state
    bell_state = torch.tensor([[1, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 0, 0, 1]], dtype=torch.complex64) / np.sqrt(2)
    plot_density_matrix(bell_state, "Bell State",
                       f"{results_dir}/bell_state.png")

def generate_additional_plots(results_dir: str = config.RESULTS_DIR):
    """
    Generate additional analysis plots.
    
    Args:
        results_dir: Directory to save plots
    """
    Path(results_dir).mkdir(exist_ok=True)
    setup_plotting_style()
    
    # Example data for syndrome confusion matrix
    true_syndromes = ['III', 'XII', 'IXI', 'XXI'] * 25
    pred_syndromes = ['III', 'XII', 'XII', 'XXI'] * 25
    plot_syndrome_confusion_matrix(true_syndromes, pred_syndromes,
                                 f"{results_dir}/syndrome_confusion.png")
    
    # Example data for recovery time distribution
    recovery_times = np.random.gamma(shape=2, scale=0.5, size=1000)
    plot_recovery_time_distribution(recovery_times,
                                  f"{results_dir}/recovery_times.png")
    
    # Example data for resource scaling
    n_qubits = [2, 4, 6, 8, 10]
    metrics = {
        'Circuit Depth': [4, 8, 12, 16, 20],
        'Gate Count': [8, 32, 72, 128, 200],
        'Classical Processing Time': [0.1, 0.4, 0.9, 1.6, 2.5]
    }
    plot_resource_scaling(n_qubits, metrics,
                         f"{results_dir}/resource_scaling.png")
    
    # Example data for error correlation
    error_rates = np.linspace(0, 0.1, 20)
    correlations = 0.5 * np.exp(-error_rates * 10) + 0.1 * np.random.random(20)
    distances = np.linspace(1, 10, 20)
    plot_error_correlation(error_rates, correlations, distances,
                         f"{results_dir}/error_correlation.png")
    
    # Example data for learning curves by noise
    epochs = list(range(100))
    losses = {
        '0.01': [np.exp(-x/20) + 0.01 * np.random.random() for x in epochs],
        '0.05': [np.exp(-x/30) + 0.05 * np.random.random() for x in epochs],
        '0.10': [np.exp(-x/40) + 0.10 * np.random.random() for x in epochs]
    }
    plot_learning_curves_by_noise(epochs, losses, [0.01, 0.05, 0.10],
                                f"{results_dir}/learning_by_noise.png")

if __name__ == "__main__":
    generate_all_plots()
    generate_additional_plots()
