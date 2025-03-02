# Configuration file for Quantum Error Correction parameters

# Number of qubits
N_QUBITS = 4

# Measurement dimensions
MEASUREMENT_DIM = 128

# Noise levels
NOISE_LEVELS = (0.0, 0.1)

# Hidden dimensions for PEMLP
HIDDEN_DIMS = [128, 128, 128]

# Dropout rate
DROPOUT_RATE = 0.1

# Training parameters
N_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Number of samples
N_SAMPLES = 10000

# Directory for saving results
RESULTS_DIR = "results"

# IBM Quantum Configuration
IBMQ_API_TOKEN = None  # Add your IBM Quantum API token here
IBMQ_HUB = 'ibm-q'    # IBM Quantum hub name
IBMQ_GROUP = 'open'    # IBM Quantum group name
IBMQ_PROJECT = 'main'  # IBM Quantum project name
IBMQ_BACKEND = 'ibmq_jakarta'  # Default backend to use
USE_REAL_DEVICE = False  # Set to True to use real quantum device, False for simulator

# Define other parameters as needed for your project
# ...

# Function to print configuration (optional)
def print_config():
    print(f"Number of Qubits: {N_QUBITS}")
    print(f"Measurement Dimension: {MEASUREMENT_DIM}")
    print(f"Noise Levels: {NOISE_LEVELS}")
    print(f"Hidden Dimensions: {HIDDEN_DIMS}")
    print(f"Dropout Rate: {DROPOUT_RATE}")
    print(f"Number of Epochs: {N_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Samples: {N_SAMPLES}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("\nIBM Quantum Configuration:")
    print(f"IBMQ Hub: {IBMQ_HUB}")
    print(f"IBMQ Group: {IBMQ_GROUP}")
    print(f"IBMQ Project: {IBMQ_PROJECT}")
    print(f"IBMQ Backend: {IBMQ_BACKEND}")
    print(f"Using Real Device: {USE_REAL_DEVICE}")
