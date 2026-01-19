#!/bin/bash
#===============================================================================
# Setup PushWorld PLR Environment on GPU Cluster
#===============================================================================
#
# Usage (can run from login node - GPU not needed for installation):
#   ./setup_cluster.sh              # Auto-detect CUDA version
#   ./setup_cluster.sh --cuda=12    # Force CUDA 12
#   ./setup_cluster.sh --cuda=11    # Force CUDA 11
#
# To verify GPU works, get an interactive GPU session:
#   srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
#   source .venv/bin/activate
#   python -c "import jax; print(jax.devices())"
#
#===============================================================================

set -e

CUDA_VERSION=""
FORCE_REINSTALL=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --cuda=*)       CUDA_VERSION="${arg#*=}" ;;
        --reinstall)    FORCE_REINSTALL=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda=VERSION    CUDA version (11 or 12)"
            echo "  --reinstall       Force reinstall even if venv exists"
            echo "  --help            Show this help"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "PushWorld PLR Cluster Setup"
echo "============================================================"

# Detect CUDA version if not specified
if [ -z "$CUDA_VERSION" ]; then
    echo "Detecting CUDA version..."
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
        CUDA_VERSION=$NVCC_VERSION
        echo "  Detected CUDA $CUDA_VERSION from nvcc"
    elif command -v nvidia-smi &> /dev/null; then
        SMI_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p')
        CUDA_VERSION=$SMI_VERSION
        echo "  Detected CUDA $CUDA_VERSION from nvidia-smi"
    else
        echo "  Could not detect CUDA version, defaulting to 12"
        CUDA_VERSION=12
    fi
fi

# Validate CUDA version
if [ "$CUDA_VERSION" != "11" ] && [ "$CUDA_VERSION" != "12" ]; then
    echo "Warning: Unexpected CUDA version $CUDA_VERSION, using 12"
    CUDA_VERSION=12
fi

echo "Using CUDA version: $CUDA_VERSION"
echo ""

# Check if venv exists
if [ -d ".venv" ] && [ "$FORCE_REINSTALL" = false ]; then
    echo "Virtual environment already exists."
    echo "Use --reinstall to recreate it."
    echo ""
    echo "To activate: source .venv/bin/activate"
    exit 0
fi

# Remove old venv if reinstalling
if [ -d ".venv" ] && [ "$FORCE_REINSTALL" = true ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Python: $(which python) ($(python --version))"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install JAX with CUDA support
echo ""
echo "Installing JAX with CUDA $CUDA_VERSION support..."
pip install -U "jax[cuda${CUDA_VERSION}]"

# Verify JAX installation (GPU check only if available)
echo ""
echo "Verifying JAX installation..."
python -c "import jax; print(f'JAX version: {jax.__version__}')"

# Check if we're on a node with GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected, verifying JAX GPU support..."
    python -c "import jax; print(f'Devices: {jax.devices()}')"
else
    echo "⚠️  No GPU on this node (login node?). JAX will use GPU when job runs on compute node."
    echo "   To verify GPU support, run an interactive GPU job:"
    echo "   srun --partition=gpu --gres=gpu:1 --pty bash"
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install flax>=0.8.0 chex>=0.1.8 optax>=0.1.7 orbax-checkpoint>=0.5.0 distrax>=0.1.5
pip install gymnax>=0.0.6
pip install hydra-core>=1.3.0 omegaconf>=2.3.0 "wandb[media]>=0.16.0"
pip install numpy>=1.24.0 pillow>=10.0.0 imageio>=2.31.0 tqdm>=4.65.0 moviepy

# Install the package in editable mode
echo ""
echo "Installing pushworld-experiments package..."
pip install -e .

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run training:"
echo "  ./submit_plr.sh"
echo ""
echo "To test locally:"
echo "  python experiments/pushworld_plr.py num_updates=100 use_wandb=false"
echo ""
