#!/bin/bash
#===============================================================================
# Submit PushWorld PLR Training Job to SLURM
#===============================================================================
#
# Usage:
#   ./submit_plr.sh                              # Use defaults (H100-96)
#   ./submit_plr.sh --gpu=a100-80                # Use A100 80GB
#   ./submit_plr.sh --gpu=h100-96 --updates=50000 # H100 with 50k updates
#   ./submit_plr.sh --dry-run                    # Show what would be submitted
#   ./submit_plr.sh --config=pushworld_plr_benchmark  # Use benchmark config
#
# GPU Options:
#   h200       - NVIDIA H200 (3h limit on gpu partition!)
#   h100-96    - NVIDIA H100 96GB (default, recommended for long training)
#   h100-47    - NVIDIA H100 47GB
#   a100-80    - NVIDIA A100 80GB
#   a100-40    - NVIDIA A100 40GB
#   nv         - NVIDIA V100/Titan/T4
#
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# Default Configuration
#-------------------------------------------------------------------------------
GPU_TYPE="h100-96"  # Default to H100-96 (available on gpu-long)
PARTITION=""        # Auto-set based on GPU type
CONFIG="pushworld_plr"  # Hydra config name (pushworld_plr or pushworld_plr_benchmark)
SCRIPT="pushworld_plr"  # Script to run (pushworld_plr or pushworld_plr_benchmark)
NUM_UPDATES=""      # Will use config default if empty
SEED=""             # Will use config default if empty
RUN_NAME=""         # Will be auto-generated if empty
TIME="24:00:00"
DRY_RUN=false
USE_WANDB=true
WANDB_PROJECT="pushworld-plr"
WANDB_ENTITY=""
EXTRA_OVERRIDES=""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE          GPU type (see below)"
    echo ""
    echo "Available GPUs:"
    echo "  h200       NVIDIA H200 141GB    (max 4 per node, 3h limit!)"
    echo "  h100-96    NVIDIA H100 96GB     (max 2 per node, recommended)"
    echo "  h100-47    NVIDIA H100 47GB     (max 4 per node)"
    echo "  a100-80    NVIDIA A100 80GB     (max 1 per node)"
    echo "  a100-40    NVIDIA A100 40GB     (max 2 per node)"
    echo "  nv         V100/Titan/T4        (max 2 per node)"
    echo ""
    echo "Training Options:"
    echo "  --benchmark         Use benchmark training (shortcut for --script/--config)"
    echo "  --script=NAME       Script to run: pushworld_plr or pushworld_plr_benchmark"
    echo "  --config=NAME       Hydra config name (default: matches script)"
    echo "  --updates=N         Number of training updates (default: from config)"
    echo "  --seed=N            Random seed (default: from config)"
    echo "  --run-name=NAME     Run name for checkpoints/wandb"
    echo "  --no-wandb          Disable wandb logging"
    echo "  --wandb-project=NAME  Wandb project name (default: pushworld-plr)"
    echo "  --wandb-entity=NAME   Wandb entity/team"
    echo ""
    echo "SLURM Options:"
    echo "  --partition=NAME    SLURM partition (auto-detected if not set)"
    echo "  --time=HH:MM:SS     Time limit (default: 24:00:00)"
    echo ""
    echo "Other:"
    echo "  --dry-run           Show generated script without submitting"
    echo "  --help              Show this help"
    echo ""
    echo "Hydra Overrides:"
    echo "  Any additional arguments are passed to Hydra as config overrides."
    echo "  Example: $0 lr=2e-4 num_train_envs=64"
    exit 0
}

for arg in "$@"; do
    case $arg in
        --gpu=*)            GPU_TYPE="${arg#*=}" ;;
        --partition=*)      PARTITION="${arg#*=}" ;;
        --config=*)         CONFIG="${arg#*=}" ;;
        --script=*)         SCRIPT="${arg#*=}" ;;
        --benchmark)        SCRIPT="pushworld_plr_benchmark"; CONFIG="pushworld_plr_benchmark" ;;
        --updates=*)        NUM_UPDATES="${arg#*=}" ;;
        --seed=*)           SEED="${arg#*=}" ;;
        --run-name=*)       RUN_NAME="${arg#*=}" ;;
        --time=*)           TIME="${arg#*=}" ;;
        --no-wandb)         USE_WANDB=false ;;
        --wandb-project=*)  WANDB_PROJECT="${arg#*=}" ;;
        --wandb-entity=*)   WANDB_ENTITY="${arg#*=}" ;;
        --dry-run)          DRY_RUN=true ;;
        --help|-h)          show_help ;;
        *)                  EXTRA_OVERRIDES="$EXTRA_OVERRIDES $arg" ;;
    esac
done

#-------------------------------------------------------------------------------
# Map GPU Type to SLURM Configuration
#-------------------------------------------------------------------------------
case $GPU_TYPE in
    h200)
        SLURM_GRES="gpu:h200-141:1"
        MEM="256G"
        DEFAULT_PARTITION="gpu"
        MAX_TIME="3:00:00"
        echo "⚠️  Warning: H200 only available on 'gpu' partition with 3-hour limit!"
        echo "   For longer training, use --gpu=h100-96 or --gpu=h100-47"
        ;;
    h100-96|h100)
        SLURM_GRES="gpu:h100-96:1"
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    h100-47)
        SLURM_GRES="gpu:h100-47:1"
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    a100-80)
        SLURM_GRES="gpu:a100-80:1"
        MEM="128G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    a100-40|a100)
        SLURM_GRES="gpu:a100-40:1"
        MEM="64G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    nv|v100|titanv|titanrtx|t4)
        SLURM_GRES="gpu:nv:1"
        MEM="32G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    *)
        echo "Error: Unknown GPU type: $GPU_TYPE"
        echo ""
        echo "Available GPUs:"
        echo "  h100-96  - H100 96GB (recommended for long training)"
        echo "  h100-47  - H100 47GB"
        echo "  a100-80  - A100 80GB"
        echo "  a100-40  - A100 40GB"
        echo "  h200     - H200 (3h limit only!)"
        echo "  nv       - V100/Titan/T4"
        exit 1
        ;;
esac

# Set partition (auto or override)
if [ -z "$PARTITION" ]; then
    PARTITION=$DEFAULT_PARTITION
fi

# Warn if time exceeds partition limit
if [ "$PARTITION" = "gpu" ] && [ "$TIME" != "3:00:00" ]; then
    echo "⚠️  Adjusting time to 3:00:00 (gpu partition limit)"
    TIME="3:00:00"
fi

# Generate run name if not specified
if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${CONFIG}_${GPU_TYPE}_${TIMESTAMP}"
fi

# Generate job name
JOB_NAME="plr-${CONFIG}-${GPU_TYPE}"

#-------------------------------------------------------------------------------
# Build Hydra Override String
#-------------------------------------------------------------------------------
HYDRA_OVERRIDES=""

if [ -n "$NUM_UPDATES" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES num_updates=$NUM_UPDATES"
fi

if [ -n "$SEED" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES seed=$SEED"
fi

HYDRA_OVERRIDES="$HYDRA_OVERRIDES run_name=$RUN_NAME"

if [ "$USE_WANDB" = true ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES use_wandb=true wandb_project=$WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        HYDRA_OVERRIDES="$HYDRA_OVERRIDES wandb_entity=$WANDB_ENTITY"
    fi
else
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES use_wandb=false"
fi

# Add any extra overrides from command line
HYDRA_OVERRIDES="$HYDRA_OVERRIDES $EXTRA_OVERRIDES"

#-------------------------------------------------------------------------------
# Create Logs Directory
#-------------------------------------------------------------------------------
mkdir -p logs

#-------------------------------------------------------------------------------
# Generate SLURM Script
#-------------------------------------------------------------------------------
SLURM_SCRIPT=$(mktemp /tmp/slurm_plr_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm_%j_${JOB_NAME}.out
#SBATCH --error=logs/slurm_%j_${JOB_NAME}.err
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --mem=${MEM}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0425887@u.nus.edu

SLURM_EOF

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'

#===============================================================================
# Job Execution
#===============================================================================

echo "============================================================"
echo "PushWorld PLR Training Job"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "Working Dir:   $(pwd)"
echo "============================================================"

# Change to project directory
cd ~/pushworld-experiments || { echo "Error: ~/pushworld-experiments not found"; exit 1; }

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set up JAX to use GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Print Python/JAX info
echo "Python: $(which python) ($(python --version 2>&1))"
echo ""

# Verify JAX can see GPU
echo "Checking JAX GPU support..."
python -c "import jax; devices = jax.devices(); print(f'JAX devices: {devices}'); assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices), 'No GPU found! Install JAX with CUDA: pip install -U \"jax[cuda12]\"'"
echo ""

SLURM_EOF

# Add the training command
cat >> "$SLURM_SCRIPT" << EOF

echo "Training Configuration:"
echo "  Script:      ${SCRIPT}.py"
echo "  Config:      ${CONFIG}"
echo "  Run Name:    ${RUN_NAME}"
echo "  GPU:         ${GPU_TYPE}"
echo "  Overrides:   ${HYDRA_OVERRIDES}"
echo ""

echo "Starting training..."

python experiments/${SCRIPT}.py \\
    --config-name=${CONFIG} \\
    ${HYDRA_OVERRIDES}

EOF

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
SLURM_EOF

#-------------------------------------------------------------------------------
# Submit or Display
#-------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SLURM Job Configuration"
echo "============================================================"
echo "Script:       ${SCRIPT}.py"
echo "GPU:          ${GPU_TYPE}"
echo "Partition:    ${PARTITION}"
echo "GRES:         ${SLURM_GRES}"
echo "Memory:       ${MEM}"
echo "Time:         ${TIME}"
echo "Config:       ${CONFIG}"
echo "Run Name:     ${RUN_NAME}"
echo "Wandb:        ${USE_WANDB}"
echo "Overrides:    ${HYDRA_OVERRIDES}"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "=== Generated SLURM Script (dry run) ==="
    echo ""
    cat "$SLURM_SCRIPT"
    echo ""
    echo "=== End of Script ==="
    rm "$SLURM_SCRIPT"
else
    echo "Submitting job..."
    JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        echo ""
        echo "✓ Job submitted successfully!"
        echo "  Job ID: $JOB_ID"
        echo ""
        echo "Useful commands:"
        echo "  squeue -j $JOB_ID              # Check job status"
        echo "  scancel $JOB_ID                # Cancel job"
        echo "  tail -f logs/slurm_${JOB_ID}_${JOB_NAME}.out  # Watch output"
        echo "  tail -f logs/slurm_${JOB_ID}_${JOB_NAME}.err  # Watch errors"
        echo ""
        
        # Save the script for reference
        cp "$SLURM_SCRIPT" "logs/submitted_${JOB_ID}.sh"
        echo "Script saved to: logs/submitted_${JOB_ID}.sh"
    else
        echo "Error: Failed to submit job"
        cat "$SLURM_SCRIPT"
        rm "$SLURM_SCRIPT"
        exit 1
    fi
    
    rm "$SLURM_SCRIPT"
fi
