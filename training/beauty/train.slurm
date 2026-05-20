#!/bin/bash
#
# Train beauty regressor on Hyak (klone).
#
# Prerequisite: bash setup_hyak.sh --run  (on a compute node via srun)
#
# Submit from this directory on a login node:
#   mkdir -p logs checkpoints
#   sbatch train.slurm
#
#SBATCH --job-name=beauty-train
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6k
#SBATCH --time=01:30:00
#SBATCH --output=logs/beauty-train-%j.out
#SBATCH --error=logs/beauty-train-%j.err

set -euo pipefail

ACCOUNT="${ACCOUNT:-intelligentsystems}"
REPO_ROOT="${REPO_ROOT:-$HOME/HCP}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/SCUT-FBP5500}"
OUT_CKPT="${OUT_CKPT:-$REPO_ROOT/training/beauty/checkpoints/beauty_regressor.pt}"
ENV_PREFIX="${ENV_PREFIX:-/gscratch/${ACCOUNT}/${USER}/conda_envs/beauty-train}"

cd "$REPO_ROOT/training/beauty"
mkdir -p logs checkpoints

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date)"
echo "Env: $ENV_PREFIX"

module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA torch not available. Run setup on a compute node:"
  echo "  srun -A $ACCOUNT -p gpu-rtx6k --gres=gpu:rtx6k --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash"
  echo "  cd $REPO_ROOT/training/beauty && bash setup_hyak.sh --run"
  exit 1
fi

python -c "import torch; print(f'torch {torch.__version__}, {torch.cuda.get_device_name(0)}')"

if [[ ! -d "$DATA_ROOT/Images" ]]; then
  echo "ERROR: dataset not found at $DATA_ROOT"
  exit 1
fi

python train.py \
  --data-root "$DATA_ROOT" \
  --epochs 25 \
  --batch-size 64 \
  --lr 1e-4 \
  --backbone resnet50 \
  --out "$OUT_CKPT"

echo "Done: $(date)"
echo "Checkpoint: $OUT_CKPT"
