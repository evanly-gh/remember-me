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
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6k:4
#SBATCH --time=02:00:00
#SBATCH --output=logs/beauty-train-%j.out
#SBATCH --error=logs/beauty-train-%j.err

set -euo pipefail

ACCOUNT="${ACCOUNT:-intelligentsystems}"
ENV_PREFIX="${ENV_PREFIX:-/gscratch/${ACCOUNT}/${USER}/conda_envs/beauty-train}"

# Resolve paths from the directory where you ran `sbatch` (not $HOME/HCP).
BEAUTY_DIR="${BEAUTY_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$BEAUTY_DIR" || ! -f "$BEAUTY_DIR/train.py" ]]; then
  for candidate in \
    "$HOME/remember-me/training/beauty" \
    "$HOME/HCP/training/beauty"; do
    if [[ -f "$candidate/train.py" ]]; then
      BEAUTY_DIR="$candidate"
      break
    fi
  done
fi
if [[ ! -f "${BEAUTY_DIR:-}/train.py" ]]; then
  echo "ERROR: cannot find training/beauty (no train.py). Set BEAUTY_DIR or run sbatch from that folder."
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$BEAUTY_DIR/../.." && pwd)}"
if [[ -z "${DATA_ROOT:-}" ]]; then
  for candidate in \
    "/gscratch/${ACCOUNT}/${USER}/data/SCUT-FBP5500" \
    "$REPO_ROOT/data/SCUT-FBP5500"; do
    if [[ -d "$candidate/Images" ]]; then
      DATA_ROOT="$candidate"
      break
    fi
  done
  DATA_ROOT="${DATA_ROOT:-/gscratch/${ACCOUNT}/${USER}/data/SCUT-FBP5500}"
fi
OUT_CKPT="${OUT_CKPT:-$BEAUTY_DIR/checkpoints/beauty_regressor.pt}"

cd "$BEAUTY_DIR"
mkdir -p logs checkpoints

echo "Beauty dir: $BEAUTY_DIR"
echo "Repo root:  $REPO_ROOT"
echo "Data root:  $DATA_ROOT"

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

python -c "import torch; print(f'torch {torch.__version__}, {torch.cuda.device_count()} CUDA device(s)')"

if [[ ! -d "$DATA_ROOT/Images" ]]; then
  echo "ERROR: dataset not found at $DATA_ROOT"
  exit 1
fi

# 4× GPU DDP: per-GPU batch 32 → global 128. Linear LR scale vs prior 1×GPU batch-64 @ 1e-4.
NPROC="${NPROC:-4}"
IMG_SIZE="${IMG_SIZE:-256}"
EPOCHS="${EPOCHS:-50}"
WARMUP="${WARMUP:-5}"
PER_GPU_BS="${PER_GPU_BS:-32}"
LR="${LR:-2e-4}"
NUM_WORKERS=$(( ( ${SLURM_CPUS_PER_TASK:-32} + NPROC - 1) / NPROC ))

torchrun --standalone --nproc_per_node="$NPROC" train.py \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --warmup-epochs "$WARMUP" \
  --batch-size "$PER_GPU_BS" \
  --lr "$LR" \
  --img-size "$IMG_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --ema-decay 0.99 \
  --backbone resnet50 \
  --out "$OUT_CKPT"

echo "Done: $(date)"
echo "Checkpoint: $OUT_CKPT"
echo "Deploy: set BEAUTY_IMG_SIZE=$IMG_SIZE in face-service if not using the default 224."
