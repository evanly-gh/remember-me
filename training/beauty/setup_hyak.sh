#!/bin/bash
#
# One-time environment setup for Hyak klone.
# Conda is NOT available on login nodes — run this via srun (see below).
#
# From a login node:
#   cd ~/path/to/HCP/training/beauty
#   bash setup_hyak.sh          # prints the srun command, then run that
#
# Or non-interactive (waits for a GPU node, ~5–15 min queue):
#   sbatch --account=intelligentsystems -p gpu-rtx6k --gres=gpu:rtx6k \
#     --cpus-per-task=4 --mem=16G --time=01:00:00 --wrap "bash setup_hyak.sh --run"

set -euo pipefail

ACCOUNT="${ACCOUNT:-intelligentsystems}"
PARTITION="${PARTITION:-gpu-rtx6k}"
GSCRATCH="${GSCRATCH:-/gscratch/${ACCOUNT}/${USER}}"
ENV_PREFIX="${ENV_PREFIX:-${GSCRATCH}/conda_envs/beauty-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_setup() {
  echo "=== Hyak beauty-train env setup on $(hostname) ==="

  if ! command -v module &>/dev/null; then
    echo "ERROR: 'module' not found. Are you on a compute node (not klone-login)?"
    exit 1
  fi

  module load conda

  mkdir -p "$(dirname "$ENV_PREFIX")"
  if [[ ! -d "$ENV_PREFIX" ]]; then
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"

  pip install --upgrade pip
  pip install -r "$SCRIPT_DIR/requirements.txt"
  # CPU-only torch from PyPI is useless on GPU nodes — install CUDA build.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo ""
  echo "Done. Env: $ENV_PREFIX"
  echo "Submit training: cd $SCRIPT_DIR && sbatch train.slurm"
}

if [[ "${1:-}" == "--run" ]]; then
  run_setup
  exit 0
fi

cat <<EOF
Hyak klone: conda only works on COMPUTE nodes, not login nodes.

Run setup in an interactive GPU shell (recommended):

  srun --account=${ACCOUNT} -p ${PARTITION} --gres=gpu:rtx6k \\
    --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

Then on the compute node:

  cd $(pwd)
  bash setup_hyak.sh --run

Env will be created at: ${ENV_PREFIX}
EOF
